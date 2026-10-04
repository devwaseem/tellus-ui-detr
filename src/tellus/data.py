from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .model import DEFAULT_COMPONENT_TYPES

DatasetSplit = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class ComponentTarget:
    type: str
    parent: int | None


@dataclass(frozen=True, slots=True)
class UIExample:
    prompt: str
    components: tuple[ComponentTarget, ...]
    input_count: int
    button_count: int


_COUNTS = tuple(
    (input_count, button_count)
    for input_count in range(1, 4)
    for button_count in range(1, 3)
)

_TEMPLATES: dict[DatasetSplit, tuple[str, ...]] = {
    "train": (
        "Create a card containing a form with {inputs} and {buttons}.",
        "Wrap a form that has {inputs} plus {buttons} inside a card.",
        "Build a card, then place a form with {inputs} and {buttons} inside it.",
        "Make a card whose child is a form containing {inputs} and {buttons}.",
        "Put {inputs} and {buttons} in a form nested under a card.",
        "Render a card around a form made of {inputs} followed by {buttons}.",
        "Add a card with a form child; the form needs {inputs} and {buttons}.",
        "The root should be a card with a form that contains {inputs} and {buttons}.",
        "Place {inputs} and {buttons} directly inside a card.",
        "Build a card containing {inputs} together with {buttons}.",
        "Use a card to group {inputs} plus {buttons}.",
        "The card needs to hold {inputs}, followed by {buttons}.",
        "Put {inputs} beside {buttons} under a card container.",
        "Render {inputs} and {buttons} as controls within one card.",
    ),
    "validation": (
        "Inside a card, nest a form consisting of {inputs} together with {buttons}.",
        "Use a card as the outer container and give its form {inputs} and {buttons}.",
        "Produce a card > form hierarchy with {inputs} and {buttons} in the form.",
        "Give a card {inputs} as well as {buttons}.",
        "The outer card contains {inputs} followed by {buttons}.",
    ),
    "test": (
        "create a form with {inputs} and {buttons} wrapped in a card",
        "A card should surround a form holding {inputs} alongside {buttons}.",
        "I need {inputs} and {buttons}; group them in a form and put that in a card.",
        "Construct the tree Card then Form, with the form owning {inputs} and {buttons}.",
        "Make one card that holds {inputs} and {buttons}.",
        "Within a card, include {inputs} plus {buttons}.",
        "Group {inputs} with {buttons} beneath a card.",
    ),
}

_INPUT_PHRASES: dict[DatasetSplit, dict[int, tuple[str, ...]]] = {
    "train": {
        1: ("one input", "a single text field"),
        2: ("two inputs", "a pair of text fields"),
        3: ("three inputs", "three text fields"),
    },
    "validation": {
        1: ("one text input",),
        2: ("2 input controls",),
        3: ("a trio of inputs",),
    },
    "test": {
        1: ("one input", "exactly one field"),
        2: ("two text inputs", "exactly 2 fields"),
        3: ("3 input controls", "exactly three fields"),
    },
}

_BUTTON_PHRASES: dict[DatasetSplit, dict[int, tuple[str, ...]]] = {
    "train": {
        1: ("a button", "one action button"),
        2: ("two buttons", "a pair of action buttons"),
    },
    "validation": {
        1: ("one button",),
        2: ("2 button controls",),
    },
    "test": {
        1: ("a button", "exactly one button"),
        2: ("two action buttons", "exactly 2 buttons"),
    },
}


def canonical_form_tree(
    input_count: int, button_count: int
) -> tuple[ComponentTarget, ...]:
    if input_count < 1 or button_count < 1:
        raise ValueError("form examples require at least one input and one button")

    return (
        ComponentTarget("Card", None),
        ComponentTarget("Form", 0),
        *(ComponentTarget("TextInput", 1) for _ in range(input_count)),
        *(ComponentTarget("Button", 1) for _ in range(button_count)),
    )


def build_form_examples(split: DatasetSplit) -> tuple[UIExample, ...]:
    templates = _TEMPLATES[split]
    examples: list[UIExample] = []
    for template_index, template in enumerate(templates):
        for input_count, button_count in _COUNTS:
            input_variants = _INPUT_PHRASES[split][input_count]
            button_variants = _BUTTON_PHRASES[split][button_count]
            prompt = template.format(
                inputs=input_variants[template_index % len(input_variants)],
                buttons=button_variants[template_index % len(button_variants)],
            )
            examples.append(
                UIExample(
                    prompt=prompt,
                    components=canonical_form_tree(input_count, button_count),
                    input_count=input_count,
                    button_count=button_count,
                )
            )
    if split == "test":
        examples.append(
            UIExample(
                prompt="create a card with three inputs and two buttons",
                components=canonical_form_tree(3, 2),
                input_count=3,
                button_count=2,
            )
        )
    return tuple(examples)


def encode_targets(
    example: UIExample,
    *,
    component_types: tuple[str, ...] = DEFAULT_COMPONENT_TYPES,
    num_slots: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(example.components) > num_slots:
        raise ValueError(
            f"example needs {len(example.components)} slots, but only {num_slots} exist"
        )

    type_to_id = {name: index for index, name in enumerate(component_types)}
    type_ids = torch.zeros(num_slots, dtype=torch.long)
    parent_ids = torch.zeros(num_slots, dtype=torch.long)
    for slot, component in enumerate(example.components):
        if component.type not in type_to_id:
            raise ValueError(f"unknown component type: {component.type}")
        if slot == 0 and component.parent is not None:
            raise ValueError("slot 0 must be the root")
        if slot > 0 and (component.parent is None or component.parent >= slot):
            raise ValueError("every non-root parent must precede its child")
        type_ids[slot] = type_to_id[component.type]
        if component.parent is not None:
            parent_ids[slot] = component.parent
    return type_ids, parent_ids
