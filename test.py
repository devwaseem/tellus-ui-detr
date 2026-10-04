from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2Model

from tellus import (
    DEFAULT_COMPONENT_TYPES,
    ParallelUIGenerator,
    build_form_examples,
    encode_targets,
)


class TinyTokenizer:
    def __call__(
        self,
        prompts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        token_rows = [
            [ord(character) % 63 + 1 for character in prompt[:max_length]]
            for prompt in prompts
        ]
        width = max(map(len, token_rows))
        input_ids = torch.zeros((len(token_rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, tokens in enumerate(token_rows):
            input_ids[row, : len(tokens)] = torch.tensor(tokens)
            attention_mask[row, : len(tokens)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_generator() -> ParallelUIGenerator:
    config = Qwen2Config.from_dict(
        {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
        }
    )
    backbone = Qwen2Model(config)
    return ParallelUIGenerator(
        backbone,
        TinyTokenizer(),  # type: ignore[arg-type]
        model_id="tiny-random-qwen2",
        d_slot=32,
        num_slots=8,
    )


class StructuralFormDatasetTests(unittest.TestCase):
    def test_raw_prompt_maps_to_canonical_card_form_tree(self) -> None:
        example = next(
            example
            for example in build_form_examples("test")
            if example.prompt
            == "create a form with one input and a button wrapped in a card"
        )

        type_ids, parent_ids = encode_targets(example)
        active_types = [
            DEFAULT_COMPONENT_TYPES[type_id]
            for type_id in type_ids.tolist()
            if type_id != 0
        ]

        self.assertEqual(active_types, ["Card", "Form", "TextInput", "Button"])
        self.assertEqual(parent_ids[:4].tolist(), [0, 0, 1, 1])

    def test_reported_implicit_form_prompt_is_held_out(self) -> None:
        example = next(
            example
            for example in build_form_examples("test")
            if example.prompt == "create a card with three inputs and two buttons"
        )

        type_ids, parent_ids = encode_targets(example)
        active_types = [
            DEFAULT_COMPONENT_TYPES[type_id]
            for type_id in type_ids.tolist()
            if type_id != 0
        ]

        self.assertEqual(
            active_types,
            [
                "Card",
                "Form",
                "TextInput",
                "TextInput",
                "TextInput",
                "Button",
                "Button",
            ],
        )
        self.assertEqual(parent_ids[:7].tolist(), [0, 0, 1, 1, 1, 1, 1])


    def test_dataset_uses_structural_descriptions_not_form_intents(self) -> None:
        splits = {
            split: build_form_examples(split)
            for split in ("train", "validation", "test")
        }
        prompt_sets = {
            split: {example.prompt for example in examples}
            for split, examples in splits.items()
        }

        self.assertTrue(all("login" not in prompt.lower() for prompts in prompt_sets.values() for prompt in prompts))
        self.assertTrue(prompt_sets["train"].isdisjoint(prompt_sets["validation"]))
        self.assertTrue(prompt_sets["train"].isdisjoint(prompt_sets["test"]))
        self.assertTrue(prompt_sets["validation"].isdisjoint(prompt_sets["test"]))


class ParallelUIGeneratorTests(unittest.TestCase):
    def test_qwen_decoder_backbone_emits_every_slot_in_one_pass(self) -> None:
        generator = make_generator()
        inputs = generator.tokenize(["login form", "dashboard"])

        type_logits, parent_logits = generator(**inputs)

        self.assertEqual(type_logits.shape, (2, 8, len(DEFAULT_COMPONENT_TYPES)))
        self.assertEqual(parent_logits.shape, (2, 8, 8))

    def test_parent_logits_only_allow_earlier_slots(self) -> None:
        generator = make_generator()
        inputs = generator.tokenize(["login form"])

        _, parent_logits = generator(**inputs)

        self.assertTrue(torch.isfinite(parent_logits[:, 0, 0]).all())
        for child_slot in range(1, generator.head.num_slots):
            self.assertTrue(
                torch.isneginf(parent_logits[:, child_slot, child_slot:]).all()
            )
            self.assertTrue(
                torch.isfinite(parent_logits[:, child_slot, :child_slot]).all()
            )

    def test_frozen_backbone_does_not_receive_gradients(self) -> None:
        generator = make_generator()
        inputs = generator.tokenize(["login form"])

        type_logits, _ = generator(**inputs)
        type_logits.sum().backward()

        self.assertTrue(all(parameter.grad is None for parameter in generator.backbone.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in generator.head.parameters()))

    def test_head_checkpoint_round_trip(self) -> None:
        source = make_generator()
        target = make_generator()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            source.save_head(path)
            target.load_head(path)

        for source_parameter, target_parameter in zip(
            source.head.parameters(), target.head.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(source_parameter, target_parameter))


if __name__ == "__main__":
    unittest.main()
