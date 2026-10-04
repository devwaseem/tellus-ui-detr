from __future__ import annotations

import copy
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import UIExample, encode_targets
from .model import ParallelUIGenerator


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    empty_class_weight: float = 1.0
    parent_loss_weight: float = 1.0
    patience: int = 100
    max_length: int = 128
    seed: int = 7


@dataclass(frozen=True, slots=True)
class Metrics:
    loss: float
    exact_tree_accuracy: float
    component_f1: float
    active_type_accuracy: float
    parent_accuracy: float


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    train: Metrics
    validation: Metrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best_epoch: int
    train: Metrics
    validation: Metrics


@dataclass(frozen=True, slots=True)
class CachedExample:
    memory: torch.Tensor
    attention_mask: torch.Tensor
    type_ids: torch.Tensor
    parent_ids: torch.Tensor


EpochCallback = Callable[[EpochResult], None]


def cache_examples(
    generator: ParallelUIGenerator,
    examples: Sequence[UIExample],
    *,
    batch_size: int = 8,
    max_length: int = 128,
) -> tuple[CachedExample, ...]:
    """Run the frozen decoder once and retain prompt features on CPU."""
    cached: list[CachedExample] = []
    for start in range(0, len(examples), batch_size):
        example_batch = examples[start : start + batch_size]
        inputs = generator.tokenize(
            [example.prompt for example in example_batch], max_length=max_length
        )
        memory_batch = generator.encode(
            inputs["input_ids"], inputs["attention_mask"]
        )
        for index, example in enumerate(example_batch):
            length = int(inputs["attention_mask"][index].sum().item())
            type_ids, parent_ids = encode_targets(
                example,
                component_types=generator.component_types,
                num_slots=generator.head.num_slots,
            )
            cached.append(
                CachedExample(
                    memory=memory_batch[index, :length].detach().cpu().contiguous(),
                    attention_mask=torch.ones(length, dtype=torch.long),
                    type_ids=type_ids,
                    parent_ids=parent_ids,
                )
            )
    return tuple(cached)


def fit_head(
    generator: ParallelUIGenerator,
    train_examples: Sequence[UIExample],
    validation_examples: Sequence[UIExample],
    *,
    config: TrainingConfig = TrainingConfig(),
    on_epoch: EpochCallback | None = None,
) -> TrainingResult:
    if not train_examples or not validation_examples:
        raise ValueError("training and validation examples must not be empty")

    torch.manual_seed(config.seed)
    train_cache = cache_examples(
        generator,
        train_examples,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    validation_cache = cache_examples(
        generator,
        validation_examples,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    optimizer = torch.optim.AdamW(
        generator.head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(generator.head.state_dict())
    best_score = (-1.0, float("-inf"))
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        generator.head.train()
        indices = list(range(len(train_cache)))
        random.Random(config.seed + epoch).shuffle(indices)
        for batch in _iter_batches(
            train_cache,
            batch_size=config.batch_size,
            device=generator.device,
            indices=indices,
        ):
            optimizer.zero_grad()
            type_logits, parent_logits = generator.head(
                batch.memory, batch.attention_mask
            )
            loss = _loss(
                type_logits,
                parent_logits,
                batch.type_ids,
                batch.parent_ids,
                empty_class_weight=config.empty_class_weight,
                parent_loss_weight=config.parent_loss_weight,
            )
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_cached(
            generator, train_cache, config=config
        )
        validation_metrics = evaluate_cached(
            generator, validation_cache, config=config
        )
        epoch_result = EpochResult(epoch, train_metrics, validation_metrics)
        if on_epoch is not None:
            on_epoch(epoch_result)

        validation_score = (
            validation_metrics.exact_tree_accuracy,
            -validation_metrics.loss,
        )
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(generator.head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    generator.head.load_state_dict(best_state)
    return TrainingResult(
        best_epoch=best_epoch,
        train=evaluate_cached(generator, train_cache, config=config),
        validation=evaluate_cached(generator, validation_cache, config=config),
    )


def evaluate_examples(
    generator: ParallelUIGenerator,
    examples: Sequence[UIExample],
    *,
    config: TrainingConfig = TrainingConfig(),
) -> Metrics:
    cached = cache_examples(
        generator,
        examples,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )
    return evaluate_cached(generator, cached, config=config)


@dataclass(frozen=True, slots=True)
class _Batch:
    memory: torch.Tensor
    attention_mask: torch.Tensor
    type_ids: torch.Tensor
    parent_ids: torch.Tensor


def _iter_batches(
    cached: Sequence[CachedExample],
    *,
    batch_size: int,
    device: torch.device,
    indices: Sequence[int] | None = None,
):
    ordered_indices = list(range(len(cached))) if indices is None else list(indices)
    for start in range(0, len(ordered_indices), batch_size):
        batch_examples = [
            cached[index] for index in ordered_indices[start : start + batch_size]
        ]
        max_tokens = max(example.memory.shape[0] for example in batch_examples)
        hidden_size = batch_examples[0].memory.shape[-1]
        memory = torch.zeros(
            len(batch_examples),
            max_tokens,
            hidden_size,
            dtype=batch_examples[0].memory.dtype,
            device=device,
        )
        attention_mask = torch.zeros(
            len(batch_examples), max_tokens, dtype=torch.long, device=device
        )
        for row, example in enumerate(batch_examples):
            length = example.memory.shape[0]
            memory[row, :length] = example.memory.to(device)
            attention_mask[row, :length] = example.attention_mask.to(device)
        yield _Batch(
            memory=memory,
            attention_mask=attention_mask,
            type_ids=torch.stack(
                [example.type_ids for example in batch_examples]
            ).to(device),
            parent_ids=torch.stack(
                [example.parent_ids for example in batch_examples]
            ).to(device),
        )



def _loss(
    type_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    type_ids: torch.Tensor,
    parent_ids: torch.Tensor,
    *,
    empty_class_weight: float,
    parent_loss_weight: float,
) -> torch.Tensor:
    class_weights = torch.ones(
        type_logits.shape[-1], dtype=type_logits.dtype, device=type_logits.device
    )
    class_weights[0] = empty_class_weight
    component_loss = nn.functional.cross_entropy(
        type_logits.flatten(0, 1), type_ids.flatten(), weight=class_weights
    )
    slot_ids = torch.arange(type_ids.shape[1], device=type_ids.device)
    active_children = (type_ids != 0) & (slot_ids.unsqueeze(0) != 0)
    if not active_children.any():
        return component_loss
    parent_loss = nn.functional.cross_entropy(
        parent_logits[active_children], parent_ids[active_children]
    )
    return component_loss + parent_loss_weight * parent_loss


@torch.inference_mode()
def evaluate_cached(
    generator: ParallelUIGenerator,
    cached: Sequence[CachedExample],
    *,
    config: TrainingConfig,
) -> Metrics:
    generator.head.eval()
    total_loss = 0.0
    example_count = 0
    exact_count = 0
    correct_components = 0
    predicted_components = 0
    target_components = 0
    correct_active_types = 0
    correct_parents = 0
    active_children_count = 0

    for batch in _iter_batches(
        cached,
        batch_size=config.batch_size,
        device=generator.device,
    ):
        type_logits, parent_logits = generator.head(
            batch.memory, batch.attention_mask
        )
        batch_size = batch.type_ids.shape[0]
        total_loss += _loss(
            type_logits,
            parent_logits,
            batch.type_ids,
            batch.parent_ids,
            empty_class_weight=config.empty_class_weight,
            parent_loss_weight=config.parent_loss_weight,
        ).item() * batch_size
        example_count += batch_size

        predicted_types = type_logits.argmax(dim=-1)
        predicted_parents = parent_logits.argmax(dim=-1)
        target_active = batch.type_ids != 0
        predicted_active = predicted_types != 0
        correct_type = predicted_types == batch.type_ids
        slot_ids = torch.arange(batch.type_ids.shape[1], device=generator.device)
        active_children = target_active & (slot_ids.unsqueeze(0) != 0)
        correct_parent = predicted_parents == batch.parent_ids

        exact_types = correct_type.all(dim=1)
        exact_parents = (correct_parent | ~active_children).all(dim=1)
        exact_count += (exact_types & exact_parents).sum().item()
        correct_components += (correct_type & target_active).sum().item()
        predicted_components += predicted_active.sum().item()
        target_components += target_active.sum().item()
        correct_active_types += (correct_type & target_active).sum().item()
        correct_parents += (correct_parent & active_children).sum().item()
        active_children_count += active_children.sum().item()

    precision = _safe_ratio(correct_components, predicted_components)
    recall = _safe_ratio(correct_components, target_components)
    component_f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return Metrics(
        loss=_safe_ratio(total_loss, example_count),
        exact_tree_accuracy=_safe_ratio(exact_count, example_count),
        component_f1=component_f1,
        active_type_accuracy=_safe_ratio(correct_active_types, target_components),
        parent_accuracy=_safe_ratio(correct_parents, active_children_count),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
