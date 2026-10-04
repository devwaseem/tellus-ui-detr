from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from tellus import (
    DEFAULT_MODEL_ID,
    EpochResult,
    ParallelUIGenerator,
    TrainingConfig,
    build_form_examples,
    evaluate_examples,
    fit_head,
)

EXAMPLE_PROMPT = "create a card with three inputs and two buttons"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Tellus to compose Card > Form > controls from raw structural "
            "descriptions."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", help="Torch device; defaults to MPS, CUDA, then CPU")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("tellus-head.pt"))
    return parser


def train_structural_forms(
    *,
    model_id: str,
    device: str | None,
    dtype: str,
    config: TrainingConfig,
    output: Path,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    generator = ParallelUIGenerator.from_pretrained(
        model_id, device=device, dtype=dtype
    )
    train_examples = build_form_examples("train")
    validation_examples = build_form_examples("validation")
    test_examples = build_form_examples("test")
    print(
        f"dataset: {len(train_examples)} train / "
        f"{len(validation_examples)} validation / {len(test_examples)} test"
    )

    def report_epoch(result: EpochResult) -> None:
        if result.epoch == 1 or result.epoch % 5 == 0:
            print(
                f"epoch {result.epoch:03d} | "
                f"train exact {result.train.exact_tree_accuracy:.1%} | "
                f"validation exact {result.validation.exact_tree_accuracy:.1%} | "
                f"validation loss {result.validation.loss:.4f}"
            )

    result = fit_head(
        generator,
        train_examples,
        validation_examples,
        config=config,
        on_epoch=report_epoch,
    )
    test_metrics = evaluate_examples(generator, test_examples, config=config)
    test_by_shape = {}
    shapes = sorted(
        {(example.input_count, example.button_count) for example in test_examples}
    )
    for input_count, button_count in shapes:
        shape_examples = tuple(
            example
            for example in test_examples
            if example.input_count == input_count
            and example.button_count == button_count
        )
        shape_key = f"{input_count}_inputs_{button_count}_buttons"
        test_by_shape[shape_key] = asdict(
            evaluate_examples(generator, shape_examples, config=config)
        )
    generator.save_head(output)
    generated = generator.generate(EXAMPLE_PROMPT)[0]

    summary: dict[str, object] = {
        "checkpoint": str(output),
        "best_epoch": result.best_epoch,
        "train": asdict(result.train),
        "validation": asdict(result.validation),
        "test": asdict(test_metrics),
        "example_prompt": EXAMPLE_PROMPT,
        "test_by_shape": test_by_shape,
        "example_output": [component.to_dict() for component in generated],
    }
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    summary = train_structural_forms(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        config=config,
        output=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
