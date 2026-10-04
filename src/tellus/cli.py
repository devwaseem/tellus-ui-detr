from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .model import DEFAULT_MODEL_ID, ParallelUIGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tellus",
        description="Generate a structured UI in one parallel model pass.",
    )
    parser.add_argument("prompt", help="Natural-language UI description")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--head", help="Path to a trained parallel-head checkpoint")
    parser.add_argument("--device", help="Torch device; defaults to MPS, CUDA, then CPU")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.head is None:
        print(
            "warning: no --head checkpoint supplied; predictions use an untrained head",
            file=sys.stderr,
        )

    generator = ParallelUIGenerator.from_pretrained(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        head_path=args.head,
    )
    components = generator.generate(args.prompt, max_length=args.max_length)[0]
    print(json.dumps([component.to_dict() for component in components], indent=2))
