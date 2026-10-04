from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_COMPONENT_TYPES = (
    "EMPTY",
    "AppShell",
    "Card",
    "CardHeader",
    "CardContent",
    "Form",
    "Button",
    "TextInput",
    "DataTable",
    "MetricCard",
    "Badge",
)


@dataclass(frozen=True, slots=True)
class UIComponent:
    slot: int
    type: str
    parent: int | None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


class ParallelSlotDecoder(nn.Module):
    """Predict a fixed set of UI nodes and their parents in one parallel pass."""

    def __init__(
        self,
        d_in: int,
        d_slot: int = 256,
        num_slots: int = 16,
        num_classes: int = len(DEFAULT_COMPONENT_TYPES),
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_slot % num_heads:
            raise ValueError("d_slot must be divisible by num_heads")

        self.num_slots = num_slots
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_slot))
        self.proj_kv = nn.Linear(d_in, d_slot)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_slot,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_slot)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_slot,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_slot)
        self.ffn = nn.Sequential(
            nn.Linear(d_slot, d_slot * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_slot * 2, d_slot),
        )
        self.norm3 = nn.LayerNorm(d_slot)
        self.type_classifier = nn.Linear(d_slot, num_classes)
        self.parent_head = nn.Linear(d_slot, d_slot)

        # Canonical trees are parent-first, so slot i can only point to slots < i.
        invalid_parents = torch.triu(
            torch.ones(num_slots, num_slots, dtype=torch.bool), diagonal=0
        )
        invalid_parents[0, 0] = False  # Root parent is ignored; keep logits finite.
        self.register_buffer("invalid_parents", invalid_parents, persistent=False)

    def forward(
        self, memory_tokens: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = memory_tokens.size(0)
        kv = self.proj_kv(memory_tokens)
        slots = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        attended, _ = self.cross_attn(
            query=slots,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.norm1(slots + attended)
        mixed, _ = self.self_attn(
            query=hidden, key=hidden, value=hidden, need_weights=False
        )
        hidden = self.norm2(hidden + mixed)
        hidden = self.norm3(hidden + self.ffn(hidden))

        type_logits = self.type_classifier(hidden)
        parent_queries = self.parent_head(hidden)
        parent_logits = torch.bmm(parent_queries, hidden.transpose(1, 2))
        parent_logits = parent_logits.masked_fill(
            self.invalid_parents.unsqueeze(0), float("-inf")
        )
        return type_logits, parent_logits


class ParallelUIGenerator(nn.Module):
    """A decoder-only language backbone with a parallel structured-UI head."""

    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        model_id: str,
        component_types: Sequence[str] = DEFAULT_COMPONENT_TYPES,
        d_slot: int = 256,
        num_slots: int = 16,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.component_types = tuple(component_types)
        self.freeze_backbone = freeze_backbone
        self.head = ParallelSlotDecoder(
            d_in=backbone.config.hidden_size,
            d_slot=d_slot,
            num_slots=num_slots,
            num_classes=len(self.component_types),
        ).to(self.device)

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype = "auto",
        component_types: Sequence[str] = DEFAULT_COMPONENT_TYPES,
        d_slot: int = 256,
        num_slots: int = 16,
        freeze_backbone: bool = True,
        head_path: str | Path | None = None,
        **model_kwargs: Any,
    ) -> ParallelUIGenerator:
        target_device = torch.device(device) if device is not None else default_device()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        if target_device.type == "mps":
            # Accelerate's direct MPS dispatch can crash while materializing weights.
            model_kwargs.pop("device_map", None)
        else:
            model_kwargs.setdefault("device_map", {"": str(target_device)})
        backbone = AutoModel.from_pretrained(
            model_id,
            dtype=dtype,
            **model_kwargs,
        )
        if target_device.type == "mps":
            backbone.to(target_device)
        generator = cls(
            backbone,
            tokenizer,
            model_id=model_id,
            component_types=component_types,
            d_slot=d_slot,
            num_slots=num_slots,
            freeze_backbone=freeze_backbone,
        )
        if head_path is not None:
            generator.load_head(head_path)
        return generator

    def train(self, mode: bool = True) -> ParallelUIGenerator:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def tokenize(
        self, prompts: Sequence[str], *, max_length: int = 512
    ) -> dict[str, torch.Tensor]:
        formatted_prompts = list(prompts)
        if getattr(self.tokenizer, "chat_template", None):
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in formatted_prompts
            ]
        encoded = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {name: tensor.to(self.device) for name, tensor in encoded.items()}

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            output = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        return output.last_hidden_state.to(self.head.proj_kv.weight.dtype)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.encode(input_ids, attention_mask)
        return self.head(memory, attention_mask)

    @torch.inference_mode()
    def generate(
        self, prompts: str | Sequence[str], *, max_length: int = 512
    ) -> list[list[UIComponent]]:
        prompt_batch = [prompts] if isinstance(prompts, str) else list(prompts)
        head_was_training = self.head.training
        self.head.eval()
        try:
            inputs = self.tokenize(prompt_batch, max_length=max_length)
            type_logits, parent_logits = self(**inputs)
            return self.decode(type_logits, parent_logits)
        finally:
            self.head.train(head_was_training)

    def decode(
        self, type_logits: torch.Tensor, parent_logits: torch.Tensor
    ) -> list[list[UIComponent]]:
        type_ids = type_logits.argmax(dim=-1).tolist()
        parent_ids = parent_logits.argmax(dim=-1).tolist()
        batches: list[list[UIComponent]] = []
        for batch_types, batch_parents in zip(type_ids, parent_ids, strict=True):
            components = [
                UIComponent(
                    slot=slot,
                    type=self.component_types[type_id],
                    parent=None if slot == 0 else batch_parents[slot],
                )
                for slot, type_id in enumerate(batch_types)
                if type_id != 0
            ]
            batches.append(components)
        return batches

    def save_head(self, path: str | Path) -> None:
        torch.save(
            {
                "model_id": self.model_id,
                "component_types": self.component_types,
                "num_slots": self.head.num_slots,
                "d_slot": self.head.slot_queries.shape[-1],
                "state_dict": self.head.state_dict(),
            },
            path,
        )

    def load_head(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint["model_id"] != self.model_id:
            raise ValueError(
                f"Head was trained for {checkpoint['model_id']!r}, not {self.model_id!r}"
            )
        if tuple(checkpoint["component_types"]) != self.component_types:
            raise ValueError("Head component types do not match this generator")
        self.head.load_state_dict(checkpoint["state_dict"])


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
