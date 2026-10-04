# Tellus

Tellus uses a frozen decoder-only language model as prompt memory and predicts every
UI slot and parent pointer in one parallel head pass. The default backbone is
`Qwen/Qwen2.5-Coder-1.5B-Instruct`; it does not autoregressively generate UI tokens.

The bundled experiment teaches a small structural grammar without form-intent labels:

```text
Card
└── Form
    ├── TextInput × 1–3
    └── Button × 1–2
```

Training descriptions state the requested structure directly. Validation and test
sets use held-out sentence templates and wording.

Train the parallel head and save its best validation checkpoint:

```bash
uv run python train.py
```

Generate from a raw structural description:

```bash
uv run tellus \
  "create a card with three inputs and two buttons" \
  --head tellus-head.pt
```

The form can be implicit: this produces `Card(parent=null)`, `Form(parent=0)`,
three `TextInput(parent=1)` nodes, and two `Button(parent=1)` nodes. Training also
reports exact-tree metrics for every input/button count combination.

Use `--model-id` on both commands to choose another decoder-only checkpoint. A head
checkpoint is tied to the backbone and component vocabulary it was trained with.