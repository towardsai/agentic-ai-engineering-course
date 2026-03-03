---
name: transfer_style_by_examples
description: Used to transfer the style to a given diagram based on a set of few shot examples only (no separate style references).
---

# Style Transfer by Examples: Unstyled Diagram → Branded Diagram

Restyle a raw diagram to match a visual brand using Nano Banana (Gemini image generation) with a few-shot approach. The brand is defined entirely by the example input/output pairs — no separate style reference images are needed.

## Prerequisites

**First time only** — install dependencies (if not already in `pyproject.toml`):

```bash
uv add google-genai Pillow
```

**Required** — set your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

Get a free key at https://aistudio.google.com/apikey

## Folder Structure

```
transfer_style_by_examples/
├── example_1/               # Few-shot pair 1
│   ├── input.png            # Raw unstyled diagram
│   └── output.png           # Correctly styled result
├── example_2/               # Few-shot pair 2
│   ├── input.png
│   └── output.png
├── not_example_1/           # Negative few-shot pair 1 (optional)
│   ├── input.png            # Raw unstyled diagram
│   └── output.png           # INCORRECT styled result (what to avoid)
├── scripts/
│   └── style_transfer.py
└── SKILL.md
```

### `example_*/` (required — at least one)

Each folder contains an `input.png` (raw unstyled diagram) and `output.png` (correctly styled result). These few-shot pairs teach the model the exact transformation: what the raw input looks like and what the expected output should be. The output side of each pair also defines the target visual brand. At least one example pair must always be present. Add more `example_3/`, `example_4/`, etc. as needed — they are auto-discovered by the `example_` prefix.

### `not_example_*/` (optional)

Each folder contains an `input.png` (raw unstyled diagram) and `output.png` (an INCORRECT styled result). These negative few-shot pairs teach the model what to avoid — wrong colors, wrong fonts, wrong shapes, or other deviations from the brand. Add `not_example_1/`, `not_example_2/`, etc. as needed — they are auto-discovered by the `not_example_` prefix. When a generated result has a recurring flaw, save it as a negative example to prevent the same mistake in future runs.

## Usage

**Basic — use bundled examples:**

```bash
uv run python scripts/style_transfer.py <target_diagram.png>
```

Output is saved as `<target_diagram>_styled.png` in the same directory.

**Custom examples directory:**

```bash
uv run python scripts/style_transfer.py <target_diagram.png> --examples-dir /path/to/my/examples
```

**Specify output path:**

```bash
uv run python scripts/style_transfer.py <target_diagram.png> --output my_styled_diagram.png
```

**Guide the transfer with custom instructions:**

```bash
uv run python scripts/style_transfer.py <target_diagram.png> --user-input "use a left-to-right layout and make the title larger"
```

**Generate multiple variations to pick from:**

```bash
uv run python scripts/style_transfer.py <target_diagram.png> --num 3
```

Output is saved as `<target_diagram>_styled_1.png`, `_styled_2.png`, `_styled_3.png`. Requests run in parallel.

**Refine a previously generated result:**

```bash
uv run python scripts/style_transfer.py <target_diagram.png> --previously-generated <target_diagram_styled.png> --user-input "make the arrows thicker and use rounded corners"
```

Output is saved as `<target_diagram>_restyled.png`. The model treats the previous output as a starting point and applies only the requested changes.

**Use a different model:**

```bash
uv run python scripts/style_transfer.py <target_diagram.png> --model gemini-2.5-flash-image
```

## How It Works

1. Auto-discovers `example_*/` folders and loads each input/output pair (at least one required).
2. Auto-discovers `not_example_*/` folders and loads each input/bad-output pair (optional).
3. Sends everything to the Gemini API in a structured multimodal prompt:
   - **Example transformations** — "here's input → output, learn the pattern and the brand"
   - **Negative examples** (if any) — "here's input → bad output, avoid these mistakes"
   - **Target** — "now transform this one"
   - **User instructions** (optional) — additional guidance for the transfer
4. Saves the generated styled diagram.

The prompt preserves all text, connections, and diagram structure — only the visual style changes.

## Image Budget

Each request sends: (2 × example pairs) + (2 × negative pairs) + 1 target. Gemini Pro supports ~16 images per request. Without style references, there is more room for example pairs compared to the original skill.

## Workflow

When the user asks to restyle a diagram:

1. Read the example pairs to understand the current brand.
2. Run the script: `uv run python scripts/style_transfer.py <path_to_target>` (add `--user-input "..."` if the user gave specific guidance).
3. If the result needs tweaking, use `--previously-generated <styled.png> --user-input "..."` for minimal refinements, or re-run from scratch with `--user-input` for bigger changes.
4. **Harvesting examples**: When the user is happy with a result, offer to save it as a new example pair (`example_N/input.png` + `example_N/output.png`) to improve future runs.
5. **Harvesting negative examples**: When a result has a recurring flaw, offer to save the original input and the bad output as a negative example pair (`not_example_N/input.png` + `not_example_N/output.png`) to prevent the same mistake in future runs.
