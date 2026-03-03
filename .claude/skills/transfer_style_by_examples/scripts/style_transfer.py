#!/usr/bin/env python3
"""
Style Transfer: Restyle an unstyled diagram to match a visual brand
using Google's Nano Banana (Gemini) image generation API.

Uses a few-shot approach with:
  1. Input/output example pairs (from example_*/ folders) — at least one required
  2. Negative example pairs (from not_example_*/ folders) — optional
  3. The target diagram to restyle

Usage:
    python style_transfer.py <target_image> [--examples-dir <dir>] [--output <path>] [--model <model>]

Examples:
    # Use bundled examples
    python style_transfer.py my_diagram.png

    # Use custom examples directory
    python style_transfer.py my_diagram.png --examples-dir /path/to/examples

    # Specify output path and model
    python style_transfer.py my_diagram.png --output styled_diagram.png --model gemini-3-pro-image-preview

Environment:
    GEMINI_API_KEY: Required. Get one at https://aistudio.google.com/apikey
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

SKILL_DIR = Path(__file__).resolve().parent.parent

STYLE_PROMPT = """\
You are a diagram styling expert. Your job is to restyle a raw, unstyled diagram to perfectly match a visual brand.

## What You Receive

1. **EXAMPLE TRANSFORMATIONS**: Pairs of (input → output) showing how raw diagrams were previously restyled into the brand. Study these carefully — they demonstrate exactly the transformation you must perform. The OUTPUT side of each pair defines the target visual brand.
2. **NEGATIVE EXAMPLES** (if provided): Pairs of (input → bad output) showing INCORRECT stylings that you must AVOID. These represent common mistakes — wrong colors, wrong fonts, wrong shapes, or other deviations from the brand. Study what went wrong in each one and make sure you do NOT repeat those mistakes.
3. **TARGET**: A raw, unstyled diagram that you must restyle to match the brand.

## What You Output

A single image: the TARGET diagram restyled to match the brand. Nothing else.

## ABSOLUTE RULE — USE ONLY OBSERVED BRAND ASSETS

**Every visual element you produce MUST come directly from what you observe in the EXAMPLE TRANSFORMATION outputs.** This includes:

- **Colors**: Use ONLY the exact colors (backgrounds, fills, borders, text, arrows) visible in the example outputs. Do NOT introduce, interpolate, or invent any color that does not appear in the references.
- **Fonts and typography**: Match the exact font family, font size, bold, italics, and all other text formatting you see. If titles are bold in the examples, they must be bold in your output. If labels use a specific size or weight, replicate it exactly. Do NOT substitute with different fonts, weights, or sizes.
- **Logos and graphics**: If the example outputs contain logos, icons, watermarks, or decorative graphics, reproduce them exactly. Do NOT omit, replace, or redesign any brand graphic.
- **Shapes and styling patterns**: Use ONLY the node shapes, border styles, arrow styles, and decorative patterns present in the example outputs. Do NOT invent new visual treatments.

**NEVER deviate from the observed brand assets.** If you are unsure about a specific visual property, pick the closest match from the example outputs — never guess or default to a generic style.

## GOLDEN RULE — STRUCTURE IS FROZEN

**You are ONLY changing the visual style. The diagram's structure is FROZEN.**

The output must be a pixel-perfect structural replica of the TARGET input. Every block, every arrow, every connection, every position, every piece of text must remain exactly the same. The ONLY thing that changes is the visual appearance (colors, fills, borders, fonts, arrow styling, background). Nothing else. NEVER deviate from this rule unless the user explicitly overrides it in the ADDITIONAL INSTRUCTIONS.

## CRITICAL — WHAT MUST STAY IDENTICAL TO THE TARGET

**BLOCKS/NODES**: Every single block/node from the TARGET MUST appear in your output. Same count, same content. Do NOT drop, merge, split, add, or invent any block.

**ARROWS/CONNECTIONS**: Every single arrow/connection from the TARGET MUST appear in your output. For each arrow:
- Same source node → same destination node (direction matters: A → B, NOT B → A).
- Same edge labels or annotations.
- Same connectivity logic (if A connects to B and C, your output must show A connecting to B and C).
Count the arrows in the TARGET. Count the arrows in your output. The numbers MUST be equal.

**LAYOUT/POSITIONS**: Every element must stay in the same spatial position. Do NOT reposition, reorder, reflow, or rearrange any node. If the TARGET flows top-to-bottom, your output flows top-to-bottom in the exact same order. If a node is on the left, it stays on the left.

**TEXT**: All text labels, node names, edge labels, titles, and annotations must stay character-for-character identical.

**BACKGROUND**: The background structure (e.g., sections, dividers, groupings) from the TARGET stays the same. Only the background COLOR changes to match the brand.

**DIAGRAM TYPE**: The diagram type (flowchart, sequence, architecture, etc.) must stay the same.

## STYLE EXTRACTION PROCESS

Before drawing anything, carefully study the output side of each example transformation. Extract the following properties. Every property MUST come from what you SEE in these images — do not invent, assume, or default to any value. If a property is not clearly visible, find the closest match from the example outputs. NEVER fall back to generic or default styles.

### Step 1: Extract the style

Go through each category below and write down (internally) what you observe:

**A. BACKGROUND**
- The canvas/background color MUST ALWAYS be solid white (#FFFFFF). This is non-negotiable regardless of what the example outputs or input use. Never use transparent, gray, dark, or any non-white background.

**B. COLOR PALETTE**
- List every distinct fill color used in nodes/blocks. Note which ROLE each color maps to (e.g., "primary action blocks use color X", "decision nodes use color Y", "start/end nodes use color Z").
- List the arrow/line color(s).
- List the text color(s) (on colored fills vs on the background).
- List any border/stroke colors.

**C. NODE / BLOCK SHAPES**
- For each node role you see (process, decision, start/end, container/group, small tag/label), note:
  - Shape (rectangle, rounded rectangle, stadium/pill, hexagon, diamond, circle)
  - Fill color (from palette above)
  - Border (visible or not, solid/dashed/dotted, color, thickness)
  - Corner rounding (sharp vs rounded)
  - Text color and weight inside the node

**D. ARROWS / CONNECTIONS**
- Color, thickness, and style (solid/dashed/dotted)
- Arrowhead shape (filled, open, none)
- Curvature: prefer smooth curves over straight lines or right-angle elbows whenever the layout allows it. Curved arrows look more polished and professional. Only use straight lines when a curve would be confusing (e.g., very short connections between adjacent nodes).
- Edge label style (font color, weight, placement)

**E. TEXT AND TYPOGRAPHY**
- Font family (sans-serif, serif, monospace)
- Title: exact size, weight (bold/regular), italics (yes/no), color, position (top-left, top-center, etc.)
- Node labels: exact size, weight, italics, color on filled blocks vs unfilled blocks
- Section/group headers: exact size, weight, italics, color, any special treatment
- For every text role, note whether it is bold, italic, or both — you MUST reproduce this exactly

**F. DECORATIVE ELEMENTS**
- Numbered step indicators (circles with numbers, etc.)
- Icons or emojis next to labels
- Any other recurring decorative patterns

**G. LAYOUT AND SPACING**
- Flow direction (top-to-bottom, left-to-right, etc.)
- Spacing density (tight vs airy)
- Overall visual feel (clean/minimal, dense/technical, etc.)

### Step 2: Learn from the example transformations

For each example pair, observe:
- How each element type in the raw input was mapped to a styled element in the output.
- How the layout was adjusted (repositioned, grouped, spaced).
- How the background, colors, shapes, and arrows changed.

### Step 3: Learn from the negative examples (if provided)

For each negative example, identify what went WRONG:
- Which colors, fonts, shapes, or styles deviate from the brand?
- What specific mistakes were made (wrong background, incorrect node shapes, mismatched typography, etc.)?
- Build an internal "DO NOT" list of these mistakes and actively avoid every one of them in your output.

### Step 4: Apply the extracted style to the target diagram

Map each element in the TARGET to its closest role (process, decision, start/end, container, etc.) and apply the corresponding extracted style from Steps 1 and 2, while avoiding every mistake identified in Step 3.

## SELF-CHECK (verify EVERY item before outputting)

**Structure — must be IDENTICAL to the TARGET (frozen):**
1. **BLOCK COUNT**: Count every block/node in the TARGET. Count in your output. Equal? → If no, REDO.
2. **ARROW COUNT**: Count every arrow in the TARGET. Count in your output. Equal? → If no, REDO.
3. **ARROW CONNECTIONS**: For each arrow, verify: same source node → same destination node, same direction, same edge label. → If any mismatch — REDO.
4. **LAYOUT**: Are all blocks in the same spatial positions and same order as the TARGET? → If anything moved — REDO.
5. **TEXT**: Is every label, title, and annotation character-for-character identical to the TARGET? → If any text differs — REDO.
6. **BACKGROUND STRUCTURE**: Are sections, groupings, and dividers in the same arrangement as the TARGET? → If anything shifted — REDO.

**Style — must match the brand references EXACTLY (no deviations):**
7. Is the background solid white (#FFFFFF)? → If not — REDO.
8. Does every node use the correct fill color, shape, and border style from the example outputs?
9. Do the arrows match the example output style (color, thickness, solid/dashed, arrowhead shape)?
10. Does the typography (font family, font size, bold/italic, weight, color) match the example outputs exactly for every text role (titles, node labels, edge labels, headers)?
11. Are ALL colors in your output taken directly from the example outputs? → If you used ANY color not present in the example outputs — REDO.
12. Are ALL logos, icons, watermarks, and decorative graphics from the brand reproduced exactly? → If any are missing, altered, or replaced — REDO.
13. Would this diagram look like it belongs in the same article as the example outputs, using the identical visual language?
14. Does your output avoid ALL mistakes shown in the negative examples? → If your output resembles any negative example — REDO.

If ANY answer is no, fix it before outputting.
"""

RESTYLE_PROMPT = """\
You are a diagram styling expert performing a minor refinement pass.

## Context

You previously generated a styled diagram from a raw input. The result was close but needs small adjustments. You will receive:

1. **EXAMPLE TRANSFORMATIONS**: Pairs of (input → output) showing the target brand style.
2. **ORIGINAL TARGET**: The raw, unstyled diagram that was the original input.
3. **PREVIOUSLY GENERATED**: The styled diagram you already produced — this is your starting point.
4. **CHANGE REQUEST**: The user's instructions describing what to fix or adjust.

## GOLDEN RULE — STRUCTURE IS FROZEN

Start from the PREVIOUSLY GENERATED image. Do NOT regenerate from scratch. Apply ONLY the changes described in the CHANGE REQUEST. Everything not explicitly mentioned in the CHANGE REQUEST must remain exactly as-is.

## CRITICAL — WHAT MUST STAY IDENTICAL

**BLOCKS/NODES**: Every block/node from the ORIGINAL TARGET must appear. Same count, same content. Do NOT drop, merge, split, add, or invent any block.

**ARROWS/CONNECTIONS**: Every arrow from the ORIGINAL TARGET must appear. Same source → same destination, same direction, same edge labels. Count must match exactly.

**LAYOUT/POSITIONS**: Do NOT move, reorder, or reflow any element unless the CHANGE REQUEST explicitly asks for it.

**BACKGROUND**: Background structure (sections, groupings, dividers) stays the same. Only change what the CHANGE REQUEST specifies. The canvas background MUST ALWAYS be solid white (#FFFFFF) — this is non-negotiable.

**TEXT**: All labels, titles, and annotations stay character-for-character identical unless the CHANGE REQUEST says otherwise.

The result must still match the brand defined by the example outputs. The background must always be solid white (#FFFFFF).

## Output

A single image: the PREVIOUSLY GENERATED diagram with the requested changes applied. Nothing else.
"""

DEFAULT_MODEL = "gemini-3-pro-image-preview"


def get_example_pairs(examples_dir: Path) -> list[tuple[Path, Path]]:
    """Discover example_*/ folders and return (input, output) path pairs."""
    example_dirs = sorted(
        d for d in examples_dir.iterdir()
        if d.is_dir() and d.name.startswith("example_")
    )
    pairs = []
    for d in example_dirs:
        input_img = next((d / f for f in ("input.png", "input.jpg", "input.jpeg", "input.webp") if (d / f).exists()), None)
        output_img = next((d / f for f in ("output.png", "output.jpg", "output.jpeg", "output.webp") if (d / f).exists()), None)
        if input_img and output_img:
            pairs.append((input_img, output_img))
        else:
            print(f"Warning: Skipping {d.name} — missing input or output image", file=sys.stderr)
    return pairs


def get_negative_examples(examples_dir: Path) -> list[tuple[Path, Path]]:
    """Discover not_example_*/ folders and return (input, bad_output) path pairs."""
    example_dirs = sorted(
        d for d in examples_dir.iterdir()
        if d.is_dir() and d.name.startswith("not_example_")
    )
    pairs = []
    for d in example_dirs:
        input_img = next((d / f for f in ("input.png", "input.jpg", "input.jpeg", "input.webp") if (d / f).exists()), None)
        output_img = next((d / f for f in ("output.png", "output.jpg", "output.jpeg", "output.webp") if (d / f).exists()), None)
        if input_img and output_img:
            pairs.append((input_img, output_img))
        else:
            print(f"Warning: Skipping {d.name} — missing input or output image", file=sys.stderr)
    return pairs


def load_image(path: str | Path) -> Image.Image:
    """Load and return a PIL Image fully read into memory, exiting on failure."""
    p = Path(path)
    if not p.exists():
        print(f"Error: Image not found: {p}", file=sys.stderr)
        sys.exit(1)
    img = Image.open(p)
    img.load()  # Force-read pixel data so the file handle is released
    return img


def _call_api(client: genai.Client, model: str, contents: list, output_path: Path, label: str) -> Path | None:
    """Send a single style-transfer request to Gemini and save the result."""
    print(f"  [{label}] Calling {model}...")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            temperature=0.2,
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,
                include_thoughts=True,
            ),
        ),
    )

    for part in response.parts:
        if part.thought and part.text is not None:
            print(f"  [{label}] Thinking: {part.text[:200]}..." if len(part.text) > 200 else f"  [{label}] Thinking: {part.text}")
        elif part.text is not None:
            print(f"  [{label}] Model response: {part.text}")
        elif part.inline_data is not None:
            image = part.as_image()
            image.save(output_path)
            print(f"  [{label}] Styled diagram saved to: {output_path}")
            return output_path

    print(f"  [{label}] Error: Model did not return an image.", file=sys.stderr)
    if hasattr(response, "prompt_feedback"):
        print(f"  [{label}] Feedback: {response.prompt_feedback}", file=sys.stderr)
    return None


def run_style_transfer(
    target_path: str | Path,
    examples_dir: Path | None = None,
    output_path: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    user_input: str | None = None,
    num: int = 1,
    previously_generated: str | Path | None = None,
) -> list[Path]:
    """
    Run style transfer using the Gemini API with few-shot examples only.

    Args:
        target_path: Path to the unstyled target diagram image.
        examples_dir: Directory containing example_*/ folders. Defaults to SKILL_DIR.
        output_path: Where to save the output. Defaults to <target>_styled.png (or _restyled.png in restyle mode).
        model: Gemini model to use.
        user_input: Optional free-text instructions to guide the style transfer.
        num: Number of variations to generate in parallel. Defaults to 1.
        previously_generated: Path to a previously generated styled image to refine. Requires --user-input.

    Returns:
        List of paths to saved output images.
    """
    target_path = Path(target_path)
    restyle_mode = previously_generated is not None

    if restyle_mode and not user_input:
        print("Error: --user-input is required when using --previously-generated.", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        suffix = "_restyled" if restyle_mode else "_styled"
        output_path = target_path.with_stem(f"{target_path.stem}{suffix}")
    output_path = Path(output_path)

    if examples_dir is None:
        examples_dir = SKILL_DIR

    if restyle_mode:
        # --- Restyle mode: refine a previously generated image ---
        previously_generated = Path(previously_generated)
        print(f"Mode: restyle (refining previously generated image)")

        # Load example pairs for brand reference in restyle mode too
        example_pairs = get_example_pairs(examples_dir)
        if not example_pairs:
            print("Error: No example pairs found. At least one example_*/ folder with input.png and output.png is required.", file=sys.stderr)
            sys.exit(1)
        print(f"Example pairs: {len(example_pairs)} transformation(s)")

        contents: list = [RESTYLE_PROMPT]

        # Example transformations (for brand reference)
        contents.append("\n--- EXAMPLE TRANSFORMATIONS ---\nEach pair below shows a raw input diagram followed by its correctly styled output. These define the target brand:\n")
        for i, (inp, out) in enumerate(example_pairs, 1):
            print(f"  Loading example {i}: {inp.parent.name}/")
            contents.append(f"\nExample {i} — INPUT (raw, unstyled):")
            contents.append(load_image(inp))
            contents.append(f"\nExample {i} — OUTPUT (correctly styled):")
            contents.append(load_image(out))

        # Original target
        contents.append("\n--- ORIGINAL TARGET ---\nThis is the raw, unstyled diagram that was the original input:\n")
        print(f"  Loading original target: {target_path}")
        contents.append(load_image(target_path))

        # Previously generated
        contents.append("\n--- PREVIOUSLY GENERATED ---\nThis is the styled diagram you already produced. Use it as your starting point — do NOT regenerate from scratch:\n")
        print(f"  Loading previously generated: {previously_generated}")
        contents.append(load_image(previously_generated))

        # Change request
        contents.append(f"\n--- CHANGE REQUEST ---\nApply ONLY these changes to the previously generated image:\n\n{user_input}\n")
        print(f"  Change request: {user_input}")

        total_images = len(example_pairs) * 2 + 2  # example pairs + original + previous
    else:
        # --- Full style transfer mode ---
        example_pairs = get_example_pairs(examples_dir)
        if not example_pairs:
            print("Error: No example pairs found. At least one example_*/ folder with input.png and output.png is required.", file=sys.stderr)
            sys.exit(1)
        print(f"Example pairs: {len(example_pairs)} transformation(s)")
        print(f"Mode: full style transfer")

        contents: list = [STYLE_PROMPT]

        # Few-shot example transformations
        contents.append("\n--- EXAMPLE TRANSFORMATIONS ---\nEach pair below shows a raw input diagram followed by its correctly styled output. Study these to understand the transformation and extract the target brand:\n")
        for i, (inp, out) in enumerate(example_pairs, 1):
            print(f"  Loading example {i}: {inp.parent.name}/")
            contents.append(f"\nExample {i} — INPUT (raw, unstyled):")
            contents.append(load_image(inp))
            contents.append(f"\nExample {i} — OUTPUT (correctly styled):")
            contents.append(load_image(out))

        # Negative few-shot examples
        negative_pairs = get_negative_examples(examples_dir)
        if negative_pairs:
            print(f"Negative examples: {len(negative_pairs)} bad transformation(s)")
            contents.append("\n--- NEGATIVE EXAMPLES (DO NOT COPY) ---\nEach pair below shows a raw input diagram followed by an INCORRECT styling. These are BAD outputs — study what went wrong and AVOID repeating these mistakes:\n")
            for i, (inp, out) in enumerate(negative_pairs, 1):
                print(f"  Loading negative example {i}: {inp.parent.name}/")
                contents.append(f"\nNegative Example {i} — INPUT (raw, unstyled):")
                contents.append(load_image(inp))
                contents.append(f"\nNegative Example {i} — BAD OUTPUT (WRONG — do NOT imitate this):")
                contents.append(load_image(out))

        # Target
        contents.append("\n--- TARGET ---\nRestyle this diagram to match the brand shown in the example outputs above. Apply everything you learned from the example transformations:\n")
        print(f"  Loading input: {target_path}")
        contents.append(load_image(target_path))

        # Optional user guidance
        if user_input:
            contents.append(f"\n--- ADDITIONAL INSTRUCTIONS ---\nThe user provided the following guidance for this specific transformation. Follow these instructions while still matching the brand style:\n\n{user_input}\n")
            print(f"  User input: {user_input}")

        total_images = len(example_pairs) * 2 + len(negative_pairs) * 2 + 1

    print(f"\nTotal images in request: {total_images}")
    print(f"Generating {num} variation(s)...")

    # Build output paths
    if num == 1:
        output_paths = [output_path]
    else:
        output_paths = [
            output_path.with_stem(f"{output_path.stem}_{i}")
            for i in range(1, num + 1)
        ]

    # Call the API (parallel when num > 1)
    client = genai.Client()
    saved_paths: list[Path] = []

    if num == 1:
        result = _call_api(client, model, contents, output_paths[0], "1/1")
        if result:
            saved_paths.append(result)
    else:
        with ThreadPoolExecutor(max_workers=num) as executor:
            futures = {
                executor.submit(
                    _call_api, client, model, contents, path, f"{i}/{num}"
                ): path
                for i, path in enumerate(output_paths, 1)
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    saved_paths.append(result)

    if not saved_paths:
        print("Error: No images were generated.", file=sys.stderr)
        sys.exit(1)

    saved_paths.sort()
    print(f"\nDone — {len(saved_paths)}/{num} variation(s) saved.")
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Restyle a diagram to match a brand using few-shot examples via Nano Banana (Gemini).",
    )
    parser.add_argument(
        "target",
        help="Path to the unstyled target diagram image",
    )
    parser.add_argument(
        "--examples-dir",
        default=None,
        help="Directory containing example_*/ folders with input/output pairs. Defaults to skill root.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the styled diagram. Defaults to <target>_styled.png",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--user-input",
        default=None,
        help="Optional free-text instructions to guide the style transfer (e.g., 'use a left-to-right layout' or 'make the title larger').",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of styled variations to generate in parallel (default: 1).",
    )
    parser.add_argument(
        "--previously-generated",
        default=None,
        help="Path to a previously generated styled image to refine. Requires --user-input. Output uses '_restyled' suffix.",
    )

    args = parser.parse_args()
    run_style_transfer(
        args.target,
        examples_dir=Path(args.examples_dir) if args.examples_dir else None,
        output_path=args.output,
        model=args.model,
        user_input=args.user_input,
        num=args.num,
        previously_generated=args.previously_generated,
    )


if __name__ == "__main__":
    main()
