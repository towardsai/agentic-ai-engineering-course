# Article Update Log — `article.md`

This log documents every change made to `article.md` to bring it in line with the
updated `notebook.ipynb`. The core driver: the notebook moved from a **text-only
embedding workaround** (describe each image with Gemini, then embed the *text* with
`gemini-embedding-001`) to **native multimodal embeddings** with `gemini-embedding-2`,
which embeds text, images, audio, video, and PDFs directly into one shared vector
space. Several new capabilities were also added (reverse search, a unified
text+image+PDF index, Matryoshka dimensions, and image-input to the agent).

> **Line numbers below refer to the ORIGINAL `article.md`** (before any edits), so they
> won't all line up with the current file — they identify *what* was changed and *where*
> it lived. Net effect: the file grew from 865 lines to ~1035 lines.

---

## 1. SDK and model IDs (Applying Multimodal LLMs section)

- **Lines 158–159** — Fixed the deprecated SDK import.
  - Old: `import google.generativeai as genai` / `from google.generativeai import types`
  - New: `from google import genai` / `from google.genai import types`
  - Why: the notebook uses the current `google-genai` SDK; the new embedding code also
    relies on `client.aio` which only exists there.

- **Line 161** — Cleaned up a malformed comment
  (`# Configure your Gemini client# genai.configure(...)`) into a single clear comment.

- **Line 163** — Model ID `gemini-2.5-flash` → `gemini-3.5-flash` to match the notebook
  (`MODEL_ID` constant).

## 2. "Implementing Multimodal RAG" — intro & rationale

- **Line 573** — Reworded the section intro: the index now holds *images, raw text
  snippets, and a PDF*, all embedded **directly** into one shared space with
  `gemini-embedding-2` (was: "a mix of images and PDF pages … treated as images").

- **Line 579** — Rewrote the architecture-description paragraph: the system now embeds
  each item (image / PDF / text) **directly** — no per-image text description; queries
  can be text *or* an image (was: "generating a textual description for each image and
  then embedding this description").

- **Lines 581–588** — Replaced the `⚠️` aside that *justified the description
  workaround* ("the Gemini API … does not currently support creating embeddings directly
  from images") with a `💡` aside explaining that `gemini-embedding-2` now embeds all
  modalities natively, so the workaround is no longer needed.

## 3. "Implementing Multimodal RAG" — code

- **Lines 611–639** — Replaced the `create_vector_index` *description-workaround*
  implementation with a new step 1: the multimodal `embed()` function
  (single text/bytes → vector via `gemini-embedding-2`, with optional
  `output_dimensionality`).

- **Lines 641–650** — Removed the "In case you start using a text-image embedding
  model, you would just have to …" hypothetical snippet — it is now the actual
  implementation. Replaced with a new step 2: the async `embed_batch()` helper
  (concurrent fan-out over the stable `client.aio.models.embed_content` endpoint, with
  a note on why we avoid the experimental `client.batches.create_embeddings`).

- **Lines 649–660** — Removed the `generate_image_description()` function (the
  workaround). Replaced with the new step 3: `create_vector_index()` that loads image
  bytes and embeds them all in one `embed_batch` call (returns items with
  `content`/`type`/`filename`/`mime_type`/`embedding`).

- **Lines 662–671** — Removed `embed_text_with_gemini()` (which used the text-only
  `gemini-embedding-001` model).

- **Lines 673–693** — Replaced `search_multimodal(query_text, …)` with a new step 4
  that accepts **either** a text query **or** a `(bytes, mime_type)` tuple, embedding
  both with the same `gemini-embedding-2` model.

- **Line 695** — Test step renumbered `3.` → `5.` and reworded to stress that the text
  query is compared **directly against image embeddings** (no description in between).

- **Line 725** — Updated the closing paragraph: "Because every item … is embedded
  **directly** into one shared vector space …" (was: "Because we normalized both
  standard images and PDF pages to images …").

## 4. NEW section — "Going Deeper with Gemini Embedding 2"

- **Inserted before "## Building Multimodal AI Agents" (originally line 727)** — A brand
  new top-level section covering capabilities added in the notebook:
  - **Image-as-Query (Reverse Search)** — feed an image into `search_multimodal`.
  - **A Truly Multimodal Index** — `add_text_to_index` / `add_pdf_to_index` helpers and
    cross-modal queries that each hit a different modality (text / image / PDF).
  - A `💡` production aside on **cross-modal score calibration** (per-modality
    normalisation, modality-aware re-ranking, hybrid scoring).
  - **Matryoshka dimensions** — using `output_dimensionality` to shrink embeddings
    (768-dim demo, ~4× storage savings) with reference [35].

## 5. "Building Multimodal AI Agents"

- **Lines 743–764** — Rewrote `multimodal_search_tool`: it now returns the **top-`k`
  candidates** and hands each back in its native form (text stays text; image/PDF
  returned as a Gemini `Part`), instead of a single result containing an image
  description + image (was: text-only `query`, `top_k=1`, `description` field).

- **Lines 768–781** — Rewrote `build_react_agent`:
  - `from langchain.agents import create_agent` (was
    `from langgraph.prebuilt import create_react_agent`).
  - Model `gemini-2.5-pro` (was `gemini-1.5-pro-latest`).
  - Replaced `messages_modifier=` / `convert_system_message_to_human=True` with the
    `create_agent(model=…, tools=…, system_prompt=…)` signature.
  - New, detailed system prompt instructing the agent to always search first and to
    inspect every candidate (cross-modal bias awareness).

- **Lines 792–816** — Rewrote the test block: uses `react_agent.invoke(...)` over **three**
  questions (image / text / PDF candidates) instead of a single `.stream()` call; removed
  the now-stale streamed `HumanMessage/AIMessage/ToolMessage` output dump.

- **Inserted after Figure 24** — New subsection **"Passing an Image Directly to the
  Agent"**, showing how to send an `image` content block in the user message
  (`load_image_as_base64` + LangChain image block) so the agent reasons over raw pixels
  before retrieving.

- **Line 823** — Updated the section's closing paragraph to mention native multimodal
  embeddings and both text and image inputs.

## 6. References

- **After reference 33** — Added three references for the new content:
  - **[34]** Gemini API — Embeddings docs.
  - **[35]** Matryoshka Representation Learning (arXiv 2205.13147).
  - **[36]** Gemini API — Batch Mode docs.
  - These are cited inline in the new embedding intro ([34]), the `embed_batch` note
    ([36]), and the Matryoshka subsection ([35]).

## 7. Reference cleanup — removed unused entries & renumbered

Audited every reference against its inline citations and **deleted the 12 that were
never cited** anywhere in the body (all were already unused in the original article; my
content changes did not orphan any citation):

- Removed: original **[4]** (HackerNoon OCR), **[7]** (Ionio guide), **[11]** (YouTube),
  **[16]–[19]** (raw GitHub image URLs), **[20]** (ColPali w/ Milvus), **[29]** (LangGraph
  agent image), **[30]** (Build Multimodal Agents), **[31]** (Colpali+Milvus+VLMs),
  **[32]** (Real-world multimodal AI).
- The remaining 24 references were **renumbered contiguously (1–24)** and every inline
  citation was remapped to match. Key remaps: `5→4`, `6→5`, `8→6`, `9→7`, `10→8`,
  `12→9` … `33→21`, and the references I added in §6 became `34→22`, `35→23`, `36→24`.
- Verified: references list is now contiguous `1..24`, every reference is cited, and
  every inline citation resolves to a listed reference.
