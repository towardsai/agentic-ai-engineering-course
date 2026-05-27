---
name: sync-code-to-lessons
description: Propagate source-code changes between two branches (dev vs test by default) from the agents_integration, research_agent_part_2, research_agent_part_3, utils and writing_workflow projects into the code snippets embedded in the course lessons. Syncs notebooks or articles (you choose at the start), applies the edits automatically, and prints a report of every sync performed.
user_invocable: true
---

You keep the course lessons in sync with the "source of truth" code. The five project folders below hold the real, runnable code. Each numbered lesson teaches that code by **embedding copies of it as fenced code blocks** inside its notebook and/or article. When the source changes on a branch, those embedded copies drift out of date — your job is to find the drift and fix it.

## Mental model

**Source-of-truth folders** (the only places a change originates):

- `lessons/agents_integration`
- `lessons/research_agent_part_2`
- `lessons/research_agent_part_3`
- `lessons/utils`
- `lessons/writing_workflow`

**Lessons** are the numbered folders `lessons/NN_*` (e.g. `lessons/22_foundations_writing_workflow`). A lesson is made of two teaching artifacts, and you sync **one kind at a time**:

- a **notebook** — `notebook.ipynb` — always in the repo.
- an **article** — `article.md` — **may not live in the repo**; the user can point you to a different location.

Both artifacts embed source code as fenced blocks. Two conventions tell you which source a block mirrors:

1. **Explicit marker** — a line like `Source: _mcp_server/src/tools/generate_next_queries_tool.py_` (path relative to its source project, or absolute from the repo root) immediately precedes the fenced block. Common in the MCP/research lessons. These blocks are usually near-verbatim copies.
2. **Prose reference** — narration like ``Let's examine the `Settings` class from `brown.config`:`` or ``From `brown.entities.guidelines`:`` precedes the block. Common in the `brown`/writing-workflow lessons. These blocks are often **deliberately simplified or trimmed** for teaching.

> [!IMPORTANT]
> Embedded snippets are frequently **abridged** versions of the source (shortened docstrings, omitted helpers, illustrative subsets). Your goal is to propagate the *real change* — a renamed symbol, a changed signature, a new/removed parameter, altered logic — **while preserving the snippet's intentional simplifications**. Do not blindly paste the full current source over a teaching snippet.

## Prerequisites

- `python3` and `git` available (both standard in this repo).
- The helper script `scripts/sync_helper.py` (read-only; it never edits). Run it with `python3` from the repo root.
- The repo is normally checked out on the **head** branch (`dev`), so reading a source file gives you its target (post-change) content. If unsure, read the exact target version with `git show dev:<path>`.

## Workflow

### Step 0 — Ask what to sync (ALWAYS do this first)

Before anything else, ask the user whether this run syncs **notebooks** or **articles**, using the `AskUserQuestion` tool. Do not assume. (If the user already stated it in their request, confirm and skip the question.)

- **Notebooks** → the targets live inside this repo at `lessons/NN_*/notebook.ipynb`.
- **Articles** → ask **where the articles live**. Default is `lessons/NN_*/article.md` in this repo, but the user may keep articles in a separate location (e.g. an Obsidian vault, a writing app export, another folder). If they give a path, you will pass it to the helper with `--root`.

Also confirm the branch pair. Default is **base `test`, head `dev`** (i.e. "what does `dev` have that `test` doesn't"). Accept overrides like "main vs dev".

### Step 1 — Find what changed in the source folders

```bash
python3 .claude/skills/sync-code-to-lessons/scripts/sync_helper.py changes --base test --head dev
```

This prints the code-relevant files (filtering out data dumps, lockfiles, caches, fixtures) that differ between the branches, grouped by project, with the `class`/`def`/constant **symbols** each `.py` file touches. This is your candidate list — not everything here is necessarily mirrored in a lesson.

Read the actual diff for anything you intend to propagate so you understand the *semantics* of the change, not just the file name:

```bash
git diff test dev -- <path/to/changed_file.py>
```

### Step 2 — Locate where each change is mirrored

For each changed file/symbol, find the embedded blocks that reference it. Pass a distinctive substring — a symbol name (best) or a path fragment — as `--pattern`:

```bash
# notebooks (in-repo)
python3 .claude/skills/sync-code-to-lessons/scripts/sync_helper.py scan --mode notebooks --pattern <symbol_or_path>

# articles in the repo
python3 .claude/skills/sync-code-to-lessons/scripts/sync_helper.py scan --mode articles --pattern <symbol_or_path>

# articles in a custom location the user gave you
python3 .claude/skills/sync-code-to-lessons/scripts/sync_helper.py scan --mode articles --root <path> --pattern <symbol_or_path>
```

The scan reports, per match: the lesson file, the **cell index** (notebooks) or **line number** (articles), the `Source:`/`ref:` hint, and a preview of the block (add `--full` for the whole block). Use the hint to confirm the block really mirrors the changed source and isn't an unrelated coincidence.

Cross-check against the lesson↔package map (a block only needs syncing if its lesson teaches that package):

- `brown` (writing_workflow) → lessons 13, 22, 23, 24, 25, 26, 28, 30, 31
- `research_agent_part_2/3`, `mcp_server`, `mcp_client` → lessons 16, 17, 18, 19, 24, 25, 26, 32, 33, 34
- `agents_integration` → lesson 25
- `utils` → most early lessons that do `from utils import ...`

### Step 3 — Decide what to propagate

Propagate a change only when it is **mirrored in a teaching artifact** of the kind you're syncing. Skip:

- changes with no embedded mirror (internal refactors, private helpers never shown, test files, eval datasets, lockfiles, config the lessons don't display);
- cosmetic diffs that don't alter what the snippet shows (e.g. an import reordering that the snippet doesn't include).

When in doubt about whether a borderline change belongs in a lesson, briefly ask the user rather than guessing.

### Step 4 — Apply the sync

For each confirmed mirror:

1. Read the **target** source (the current file on `dev`, or `git show dev:<path>`) and the **embedded** block.
2. Determine the minimal edit that brings the snippet in line with the real change while keeping its teaching shape (preserve trims, abbreviated docstrings, `...` elisions, illustrative ordering).
3. Apply it:
   - **Notebooks** → use the `NotebookEdit` tool to replace the source of the specific cell (by the index the scan reported). The fenced block lives inside a markdown cell (or *is* a code cell); replace the whole cell source with the corrected version.
   - **Articles** → use the `Edit` tool on the `.md` file, matching the old fenced block exactly and replacing it.
4. If the change renamed or removed a symbol, also fix the **surrounding prose** that names it (headings, `from x import y` narration, inline mentions) so the lesson stays coherent.

Make the edits automatically — this skill syncs, it doesn't just report. After editing, the user can review with `git diff`.

### Step 5 — Report every sync

End with a concise report. List, per lesson artifact:

- the **source change** (file + symbol),
- **where** it landed (lesson, cell index / line),
- **what** you changed (one phrase), and
- anything you intentionally **skipped** and why (e.g. "internal helper, not shown in any lesson").

Use a table or bullets. Example:

```
Synced (notebooks, test → dev)

| Source change | Lesson | Location | Edit |
|---|---|---|---|
| research_agent_part_2/.../scrape_research_urls_tool.py · validate_and_read_urls_file | 19_final_outputs | notebook.ipynb cell 8 | updated renamed param `path`→`urls_file` |
| writing_workflow/.../config.py · Settings | 22_foundations… | notebook.ipynb cell 21 | added new `GOOGLE_API_KEY` field |

Skipped: writing_workflow tests/* (test code, not taught); utils/env.py (no embedded mirror).
```

## Notes & guardrails

- **One artifact kind per run.** Notebooks and articles are synced separately because articles may live elsewhere.
- **Verify hints.** A `--pattern` match is a candidate, not proof. Read the block and its `Source:`/prose hint before editing.
- **Don't over-write teaching snippets.** Preserve intentional simplifications; propagate the real change only.
- **Notebook outputs.** Don't fabricate or alter cell outputs; only fix source. If a change would invalidate a shown output, note it in the report rather than inventing a new output.
- **Scope.** Only the five source folders are sources; only `lessons/NN_*` (and the user-provided article location) are targets. The helper already filters fixtures and data; if you scan manually, do the same.
