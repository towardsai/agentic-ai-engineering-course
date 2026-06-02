---
name: release-dev-test
description: Release a chosen subset of files from the `dev` branch to the `test` staging branch, following the team's Git content release process (checkout test, pull just the selected files from dev, commit, push, return to dev), then guide opening a `test → main` pull request. The user picks which files to release. HARD REQUIREMENT - this skill only runs when the current branch is `dev`; it aborts otherwise. User-invoked via /release-dev-test.
user_invocable: true
---

You stage content for release by copying a **user-selected subset of files** from the `dev` branch onto the `test` branch, then push `test` so a `test → main` pull request can be opened on GitHub. You do **not** release everything on `dev` — only the exact files the user names.

## Branch model

- **`dev`** — where all work happens (article guidelines, articles, notebooks, source code… everything). This is the source you copy *from*.
- **`test`** — the staging branch. You copy the selected files *onto* it, commit, and push. This is what gets PR'd into `main`.
- **`main`** — what students clone (notebooks etc., no article guidelines/articles). You never touch `main` directly here; it only receives content via a reviewed PR from `test`.

> There are no branch protections on this repo (free plan, private repo). That means nothing mechanically stops a bad push — **you** are the safeguard. Be deliberate, confirm before committing/pushing, and never force-push.

## Release content policy for `lessons/NN_*` (ALWAYS enforce)

Students clone `main`. From the **numbered course lessons** (`lessons/NN_<name>/`, e.g. `lessons/06_tools/`) we ship a runnable lesson, **not** the authoring artifacts. So whenever the release set touches any `lessons/NN_*` path, apply this filter automatically — it is not optional and does not depend on the user listing individual files:

**Default for every numbered lesson — include ONLY:**
1. `notebook.ipynb`, and
2. the **supporting media the notebook actually references** — images, PDFs, audio, etc. Infer this set by scanning the notebook for local media paths (do **not** ship the whole directory blindly):

   ```bash
   # List media paths referenced inside a lesson's notebook:
   grep -oE '[A-Za-z0-9_./-]+\.(png|jpe?g|gif|svg|webp|bmp|pdf|mp3|wav|m4a|mp4|mov)' \
     lessons/NN_<name>/notebook.ipynb | sort -u
   ```
   The grep is a **heuristic** — it also emits noise (bare filenames, URL fragments, paths from markdown output cells). Treat a hit as real media only when it resolves to a tracked file in the lesson directory; cross-check against the actual tree:
   ```bash
   git ls-tree -r --name-only dev -- lessons/NN_<name>
   ```
   Keep the relative paths that exist as files under the lesson (e.g. `images/attention_is_all_you_need_1.jpeg`, `pdfs/attention_is_all_you_need_paper.pdf`); drop hits that don't resolve. Lesson **11** (`11_multimodal`) is the canonical example: it pulls in `lessons/11_multimodal/images/*` and `lessons/11_multimodal/pdfs/*`. Present the final inferred media list to the user before staging.

**ALWAYS exclude** from numbered lessons (never release): `article.md`, `article_*.md`, `article_guideline.md`, `article_outline.md`, `*.metadata.json`, `research.md`, `research.xml`, `*_review*.md`, `reflection_scores.md`, `style_guideline.md`, `checkpoints/`, and any other article/research/eval authoring artifacts.

### Stored edge cases (override the default above)

- **`lessons/25_integrate_agents/`** — in addition to `notebook.ipynb`, also release the **MCP server setup** config files the lesson requires: `mcp_servers_config_http.json` and `mcp_composed_server_config_http.json`. (Still exclude `article*.md`, `research.md`, `article_guideline.md`.)
- **`lessons/26_end_to_end_demo/`** — release **ALL** files in the lesson directory. We deliberately keep every output artifact produced during the end-to-end demo (`article.md`, `article_000.md`, `article_001.md`, `article_002.md`, `research.md`, `article_guideline.md`, `notebook.ipynb`, plus anything else present). Do **not** apply the notebook-only filter here.

This policy applies **only** to `lessons/NN_*`. Source-code projects under `lessons/` (`writing_workflow`, `research_agent_part_*`, `agents_integration`, `utils`), root/infra files, and `assets/` are released as-is per the user's selection. `.claude/` is never released.

## Step 0 — Hard precondition: must be on `dev` (ASSERTION)

This is a blocking gate. Run it **first**, before any other git command:

```bash
git rev-parse --abbrev-ref HEAD
```

- If the output is **not exactly `dev`**, you must **STOP immediately**. Do not checkout, fetch, or modify anything. Tell the user: *"This skill can only be run from the `dev` branch. You are on `<branch>`. Switch with `git checkout dev` and re-run."* End the run.
- Only if the branch is exactly `dev` do you continue.

Also assert a **clean working tree** before going further:

```bash
git status --porcelain
```

- If the output is non-empty, there are uncommitted/untracked changes. Switching branches could carry or clobber them. **STOP** and ask the user to commit, stash, or discard first — do not proceed automatically. (Untracked files that don't collide are usually harmless; use judgement, but when in doubt, stop and ask.)

## Step 0.5 — Optional: bump (and publish) the package version — ALWAYS ask, NEVER automatic

After the `dev` + clean-tree gate passes, **ask the user** whether they want to bump the package version before releasing — e.g. *"Want me to bump the package version with `make build` (and optionally publish to PyPI with `make publish`) before we stage the release?"*

> **HARD RULE:** never run `make build` or `make publish` on your own initiative. Only run them when the user explicitly says yes in this step. If the user doesn't ask for a bump, skip straight to Step 1 and leave the version untouched.

If the user says **no**, skip to Step 1.

If the user says **yes**:

1. **Confirm the bump level.** `make build` runs `uv version --bump $(VERSION_BUMP)` (default `patch`) then `uv build`. Ask whether they want `patch` (default), `minor`, or `major`, then run:

   ```bash
   make build                       # patch bump (default)
   # or, for a different level:
   make build VERSION_BUMP=minor    # or major
   ```

   This edits `pyproject.toml` and `uv.lock` in the working tree (and produces a build under `dist/`).

2. **Ask before publishing.** Publishing to PyPI is outward-facing and irreversible — a version can't be re-uploaded. Only if the user explicitly wants it:

   ```bash
   make publish                     # uv publish --token $(PYPI_TOKEN)
   ```

   Skip this if they only want the version bump (e.g. they'll publish later or from CI).

3. **Commit the bump on `dev`.** The bump dirtied the tree that Step 0 asserted clean. Commit it on `dev` so the new version is part of the release source:

   ```bash
   git add pyproject.toml uv.lock
   git commit -m "chore: bump package version to <new-version>"
   ```

   (Read the new version from `pyproject.toml` after the bump.) Make sure the tree is clean again before continuing.

4. **Add the version files to the release set.** Ensure `pyproject.toml` and `uv.lock` are included in `<files>` for the steps below, so the bump actually reaches `test` → `main`.

## Step 1 — Decide which files to release

The user controls the file list. Resolve it in this order:

1. **If the user passed files as arguments** to the command, treat those as the release set.
2. **Otherwise, ask the user** which files to release. To help them choose, show the files that actually differ between `test` and `dev` (candidates worth releasing):

   ```bash
   git fetch origin
   git diff --name-only origin/test...dev    # files dev has that test doesn't (or differs)
   ```

   Present that list and let the user pick a subset (or type their own paths). Do not assume "release everything that differs" — wait for the user's explicit selection.

3. **Validate** every selected path exists on `dev` before continuing:

   ```bash
   git cat-file -e dev:<file> 2>/dev/null && echo "ok: <file>" || echo "MISSING on dev: <file>"
   ```

   If any path is missing on `dev`, surface it and ask the user to correct the list rather than guessing.

4. **Apply the `lessons/NN_*` release content policy** (see the section above) to the resolved set. For any numbered lesson in the selection, narrow it to `notebook.ipynb` + inferred supporting media, honoring the stored edge cases for lessons **25** and **26**. Show the user the before/after for those lessons (what you dropped and why) so the filtering is transparent, then continue with the filtered set. This filter is mandatory even when the user said "all the notebooks" or named a whole lesson directory.

Keep the confirmed list as `<files>` for the steps below (space-separated, each path quoted if it contains spaces).

## Step 2 — Switch to `test` and bring it up to date

```bash
git checkout test
git pull --ff-only origin test
```

- If `git checkout test` fails, stop and report (most likely an unclean tree slipped through Step 0).
- If `git pull --ff-only` fails (diverged history), stop and ask the user how to reconcile rather than force-merging.

## Step 3 — Copy the selected files from `dev`

Pull *only* the chosen files from `dev` onto `test` (this also stages them):

```bash
git checkout dev -- <files>
```

Then show the user exactly what will be released, and **confirm before committing**:

```bash
git status
git --no-pager diff --staged --stat
```

If the staged set doesn't match what the user asked for, fix it (`git restore --staged <file>` / re-checkout) before moving on.

## Step 4 — Commit and push

Use a clear, content-focused commit message. If the user gave one, use it; otherwise propose one (e.g. `release: <short summary of what's being released>`) and confirm.

```bash
git add <files>          # explicit; safe even though checkout already staged them
git commit -m "<message>"
git push origin test
```

Committing and pushing are outward-facing — get the user's go-ahead on the file set and message first. Never `git push --force`.

## Step 5 — Return to `dev`

Always end on `dev`, even if a previous step failed midway:

```bash
git checkout dev
```

Confirm with `git rev-parse --abbrev-ref HEAD` that you're back on `dev`.

## Step 6 — Optional: audit `dev` ↔ `test` synchronization

After the release is pushed and you're back on `dev`, **ask the user** whether they want a synchronization check between the two branches — e.g. *"Want me to run the `dev-test-sync-checker` to see how `dev` and `test` still differ (both directions)?"*

- If **no**, skip to Step 7.
- If **yes**, launch the **`dev-test-sync-checker`** subagent (via the Agent tool, `subagent_type: dev-test-sync-checker`). It is **read-only**: it compares the two branch tips both ways and returns a categorized report —
  - what `dev` has that `test` doesn't (unreleased / intentionally-excluded artifacts),
  - what `test` has that `dev` doesn't (likely stale leftovers), and
  - files present on both but differing.

  Relay its report to the user. It never commits, pushes, or deletes — if its report suggests cleanup (e.g. stale files on `test`, or source files that drifted), surface those as **follow-up suggestions** and let the user decide. Acting on them (deletions on `test`, re-releasing drifted files) is a separate, explicitly-confirmed operation, not part of this skill's automatic flow.

## Step 7 — Open the `test → main` pull request

The PR is reviewed (diffs + reviewers) and merged on GitHub. Offer the user two options:

- **GitHub UI (matches the team's usual flow):** print the compare link and let them open it, review the diff, assign reviewers, and merge:

  ```
  https://github.com/<owner>/<repo>/compare/main...test
  ```

  (Derive `<owner>/<repo>` from `git remote get-url origin`.)

- **`gh` CLI (optional, only if the user wants it):**

  ```bash
  gh pr create --base main --head test --title "<title>" --body "<body>"
  ```

  Don't auto-merge — leave the review/merge decision to the user.

## Step 8 — Report

Summarize concisely:

- the branch you released from (`dev`) and to (`test`),
- whether a version bump/publish (Step 0.5) was run, and if so the new version and whether it was published to PyPI,
- the exact files released,
- the commit message and that the push succeeded,
- confirmation you're back on `dev`,
- whether the sync check (Step 6) was run, and a one-line takeaway if it was,
- the PR link (or the PR that `gh` created), and
- any step you skipped or stopped on, and why.

## Guardrails

- **Never run anything if Step 0's branch assertion fails.** `dev`-only, no exceptions.
- Only release the files the user explicitly selected — never a blanket "everything that differs."
- **Always enforce the `lessons/NN_*` release content policy** (notebooks + inferred supporting media only; lesson 25 also ships its MCP configs; lesson 26 ships everything). Never release lesson article/research/metadata authoring artifacts.
- Confirm with the user before `git commit` and `git push`.
- **Never bump or publish automatically.** Run `make build` / `make publish` only when the user explicitly asks in Step 0.5; `make publish` is irreversible (a PyPI version can't be re-uploaded), so confirm it separately from the bump.
- Never force-push; never touch `main` directly.
- The `dev-test-sync-checker` subagent is read-only — it reports divergence but never commits, pushes, or deletes. Any cleanup it suggests needs explicit user confirmation as a separate step.
- If any step errors, stop, report, and make sure the repo is left back on `dev`.
