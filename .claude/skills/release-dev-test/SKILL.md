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

## Step 6 — Open the `test → main` pull request

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

## Step 7 — Report

Summarize concisely:

- the branch you released from (`dev`) and to (`test`),
- the exact files released,
- the commit message and that the push succeeded,
- confirmation you're back on `dev`,
- the PR link (or the PR that `gh` created), and
- any step you skipped or stopped on, and why.

## Guardrails

- **Never run anything if Step 0's branch assertion fails.** `dev`-only, no exceptions.
- Only release the files the user explicitly selected — never a blanket "everything that differs."
- Confirm with the user before `git commit` and `git push`.
- Never force-push; never touch `main` directly.
- If any step errors, stop, report, and make sure the repo is left back on `dev`.
