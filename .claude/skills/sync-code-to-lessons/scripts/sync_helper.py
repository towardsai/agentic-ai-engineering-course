#!/usr/bin/env python3
"""Helper for the `sync-code-to-lessons` skill.

Two read-only subcommands that do the tedious, error-prone parts of the sync so
the agent can focus on judgement and editing:

  changes  Show code-relevant files that differ between two branches inside the
           five "source of truth" project folders, plus the class/def/constant
           symbols each changed `.py` file touches. This is the list of changes
           that *might* need to be propagated into the lessons.

  scan     Extract every embedded ```python (or other) code block from lesson
           notebooks and/or articles, together with the "Source: _<path>_"
           marker or module reference that precedes it. Optionally filter to
           blocks matching a pattern (a symbol name or a source path), so you
           can locate exactly where a changed source file is mirrored.

The script only reads; it never edits. The agent performs the edits.

Usage examples
--------------
  # What changed in the source folders (default: test -> dev)?
  python scripts/sync_helper.py changes
  python scripts/sync_helper.py changes --base test --head dev

  # Where in the notebooks is `generate_next_queries_tool.py` mirrored?
  python scripts/sync_helper.py scan --mode notebooks --pattern generate_next_queries_tool

  # List every embedded code block in the notebooks (no filter)
  python scripts/sync_helper.py scan --mode notebooks

  # Scan articles that live outside the repo
  python scripts/sync_helper.py scan --mode articles --root /path/to/articles --pattern ContextMixin
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# The five "source of truth" project folders the skill watches.
SOURCE_DIRS = [
    "lessons/agents_integration",
    "lessons/research_agent_part_2",
    "lessons/research_agent_part_3",
    "lessons/utils",
    "lessons/writing_workflow",
]

# Files that are real, mirror-able code/config. Everything else is noise.
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json", ".sh", ".ini", ".mako"}
CODE_FILENAMES = {"Dockerfile"}

# Paths that never get mirrored into a lesson (data dumps, locks, caches, ...).
NOISE_RE = re.compile(
    r"(/data/|/inputs/|/tests/fixtures/|\.lock$|__pycache__|\.ruff_cache"
    r"|\.pytest_cache|\.ipynb_checkpoints|\.DS_Store|\.egg-info)"
)


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


# ---------------------------------------------------------------------------
# `changes` subcommand
# ---------------------------------------------------------------------------

SYMBOL_RE = re.compile(r"^[+-]\s*(?:async\s+)?(class|def)\s+([A-Za-z_]\w*)")
CONST_RE = re.compile(r"^[+-]\s*([A-Z_][A-Z0-9_]{2,})\s*[:=]")


def is_code_file(path: str) -> bool:
    if NOISE_RE.search(path):
        return False
    name = path.rsplit("/", 1)[-1]
    if name in CODE_FILENAMES:
        return True
    return any(path.endswith(suf) for suf in CODE_SUFFIXES)


def cmd_changes(args: argparse.Namespace) -> int:
    root = repo_root()
    names = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{args.base}", f"{args.head}", "--", *SOURCE_DIRS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    files = sorted(p for p in names if is_code_file(p))
    if not files:
        print(f"No code-relevant changes between {args.base} and {args.head} in the source folders.")
        return 0

    # Group by source project for a readable report.
    groups: dict[str, list[str]] = {}
    for f in files:
        proj = next((d for d in SOURCE_DIRS if f.startswith(d + "/") or f == d), "other")
        groups.setdefault(proj, []).append(f)

    print(f"# Source changes: {args.base} -> {args.head}\n")
    print(f"{len(files)} code-relevant file(s) changed across {len(groups)} project(s).\n")

    for proj in SOURCE_DIRS:
        if proj not in groups:
            continue
        print(f"## {proj}\n")
        for f in groups[proj]:
            symbols = changed_symbols(root, args.base, args.head, f) if f.endswith(".py") else []
            rel = f[len(proj) + 1 :]
            if symbols:
                print(f"- `{rel}`  →  symbols: {', '.join(sorted(symbols))}")
            else:
                print(f"- `{rel}`")
        print()
    return 0


def changed_symbols(root: Path, base: str, head: str, path: str) -> set[str]:
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", f"{base}", f"{head}", "--", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    symbols: set[str] = set()
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        m = SYMBOL_RE.match(line)
        if m:
            symbols.add(m.group(2))
            continue
        # Hunk headers carry enclosing-def context after the second @@.
        if line.startswith("@@"):
            ctx = line.split("@@")[-1]
            cm = re.search(r"(?:class|def)\s+([A-Za-z_]\w*)", ctx)
            if cm:
                symbols.add(cm.group(1))
            continue
        cm = CONST_RE.match(line)
        if cm:
            symbols.add(cm.group(1))
    return symbols


# ---------------------------------------------------------------------------
# `scan` subcommand
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
# A "Source:" marker is captured line-based: everything after "Source:" up to the
# end of the line. We then strip markdown wrappers (_italics_ / `code`) and keep
# the first whitespace-free token — robust to underscores inside file paths.
SOURCE_LINE_RE = re.compile(r"Source:\s*(.+)", re.IGNORECASE)
MODULE_REF_RE = re.compile(r"(?:from|import)\s+([\w.]+)|`([\w.]+\.[\w.]+)`")


def _clean_marker(raw: str) -> str:
    s = raw.strip()
    # Drop a single matched pair of italic/code wrappers around the whole token.
    for ch in ("_", "`", "*"):
        if len(s) >= 2 and s.startswith(ch) and s.endswith(ch):
            s = s[1:-1]
    # Paths and URLs have no spaces; keep the first token.
    return s.split()[0].strip("_`*") if s.split() else ""


def find_source_marker(text: str) -> str:
    """Return the path from the last 'Source: ...' marker in `text`, or ''."""
    matches = SOURCE_LINE_RE.findall(text)
    return _clean_marker(matches[-1]) if matches else ""


def collect_text_blocks(text: str):
    """Yield (lang, code, hint) for every fenced block in a markdown string.

    `hint` is the nearest preceding "Source: _..._" marker, or the first module
    reference inside the block, whichever is found first.
    """
    for m in FENCE_RE.finditer(text):
        lang = m.group(1) or ""
        code = m.group(2)
        preceding = text[: m.start()]
        hint = ""
        marker = find_source_marker(preceding)
        if marker:
            hint = "Source: " + marker
        else:
            ref = MODULE_REF_RE.search(code)
            if ref:
                hint = "ref: " + (ref.group(1) or ref.group(2))
        # Line number of the fence opening within `text`.
        line_no = preceding.count("\n") + 1
        yield lang, code, hint, line_no


def article_files(root: Path, in_repo: bool):
    """Markdown files to treat as articles.

    In-repo: only the canonical `lessons/NN_*/article.md`. Custom root: every
    `.md` under it (the user pointed us there deliberately), minus obvious noise.
    """
    if in_repo:
        return sorted(root.glob("lessons/[0-9]*/article.md"))
    return [
        p
        for p in sorted(root.glob("**/*.md"))
        if not NOISE_RE.search(str(p)) and "/.nova/" not in str(p)
    ]


def scan_articles(root: Path, pattern: str | None, in_repo: bool):
    results = []
    for md in article_files(root, in_repo):
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lang, code, hint, line_no in collect_text_blocks(text):
            if pattern and pattern not in code and pattern not in hint:
                continue
            results.append((str(md), f"L{line_no}", lang, hint, code))
    return results


def notebook_files(root: Path, in_repo: bool):
    if in_repo:
        return sorted(root.glob("lessons/[0-9]*/notebook.ipynb"))
    return [p for p in sorted(root.glob("**/notebook.ipynb")) if not NOISE_RE.search(str(p))]


def scan_notebooks(root: Path, pattern: str | None, in_repo: bool):
    results = []
    for nb_path in notebook_files(root, in_repo):
        if ".ipynb_checkpoints" in str(nb_path):
            continue
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for idx, cell in enumerate(nb.get("cells", [])):
            src = "".join(cell.get("source", []))
            ctype = cell.get("cell_type")
            if ctype == "markdown":
                for lang, code, hint, _ in collect_text_blocks(src):
                    if pattern and pattern not in code and pattern not in hint:
                        continue
                    results.append((str(nb_path), f"cell {idx} (md fence)", lang, hint, code))
            elif ctype == "code":
                if pattern and pattern not in src:
                    continue
                # A code cell is itself the "block".
                ref = MODULE_REF_RE.search(src)
                hint = ("ref: " + (ref.group(1) or ref.group(2))) if ref else ""
                results.append((str(nb_path), f"cell {idx} (code)", "python", hint, src))
    return results


def cmd_scan(args: argparse.Namespace) -> int:
    in_repo = not args.root
    root = Path(args.root).expanduser() if args.root else repo_root()
    if args.mode == "notebooks":
        results = scan_notebooks(root, args.pattern, in_repo)
    else:
        results = scan_articles(root, args.pattern, in_repo)

    if not results:
        where = f" matching '{args.pattern}'" if args.pattern else ""
        print(f"No embedded {args.mode} code blocks{where} found under {root}.")
        return 0

    filt = f" matching '{args.pattern}'" if args.pattern else ""
    print(f"# Embedded code blocks in {args.mode}{filt}\n")
    print(f"{len(results)} block(s) found.\n")
    for path, loc, lang, hint, code in results:
        rel = path
        try:
            rel = str(Path(path).relative_to(repo_root()))
        except ValueError:
            pass
        head = f"## {rel} — {loc}"
        if hint:
            head += f"  [{hint}]"
        print(head)
        preview = code if args.full else "\n".join(code.splitlines()[:20])
        truncated = (not args.full) and code.count("\n") > 20
        print(f"```{lang}\n{preview}{'' if preview.endswith(chr(10)) else chr(10)}```"
              + ("\n_(truncated; rerun with --full)_" if truncated else ""))
        print()
    return 0


# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("changes", help="List code-relevant source changes between two branches.")
    pc.add_argument("--base", default="test", help="Base branch (default: test).")
    pc.add_argument("--head", default="dev", help="Head branch (default: dev).")
    pc.set_defaults(func=cmd_changes)

    ps = sub.add_parser("scan", help="Find embedded code blocks in lessons.")
    ps.add_argument("--mode", choices=["notebooks", "articles"], required=True)
    ps.add_argument("--pattern", default=None, help="Only show blocks containing this substring.")
    ps.add_argument("--root", default=None, help="Root dir to scan (for articles outside the repo).")
    ps.add_argument("--full", action="store_true", help="Print full blocks instead of a 20-line preview.")
    ps.set_defaults(func=cmd_scan)

    args = p.parse_args()
    try:
        return args.func(args)
    except subprocess.CalledProcessError as e:
        print(f"git error: {e.stderr or e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
