# CI Guidelines

## Global Context of the Lesson

### What We Are Planning to Share

In this lesson, we’ll set up a professional Continuous Integration (CI) pipeline for our agent repo, built around a simple idea:

> The same quality checks you run locally should be enforced automatically in CI.
> 

We will introduce:

- **pre-commit hooks** to run automated checks before commits.
- **Linting + formatting with Ruff** to keep style consistent and prevent common bugs.
- **Unit tests with pytest** to verify deterministic parts of agent code.
- A **GitHub Actions CI workflow** that runs format/lint checks and tests on pull requests and pushes.

We will also show how **Brown** tests avoid real LLM/API calls by using a **fake model**, which makes tests deterministic and CI-friendly.

### Why We Think It’s Valuable

Agent repos have two problems that show up early:

1. They change fast (prompt changes, routing changes, schema changes).
2. They break in subtle ways (format drift, import issues, regressions in parsing/rendering, flaky tests tied to external APIs).

A minimal CI pipeline prevents “silent breakage,” keeps PRs reviewable, and makes the repo safe to collaborate on.

### Expected Length of the Lesson

5,000–6,000 words (main instructional content, excluding code examples and hands-on demonstrations).

### Point of View

Use **we/us/our** for the instructors and **you/your** for the learner.

---

## Anchoring the Lesson in the Course

### Where This Lesson Fits

This lesson is part of the “professional engineering” track of the course: you’ve built agent systems (Nova/Brown), and now we make the repo **maintainable**.

### Lesson Scope

In scope:

- pre-commit setup + how we use it in this repo
- Ruff lint + format (and where it’s configured)
- pytest unit testing workflow
- mocking LLM calls the “Brown way”
- GitHub Actions pipeline (`.github/workflows/ci.yml`) and when it runs

Out of scope (mention briefly, point forward):

- evaluation pipelines beyond unit tests
- deployment pipelines
- observability/tracing pipelines

### Prerequisites

- You can run commands from the repo root.
- You have `uv` installed and can run `uv sync`.
- You know basic `git` workflow (commit, push, PR).

---

## Narrative Flow of the Lesson

Use the following story arc:

1. **Problem:** “We merged a harmless change; CI would’ve caught the breakage immediately.”
2. **Local guardrails:** pre-commit + Ruff keeps your diffs clean and code readable.
3. **Correctness guardrails:** unit tests validate deterministic logic.
4. **Enforcement:** GitHub Actions makes these checks non-optional.
5. **Workflow:** what you do before committing, before pushing, and in PRs.

---

## Lesson Outline

1. **Introduction: What is Continuous Integration?**
   - Anchor in course (Lessons 27-29 connection)
   - Define CI fundamentals and core principles
   - CI for AI Agent Systems
   - Scope of this lesson (what's included, what's excluded)
   - Companion notebook mention

2. **Why Repos Need CI Early**
   - Three concrete failure modes
   - Standard CI Pipeline Components for Python
   - The Three-Tier CI Model for AI Agents
     - Tier 1: Formatting and Linting (Always Run)
     - Tier 2: Unit Tests with Mocked LLMs (Always Run)
     - Tier 3: AI Evaluations (Manual/Release)
   - Mermaid diagram showing three tiers
   - What we didn't include (but you might)

3. **Pre-commit Hooks: Automated Local Guardrails**
   - The Pre-commit Framework
   - Example pre-commit configuration
   - Brown's actual pre-commit configuration
   - Setting up pre-commit
   - The pre-commit workflow

4. **Ruff: Fast Python Linting and Formatting**
   - Formatting vs. Linting distinction
   - Ruff configuration for AI agent projects
   - Brown's Ruff configuration in pyproject.toml
   - Running Ruff locally
   - Brown's Makefile QA targets
   - Hands-On Example 1: Fixing Formatting Issues
   - Hands-On Example 2: Fixing Linting Issues

5. **Unit Tests for Agent Repos**
   - What to test in AI agents
   - Standard practice: Mocking external dependencies
   - Common LLM mocking approaches (3 patterns)
   - Our implementation: The FakeModel pattern (3 parts)
   - Brown's test structure
   - Example: Testing deterministic components
   - Example: Testing nodes with mocked responses
   - Running Brown's tests

6. **CI Workflows: Automated Enforcement**
   - CI platforms overview
   - Example GitHub Actions workflow
   - Key workflow components (triggers, jobs, steps, actions)
   - Why split jobs?
   - Making CI mirror local development
   - Common CI enhancements

7. **AI Evaluations as Regression Tests**
   - Why AI evals are unique to AI systems
   - Cost analysis with realistic numbers
   - Running AI evals (Brown's approach)
   - Manual-trigger CI workflow for AI evals
   - When to run AI evals (by development phase)

8. **Daily Development Workflow & Conclusion**
   - Step-by-step daily workflow checklist
   - Key takeaways
   - Companion notebook exercises
   - What we covered (and what we didn't)
   - Forward reference to Lessons 32-34

---

# Section-by-Section Writing Instructions

## Section 1 — Introduction: What is Continuous Integration?

**Source reference:** Reference the main introduction paragraphs, "What is Continuous Integration?", "CI for AI Agent Systems", and "Scope of This Lesson" subsections in `ci_source.md`

- **Quick reference to what we've learned in previous lessons:**  One sentence on what we’ve learnt in previous lessons, with a focus on lessons 28 and 29 as they are part of understanding and implementing Brown's AI evals layer.

- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` section and highlight the importance and existence of the lesson based on the `Why We Think It's Valuable` section.

Include 3 concrete failure modes:

- formatting churn and noisy diffs
- “works locally” but fails in PR because checks weren’t run
- flaky tests when they hit LLM APIs or depend on network

Make the key distinction:

- **Local checks** = fast feedback loop
- **CI checks** = shared enforcement

---

## Section 2 — pre-commit in this repo (what it is, how to use it)

### What to explain

- pre-commit runs checks automatically via git hooks.
- It prevents low-quality commits from being created in the first place.

### Show the repo’s config and interpret it

Open and walk through: **`.pre-commit-config.yaml`**.

Explain each hook in *plain English* and what it protects:

1. `validate-pyproject` (repo: `abravalheri/validate-pyproject`, `rev: v0.24.1`)
    - Ensures `pyproject.toml` is structurally valid.
2. `prettier` (repo: `pre-commit/mirrors-prettier`, `rev: v3.1.0`, `types_or: [yaml, json5]`)
    - Enforces consistent formatting for YAML/JSON-like config files.
3. Ruff hooks (repo: `astral-sh/ruff-pre-commit`, `rev: v0.12.1`)
    - `ruff-check` with `-fix` and `-exit-non-zero-on-fix`
    - `ruff-format`

Important nuance to teach (very practical):

- `-exit-non-zero-on-fix` means **Ruff can auto-fix**, but it will still fail the hook so you notice changes and re-stage them.
- Lint (`ruff-check --fix`) runs before formatting (`ruff-format`), which matches Ruff’s recommended ordering when using fixes. [GitHub+1](https://github.com/astral-sh/ruff-pre-commit?utm_source=chatgpt.com)

### Teach the workflow (using your Makefile)

From `Makefile` + README, show the intended commands:

- Run hooks manually anytime:
    - `make pre-commit` → `uv run pre-commit run --all-files`
- Check-only workflows:
    - `make format-check` → `uv run ruff format --check …`
    - `make lint-check` → `uv run ruff check …`
- Auto-fix workflows:
    - `make format-fix`
    - `make lint-fix`

Also teach the “first thing after cloning”:

- `pre-commit install` installs hooks into `.git/hooks/` (this is standard pre-commit usage). [Pre-commit](https://pre-commit.com/?utm_source=chatgpt.com)
- Optional: If they want uv-managed installation, point to uv’s pre-commit integration docs (good “golden source”). [Astral Docs+1](https://docs.astral.sh/uv/guides/integration/pre-commit/?utm_source=chatgpt.com)

---

## Section 3 — Ruff linting + formatting (repo-specific)

### What to explain

Give the beginner-friendly distinction:

- **Formatting** = consistent code style (automatic rewrite)
- **Linting** = catch bugs + suspicious code patterns

### Show the repo’s Ruff config

Open `pyproject.toml` and explicitly call out the settings that matter:

- `target-version = "py312"`
- `line-length = 140`
- Lint selection: `F`, `E`, `I` (pyflakes, pycodestyle errors, import sorting)
- isort config: `known-first-party = ["src", "tests"]`

Then connect it directly to workflow:

- Why line-length matters
- Why import sorting matters for readable diffs and fewer merge conflicts

### Concrete commands to show

Match the repo + CI usage (don’t invent new ones):

Local:

- `make format-check`
- `make lint-check`
- `make format-fix`
- `make lint-fix`

CI equivalent (preview what CI will do):

- `uv run ruff format --check …`
- `uv run ruff check …`

If you want to add a short external reference for readers, use official Ruff docs. [GitHub+1](https://github.com/astral-sh/ruff?utm_source=chatgpt.com)

---

## Section 4 — Unit tests for agent repos (and what to test)

### What to teach

Unit tests for agents should target **deterministic** code:

- parsing and rendering (markdown loaders, guideline parsing, etc.)
- schema validation and entity logic
- routing decisions (given input → expected decision)
- utilities (network helpers, transformations)

Make it explicit:

> We do not unit test the LLM. We test our code around it.
> 

### Show real tests from this repo

Use two examples:

**Example A: deterministic entity test**

- `tests/brown/domain/test_guidelines.py`
    - Demonstrates simple, deterministic test cases for `ArticleGuideline`.

**Example B: node test with LLM mocked**

- `tests/brown/nodes/test_article_writer.py`
    - Show how the model is created and then a response is injected:
        - `model, _ = build_model(app_config, node="write_article")`
        - `model.responses = [mock_response]`
    - Then the writer uses that model and the test asserts the behavior.

Teach why this is good:

- No network
- No API keys required
- Fully deterministic output
- Works in CI

### Show where the fake model comes from (Brown internals)

Walk through these files conceptually (no need to deep-dive, but show the mechanism):

- `configs/debug.yaml` uses `model_id: "fake"` for Brown nodes (ex: `write_article`, `review_article`, etc.)
- `src/brown/models/get_model.py` returns a `FakeModel` when the configured model is `FAKE_MODEL`
- `src/brown/models/fake_model.py` extends `FakeListChatModel` and supports structured output expectations

This is the key take-away:

> The repo is designed so unit tests can run with a fake model by default, and individual tests can inject specific responses when needed.
> 

### How to run tests (repo workflow)

Use the Makefile and CI parity:

Local:

- `make tests` (note: uses `CONFIG_FILE=./configs/debug.yaml uv run pytest`)

CI:

- same `CONFIG_FILE=configs/debug.yaml uv run pytest`

---

## Section 5 — GitHub Actions CI workflow walkthrough

### What to explain

- CI makes the rules non-optional.
- CI should mirror local workflow.

### Walk through `.github/workflows/ci.yml` (repo-specific)

Explain:

**Triggers**

- Runs on:
    - `pull_request` to `main` and `dev`
    - `push` to `main`

**Environment**

- `QA_FOLDERS: "src/brown src/nova scripts/ tests/"`

**Jobs**

There are two jobs:

1. `qa` job (format + lint)
- Checkout repo
- Install uv using `astral-sh/setup-uv@v4` [GitHub](https://github.com/astral-sh/setup-uv?utm_source=chatgpt.com)
- Set up Python using `actions/setup-python@v5` and `python-version-file: ".python-version"` (so CI uses 3.12.11)
- Install dependencies with `uv sync --dev`
- Run:
    - `uv run ruff format --check $QA_FOLDERS`
    - `uv run ruff check $QA_FOLDERS`
1. `tests` job (pytest)
- Checkout repo
- Install uv
- Set up Python from `.python-version`
- Install dependencies with `uv sync`
- Run:
    - `CONFIG_FILE=configs/debug.yaml uv run pytest`

Explain why splitting jobs is helpful:

- QA failures show up fast and clearly (format/lint), separate from test failures.

Optional “golden source” for learners who want to adapt CI:

- uv’s official GitHub Actions integration guide. [Astral Docs](https://docs.astral.sh/uv/guides/integration/github/?utm_source=chatgpt.com)

---

## Section 6 — Daily workflow checklist and conclusion

End with a practical checklist:

**Daily dev loop**

- Write code
- Run quick checks:
    - `make lint-check` / `make format-check` (or just `make pre-commit`)
- Run tests when changing logic:
    - `make tests`
- Commit (pre-commit runs automatically if installed)
- Push + open PR
- CI enforces the same checks on GitHub

Close with:

- “Now your repo is safer to collaborate on.”
- Point forward to future course topics (evaluation harnesses, deployment, etc.) without introducing new mechanics.

---

## Article Code (files to reference explicitly)

- `ci_source.md` (comprehensive source article covering all CI topics - to be used as primary reference for each section)