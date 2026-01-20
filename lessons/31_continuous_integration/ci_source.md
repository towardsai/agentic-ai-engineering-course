# Lesson 31: Continuous Integration (CI) for AI Engineering

In Lesson 27 and Lesson 28, we introduced agent observability with Opik and started building our offline evaluation dataset. We learned that improving a system requires seeing what it does and capturing data to test against. In Lesson 29, we covered the evaluation-driven development framework, including the optimization flywheel and using AI evaluations as regression tests.

Now, we shift focus to Continuous Integration (CI): the automated infrastructure that keeps your codebase maintainable and prevents regressions from reaching production.

## What is Continuous Integration?

Continuous Integration is a software development practice where developers frequently merge code changes into a shared repository, with each merge triggering automated checks to detect integration issues early. The core principle: **catch bugs quickly and cheaply by testing every change automatically.**

Traditional CI pipelines include several standard components:

- **Build verification:** Ensure the code compiles and dependencies resolve correctly.
- **Automated testing:** Run unit tests, integration tests, and sometimes end-to-end tests.
- **Code quality checks:** Linting (catching bugs and style violations) and formatting (ensuring consistent style).
- **Security scanning:** Detect vulnerable dependencies and security issues.
- **Deployment preparation:** Build artifacts ready for staging or production.

Popular CI/CD platforms include **GitHub Actions**, **GitLab CI**, **Jenkins**, **CircleCI**, and **Travis CI**. While each has different syntax and features, they all support the same fundamental practices.

CI best practices that apply universally:

- **Fail fast:** Run quick checks first (linting, formatting) to catch simple issues before expensive tests.
- **Keep builds reproducible:** Lock dependency versions, use consistent environments, document setup.
- **Test automation:** Every testable aspect of the system should be tested automatically.
- **Small, frequent commits:** Smaller changes are easier to review, test, and debug when issues arise.
- **Treat CI failures seriously:** If the build is red, fixing it becomes the top priority.

### CI for AI Agent Systems

For AI agent projects, CI follows the same principles but with unique challenges. Prompts change frequently, schemas evolve, routing logic shifts, and LLM outputs are non-deterministic. Traditional testing approaches—asserting exact outputs, using real API calls—don't work well for LLM-powered systems. Without proper CI, your repository becomes a minefield where small changes break existing features in subtle, hard-to-detect ways.

The key difference: AI agents require an additional tier of validation—**AI evaluations**—that supplements traditional testing but is too expensive to run on every commit.

### Scope of This Lesson

This lesson covers CI essentials for building production-ready AI agents. We focus on practical techniques you'll use daily: automated quality checks, testing with mocked LLMs, and structuring CI pipelines around cost constraints.

We deliberately keep the scope focused on what AI engineers need for agent development. Comprehensive DevOps topics—container orchestration with Kubernetes, infrastructure as code with Terraform, advanced monitoring and alerting, multi-region deployments—are beyond this course. Those topics deserve dedicated courses. Instead, we teach what you need to move from prototype to production-ready agents with confidence.

In this lesson, we will cover:

- Why agent repositories need CI infrastructure from day one.
- Setting up pre-commit hooks to enforce code quality automatically.
- Configuring Ruff for linting and formatting.
- Writing unit tests for deterministic agent code with mocked LLM responses.
- Building a CI pipeline that runs automatically on every change.
- Using AI evaluations as selective regression tests in CI.

**Companion Notebook:** This lesson includes a hands-on notebook where you'll practice running formatting checks, linting, and tests on Brown, our writing agent. The notebook lets you create files with issues and fix them automatically using the tools discussed in this lesson. [Access the notebook here](notebook.ipynb).

## Why Repos Need CI Early

Traditional software engineering has learned the hard way that deferring quality infrastructure leads to technical debt. For all software projects—and especially AI agent repositories—the problem compounds quickly. Agent codebases change rapidly: you tweak prompts, adjust schemas, modify routing logic, and refactor tool definitions. Each change can have cascading effects across your system.

Here are three failure modes that emerge quickly without CI:

**Formatting churn and noisy diffs.** When team members use different formatters or no formatter at all, every pull request becomes a mess of whitespace changes and style inconsistencies. Code reviews waste time debating indentation instead of logic. Worse, merge conflicts multiply because the same lines get reformatted differently. This is a universal software problem, not unique to AI.

**"Works locally" syndrome.** A developer makes a change, tests it locally, and pushes code. Hours later, the build fails in CI—or worse, there is no CI, and the broken code ships to production. The root cause? They ran formatting checks locally but forgot to run tests. Or they ran tests but didn't notice a linting error. Without enforcement, manual checks get skipped. Again, this affects all software projects.

**Flaky tests dependent on external APIs.** Traditional software mocks databases and external APIs to keep tests fast and deterministic. For AI agents, the challenge intensifies: the temptation is to call real LLMs in tests to "verify actual behavior." But real API calls make tests slow, expensive, and non-deterministic. Your test suite becomes flaky: it passes one run and fails the next because the LLM returned a slightly different response. This destroys confidence in your tests and makes CI unreliable.

These problems compound quickly. As your team grows and the codebase matures, the cost of fixing them increases exponentially.

### Standard CI Pipeline Components

Standard CI pipelines for Python projects typically include:

- **Dependency installation:** Installing packages with pinned versions for reproducibility.
- **Linting and formatting:** Tools like Ruff, Black, isort, Flake8, or pylint check code quality.
- **Unit and integration tests:** Running pytest or unittest suites.
- **Type checking:** Using mypy or pyright for static type analysis.
- **Security scanning:** Tools like Dependabot, Snyk, or Safety check for vulnerable dependencies.
- **Code coverage:** Measuring what percentage of code is tested.
- **Documentation generation:** Building docs with Sphinx or MkDocs.
- **Build artifacts:** Packaging wheels or Docker images for deployment.

Our Nova and Brown agent projects use many of these standard components: dependency installation (uv), linting and formatting (Ruff), and unit tests (pytest). We skip some common additions—type checking with mypy, security scanning, code coverage reports, documentation generation—to keep the CI pipeline focused and fast. You can add these as your project matures.

### The Three-Tier CI Model for AI Agents

For AI agent projects, we adapt standard CI practices into three tiers based on cost and speed:

**Tier 1: Formatting and Linting (Always Run).** These checks are fast (seconds) and cheap (no API calls). They catch syntactic issues, enforce style consistency, and validate configuration files. Every commit should pass these checks before it even enters version control. **This tier is identical to traditional CI.**

**Tier 2: Unit and Integration Tests (Always Run).** These verify deterministic logic—parsing, schema validation, routing decisions, and utilities—without calling external APIs. By mocking LLM responses (just as traditional apps mock databases), tests run quickly (under a minute) and reliably. Every pull request should pass the full test suite. **This tier adapts traditional CI practices to AI's non-deterministic LLM dependency.**

**Tier 3: AI Evaluations (Manual/Release).** This tier is **unique to AI systems**. Traditional CI doesn't have an equivalent because unit tests are fast enough to run on every commit. For AI agents, we need a separate tier of expensive, LLM-based quality checks. These use real LLM calls to evaluate agent quality on a curated dataset. They are expensive (multiple API calls per sample) and slow (minutes to hours depending on dataset size). We run these selectively: before major releases, after significant prompt changes, or when debugging regressions. They serve as the ultimate regression test but are too costly for every commit.

This three-tier model balances speed, cost, and confidence. Fast checks run always; expensive checks run strategically.

```mermaid
flowchart TD
    subgraph tier1 [Tier 1: Every Commit - Seconds]
        A[Format Check] --> B[Lint Check]
    end
    
    subgraph tier2 [Tier 2: Every PR - Under 1 Minute]
        C[Unit Tests]
    end
    
    subgraph tier3 [Tier 3: Manual/Release - Minutes to Hours]
        D[AI Evaluations]
    end
    
    tier1 --> tier2
    tier2 --> tier3
    
    style tier1 fill:#e8f5e9
    style tier2 fill:#fff3e0
    style tier3 fill:#fce4ec
```

**What We Didn't Include (But You Might):**

Common CI components we omitted from our agents to keep the pipeline simple:
- **Type checking with mypy/pyright:** Useful for large teams and complex type hierarchies.
- **Security scanning:** Tools like Dependabot or Snyk detect vulnerable dependencies.
- **Code coverage reporting:** Tracks what percentage of code is tested.
- **Performance benchmarks:** Detect performance regressions in critical paths.
- **Documentation generation:** Auto-build docs on every commit.
- **Integration tests with staging:** Test against real services in a non-production environment.

You can add these as your project matures, but start with the essentials: format, lint, test.

Let's implement each tier for an AI agent project, using our Nova and Brown agents as concrete examples.

## Pre-commit Hooks: Automated Local Guardrails

Pre-commit hooks are Git hooks that run automatically before you create a commit. They catch issues immediately, in your local development environment, before the code ever touches CI. This fast feedback loop is invaluable: you fix problems within seconds, not minutes or hours later when CI fails.

### The Pre-commit Framework

**Pre-commit** (https://pre-commit.com/) is an open-source framework that manages Git hooks using a declarative YAML configuration. Instead of writing custom shell scripts in `.git/hooks/`, you define hooks in `.pre-commit-config.yaml` and the framework handles installation and execution.

Pre-commit's key innovation: hooks are defined as references to external repositories. This means the community maintains hooks for popular tools, and you just reference them. When you run `pre-commit install`, it downloads the specified hooks and sets up your local Git hooks automatically.

### Example Pre-commit Configuration

Here's a minimal pre-commit configuration for an AI agent project:

```yaml
# .pre-commit-config.yaml
fail_fast: false

repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.1  # Use latest stable version
    hooks:
      - id: ruff-check
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

This configuration runs Ruff's linter and formatter on every commit. Let's add more sophisticated hooks:

```yaml
# .pre-commit-config.yaml for an AI agent project
fail_fast: false

repos:
  # Validate pyproject.toml structure
  - repo: https://github.com/abravalheri/validate-pyproject
    rev: v0.24.1
    hooks:
      - id: validate-pyproject

  # Format YAML/JSON configuration files
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
      - id: prettier
        types_or: [yaml, json5]

  # Lint and format Python code
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.1
    hooks:
      - id: ruff-check
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

Let's understand each hook:

**`validate-pyproject` (from `abravalheri/validate-pyproject`):** This tool validates that your `pyproject.toml` file is structurally correct according to PEP 518 and PEP 621 standards. It's maintained by contributors to the Python Packaging Authority ecosystem. Since `pyproject.toml` defines dependencies, build configuration, and tool settings, a malformed file breaks the entire project. This hook catches syntax errors immediately.

**`prettier` (from `pre-commit/mirrors-prettier`):** Prettier is a popular opinionated code formatter originally built for JavaScript but with support for multiple languages including YAML, JSON, Markdown, and more. The `pre-commit/mirrors-prettier` repository is an official mirror maintained by the pre-commit team, providing Prettier as a pre-commit hook. We use it to format configuration files like `.github/workflows/ci.yml` and `configs/agent-config.yaml`. Consistent formatting makes these files readable and reduces merge conflicts.

**`ruff-check` and `ruff-format` (from `astral-sh/ruff-pre-commit`):** Ruff is a modern Python linter and formatter written in Rust by Astral (the creators of uv). The `astral-sh/ruff-pre-commit` repository provides official pre-commit hooks for Ruff. We'll discuss Ruff in detail in the next section, but key flags:
- `--fix` automatically fixes issues it can (unused imports, incorrect indentation, etc.)
- `--exit-non-zero-on-fix` causes the hook to fail even when it auto-fixes issues, forcing you to review and re-stage the changes

This ensures you see what was fixed and can verify it makes sense.

**Important:** The `ruff-format` hook always runs after `ruff-check` (per Ruff's recommendation) because linting can modify code that then needs reformatting.

**In our Nova and Brown agents, we use this exact configuration.** You can adapt it by adding or removing hooks based on your project's needs.

### Brown's Pre-commit Configuration

Our Brown writing agent uses this exact pre-commit configuration. You can find it at `lessons/writing_workflow/.pre-commit-config.yaml`:

```yaml
fail_fast: false

repos:
  - repo: https://github.com/abravalheri/validate-pyproject
    rev: v0.24.1
    hooks:
      - id: validate-pyproject

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.1.0
    hooks:
      - id: prettier
        types_or: [yaml, json5]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.1
    hooks:
      - id: ruff-check
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```

This configuration ensures that every commit to Brown's codebase is validated, formatted, and linted automatically. When you work with Brown in the accompanying notebook for this lesson, you'll see these hooks in action.

**Try it yourself:** The companion notebook walks you through running these pre-commit hooks manually on Brown's codebase. You'll see exactly what checks run and how they catch issues before code enters version control.

### Setting Up Pre-commit

After cloning an agent repository, set up pre-commit hooks with:

```bash
# Install dependencies (includes pre-commit)
uv sync --dev  # or: pip install pre-commit

# Install the Git hooks
pre-commit install
```

The `pre-commit install` command creates a Git hook at `.git/hooks/pre-commit`. Now every time you run `git commit`, pre-commit runs automatically.

You can also run hooks manually anytime:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Or use a convenience wrapper (if you have a Makefile)
make pre-commit
```

This is useful when you want to check your changes before committing or after pulling changes from others.

### The Pre-commit Workflow

The workflow becomes:

1. Make code changes.
2. Run `git add` to stage files.
3. Run `git commit`. Pre-commit hooks execute automatically.
4. If hooks fail:
   - Review the changes and error messages.
   - Fix any issues (or accept auto-fixes).
   - Run `git add` again to re-stage modified files.
   - Run `git commit` again.
5. If hooks pass, the commit is created successfully.

This tight feedback loop catches formatting and linting issues within seconds, before they ever reach CI.

## Ruff: Fast Python Linting and Formatting

Ruff is an extremely fast Python linter and formatter written in Rust. Over the past two years, Ruff has rapidly become the Python community standard, replacing a collection of older tools:
- **Black** (formatter)
- **isort** (import sorter)
- **Flake8** (linter)
- **pyupgrade** (Python version upgrade helper)
- **pydocstyle** (docstring linter)

By consolidating these tools into a single binary, Ruff dramatically speeds up checks. In our repositories, Ruff runs in milliseconds even on hundreds of files. For comparison, the old tool chain (Black + isort + Flake8) could take several seconds on the same codebase.

### Formatting vs. Linting

It's important to understand the distinction:

**Formatting** rewrites code to follow consistent style rules (indentation, line breaks, quotes, spacing). It's automatic and opinionated. There's no "right" or "wrong"—just consistency. Running a formatter twice on the same code produces identical output (idempotent).

**Linting** analyzes code for bugs, suspicious patterns, and violations of best practices. It flags issues like unused variables, missing imports, complex expressions, and potential errors. Some issues can be auto-fixed; others require manual intervention.

Both are essential. Formatting keeps diffs clean; linting catches bugs before they reach production.

### Ruff Configuration for AI Agent Projects

Ruff configuration lives in `pyproject.toml`. Here's a template for an AI agent project:

```toml
# pyproject.toml
[tool.ruff]
target-version = "py312"  # Adjust to your Python version
line-length = 140          # Adjust based on team preference (88 is Black default, 120-140 common for modern teams)

[tool.ruff.lint]
select = [
    "F",    # Pyflakes - catches common bugs
    "E",    # pycodestyle errors - enforces PEP 8
    "I",    # isort - organizes imports
]

[tool.ruff.lint.isort]
known-first-party = ["src", "tests"]  # Adjust to your project structure
```

Let's break down each setting:

**`target-version = "py312"`** tells Ruff which Python version you're using, enabling syntax checks specific to that version. If you use Python 3.11, set this to `"py311"`.

**`line-length = 140`** sets the maximum line length. While PEP 8 recommends 79 characters, modern screens and syntax-aware editors make longer lines readable. Common choices: 88 (Black default), 120, or 140. Pick what works for your team.

**`select = ["F", "E", "I"]`** enables three rule sets:
- **F (Pyflakes):** Catches common bugs like undefined names, unused imports, and duplicate keys.
- **E (pycodestyle errors):** Enforces PEP 8 style guidelines like proper indentation and whitespace.
- **I (isort):** Sorts and groups imports consistently.

**Other popular lint rules you might add:**
- **B (bugbear):** Catches likely bugs and design problems (e.g., mutable default arguments).
- **N (naming):** Enforces PEP 8 naming conventions.
- **UP (pyupgrade):** Suggests modern Python syntax (e.g., type hints instead of comments).
- **W (pycodestyle warnings):** Additional style warnings beyond errors.
- **C90 (mccabe):** Checks code complexity.

**`known-first-party = ["src", "tests"]`** tells isort that `src` and `tests` are first-party modules (your code), ensuring imports are grouped correctly:
1. Standard library imports
2. Third-party imports
3. First-party imports (your code)
4. Local imports (relative imports)

**In our Nova and Brown agents, we use `line-length=140` and lint rules F, E, I.** Your project might need different settings based on team preferences and existing codebases.

### Brown's Ruff Configuration

Brown's actual Ruff configuration is defined in `lessons/writing_workflow/pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 140

[tool.ruff.lint]
select = [
    "F",    # Pyflakes
    "E",    # pycodestyle errors
    "I",    # isort
]

[tool.ruff.lint.isort]
known-first-party = ["src", "tests"]
```

This configuration uses Python 3.12, allows longer lines (140 characters) for readability on modern screens, and enables the three essential rule sets. When you work with Brown's codebase, these rules ensure consistent code quality across all contributions.

### Running Ruff Locally

Ruff provides two main commands:

**Check without changes:**
```bash
ruff format --check src/ tests/  # Check if code is formatted correctly
ruff check src/ tests/            # Check for linting issues
```

These commands exit with an error if issues are found but don't modify files. Useful for verifying everything is clean (e.g., in CI).

**Auto-fix issues:**
```bash
ruff format src/ tests/  # Reformat code automatically
ruff check --fix src/ tests/  # Fix linting issues automatically
```

These commands modify files in place. Run them before committing to ensure your changes pass CI.

**Using Make targets (convenience wrappers):**

For convenience, you can wrap these in Make targets or shell scripts:

```makefile
# Makefile
QA_FOLDERS := src/ tests/ scripts/

format-check:
	ruff format --check $(QA_FOLDERS)

lint-check:
	ruff check $(QA_FOLDERS)

format-fix:
	ruff format $(QA_FOLDERS)

lint-fix:
	ruff check --fix $(QA_FOLDERS)
```

Then run:
```bash
make format-check  # or: make lint-check
make format-fix    # or: make lint-fix
```

### Brown's Makefile QA Targets

Brown provides these exact convenience wrappers in `lessons/writing_workflow/Makefile`. Here are the key QA-related targets:

```makefile
# --- Tests & QA ---

QA_FOLDERS := src/ tests/ scripts/

tests: # Run tests.
	CONFIG_FILE=configs/debug.yaml uv run pytest

pre-commit: # Run pre-commit hooks.
	uv run pre-commit run --all-files

format-fix: # Auto-format Python code using ruff formatter.
	uv run ruff format $(QA_FOLDERS)

lint-fix: # Auto-fix linting issues using ruff linter.
	uv run ruff check --fix $(QA_FOLDERS)

format-check: # Check code formatting without making changes.
	uv run ruff format --check $(QA_FOLDERS) 

lint-check: # Check code for linting issues without fixing them.
	uv run ruff check $(QA_FOLDERS)
```

The `QA_FOLDERS` variable defines which directories to check (source code, tests, and scripts). Each target uses `uv run` to execute commands within the virtual environment. You can run these targets from the `writing_workflow/` directory:

```bash
cd lessons/writing_workflow
make format-check   # Check formatting
make lint-check     # Check linting
make format-fix     # Auto-fix formatting
make lint-fix       # Auto-fix linting issues
make tests          # Run test suite
```

The typical workflow:

1. Write code.
2. Run `ruff check --fix .` and `ruff format .` periodically to see if you've introduced issues.
3. Before committing, run these commands one final time.
4. Review the changes (with `git diff`) to ensure the fixes are correct.
5. Commit. Pre-commit hooks run the checks again as a safety net.

This iterative process becomes second nature and takes seconds per iteration.

### Hands-On Example: Fixing Formatting Issues

To understand Ruff's formatter in practice, let's create a simple Python file with formatting issues and fix them. This example demonstrates what the formatter does automatically.

**Step 1: Create a file with formatting issues**

```python
# test_formatting.py - intentionally poorly formatted
def  badly_formatted_function(x,y,z):
    result=x+y+z
    my_list=[1,2,3,4,5,6,7,8,9,10]
    my_dict={"key1":"value1","key2":"value2","key3":"value3"}
    if result>10:
        print("Result is greater than 10")
    else:
        print("Result is 10 or less")
    return result

class   BadlyFormattedClass:
    def __init__(self,name,age):
        self.name=name
        self.age=age
    def get_info(self):
        return f"{self.name} is {self.age} years old"
```

**Step 2: Check formatting (without fixing)**

```bash
ruff format --check test_formatting.py
```

Output:
```
Would reformat: test_formatting.py
1 file would be reformatted
```

The `--check` flag reports issues without modifying the file.

**Step 3: Auto-fix formatting**

```bash
ruff format test_formatting.py
```

Output:
```
1 file reformatted
```

**Step 4: View the result**

```python
# test_formatting.py - after Ruff formatting
def badly_formatted_function(x, y, z):
    result = x + y + z
    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    my_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}
    if result > 10:
        print("Result is greater than 10")
    else:
        print("Result is 10 or less")
    return result


class BadlyFormattedClass:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_info(self):
        return f"{self.name} is {self.age} years old"
```

Notice how Ruff automatically:
- Fixed spacing around operators (`x+y+z` → `x + y + z`)
- Added proper spacing in function signatures
- Formatted lists and dictionaries consistently
- Fixed class definition spacing
- Added proper line breaks between class methods

This is the power of automated formatting: consistent style with zero manual effort. You can find this hands-on example in the companion notebook.

### Hands-On Example: Fixing Linting Issues

Linting goes beyond formatting—it checks for bugs and code quality issues. Let's see what Ruff's linter catches.

**Step 1: Create a file with linting issues**

```python
# test_linting.py
import os
import sys
import json  # Unused import

def calculate_sum(numbers):
    """Calculate sum of numbers."""
    total = 0
    for num in numbers:
        total = total + num
    return total

def process_data(data):
    """Process some data."""
    result = calculate_sum(data)
    print(f"Result: {result}")
    undefined_variable = some_undefined_function()  # Using undefined name
    return result

# Duplicate import at the bottom
import sys
```

**Step 2: Check for linting issues**

```bash
ruff check test_linting.py
```

Output:
```
test_linting.py:3:8: F401 [*] `json` imported but unused
test_linting.py:17:26: F821 Undefined name `some_undefined_function`
test_linting.py:21:8: F811 [*] Redefinition of unused `sys` from line 2
Found 3 errors.
[*] 2 fixable with the `--fix` option.
```

Ruff reports:
- **F401**: Unused import (`json`)
- **F821**: Undefined name (`some_undefined_function`)
- **F811**: Duplicate import (`sys`)

**Step 3: Auto-fix what can be fixed**

```bash
ruff check --fix test_linting.py
```

Output:
```
Fixed 2 errors:
- test_linting.py:3:8: F401 [*] `json` imported but unused
- test_linting.py:21:8: F811 [*] Redefinition of unused `sys` from line 2
Found 1 error (2 fixed, 1 remaining).
```

Ruff automatically removed the unused and duplicate imports. The undefined name requires manual intervention—it's a logic error, not a style issue.

**Step 4: View the result**

```python
# test_linting.py - after auto-fix
import os
import sys

def calculate_sum(numbers):
    """Calculate sum of numbers."""
    total = 0
    for num in numbers:
        total = total + num
    return total

def process_data(data):
    """Process some data."""
    result = calculate_sum(data)
    print(f"Result: {result}")
    undefined_variable = some_undefined_function()  # Still an error - requires manual fix
    return result
```

The unused `json` import and duplicate `sys` import are gone. The undefined function remains because Ruff can't guess what you intended.

**This demonstrates the two-tier approach:**
1. **Auto-fixable issues** (imports, formatting): Ruff handles automatically
2. **Logic errors** (undefined names): Require developer attention

You can try this example yourself in the companion notebook.

## Unit Tests for Agent Repos

Unit tests verify that your code behaves correctly under controlled conditions. For AI agents, this is tricky: the core logic involves calling LLMs, which return non-deterministic outputs. You can't assert exact outputs, and real API calls make tests slow and flaky.

The solution mirrors traditional software engineering: **mock external dependencies.** Just as traditional apps mock databases, APIs, and external services, AI agents must mock LLM calls.

### Standard Practice: Mocking External Dependencies

In traditional CI, mocking external dependencies is standard practice:
- **Database mocking:** Use in-memory SQLite instead of PostgreSQL, or mock the database layer entirely.
- **API mocking:** Return pre-scripted responses instead of calling real APIs.
- **File system mocking:** Use temporary directories that are cleaned up after tests.
- **Time mocking:** Control the clock to test time-dependent logic.

The reasons are universal:
- **Speed:** Real external calls add seconds or minutes to test runs.
- **Reliability:** External services can be down, slow, or rate-limited.
- **Determinism:** Mocked responses are consistent; real services return variable data.
- **Cost:** Some APIs charge per call.

For AI agents, the LLM is the most critical external dependency. Mocking it is essential.

### What to Test in AI Agents

Focus on deterministic parts of your agent:

- **Parsing and rendering:** Does your markdown loader correctly extract articles? Does your renderer format output properly?
- **Schema validation:** Does your Pydantic model reject invalid data? Do required fields enforce constraints?
- **Routing decisions:** Given a specific state, does your workflow route to the correct node?
- **Utilities:** Do helper functions like URL extraction, text cleaning, and file I/O work correctly?
- **State management:** Does your workflow correctly update and pass state between nodes?

These components have clear inputs and expected outputs. You can write fast, reliable tests without any LLM involvement.

For nodes that do call LLMs (writers, reviewers, editors, tool-calling agents), you mock the LLM responses. This lets you test the node's logic—how it constructs prompts, processes responses, updates state—without making expensive, non-deterministic API calls.

### Common LLM Mocking Approaches

There are several patterns for mocking LLM calls in tests:

**1. Response Injection (simplest):**
Create a fake model class that returns pre-scripted responses. This is what we use in our agents.

```python
class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
```

**2. Fixture-based Mocking:**
Use pytest fixtures to provide consistent test data.

```python
@pytest.fixture
def mock_llm_response():
    return "This is a mocked LLM response."

def test_agent_node(mock_llm_response):
    agent = Agent(llm=FakeLLM([mock_llm_response]))
    result = agent.run("user input")
    assert "mocked" in result
```

**3. VCR/Cassette Pattern (record/replay):**
Record real LLM responses once, then replay them in tests. Libraries like `vcr.py` or `pytest-vcr` support this. Useful for testing against actual API behavior without repeated calls.

```python
@pytest.mark.vcr
def test_agent_with_real_response_once():
    # First run: makes real API call and records response
    # Subsequent runs: replays recorded response
    agent = Agent(llm=RealLLM())
    result = agent.run("test input")
    assert result is not None
```

For most AI agent projects, **response injection** (approach 1) provides the best balance of simplicity and control.

### Example Mocking Pattern

Here's a general pattern for testing an AI agent node with mocked responses:

```python
import pytest

# Generic agent node that calls an LLM
class AgentNode:
    def __init__(self, llm, guidelines: str, context: str):
        self.llm = llm
        self.guidelines = guidelines
        self.context = context
    
    async def execute(self) -> str:
        prompt = f"Guidelines: {self.guidelines}\nContext: {self.context}\nGenerate output:"
        response = await self.llm.generate(prompt)
        return self.process_response(response)
    
    def process_response(self, response: str) -> str:
        # Your processing logic here
        return response.strip()

# Simple fake LLM for testing
class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.index = 0
    
    async def generate(self, prompt: str) -> str:
        response = self.responses[self.index]
        self.index += 1
        return response

# Test using the fake LLM
@pytest.mark.asyncio
async def test_agent_node_with_mocked_llm():
    mock_response = "Generated output based on guidelines"
    fake_llm = FakeLLM(responses=[mock_response])
    
    node = AgentNode(
        llm=fake_llm,
        guidelines="Write a summary",
        context="AI agents are useful"
    )
    
    result = await node.execute()
    
    assert result == "Generated output based on guidelines"
    assert fake_llm.index == 1  # Verify LLM was called once
```

### Our Implementation: The FakeModel Pattern

**In our Brown agent, we implement response injection with a `FakeModel` class that extends LangChain's `FakeListChatModel`.** This provides compatibility with LangChain's interface while allowing us to inject responses.

The pattern has three parts:

**Part 1: Configuration specifies the fake model**

Brown uses a YAML configuration file to specify which model to use for each node. The `debug.yaml` configuration at `lessons/writing_workflow/configs/debug.yaml` sets all nodes to use the fake model:

```yaml
nodes:
  write_article:
    model_id: "fake"
    model_config:
      temperature: 0.7
      include_thoughts: false
      thinking_budget: 6144
  review_article:
    model_id: "fake"
    model_config:
      temperature: 0.0
      include_thoughts: false
      thinking_budget: null
  edit_article:
    model_id: "fake"
    model_config:
      temperature: 0.1
      include_thoughts: false
      thinking_budget: 8192
```

When `model_id: "fake"` is specified, the model builder returns a `FakeModel` instance instead of a real LLM.

**Part 2: Model factory returns FakeModel when configured**

Brown's `FakeModel` class is defined in `lessons/writing_workflow/src/brown/models/fake_model.py`:

```python
class FakeModel(FakeListChatModel):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses=responses)
        self._structured_output_type: Type[Any] | None = None
        self._include_raw: bool = False

    def bind_tools(self, tools, *args, **kwargs) -> Self:
        return self

    def with_structured_output(self, output_type: Type[Any], include_raw: bool = False) -> Self:
        self._structured_output_type = output_type
        self._include_raw = include_raw
        return self

    async def ainvoke(self, inputs, *args, **kwargs) -> Any:
        if len(self.responses) == 0:
            return []
        
        if self._structured_output_type is not None:
            response_content = self.responses[0]
            
            if isinstance(response_content, dict):
                structured_response = self._structured_output_type(**response_content)
            elif isinstance(response_content, str):
                data = json.loads(response_content)
                structured_response = self._structured_output_type(**data)
            # ... handle raw output if needed
            
            return structured_response
        
        # For non-structured output, use parent's implementation
```

This `FakeModel` supports both structured outputs (Pydantic models) and tool calling, making it compatible with Brown's various node types.

**Part 3: Tests inject specific responses**

Brown's test structure is organized in `lessons/writing_workflow/tests/`:

```
tests/
├── brown/
│   ├── conftest.py           # Brown-specific fixtures
│   ├── domain/               # Tests for domain entities
│   │   ├── test_articles.py
│   │   ├── test_guidelines.py
│   │   ├── test_research.py
│   │   └── test_reviews.py
│   ├── evals/                # Tests for evaluation code
│   │   └── metrics/
│   ├── nodes/                # Tests for agent nodes (with mocked LLMs)
│   │   ├── test_article_writer.py
│   │   ├── test_article_reviewer.py
│   │   └── test_media_generator.py
│   └── utils/                # Tests for utility functions
└── conftest.py               # Root fixtures
```

Here's an example test from `lessons/writing_workflow/tests/brown/nodes/test_article_writer.py`:

```python
@pytest.mark.asyncio
async def test_article_writer_ainvoke_success(
    mock_article_guideline: ArticleGuideline,
    mock_research: Research,
    mock_article_profiles: ArticleProfiles,
    mock_media_items: MediaItems,
    mock_article_examples: ArticleExamples,
) -> None:
    """Test article generation with mocked response."""
    mock_response = '{"content": "# Generated Article\\n### Mock Subtitle\\n\\nThis is a generated article about AI.\\n\\n## Section 1\\nMock section 1.\\n\\n## Conclusion\\nMock conclusion.\\n"}'
    
    app_config = get_app_config()
    model, _ = build_model(app_config, node="write_article")
    model.responses = [mock_response]
    
    writer = ArticleWriter(
        article_guideline=mock_article_guideline,
        research=mock_research,
        article_profiles=mock_article_profiles,
        media_items=mock_media_items,
        article_examples=mock_article_examples,
        model=model,
    )
    
    result = await writer.ainvoke()
    
    assert isinstance(result, Article)
    assert "# Generated Article" in result.content
```

The test:
1. Creates a mock JSON response matching the expected output schema
2. Builds a fake model using Brown's configuration system
3. Injects the mock response into the model's `responses` list
4. Instantiates the `ArticleWriter` node with the fake model
5. Calls the node and asserts on the output

This pattern keeps tests fast (no API calls), deterministic (same response every time), and free (no costs).

This three-part pattern is used throughout Brown's test suite. You'll see these patterns in action in the accompanying notebook for this lesson.

### Example: Testing Deterministic Components

Here's a simple test for a deterministic entity (no LLM involved):

```python
from pydantic import BaseModel

class AgentGuideline(BaseModel):
    content: str
    
    def to_prompt(self) -> str:
        return f"<guidelines>{self.content}</guidelines>"

def test_guideline_to_prompt():
    guideline = AgentGuideline(content="Write about AI")
    prompt = guideline.to_prompt()
    
    assert "<guidelines>" in prompt
    assert "</guidelines>" in prompt
    assert "Write about AI" in prompt
```

No mocking needed—just pure input/output testing.

### Example: Testing Nodes with Mocked Responses

Here's a pattern for testing a node that calls an LLM:

```python
@pytest.mark.asyncio
async def test_agent_node_generates_output():
    # Arrange: set up fake LLM with mocked response
    mock_response = '{"output": "Generated content about AI agents"}'
    fake_llm = FakeLLM(responses=[mock_response])
    
    # Arrange: create node with fake LLM
    node = AgentNode(
        llm=fake_llm,
        guidelines="Write about agents",
        context="AI context..."
    )
    
    # Act: execute the node
    result = await node.execute()
    
    # Assert: verify behavior
    assert "Generated content" in result
    assert fake_llm.call_count == 1
```

We're not testing whether the LLM generates good content—that's what AI evaluations are for (covered later). We're testing that our node correctly:
1. Constructs prompts from inputs
2. Calls the LLM
3. Parses and processes the response
4. Returns the expected output format

### Running Tests

Run your test suite with pytest:

```bash
pytest                          # Run all tests
pytest tests/test_nodes.py      # Run specific test file
pytest -v                       # Verbose output
pytest -k "test_agent"          # Run tests matching pattern
```

**Using configuration for test mode:**

```bash
# Set config to use fake models
CONFIG_FILE=configs/test.yaml pytest
```

Or in a Makefile:

```makefile
tests:
	CONFIG_FILE=configs/test.yaml pytest
```

**In our Nova and Brown agents, we run tests with:**

```bash
CONFIG_FILE=./configs/debug.yaml pytest
```

The `debug.yaml` configuration specifies fake models for all nodes, ensuring tests never call real LLMs.

### Running Brown's Tests

To run Brown's test suite, use the command specified in Brown's Makefile at `lessons/writing_workflow/Makefile`:

```bash
# From the writing_workflow directory
cd lessons/writing_workflow
CONFIG_FILE=configs/debug.yaml uv run pytest
```

Or use the convenience target:

```bash
make tests
```

Brown's test suite includes:
- **Domain tests**: Testing Pydantic models and data structures (no LLM calls)
- **Node tests**: Testing agent nodes with mocked LLM responses
- **Utility tests**: Testing helper functions and utilities
- **Evaluation tests**: Testing evaluation metrics and dataset handling

The complete test suite runs in under a minute and requires no API keys. Every test is deterministic: run it 100 times, get the same result every time. This reliability is essential for CI.

**Try it yourself:** The companion notebook includes exercises where you'll run Brown's test suite and see the three-tier CI model in action. You'll observe the speed difference between Tier 1 (formatting/linting in seconds), Tier 2 (tests in under a minute), and understand why Tier 3 (AI evals) must be run selectively.

## CI Workflows: Automated Enforcement

Local checks (pre-commit, tests) provide fast feedback, but they're optional. A developer can skip them—intentionally or accidentally—and push broken code. CI provides enforcement: it runs the same checks automatically on every push and pull request. If checks fail, the PR can't be merged.

### CI Platforms

While we use GitHub Actions in our agents, these principles apply to any CI platform:

- **GitHub Actions:** Native to GitHub, YAML-based, generous free tier, marketplace of reusable actions.
- **GitLab CI:** Native to GitLab, YAML-based, integrated with GitLab features.
- **Jenkins:** Self-hosted, highly customizable, scriptable, requires infrastructure management.
- **CircleCI:** Cloud-hosted, YAML-based, Docker-first, good free tier for open source.
- **Travis CI:** Cloud-hosted, YAML-based, popular for open source projects.

The core concepts translate across platforms: triggers, jobs/stages, steps/commands, environment variables, artifacts, and caching.

### Example GitHub Actions Workflow

Here's a CI workflow template for an AI agent project:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main, dev]  # Run on PRs to these branches
  push:
    branches: [main]       # Run on direct pushes to main

env:
  # Customize these folders for your project
  QA_FOLDERS: "src/ tests/ scripts/"

jobs:
  # Job 1: Code quality checks (fast)
  qa:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv (or pip/poetry)
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: ".python-version"  # Or: python-version: "3.12"

      - name: Install dependencies
        run: uv sync --dev  # Or: pip install -r requirements-dev.txt

      - name: Format check
        run: uv run ruff format --check $QA_FOLDERS

      - name: Lint check
        run: uv run ruff check $QA_FOLDERS

  # Job 2: Run tests
  tests:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: ".python-version"

      - name: Install dependencies
        run: uv sync  # Or: pip install -r requirements.txt

      - name: Run tests
        run: |
          CONFIG_FILE=configs/test.yaml uv run pytest
        # Or without config: pytest
```

### Key Workflow Components

**Triggers (`on:`):**
- `pull_request`: Runs when PRs are opened or updated.
- `push`: Runs on direct commits to specified branches.
- `schedule`: Runs on a cron schedule (e.g., nightly tests).
- `workflow_dispatch`: Manual trigger from GitHub UI.

**Jobs:**
- Jobs run in parallel by default (unless dependencies are specified).
- Each job runs in a fresh VM, isolated from other jobs.
- Use `needs:` to create dependencies between jobs.

**Steps:**
- Steps within a job run sequentially.
- Each step either runs a command (`run:`) or uses an action (`uses:`).
- Actions are reusable workflow components from the GitHub Marketplace.

**Common Actions:**
- `actions/checkout@v4`: Checks out your repository code.
- `actions/setup-python@v5`: Installs a specific Python version.
- `actions/cache@v4`: Caches dependencies for faster builds.
- `actions/upload-artifact@v4`: Uploads build artifacts (test reports, logs).

### Why Split Jobs?

We split QA and tests into separate jobs for two reasons:

**Parallelization:** Both jobs run simultaneously, reducing total CI time. If QA takes 30 seconds and tests take 60 seconds, total time is 60 seconds, not 90.

**Clear feedback:** If formatting fails, you see "QA job failed" immediately. You don't have to dig through test logs to find the formatting error. This speeds up debugging.

### Making CI Mirror Local Development

**Critical principle:** CI should run exactly the same commands you run locally. This eliminates "works on my machine" problems.

**Local commands:**
```bash
ruff format --check src/ tests/
ruff check src/ tests/
pytest
```

**CI commands (should be identical):**
```yaml
run: ruff format --check src/ tests/
run: ruff check src/ tests/
run: pytest
```

If tests pass locally, they'll pass in CI. If CI fails, you can reproduce the issue locally by running the same commands.

### Common CI Enhancements

**Dependency caching (speeds up builds):**
```yaml
- name: Cache dependencies
  uses: actions/cache@v4
  with:
    path: ~/.cache/uv  # Or ~/.cache/pip
    key: ${{ runner.os }}-uv-${{ hashFiles('**/pyproject.toml') }}
```

**Matrix testing (test multiple Python versions):**
```yaml
strategy:
  matrix:
    python-version: ["3.11", "3.12"]
steps:
  - uses: actions/setup-python@v5
    with:
      python-version: ${{ matrix.python-version }}
```

**Artifact uploads (save test reports):**
```yaml
- name: Upload test results
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: test-results
    path: test-reports/
```

**In our Nova and Brown agents, we use a two-job workflow:** one for QA (format + lint) and one for tests. The workflow uses `uv` for dependency management and reads the Python version from `.python-version` for reproducibility.

## AI Evals as Regression Tests

In Lesson 29, we discussed using AI evaluations to quantify system quality and guide optimization. But evaluations serve another critical purpose: regression testing.

Recall from Lesson 29:

> Your new feature might work perfectly on its own. However, because you touched parts of the code that affect other features, such as modifying a shared system_prompt to support a new tool, the likelihood of breaking other functionality is relatively high.

This is the regression testing problem. Every change risks breaking existing features. In traditional software, we write unit tests that verify specific behaviors and run them on every commit. **In AI systems, we supplement unit tests with AI evaluations that verify semantic quality.**

### Why AI Evals Are Unique to AI Systems

**Traditional CI doesn't have an equivalent to Tier 3 evaluations.** In traditional software:
- Unit tests are fast (milliseconds to seconds).
- Integration tests are moderate (seconds to minutes).
- End-to-end tests are slower (minutes) but still manageable.

All of these can run on every commit without breaking the bank.

**For AI agents, semantic quality testing requires real LLM calls:**
- Running your agent on evaluation inputs (LLM calls for generation).
- Computing metrics with LLM judges (more LLM calls for evaluation).
- Each sample might require 5-10 LLM calls.
- A modest dataset of 10 samples = 50-100 LLM calls per evaluation run.

### Cost Analysis

Let's do the math for a realistic AI agent evaluation:

**Assumptions:**
- Dataset: 10 test samples
- Per sample: 2 LLM calls for generation (with large context + thinking tokens), 2 metrics × 1 LLM call each for evaluation
- Total: ~40 LLM calls per evaluation run
- Average cost per call (Gemini 2.5 Pro with thinking): $0.05
- **Total cost per run: ~$2.00**

**Frequency implications:**
- Run on every commit (50/day for a team of 5): **$100/day = $2,000/month**
- Run before each release (weekly): **$8/month**
- Run manually when needed: **~$20/month**

For a team making dozens of commits daily, running evals on every commit would cost thousands of dollars monthly and slow development significantly (each run takes several minutes).

**This is why Tier 3 exists:** AI evaluations are too expensive and slow for continuous execution, so we run them selectively.

### Running AI Evals

For an AI agent project, you might run evaluations with a command like:

```bash
# Generic pattern
python -m scripts.run_eval \
    --dataset-name my-eval-dataset \
    --metrics metric1 \
    --metrics metric2 \
    --config configs/production.yaml
```

Or wrapped in a convenience command:

```bash
make run-eval
```

**In our Brown agent, we run evaluations with:**

```bash
CONFIG_FILE=./configs/course-gemini-2.5-pro.yaml python -m scripts.brown_run_eval \
    --dataset-name brown-course-lessons \
    --metrics follows_gt \
    --metrics user_intent \
    --split test \
    --cache-dir outputs/evals \
    --workers 1
```

Key points:
- Uses the production config (real model, not fake), to test actual behavior.
- Evaluates custom metrics relevant to our use case.
- Caches results to avoid redundant regeneration during iteration.

Results are typically saved locally and uploaded to an observability platform (like Opik) for analysis.

### Manual-Trigger CI Workflow for AI Evals

While we don't run AI evals automatically, we can create a CI workflow that runs them on demand. Here's a template:

```yaml
# .github/workflows/eval.yml
name: AI Evaluations

on:
  workflow_dispatch:  # Manual trigger only
    inputs:
      dataset:
        description: 'Dataset to evaluate'
        required: false
        default: 'production-eval-dataset'
      config:
        description: 'Config file'
        required: false
        default: 'configs/production.yaml'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Install dependencies
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version-file: ".python-version"

      - name: Install project
        run: uv sync

      - name: Run evaluations
        env:
          # Securely provide API keys via GitHub Secrets
          LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
          EVAL_API_KEY: ${{ secrets.EVAL_API_KEY }}
        run: |
          python -m scripts.run_eval \
            --dataset ${{ github.event.inputs.dataset }} \
            --config ${{ github.event.inputs.config }}

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: outputs/evals/
```

This workflow:
- Triggers manually via GitHub Actions UI ("Run workflow" button).
- Accepts input parameters (which dataset, which config).
- Runs the full evaluation suite with real LLM calls.
- Uploads results as downloadable artifacts.
- Stores API keys securely in GitHub Secrets (never hardcoded).

**Alternative: Scheduled Evaluations**

Some teams run nightly evaluations:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Every day at 2 AM UTC
  workflow_dispatch:     # Also allow manual trigger
```

This catches regressions daily without blocking development. The trade-off: you discover issues hours after they're introduced rather than immediately.

### When to Run AI Evals

The decision depends on your project's maturity and risk tolerance:

**Early development (prototype phase):**
- Run manually when you want to measure progress.
- Frequency: weekly or after major changes.
- Goal: track improvement over time.

**Active development (building features):**
- Run before merging significant changes (multi-day PRs, prompt refactors, schema changes).
- Frequency: a few times per week.
- Goal: catch regressions before they reach main branch.

**Mature product (in production):**
- Run as part of your release process. Treat evaluations as release gates.
- Frequency: before every release (weekly, biweekly, monthly).
- Goal: ensure production quality never degrades.

**During optimization:**
- Run after each change when actively optimizing (as described in Lesson 29's optimization flywheel).
- Goal: measure improvement from each iteration.

The key is making AI evals systematic without making them a bottleneck. Manual triggers give you control while keeping costs manageable.

**In our project:** We run evaluations manually before releases and when making significant changes to prompts or agent architecture. This balances cost, speed, and confidence.

## Daily Development Workflow

With all these tools in place, here's a recommended daily development workflow for AI agent projects:

**1. Start your work session:**
```bash
git checkout -b feature/my-feature
```

**2. Write code:**
- Make your changes to source files.
- Write or update tests as needed.

**3. Run quick checks periodically:**
```bash
# Direct commands (works for any project)
ruff format --check src/ tests/
ruff check src/ tests/

# Or using convenience wrappers (if available)
make lint-check
make format-check
```

This gives you fast feedback (< 1 second) on style issues.

**4. Before committing, fix issues:**
```bash
# Direct commands
ruff format src/ tests/
ruff check --fix src/ tests/

# Or using convenience wrappers
make format-fix
make lint-fix
```

Review changes with `git diff` to ensure auto-fixes are correct.

**5. Run tests if you changed logic:**
```bash
# Direct command
CONFIG_FILE=configs/test.yaml pytest

# Or using convenience wrapper
make tests
```

This takes under a minute and catches regressions in deterministic code.

**6. Commit your changes:**
```bash
git add .
git commit -m "Add feature X"
```

Pre-commit hooks run automatically (if installed). If they fail, fix issues and re-commit.

**7. Push and open a PR:**
```bash
git push origin feature/my-feature
```

Open a pull request on your platform (GitHub, GitLab, etc.). CI runs automatically.

**8. CI provides feedback:**
- If QA or tests fail, click the failing job to see logs.
- Fix issues locally, push again. CI re-runs automatically.
- Once CI passes and reviewers approve, merge the PR.

**9. Before releasing, run AI evals:**
```bash
# Generic command for your project
python -m scripts.run_eval --dataset your-dataset

# Or using convenience wrapper
make run-eval
```

Review metrics in your observability platform. If quality has regressed, investigate before releasing.

This workflow takes seconds for most commits and catches issues early. The faster the feedback loop, the cheaper bugs are to fix.

**Practice this workflow:** The companion notebook walks you through this entire workflow using Brown as a concrete example. You'll create files with issues, run checks, fix them automatically, and run tests—experiencing the complete CI cycle hands-on.

## Conclusion

Continuous Integration for AI agents adapts traditional software engineering practices to the unique challenges of LLM-powered systems. The core CI principles—automated testing, reproducible builds, fast feedback—remain the same. The adaptation is in the three-tier model: fast checks (always), deterministic tests with mocked LLMs (always), and expensive AI evaluations (selectively).

**Key takeaways:**

- **CI fundamentals apply to AI agents:** Formatting, linting, unit tests, and automated enforcement are standard practices we inherit from traditional software engineering.
- **Pre-commit hooks** catch formatting and linting issues in seconds, before code enters version control.
- **Ruff** consolidates multiple Python tools (Black, isort, Flake8) into a single fast binary.
- **Mocking LLMs** in tests (using response injection, fixtures, or VCR patterns) enables fast, deterministic testing without API calls or costs.
- **CI platforms** (GitHub Actions, GitLab CI, Jenkins, etc.) enforce checks automatically, preventing broken code from reaching production.
- **AI evaluations** are unique to AI systems: too expensive for every commit, they serve as selective regression tests before releases and major changes.

This infrastructure is not optional for production AI systems. The investment in CI pays for itself the first time it catches a regression before customers see it.

**Hands-on practice:** Don't just read about CI—experience it. The [companion notebook](notebook.ipynb) lets you practice every concept from this lesson:
- Run formatting and linting checks on real code
- Create files with issues and fix them automatically
- Run Brown's test suite with mocked LLMs
- See the three-tier CI model in action with actual timing measurements

The notebook transforms these concepts from theory into muscle memory.

### What We Covered (and What We Didn't)

In this lesson, we covered CI essentials for building production-ready AI agents:
- Automated quality checks (pre-commit, linting, formatting)
- Testing patterns for non-deterministic systems (mocked LLMs)
- CI workflow structure and best practices
- Cost-aware evaluation strategies (three-tier model)

**Topics beyond this lesson's scope:**

These important DevOps topics deserve dedicated courses and are beyond our focus on AI agent development:
- **Container orchestration:** Kubernetes, Docker Swarm, service meshes
- **Infrastructure as code:** Terraform, Pulumi, CloudFormation
- **Advanced monitoring:** Distributed tracing, log aggregation, alerting strategies
- **Multi-region deployment:** Load balancing, failover, geo-distribution
- **Security hardening:** Secrets management, network policies, compliance scanning
- **Performance optimization:** Caching strategies, CDNs, database tuning

**In Lessons 32-34, we'll cover deployment essentials:** Docker containerization, database state management, authentication with Descope, and continuous deployment with GitHub Actions. These lessons focus on getting your agents from localhost to production with confidence, building on the CI foundation we established today.

---

**Additional Resources:**

- [Ruff Documentation](https://docs.astral.sh/ruff/) - Official Ruff docs with comprehensive rule listings
- [Pre-commit Framework](https://pre-commit.com/) - Pre-commit hook framework documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions) - Complete GitHub Actions guide
- [uv GitHub Actions Integration](https://docs.astral.sh/uv/guides/integration/github/) - Using uv in CI/CD
- [pytest Documentation](https://docs.pytest.org/) - Testing framework used in most Python projects
- Lesson 29: Evaluation processes and optimization flywheel
- Lesson 28: Building the evaluation dataset
- Lesson 27: Agent observability with Opik
