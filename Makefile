# --- Deploying Python Package ---

VERSION_BUMP ?= patch

build:
	uv version --bump $(VERSION_BUMP) # or minor/major
	uv build

publish:
	uv publish --token $(PYPI_TOKEN)

clean:
	rm -rf ./dist


# --- Tests & QA ---

# Tooling comes from the project's dev dependency group (pinned via uv.lock),
# matching the CI workflows (.github/workflows/*.yml).
RUFF := uv run ruff

# QA groups mirror the CI jobs in .github/workflows/ci.yml.
QA_WRITING  := lessons/writing_workflow/src lessons/writing_workflow/scripts lessons/writing_workflow/tests
QA_RESEARCH := lessons/research_agent_part_2/mcp_client lessons/research_agent_part_2/mcp_server lessons/research_agent_part_3/src lessons/research_agent_part_3/tests
QA_CORE     := lessons/agents_integration/mcp_client lessons/agents_integration/mcp_server lessons/utils scripts tests
QA_FOLDERS  := $(QA_WRITING) $(QA_RESEARCH) $(QA_CORE)
NOTEBOOKS   := $(shell git ls-files '*.ipynb' | grep -v '/inputs/')

format-fix: # Auto-format code and notebooks using ruff formatter.
	$(RUFF) format $(QA_FOLDERS) $(NOTEBOOKS)

lint-fix: # Auto-fix linting issues in code and notebooks using ruff linter.
	$(RUFF) check --fix $(QA_FOLDERS) $(NOTEBOOKS)

format-check: # Check formatting of code and notebooks without making changes.
	$(RUFF) format --check $(QA_FOLDERS) $(NOTEBOOKS)

lint-check: # Check code and notebooks for linting issues without fixing them.
	$(RUFF) check $(QA_FOLDERS) $(NOTEBOOKS)

notebooks-validate: # Verify every plain (tracked) lesson notebook conforms to the nbformat schema.
	uv run python -c "import sys, nbformat; [nbformat.validate(nbformat.read(f, as_version=4)) for f in sys.argv[1:]]" $(NOTEBOOKS)

test: # Run the writing_workflow unit tests (offline, mocked config).
	cd lessons/writing_workflow && CONFIG_FILE=configs/debug.yaml uv run pytest

test-nova: # Run the Nova research agent unit tests (offline, mocked).
	cd lessons/research_agent_part_2/mcp_server && uv run pytest
	cd lessons/research_agent_part_3 && uv run pytest

test-root: # Run the root-level tests (notebook env selector logic).
	uv run --no-project --with pytest --with python-dotenv --with google-genai pytest tests/

ci-local: format-check lint-check notebooks-validate test test-nova test-root # Run the full CI suite locally.


# --- Course data archives (served raw from GitHub) ---

DATA_SRC := lessons/writing_workflow
DATA_OUT := data

compress-configs: # Zip writing_workflow/configs into data/configs.zip (archive root: configs/).
	cd $(DATA_SRC) && rm -f configs.zip && zip -r configs.zip configs -x "*.DS_Store"
	mkdir -p $(DATA_OUT) && mv -f $(DATA_SRC)/configs.zip $(DATA_OUT)/configs.zip

compress-inputs: # Zip writing_workflow/inputs into data/inputs.zip (archive root: inputs/).
	cd $(DATA_SRC) && rm -f inputs.zip && zip -r inputs.zip inputs -x "*.DS_Store"
	mkdir -p $(DATA_OUT) && mv -f $(DATA_SRC)/inputs.zip $(DATA_OUT)/inputs.zip

compress-outputs: # Zip writing_workflow/outputs-cached into data/outputs.zip (archive root: outputs/).
	cd $(DATA_SRC) && rm -rf outputs && cp -R outputs-cached outputs && \
		rm -f outputs.zip && zip -r outputs.zip outputs -x "*.DS_Store" && rm -rf outputs
	mkdir -p $(DATA_OUT) && mv -f $(DATA_SRC)/outputs.zip $(DATA_OUT)/outputs.zip

compress-all: compress-configs compress-inputs compress-outputs # Rebuild all three data archives.