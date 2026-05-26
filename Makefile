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

QA_FOLDERS := lessons/research_agent_part_2/ lessons/writing_workflow/ lessons/utils/

format-fix: # Auto-format Python code using ruff formatter.
	uv run ruff format $(QA_FOLDERS)

lint-fix: # Auto-fix linting issues using ruff linter.
	uv run ruff check --fix $(QA_FOLDERS)

format-check: # Check code formatting without making changes using ruff formatter.
	uv run ruff format --check $(QA_FOLDERS) 

lint-check: # Check code for linting issues without fixing them using ruff linter.
	uv run ruff check $(QA_FOLDERS)


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