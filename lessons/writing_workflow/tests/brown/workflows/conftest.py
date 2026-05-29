"""Fixtures for the Brown workflow integration tests.

These tests run the three Brown workflows (generate article, edit article, edit
selected text) end-to-end while mocking every LLM call. The mocking comes "for
free" from the config: `tests/fixtures/configs/integration.yaml` mirrors
`configs/debug.yaml` and assigns the `fake` model to every node. Empty fake
models are filled with canned, schema-valid responses by each node's
`_set_default_model_mocked_responses`, so a full run completes in milliseconds
with no network access.

The workflow modules capture `app_config = get_app_config()` as a module-level
global at import time, so we inject the fake config by overwriting that global
directly. This keeps the tests deterministic regardless of the `CONFIG_FILE`
env var or `lru_cache` state when the modules were first imported.
"""

import importlib
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from brown.builders import build_short_term_memory
from brown.config_app import AppConfig

# .../writing_workflow -- the writing_workflow project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_CONFIG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "configs" / "integration.yaml"
SAMPLE_INPUTS_DIR = PROJECT_ROOT / "inputs" / "tests" / "00_sample_tiny"

# Workflow modules that bind `app_config` at import time and must be patched.
_WORKFLOW_MODULES = (
    "brown.workflows.generate_article",
    "brown.workflows.edit_article",
    "brown.workflows.edit_selected_text",
)


@pytest.fixture
def integration_config(monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    """Load the all-`fake` debug-style config and inject it into every workflow."""
    # The config's profile/example dirs are declared as relative paths and are
    # validated against the CWD, so anchor the CWD to the project root.
    monkeypatch.chdir(PROJECT_ROOT)

    config = AppConfig.from_yaml(INTEGRATION_CONFIG_PATH)
    for module_name in _WORKFLOW_MODULES:
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "app_config", config)

    return config


@pytest.fixture
def workflow_dir(tmp_path: Path) -> Path:
    """Copy the tiny sample article inputs into an isolated working directory.

    Provides `article_guideline.md`, `research.md` and `article.md` so the
    loaders have everything they need. Working on a copy keeps generated
    artefacts (`article_000.md`, `article.md`, ...) out of the repo.
    """
    for file_name in ("article_guideline.md", "research.md", "article.md"):
        shutil.copy(SAMPLE_INPUTS_DIR / file_name, tmp_path / file_name)

    return tmp_path


@pytest.fixture
def run_workflow(integration_config: AppConfig) -> Callable:
    """Return an async helper that drives a workflow to completion.

    Mirrors how the MCP server runs the workflows: an in-memory checkpointer plus
    a streamed run over the `custom` (progress) and `values` (result) modes, where
    the last `values` chunk is the workflow's return value. Pulling in
    `integration_config` guarantees the fake config is injected before the run.
    """

    async def _run(build_workflow: Callable, inputs: dict[str, Any]) -> Any:
        async with build_short_term_memory(integration_config) as checkpointer:
            workflow = build_workflow(checkpointer=checkpointer)
            run_config = {"configurable": {"thread_id": "integration-test"}}

            final_result: Any = None
            async for mode, chunk in workflow.astream(inputs, config=run_config, stream_mode=["custom", "values"]):
                if mode == "values":
                    final_result = chunk

        return final_result

    return _run
