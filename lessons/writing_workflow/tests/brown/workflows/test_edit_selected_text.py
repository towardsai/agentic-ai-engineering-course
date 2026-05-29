"""End-to-end integration test for the edit-selected-text workflow.

Runs the full workflow with every LLM call mocked (see this package's
`conftest.py`), exercising the real loading, model-building, review/edit loop and
rendering code paths -- fast and fully offline.
"""

from pathlib import Path
from typing import Callable

import pytest

from brown.workflows.edit_selected_text import build_edit_selected_text_workflow


class TestEditSelectedTextWorkflow:
    """Smoke-level end-to-end run of the edit-selected-text workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edit_selected_text_workflow(self, workflow_dir: Path, run_workflow: Callable) -> None:
        """Editing a selected snippet returns the edited text wrapped in instructions."""
        selected_text = "The distinction between an AI workflow and an AI agent is best understood as a spectrum of autonomy."
        result = await run_workflow(
            build_edit_selected_text_workflow,
            {
                "dir_path": workflow_dir,
                "human_feedback": "Tighten this sentence.",
                "selected_text": selected_text,
                "number_line_before_selected_text": 5,
                "number_line_after_selected_text": 5,
            },
        )

        assert isinstance(result, str)
        assert "Here is the edited selected text:" in result
