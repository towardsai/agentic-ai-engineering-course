"""End-to-end integration test for the edit-article workflow.

Runs the full workflow with every LLM call mocked (see this package's
`conftest.py`), exercising the real loading, model-building, review/edit loop and
rendering code paths -- fast and fully offline.
"""

from pathlib import Path
from typing import Callable

import pytest

from brown.workflows.edit_article import build_edit_article_workflow


class TestEditArticleWorkflow:
    """Smoke-level end-to-end run of the edit-article workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_edit_article_workflow(self, workflow_dir: Path, run_workflow: Callable) -> None:
        """Editing a whole article returns the edited article wrapped in instructions."""
        result = await run_workflow(
            build_edit_article_workflow,
            {"dir_path": workflow_dir, "human_feedback": "Make the introduction more concise."},
        )

        assert isinstance(result, str)
        assert "Here is the edited article:" in result
