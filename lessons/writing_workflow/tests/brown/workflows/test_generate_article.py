"""End-to-end integration test for the generate-article workflow.

Runs the full workflow with every LLM call mocked (see this package's
`conftest.py`), exercising the real loading, model-building, review/edit loop and
rendering code paths -- fast and fully offline.
"""

from pathlib import Path
from typing import Callable

import pytest

from brown.workflows.generate_article import build_generate_article_workflow


class TestGenerateArticleWorkflow:
    """Smoke-level end-to-end run of the generate-article workflow."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generate_article_workflow(self, workflow_dir: Path, run_workflow: Callable) -> None:
        """Generating an article writes a non-empty final article plus the raw draft."""
        result = await run_workflow(build_generate_article_workflow, {"dir_path": workflow_dir})

        final_article = workflow_dir / "article.md"
        assert final_article.exists()
        assert final_article.read_text().strip()
        # The raw (pre-review) article is persisted as `article_000.md`.
        assert (workflow_dir / "article_000.md").exists()
        assert isinstance(result, str)
