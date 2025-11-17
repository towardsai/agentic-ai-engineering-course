"""Brown module-specific pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def mock_workflow_directory(tmp_path: Path) -> Path:
    """Create a mock workflow directory with required files."""
    # Create required files
    (tmp_path / "article_guideline.md").write_text("# Guideline\n\nWrite about AI")
    (tmp_path / "research.md").write_text("# Research\n\nAI research content")
    (tmp_path / "article.md").write_text("# Article\n\nGenerated article content")

    return tmp_path
