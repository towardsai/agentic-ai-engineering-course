"""Tests for brown.data.loaders module."""

from pathlib import Path

import pytest

from brown.entities.exceptions import InputNotFoundException
from brown.loaders import (
    MarkdownArticleExampleLoader,
    MarkdownArticleGuidelineLoader,
    MarkdownArticleLoader,
    MarkdownArticleProfilesLoader,
    MarkdownResearchLoader,
)


class TestMarkdownArticleLoader:
    """Test the MarkdownArticleLoader class."""

    def test_article_loader_success(self, tmp_path: Path) -> None:
        """Test loading a valid article."""
        article_file = tmp_path / "article.md"
        article_content = "# Test Article\n\nThis is test content."
        article_file.write_text(article_content)

        loader = MarkdownArticleLoader(uri="article.md")
        article = loader.load(working_uri=tmp_path)

        assert article.content == article_content

    def test_article_loader_file_not_found(self, tmp_path: Path) -> None:
        """Test loading a non-existent article file."""
        loader = MarkdownArticleLoader(uri="non_existent.md")
        with pytest.raises(InputNotFoundException):
            loader.load(working_uri=tmp_path)


class TestMarkdownArticleGuidelineLoader:
    """Test the MarkdownArticleGuidelineLoader class."""

    def test_article_guideline_loader_success(self, tmp_path: Path) -> None:
        """Test loading a valid article guideline."""
        guideline_file = tmp_path / "guideline.md"
        guideline_content = "# Guideline\n\nWrite about AI topics."
        guideline_file.write_text(guideline_content)

        loader = MarkdownArticleGuidelineLoader(uri="guideline.md")
        guideline = loader.load(working_uri=tmp_path)

        assert guideline.content == guideline_content

    def test_article_guideline_loader_file_not_found(self, tmp_path: Path) -> None:
        """Test loading a non-existent guideline file."""
        loader = MarkdownArticleGuidelineLoader(uri="non_existent.md")
        with pytest.raises(InputNotFoundException):
            loader.load(working_uri=tmp_path)


class TestMarkdownResearchLoader:
    """Test the MarkdownResearchLoader class."""

    def test_research_loader_success(self, tmp_path: Path) -> None:
        """Test loading valid research content."""
        research_file = tmp_path / "research.md"
        research_content = "# Research\n\nThis is research content."
        research_file.write_text(research_content)

        loader = MarkdownResearchLoader(uri="research.md")
        research = loader.load(working_uri=tmp_path)

        assert research.content == research_content

    def test_research_loader_with_markdown_links(self, tmp_path: Path) -> None:
        """Test loading research with markdown links that get cleaned."""
        research_file = tmp_path / "research.md"
        research_content = "Check [this link](https://example.com) for more info."
        research_file.write_text(research_content)

        loader = MarkdownResearchLoader(uri="research.md")
        research = loader.load(working_uri=tmp_path)

        # Links should be cleaned
        assert "https://example.com" in research.content
        assert "[this link]" not in research.content

    def test_research_loader_file_not_found(self, tmp_path: Path) -> None:
        """Test loading a non-existent research file."""
        loader = MarkdownResearchLoader(uri="non_existent.md")
        with pytest.raises(InputNotFoundException):
            loader.load(working_uri=tmp_path)


class TestMarkdownArticleProfilesLoader:
    """Test the ArticleProfilesLoader class."""

    def test_profiles_loader_success(self, tmp_path: Path) -> None:
        """Test loading all profiles successfully."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "character_profiles").mkdir()

        # Create profile files
        profile_files = {
            "article": profiles_dir / "article_profile.md",
            "character": profiles_dir / "character_profiles" / "paul_iusztin.md",
            "mechanics": profiles_dir / "mechanics_profile.md",
            "structure": profiles_dir / "structure_profile.md",
            "terminology": profiles_dir / "terminology_profile.md",
            "tonality": profiles_dir / "tonality_profile.md",
        }

        profile_contents = {
            "article": "# Article Profile\n\nWrite technical articles.",
            "character": "# Character Profile\n\nPaul Iusztin's style.",
            "mechanics": "# Mechanics Profile\n\nGrammar and formatting rules.",
            "structure": "# Structure Profile\n\nArticle structure guidelines.",
            "terminology": "# Terminology Profile\n\nTechnical terminology rules.",
            "tonality": "# Tonality Profile\n\nTone and voice guidelines.",
        }

        for profile_type, file_path in profile_files.items():
            file_path.write_text(profile_contents[profile_type])

        loader = MarkdownArticleProfilesLoader(uri=profile_files)
        profiles = loader.load(working_uri=tmp_path)

        assert profiles.article.content == profile_contents["article"]
        assert profiles.character.content == profile_contents["character"]
        assert profiles.mechanics.content == profile_contents["mechanics"]
        assert profiles.structure.content == profile_contents["structure"]
        assert profiles.terminology.content == profile_contents["terminology"]
        assert profiles.tonality.content == profile_contents["tonality"]

    def test_profiles_loader_missing_file(self, tmp_path: Path) -> None:
        """Test loading profiles with a missing file."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        profile_files = {
            "article": profiles_dir / "article_profile.md",
            "character": profiles_dir / "character_profiles" / "paul_iusztin.md",  # This will be missing
            "mechanics": profiles_dir / "mechanics_profile.md",
            "structure": profiles_dir / "structure_profile.md",
            "terminology": profiles_dir / "terminology_profile.md",
            "tonality": profiles_dir / "tonality_profile.md",
        }

        # Create only some files
        (profiles_dir / "article_profile.md").write_text("# Article Profile")
        (profiles_dir / "mechanics_profile.md").write_text("# Mechanics Profile")
        (profiles_dir / "structure_profile.md").write_text("# Structure Profile")
        (profiles_dir / "terminology_profile.md").write_text("# Terminology Profile")
        (profiles_dir / "tonality_profile.md").write_text("# Tonality Profile")

        loader = MarkdownArticleProfilesLoader(uri=profile_files)
        with pytest.raises(InputNotFoundException):
            loader.load(working_uri=tmp_path)

    def test_profiles_loader_get_supported_profiles(self) -> None:
        """Test getting supported profile types."""
        supported = MarkdownArticleProfilesLoader.get_supported_profiles()
        expected = ["character", "article", "structure", "mechanics", "terminology", "tonality"]
        assert set(supported) == set(expected)


class TestMarkdownArticleExampleLoader:
    """Test the ArticleExampleLoader class."""

    def test_article_example_loader_success(self, tmp_path: Path) -> None:
        """Test loading example articles successfully."""
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()

        # Create example files
        (examples_dir / "example1.md").write_text("# Example 1\n\nContent 1")
        (examples_dir / "example2.md").write_text("# Example 2\n\nContent 2")
        (examples_dir / "example3.md").write_text("# Example 3\n\nContent 3")

        loader = MarkdownArticleExampleLoader(uri=examples_dir)
        examples = loader.load(working_uri=tmp_path)

        assert len(examples.examples) == 3

        # Check that all expected contents are present (order may vary)
        contents = [example.content for example in examples.examples]
        assert "# Example 1\n\nContent 1" in contents
        assert "# Example 2\n\nContent 2" in contents
        assert "# Example 3\n\nContent 3" in contents

    def test_article_example_loader_empty_dir(self, tmp_path: Path) -> None:
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loader = MarkdownArticleExampleLoader(uri=empty_dir)
        examples = loader.load(working_uri=tmp_path)

        assert len(examples.examples) == 0

    def test_article_example_loader_dir_not_found(self, tmp_path: Path) -> None:
        """Test loading from non-existent directory."""
        non_existent_dir = tmp_path / "non_existent"

        loader = MarkdownArticleExampleLoader(uri=non_existent_dir)
        with pytest.raises(InputNotFoundException):
            loader.load(working_uri=tmp_path)

    def test_article_example_loader_ignores_non_md_files(self, tmp_path: Path) -> None:
        """Test that loader ignores non-markdown files."""
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()

        # Create mixed files
        (examples_dir / "example.md").write_text("# Example\n\nContent")
        (examples_dir / "readme.txt").write_text("This is a text file")
        (examples_dir / "config.json").write_text('{"key": "value"}')

        loader = MarkdownArticleExampleLoader(uri=examples_dir)
        examples = loader.load(working_uri=tmp_path)

        # Should only load the .md file
        assert len(examples.examples) == 1
        assert examples.examples[0].content == "# Example\n\nContent"
