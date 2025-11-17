"""Tests for brown.entities.guidelines module."""

from brown.entities.guidelines import ArticleGuideline


class TestArticleGuideline:
    """Test the ArticleGuideline class."""

    def test_article_guideline_creation(self) -> None:
        """Test creating an article guideline."""
        content = "# Guideline\n\nWrite about AI topics."
        guideline = ArticleGuideline(content=content)
        assert guideline.content == content

    def test_article_guideline_to_context(self) -> None:
        """Test guideline context generation."""
        guideline = ArticleGuideline(content="# Guideline\n\nWrite about AI.")
        context = guideline.to_context()

        assert "<article_guideline>" in context
        assert "</article_guideline>" in context
        assert "# Guideline\n\nWrite about AI." in context

    def test_article_guideline_empty_content(self) -> None:
        """Test guideline with empty content."""
        guideline = ArticleGuideline(content="")
        assert guideline.content == ""

        context = guideline.to_context()
        assert "<article_guideline>" in context
        assert "</article_guideline>" in context

    def test_article_guideline_complex_content(self) -> None:
        """Test guideline with complex content."""
        content = """# Article Guideline: AI Systems

## Objective
Write a comprehensive article about AI systems.

## Target Audience
- Software engineers
- AI practitioners

## Key Requirements
- Explain core concepts
- Include practical examples
- Provide code snippets

## Structure
1. Introduction
2. Technical details
3. Examples
4. Conclusion
"""
        guideline = ArticleGuideline(content=content)
        assert guideline.content == content

        context = guideline.to_context()
        assert "AI Systems" in context
        assert "Target Audience" in context
        assert "Key Requirements" in context
