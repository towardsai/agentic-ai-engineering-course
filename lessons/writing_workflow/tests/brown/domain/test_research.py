"""Tests for brown.entities.research module."""

from brown.entities.research import Research


class TestResearch:
    """Test the Research class."""

    def test_research_creation(self) -> None:
        """Test creating a research object."""
        content = "# Research\n\nThis is research content."
        research = Research(content=content)
        assert research.content == content

    def test_research_to_context(self) -> None:
        """Test research context generation."""
        research = Research(content="# Research\n\nAI research findings")
        context = research.to_context()

        assert "<research>" in context
        assert "</research>" in context
        assert "# Research\n\nAI research findings" in context

    def test_research_empty_content(self) -> None:
        """Test research with empty content."""
        research = Research(content="")
        assert research.content == ""

        context = research.to_context()
        assert "<research>" in context
        assert "</research>" in context

    def test_research_complex_content(self) -> None:
        """Test research with complex content."""
        content = """# Research: Machine Learning

## Overview
Machine learning is a subset of AI.

## Key Concepts
- Supervised learning
- Unsupervised learning
- Reinforcement learning

## Applications
- Image recognition
- Natural language processing
- Recommendation systems

## References
- [Deep Learning Book](https://example.com/book)
- [ML Course](https://example.com/course)
"""
        research = Research(content=content)
        assert research.content == content

        context = research.to_context()
        assert "Machine Learning" in context
        assert "Key Concepts" in context
        assert "Applications" in context
        assert "References" in context

    def test_research_str_representation(self) -> None:
        """Test string representation."""
        research = Research(content="Test research content")
        str_repr = str(research)

        assert "Research(len_content=21, len_image_urls=0)" in str_repr
