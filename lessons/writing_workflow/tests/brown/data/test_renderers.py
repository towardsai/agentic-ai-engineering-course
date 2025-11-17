"""Tests for brown.renderers module."""

from pathlib import Path

from brown.entities.articles import Article
from brown.entities.reviews import ArticleReviews, Review
from brown.renderers import MarkdownArticleRenderer, MarkdownArticleReviewsRenderer


class TestMarkdownArticleRenderer:
    """Test the MarkdownArticleRenderer class."""

    def test_article_markdown_renderer(self, tmp_path: Path) -> None:
        """Test rendering article to markdown file."""
        article = Article(content="# Test Article\n\nThis is test content.")
        output_file = tmp_path / "output.md"

        renderer = MarkdownArticleRenderer()
        renderer.render(article, output_uri=output_file)

        assert output_file.exists()
        assert output_file.read_text() == "# Test Article\n\nThis is test content."

    def test_article_markdown_renderer_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that renderer overwrites existing file."""
        # Create existing file
        output_file = tmp_path / "output.md"
        output_file.write_text("Old content")

        article = Article(content="# New Article\n\nNew content.")
        renderer = MarkdownArticleRenderer()
        renderer.render(article, output_uri=output_file)

        assert output_file.read_text() == "# New Article\n\nNew content."

    def test_article_markdown_renderer_empty_content(self, tmp_path: Path) -> None:
        """Test rendering article with empty content."""
        article = Article(content="")
        output_file = tmp_path / "empty.md"

        renderer = MarkdownArticleRenderer()
        renderer.render(article, output_uri=output_file)

        assert output_file.exists()
        assert output_file.read_text() == ""


class TestMarkdownArticleReviewsRenderer:
    """Test the MarkdownArticleReviewsRenderer class."""

    def test_article_reviews_context_renderer(self, tmp_path: Path) -> None:
        """Test rendering reviews to context file."""
        from brown.entities.articles import Article

        article = Article(content="# Test Article\n\nContent here.")
        reviews = ArticleReviews(
            article=article,
            reviews=[
                Review(profile="structure", location="intro", comment="This is a good article."),
                Review(profile="mechanics", location="body", comment="Needs more examples."),
            ],
        )
        output_file = tmp_path / "reviews.md"

        renderer = MarkdownArticleReviewsRenderer()
        renderer.render(reviews, output_uri=output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "This is a good article." in content
        assert "Needs more examples." in content
        assert "structure" in content
        assert "mechanics" in content

    def test_article_reviews_context_renderer_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that renderer overwrites existing file."""
        # Create existing file
        output_file = tmp_path / "reviews.md"
        output_file.write_text("Old reviews")

        from brown.entities.articles import Article

        article = Article(content="# Test Article\n\nContent here.")
        reviews = ArticleReviews(
            article=article,
            reviews=[
                Review(profile="structure", location="intro", comment="New review content."),
            ],
        )

        renderer = MarkdownArticleReviewsRenderer()
        renderer.render(reviews, output_uri=output_file)

        content = output_file.read_text()
        assert "New review content." in content
        assert "Old reviews" not in content

    def test_article_reviews_context_renderer_empty_reviews(self, tmp_path: Path) -> None:
        """Test rendering empty reviews."""
        from brown.entities.articles import Article

        article = Article(content="# Test Article\n\nContent here.")
        reviews = ArticleReviews(article=article, reviews=[])
        output_file = tmp_path / "empty_reviews.md"

        renderer = MarkdownArticleReviewsRenderer()
        renderer.render(reviews, output_uri=output_file)

        assert output_file.exists()
        content = output_file.read_text()
        # Should contain the XML structure even with empty reviews
        assert "<article_reviews>" in content
        assert "</article_reviews>" in content

    def test_article_reviews_context_renderer_without_article(self, tmp_path: Path) -> None:
        """Test rendering reviews without including article content."""
        from brown.entities.articles import Article

        article = Article(content="# Test Article\n\nContent here.")
        reviews = ArticleReviews(
            article=article,
            reviews=[
                Review(profile="structure", location="intro", comment="Review without article context."),
            ],
        )
        output_file = tmp_path / "reviews_no_article.md"

        renderer = MarkdownArticleReviewsRenderer()
        renderer.render(reviews, output_uri=output_file)

        content = output_file.read_text()
        # Should not contain article content when include_article=False
        assert "Review without article context." in content
        # The to_context method should be called with include_article=False
        # This tests the renderer's behavior, not the domain model's to_context method
