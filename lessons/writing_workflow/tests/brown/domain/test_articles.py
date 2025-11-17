"""Tests for brown.entities.articles module."""

from brown.entities.articles import Article, ArticleExample, ArticleExamples, SelectedText


class TestArticle:
    """Test the Article class."""

    def test_article_creation(self) -> None:
        """Test creating an article."""
        content = "# Test Article\n\nThis is test content."
        article = Article(content=content)
        assert article.content == content

    def test_article_to_context(self) -> None:
        """Test XML context generation."""
        article = Article(content="# Test\n\nContent")
        context = article.to_context()

        assert "<article>" in context
        assert "</article>" in context
        assert "# Test\n\nContent" in context

    def test_article_to_markdown(self) -> None:
        """Test markdown conversion."""
        content = "# Test Article\n\nThis is test content."
        article = Article(content=content)
        markdown = article.to_markdown()

        assert markdown == content

    def test_article_str_representation(self) -> None:
        """Test string representation."""
        article = Article(content="Test content")
        str_repr = str(article)

        assert "Article(len_content=12)" in str_repr

    def test_article_empty_content(self) -> None:
        """Test article with empty content."""
        article = Article(content="")
        assert article.content == ""
        assert article.to_markdown() == ""


class TestSelectedText:
    """Test the SelectedText class."""

    def test_selected_text_creation(self) -> None:
        """Test creating selected text."""
        article = Article(content="Test article content")
        selected_text = SelectedText(article=article, content="This is selected text", first_line_number=10, last_line_number=15)

        assert selected_text.content == "This is selected text"
        assert selected_text.first_line_number == 10
        assert selected_text.last_line_number == 15
        assert selected_text.article == article

    def test_selected_text_to_context(self) -> None:
        """Test selected text context generation."""
        article = Article(content="Test article content")
        selected_text = SelectedText(article=article, content="Selected content", first_line_number=5, last_line_number=8)
        context = selected_text.to_context()

        assert "<selected_text>" in context
        assert "</selected_text>" in context
        assert "Selected content" in context
        assert "<first_line_number>5</first_line_number>" in context
        assert "<last_line_number>8</last_line_number>" in context

    def test_selected_text_single_line(self) -> None:
        """Test selected text with single line."""
        article = Article(content="Test article content")
        selected_text = SelectedText(article=article, content="Single line", first_line_number=10, last_line_number=10)
        context = selected_text.to_context()

        assert "<first_line_number>10</first_line_number>" in context
        assert "<last_line_number>10</last_line_number>" in context


class TestArticleExample:
    """Test the ArticleExample class."""

    def test_article_example_creation(self) -> None:
        """Test creating an article example."""
        content = "# Example Article\n\nThis is an example."
        example = ArticleExample(content=content)
        assert example.content == content

    def test_article_example_to_context(self) -> None:
        """Test example context generation."""
        example = ArticleExample(content="# Example\n\nContent")
        context = example.to_context()

        assert "<article_example>" in context
        assert "</article_example>" in context
        assert "# Example\n\nContent" in context

    def test_article_example_str_representation(self) -> None:
        """Test string representation."""
        example = ArticleExample(content="Example content")
        str_repr = str(example)

        assert "ArticleExample(len_content=15)" in str_repr


class TestArticleExamples:
    """Test the ArticleExamples class."""

    def test_article_examples_creation(self) -> None:
        """Test creating article examples."""
        examples = [
            ArticleExample(content="# Example 1\n\nContent 1"),
            ArticleExample(content="# Example 2\n\nContent 2"),
        ]
        article_examples = ArticleExamples(examples=examples)

        assert len(article_examples.examples) == 2
        assert article_examples.examples[0].content == "# Example 1\n\nContent 1"
        assert article_examples.examples[1].content == "# Example 2\n\nContent 2"

    def test_article_examples_to_context(self) -> None:
        """Test examples context generation."""
        examples = [
            ArticleExample(content="# Example 1\n\nContent 1"),
            ArticleExample(content="# Example 2\n\nContent 2"),
        ]
        article_examples = ArticleExamples(examples=examples)
        context = article_examples.to_context()

        assert "<article_examples>" in context
        assert "</article_examples>" in context
        assert "# Example 1\n\nContent 1" in context
        assert "# Example 2\n\nContent 2" in context

    def test_article_examples_empty(self) -> None:
        """Test empty examples list."""
        article_examples = ArticleExamples(examples=[])
        context = article_examples.to_context()

        assert "<article_examples>" in context
        assert "</article_examples>" in context
        # Should not contain any example content
        assert "Example" not in context

    def test_article_examples_single(self) -> None:
        """Test single example."""
        example = ArticleExample(content="# Single Example\n\nContent")
        article_examples = ArticleExamples(examples=[example])
        context = article_examples.to_context()

        assert "# Single Example\n\nContent" in context
