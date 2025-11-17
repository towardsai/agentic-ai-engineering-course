"""Tests for brown.entities.reviews module."""

from brown.entities.articles import Article, SelectedText
from brown.entities.reviews import ArticleReviews, HumanFeedback, Review, SelectedTextReviews


class TestReview:
    """Test the Review class."""

    def test_review_creation(self) -> None:
        """Test creating a review."""
        review = Review(profile="structure", location="introduction", comment="This is a good article.")
        assert review.profile == "structure"
        assert review.location == "introduction"
        assert review.comment == "This is a good article."

    def test_review_to_context(self) -> None:
        """Test review context generation."""
        review = Review(profile="tonality", location="conclusion", comment="Great article!")
        context = review.to_context()

        assert "<review>" in context
        assert "</review>" in context
        assert "<profile>tonality</profile>" in context
        assert "<location>conclusion</location>" in context
        assert "<comment>Great article!</comment>" in context

    def test_review_score_validation(self) -> None:
        """Test review creation with different profiles."""
        # Different profiles
        review1 = Review(profile="structure", location="intro", comment="Good structure")
        review2 = Review(profile="mechanics", location="body", comment="Good mechanics")
        review3 = Review(profile="terminology", location="conclusion", comment="Good terminology")

        assert review1.profile == "structure"
        assert review2.profile == "mechanics"
        assert review3.profile == "terminology"

    def test_review_str_representation(self) -> None:
        """Test string representation."""
        review = Review(profile="test", location="test", comment="Test review")
        str_repr = str(review)

        # Review doesn't have a custom __str__ method, so it uses the default
        assert "profile='test'" in str_repr
        assert "comment='Test review'" in str_repr


class TestArticleReviews:
    """Test the ArticleReviews class."""

    def test_article_reviews_creation(self) -> None:
        """Test creating article reviews."""
        article = Article(content="# Test Article\n\nContent here.")
        reviews = [
            Review(profile="structure", location="intro", comment="Good article"),
            Review(profile="mechanics", location="body", comment="Needs improvement"),
        ]
        article_reviews = ArticleReviews(article=article, reviews=reviews)

        assert len(article_reviews.reviews) == 2
        assert article_reviews.reviews[0].comment == "Good article"
        assert article_reviews.reviews[1].comment == "Needs improvement"

    def test_article_reviews_to_context(self) -> None:
        """Test article reviews context generation."""
        article = Article(content="# Test Article\n\nContent here.")
        reviews = [
            Review(profile="structure", location="intro", comment="Excellent work"),
            Review(profile="mechanics", location="body", comment="Good but could be better"),
        ]
        article_reviews = ArticleReviews(article=article, reviews=reviews)
        context = article_reviews.to_context()

        assert "<article_reviews>" in context
        assert "</article_reviews>" in context
        assert "<comment>Excellent work</comment>" in context
        assert "<comment>Good but could be better</comment>" in context
        assert "<profile>structure</profile>" in context
        assert "<profile>mechanics</profile>" in context

    def test_article_reviews_with_article(self) -> None:
        """Test article reviews with article content."""
        article = Article(content="# Test Article\n\nThis is the article content.")
        reviews = [Review(profile="structure", location="intro", comment="Good article")]
        article_reviews = ArticleReviews(article=article, reviews=reviews)

        context = article_reviews.to_context(include_article=True)

        # Should include article content (as string representation)
        assert "Article(len_content=" in context
        # Should include reviews
        assert "<article_reviews>" in context
        assert "Good article" in context

    def test_article_reviews_without_article(self) -> None:
        """Test article reviews without article content."""
        article = Article(content="# Test Article\n\nThis is the article content.")
        reviews = [Review(profile="structure", location="intro", comment="Review without article")]
        article_reviews = ArticleReviews(article=article, reviews=reviews)

        context = article_reviews.to_context(include_article=False)

        # Should not include article content
        assert "# Test Article" not in context
        # Should include reviews
        assert "<article_reviews>" in context
        assert "Review without article" in context

    def test_article_reviews_empty(self) -> None:
        """Test empty reviews collection."""
        article = Article(content="# Test Article\n\nContent here.")
        article_reviews = ArticleReviews(article=article, reviews=[])
        context = article_reviews.to_context()

        assert "<article_reviews>" in context
        assert "</article_reviews>" in context
        # Should not contain any review content
        assert "<comment>" not in context

    def test_article_reviews_single(self) -> None:
        """Test single review."""
        article = Article(content="# Test Article\n\nContent here.")
        review = Review(profile="structure", location="intro", comment="Single review")
        article_reviews = ArticleReviews(article=article, reviews=[review])
        context = article_reviews.to_context()

        assert "Single review" in context
        assert "<profile>structure</profile>" in context

    def test_article_reviews_str_representation(self) -> None:
        """Test string representation."""
        article = Article(content="# Test Article\n\nContent here.")
        reviews = [
            Review(profile="structure", location="intro", comment="Review 1"),
            Review(profile="mechanics", location="body", comment="Review 2"),
        ]
        article_reviews = ArticleReviews(article=article, reviews=reviews)
        str_repr = str(article_reviews)

        assert "Reviews(len_reviews=2)" in str_repr


class TestHumanFeedback:
    """Test the HumanFeedback class."""

    def test_human_feedback_creation(self) -> None:
        """Test creating human feedback."""
        feedback = HumanFeedback(content="Please improve the introduction")
        assert feedback.content == "Please improve the introduction"

    def test_human_feedback_to_context(self) -> None:
        """Test human feedback context generation."""
        feedback = HumanFeedback(content="Make it more engaging")
        context = feedback.to_context()

        assert "<human_feedback>" in context
        assert "</human_feedback>" in context
        assert "Make it more engaging" in context


class TestSelectedTextReviews:
    """Test the SelectedTextReviews class."""

    def test_selected_text_reviews_creation(self) -> None:
        """Test creating selected text reviews."""
        article = Article(content="# Test Article\n\nThis is the body content.")
        selected_text = SelectedText(article=article, content="This is the body content.", first_line_number=2, last_line_number=2)
        reviews = [
            Review(profile="structure", location="body", comment="Good structure"),
        ]
        selected_text_reviews = SelectedTextReviews(article=article, selected_text=selected_text, reviews=reviews)

        assert len(selected_text_reviews.reviews) == 1
        assert selected_text_reviews.reviews[0].comment == "Good structure"

    def test_selected_text_reviews_to_context(self) -> None:
        """Test selected text reviews context generation."""
        article = Article(content="# Test Article\n\nThis is the body content.")
        selected_text = SelectedText(article=article, content="This is the body content.", first_line_number=2, last_line_number=2)
        reviews = [
            Review(profile="structure", location="body", comment="Good structure"),
        ]
        selected_text_reviews = SelectedTextReviews(article=article, selected_text=selected_text, reviews=reviews)
        context = selected_text_reviews.to_context()

        assert "<selected_text_reviews>" in context
        assert "</selected_text_reviews>" in context
        assert "<selected_text>" in context
        assert "This is the body content" in context
        assert "Good structure" in context
