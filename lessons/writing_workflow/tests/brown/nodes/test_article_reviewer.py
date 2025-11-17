"""Tests for brown.nodes.article_reviewer module."""

import pytest

from brown.builders import build_model
from brown.config_app import get_app_config
from brown.entities.articles import Article, SelectedText
from brown.entities.guidelines import ArticleGuideline
from brown.entities.profiles import (
    ArticleProfile,
    ArticleProfiles,
    CharacterProfile,
    MechanicsProfile,
    StructureProfile,
    TerminologyProfile,
    TonalityProfile,
)
from brown.entities.reviews import ArticleReviews, HumanFeedback, Review, SelectedTextReviews
from brown.nodes.article_reviewer import ArticleReviewer


class TestArticleReviewer:
    """Test the ArticleReviewer class."""

    def test_article_reviewer_initialization(self) -> None:
        """Test creating reviewer."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(reviews=[Review(profile="test_profile", location="test_location", comment="This is a good article.")])

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_article")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        reviewer = ArticleReviewer(
            to_review=article,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )

        assert reviewer.model == model

    @pytest.mark.asyncio
    async def test_article_reviewer_ainvoke_success(self) -> None:
        """Test review generation with mocked response."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(
            reviews=[
                Review(profile="test_profile", location="test_location", comment="This article is well-written and informative."),
                Review(profile="test_profile", location="test_location", comment="Could use more examples in section 2."),
            ]
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_article")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        reviewer = ArticleReviewer(
            to_review=article,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )

        # Test input data
        article = Article(content="# Test Article\n\nThis is test content.")

        result = await reviewer.ainvoke()

        assert isinstance(result, ArticleReviews)
        assert len(result.reviews) == 2
        assert result.reviews[0].comment == "This article is well-written and informative."
        assert result.reviews[0].profile == "test_profile"
        assert result.reviews[1].comment == "Could use more examples in section 2."
        assert result.reviews[1].profile == "test_profile"

    @pytest.mark.asyncio
    async def test_article_reviewer_structured_output(self) -> None:
        """Test that reviewer returns structured output."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(
            reviews=[Review(profile="test_profile", location="test_location", comment="Excellent article with clear structure.")]
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_article")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        reviewer = ArticleReviewer(
            to_review=article,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )

        article = Article(content="# Test Article\n\nContent here.")

        result = await reviewer.ainvoke()

        # Verify structure
        assert isinstance(result, ArticleReviews)
        assert len(result.reviews) == 1

        review = result.reviews[0]
        assert hasattr(review, "comment")
        assert hasattr(review, "profile")
        assert hasattr(review, "location")
        assert isinstance(review.comment, str)
        assert isinstance(review.profile, str)
        assert isinstance(review.location, str)

    @pytest.mark.asyncio
    async def test_article_reviewer_empty_article(self) -> None:
        """Test reviewer with empty article."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(
            reviews=[Review(profile="test_profile", location="test_location", comment="Article is too short and lacks content.")]
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_article")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        reviewer = ArticleReviewer(
            to_review=article,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )

        result = await reviewer.ainvoke()

        assert isinstance(result, ArticleReviews)
        assert len(result.reviews) == 1
        assert "too short" in result.reviews[0].comment

    @pytest.mark.asyncio
    async def test_article_reviewer_multiple_reviews(self) -> None:
        """Test reviewer generating multiple reviews."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(
            reviews=[
                Review(profile="test_profile", location="test_location", comment="Good introduction section."),
                Review(profile="test_profile", location="test_location", comment="Body section needs more detail."),
                Review(profile="test_profile", location="test_location", comment="Conclusion is well-written."),
            ]
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_article")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        reviewer = ArticleReviewer(
            to_review=article,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )

        result = await reviewer.ainvoke()

        assert isinstance(result, ArticleReviews)
        assert len(result.reviews) == 3

        # Check that all reviews have valid structure
        for review in result.reviews:
            assert isinstance(review.comment, str)
            assert isinstance(review.profile, str)
            assert isinstance(review.location, str)
            assert len(review.comment) > 0

    def test_article_reviewer_requires_mocked_response_for_fake_model(self) -> None:
        """Test that fake model requires mocked response."""
        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_article")
        # Don't set responses - should use default

        article = Article(content="# Test Article\n\nContent here.")
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        # Should not raise an error - fake model uses default response when mocked_response is None
        reviewer = ArticleReviewer(
            to_review=article,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )

        # Verify it was created successfully
        assert reviewer.model == model

    def test_article_reviewer_selected_text_initialization(self) -> None:
        """Test creating reviewer with selected text."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(
            reviews=[Review(profile="test_profile", location="test_location", comment="This is a good selected text.")]
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_selected_text")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        selected_text = SelectedText(article=article, content="Selected text content", first_line_number=10, last_line_number=15)
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        reviewer = ArticleReviewer(
            to_review=selected_text,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )

        assert reviewer.model == model
        assert reviewer.is_selected_text is True
        assert reviewer.is_article is False

    @pytest.mark.asyncio
    async def test_article_reviewer_selected_text_ainvoke(self) -> None:
        """Test review generation for selected text with mocked response."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(
            reviews=[
                Review(profile="test_profile", location="test_location", comment="This selected text is well-written and informative."),
                Review(profile="test_profile", location="test_location", comment="Could use more examples in this section."),
            ]
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_selected_text")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        selected_text = SelectedText(article=article, content="Selected text content", first_line_number=10, last_line_number=15)
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        reviewer = ArticleReviewer(
            to_review=selected_text,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )

        result = await reviewer.ainvoke()

        assert isinstance(result, SelectedTextReviews)
        assert result.article == article
        assert result.selected_text == selected_text
        assert len(result.reviews) == 2
        assert result.reviews[0].comment == "This selected text is well-written and informative."
        assert result.reviews[1].comment == "Could use more examples in this section."

    @pytest.mark.asyncio
    async def test_article_reviewer_selected_text_with_human_feedback(self) -> None:
        """Test selected text reviewer with human feedback."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(
            reviews=[
                Review(profile="human_feedback", location="Selected text level", comment="Add more technical details to this section."),
                Review(profile="test_profile", location="test_location", comment="Good structure but needs more depth."),
            ]
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_selected_text")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        selected_text = SelectedText(article=article, content="Selected text content", first_line_number=10, last_line_number=15)
        human_feedback = HumanFeedback(content="This section needs more technical depth")
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        reviewer = ArticleReviewer(
            to_review=selected_text,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            human_feedback=human_feedback,
            model=model,
        )

        result = await reviewer.ainvoke()

        assert isinstance(result, SelectedTextReviews)
        assert result.article == article
        assert result.selected_text == selected_text
        assert len(result.reviews) == 2
        # Check that human feedback was processed
        human_feedback_reviews = [r for r in result.reviews if r.profile == "human_feedback"]
        assert len(human_feedback_reviews) == 1
        assert "technical details" in human_feedback_reviews[0].comment

    def test_article_reviewer_article_property(self) -> None:
        """Test that article property returns correct article for both Article and SelectedText inputs."""
        from brown.nodes.article_reviewer import ReviewsOutput

        mock_response = ReviewsOutput(reviews=[Review(profile="test_profile", location="test_location", comment="Test comment")])

        app_config = get_app_config()
        model, _ = build_model(app_config, node="review_article")
        model.responses = [mock_response.model_dump_json()]

        article = Article(content="# Test Article\n\nContent here.")
        selected_text = SelectedText(article=article, content="Selected text content", first_line_number=10, last_line_number=15)
        article_guideline = ArticleGuideline(content="Write a test article.")
        article_profiles = ArticleProfiles(
            character=CharacterProfile(name="test", content="Test character"),
            article=ArticleProfile(name="test", content="Test article"),
            structure=StructureProfile(name="test", content="Test structure"),
            mechanics=MechanicsProfile(name="test", content="Test mechanics"),
            terminology=TerminologyProfile(name="test", content="Test terminology"),
            tonality=TonalityProfile(name="test", content="Test tonality"),
        )

        # Test with Article input
        article_reviewer = ArticleReviewer(
            to_review=article,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )
        assert article_reviewer.article == article

        # Test with SelectedText input
        selected_text_reviewer = ArticleReviewer(
            to_review=selected_text,
            article_guideline=article_guideline,
            article_profiles=article_profiles,
            model=model,
        )
        assert selected_text_reviewer.article == article
