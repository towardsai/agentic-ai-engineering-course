"""Tests for brown.nodes.article_writer module."""

import pytest

from brown.builders import build_model
from brown.config_app import get_app_config
from brown.entities.articles import Article, ArticleExamples, SelectedText
from brown.entities.guidelines import ArticleGuideline
from brown.entities.media_items import MediaItems
from brown.entities.profiles import ArticleProfiles
from brown.entities.research import Research
from brown.entities.reviews import ArticleReviews, Review, SelectedTextReviews
from brown.nodes.article_writer import ArticleWriter


class TestArticleWriter:
    """Test the ArticleWriter class."""

    def test_article_writer_initialization(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test creating writer with fake model."""
        mock_response = {
            "content": """
# Mock Title
### Mock Subtitle

Mock intro.

## Section 1
Mock section 1.

## Section 2
Mock section 2.

## Section 3
Mock section 3.

## Conclusion
Mock conclusion.

## References
Mock references.
"""
        }

        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            model=model,
        )

        assert writer.model == model

    @pytest.mark.asyncio
    async def test_article_writer_ainvoke_success(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test article generation with mocked response."""
        mock_response = '{"content": "# Generated Article\\n### Mock Subtitle\\n\\nThis is a generated article about AI.\\n\\n## Section 1\\nMock section 1.\\n\\n## Section 2\\nMock section 2.\\n\\n## Section 3\\nMock section 3.\\n\\n## Conclusion\\nMock conclusion.\\n\\n## References\\nMock references.\\n"}'  # noqa: E501
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            model=model,
        )

        result = await writer.ainvoke()

        assert isinstance(result, Article)
        assert "# Generated Article" in result.content

    @pytest.mark.asyncio
    async def test_article_writer_with_media_items(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test article generation with media items."""
        mock_response = '{"content": "# Article with Media\\n### Mock Subtitle\\n\\nThis article includes diagrams.\\n\\n## Section 1\\nMock section 1.\\n\\n## Section 2\\nMock section 2.\\n\\n## Section 3\\nMock section 3.\\n\\n## Conclusion\\nMock conclusion.\\n\\n## References\\nMock references.\\n"}'  # noqa: E501
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        from brown.entities.media_items import MermaidDiagram

        media_items = MediaItems(
            media_items=[MermaidDiagram(location="Introduction", content="graph TD\n    A --> B", caption="Test diagram")]
        )

        writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=media_items,
            article_examples=mock_article_examples,
            model=model,
        )

        result = await writer.ainvoke()

        assert isinstance(result, Article)
        assert "Article with Media" in result.content

    @pytest.mark.asyncio
    async def test_article_writer_with_reviews(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test article generation with reviews (editing mode)."""
        mock_response = '{"content": "# Revised Article\\n### Mock Subtitle\\n\\nThis is a revised article based on reviews.\\n\\n## Section 1\\nMock section 1.\\n\\n## Section 2\\nMock section 2.\\n\\n## Section 3\\nMock section 3.\\n\\n## Conclusion\\nMock conclusion.\\n\\n## References\\nMock references.\\n"}'  # noqa: E501
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        reviews = ArticleReviews(
            article=Article(content="Test article"),
            reviews=[Review(profile="test_profile", location="test_location", comment="Needs improvement")],
        )

        writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            reviews=reviews,
            model=model,
        )

        result = await writer.ainvoke()

        assert isinstance(result, Article)
        assert "Revised Article" in result.content

    @pytest.mark.asyncio
    async def test_article_writer_empty_input(
        self,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test article writer with empty input."""
        mock_response = '{"content": "# Mock Title\\n### Mock Subtitle\\n\\nEmpty article\\n\\n## Section 1\\nMock section 1.\\n\\n## Section 2\\nMock section 2.\\n\\n## Section 3\\nMock section 3.\\n\\n## Conclusion\\nMock conclusion.\\n\\n## References\\nMock references.\\n"}'  # noqa: E501
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        article_guideline = ArticleGuideline(content="")
        research = Research(content="")

        writer = ArticleWriter(
            article_guideline=article_guideline,
            research=research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            model=model,
        )

        result = await writer.ainvoke()

        assert isinstance(result, Article)
        # The mocked response will return the default mocked article content
        # The exact content depends on the mocked responses in the ArticleWriter node

    def test_article_writer_requires_mocked_response_for_fake_model(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test that fake model requires mocked response."""
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        # Don't set responses - should use default

        # The mocked response will return the default mocked article content
        # The exact content depends on the mocked responses in the ArticleWriter node
        writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            model=model,
        )

        assert writer.model == model

    @pytest.mark.asyncio
    async def test_article_writer_with_selected_text_reviews(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test article writer with selected text reviews returns SelectedText."""
        mock_response = '{"content": "## Mock Section Title\\n\\nMock selected text content that has been edited based on the human feedback and reviews.\\n\\nThis is the edited version of the selected text that addresses the specific issues identified in the reviews while\\nincorporating the human feedback.\\n"}'  # noqa: E501
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        article = Article(content="Test article")
        selected_text = SelectedText(article=article, content="Original selected text", first_line_number=10, last_line_number=15)
        reviews = SelectedTextReviews(
            article=article,
            selected_text=selected_text,
            reviews=[Review(profile="test_profile", location="test_location", comment="Needs improvement")],
        )

        writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            reviews=reviews,
            model=model,
        )

        result = await writer.ainvoke()

        assert isinstance(result, SelectedText)
        assert result.article == article
        assert result.first_line_number == 10
        assert result.last_line_number == 15
        assert "Mock selected text content" in result.content

    @pytest.mark.asyncio
    async def test_article_writer_selected_text_preserves_line_numbers(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test that line numbers are preserved in selected text output."""
        mock_response = '{"content": "## Mock Section Title\\n\\nCompletely rewritten content that has been edited based on the human feedback and reviews.\\n\\nThis is the edited version of the selected text that addresses the specific issues identified in the reviews while\\nincorporating the human feedback.\\n"}'  # noqa: E501
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        article = Article(content="Test article")
        selected_text = SelectedText(article=article, content="Original selected text", first_line_number=25, last_line_number=40)
        reviews = SelectedTextReviews(
            article=article,
            selected_text=selected_text,
            reviews=[Review(profile="test_profile", location="test_location", comment="Rewrite completely")],
        )

        writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            reviews=reviews,
            model=model,
        )

        result = await writer.ainvoke()

        assert isinstance(result, SelectedText)
        assert result.article == article
        assert result.first_line_number == 25
        assert result.last_line_number == 40
        assert "Completely rewritten content" in result.content

    @pytest.mark.asyncio
    async def test_article_writer_selected_text_vs_article_output(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test that ArticleWriter returns correct type based on review type."""
        mock_response = '{"content": "# Mock Title\\n### Mock Subtitle\\n\\nMock content based on review type.\\n"}'
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        article = Article(content="Test article")
        selected_text = SelectedText(article=article, content="Original selected text", first_line_number=10, last_line_number=15)

        # Test with ArticleReviews - should return Article
        article_reviews = ArticleReviews(
            article=article,
            reviews=[Review(profile="test_profile", location="test_location", comment="Article needs improvement")],
        )

        article_writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            reviews=article_reviews,
            model=model,
        )

        article_result = await article_writer.ainvoke()
        assert isinstance(article_result, Article)
        assert "Mock Title" in article_result.content

        # Test with SelectedTextReviews - should return SelectedText
        selected_text_reviews = SelectedTextReviews(
            article=article,
            selected_text=selected_text,
            reviews=[Review(profile="test_profile", location="test_location", comment="Selected text needs improvement")],
        )

        selected_text_writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            reviews=selected_text_reviews,
            model=model,
        )

        selected_text_result = await selected_text_writer.ainvoke()
        assert isinstance(selected_text_result, SelectedText)
        assert selected_text_result.article == article
        assert selected_text_result.first_line_number == 10
        assert selected_text_result.last_line_number == 15

    @pytest.mark.asyncio
    async def test_article_writer_selected_text_content(
        self,
        mock_article_guideline: ArticleGuideline,
        mock_research: Research,
        mock_article_profiles: ArticleProfiles,
        mock_media_items: MediaItems,
        mock_article_examples: ArticleExamples,
    ) -> None:
        """Test that selected text content is properly extracted and returned."""
        mock_response = '{"content": "## Enhanced Section\\n\\nThis is the enhanced selected text content with more technical details and examples.\\n\\nThe content has been significantly improved based on the feedback provided.\\n"}'  # noqa: E501
        app_config = get_app_config()
        model, _ = build_model(app_config, node="write_article")
        model.responses = [mock_response]

        article = Article(content="Test article")
        selected_text = SelectedText(article=article, content="Original selected text", first_line_number=5, last_line_number=12)
        reviews = SelectedTextReviews(
            article=article,
            selected_text=selected_text,
            reviews=[Review(profile="test_profile", location="test_location", comment="Add technical details")],
        )

        writer = ArticleWriter(
            article_guideline=mock_article_guideline,
            research=mock_research,
            article_profiles=mock_article_profiles,
            media_items=mock_media_items,
            article_examples=mock_article_examples,
            reviews=reviews,
            model=model,
        )

        result = await writer.ainvoke()

        assert isinstance(result, SelectedText)
        assert result.article == article
        assert "Enhanced Section" in result.content
        assert "technical details and examples" in result.content
        assert "significantly improved" in result.content
