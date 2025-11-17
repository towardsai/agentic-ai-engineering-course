"""Tests for brown.nodes.media_generator module."""

import pytest

from brown.builders import build_model
from brown.config_app import get_app_config
from brown.nodes.media_generator import MediaGeneratorOrchestrator
from brown.nodes.tool_nodes import MermaidDiagramGenerator


class TestMermaidDiagramGenerator:
    """Test the MermaidDiagramGenerator class."""

    def test_mermaid_diagram_generator_initialization(self) -> None:
        """Test creating Mermaid diagram generator."""
        from brown.nodes.tool_nodes import GeneratedMermaidDiagram

        mock_response = GeneratedMermaidDiagram(
            content="```mermaid\ngraph TD\n    A[Start] --> B[End]\n```",
            caption="Test diagram",
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="generate_media_items")
        model.responses = [mock_response.model_dump_json()]

        generator = MermaidDiagramGenerator(model=model)

        assert generator.model == model

    @pytest.mark.asyncio
    async def test_mermaid_diagram_generator_ainvoke(self) -> None:
        """Test Mermaid diagram generation."""
        from brown.nodes.tool_nodes import GeneratedMermaidDiagram

        mock_response = GeneratedMermaidDiagram(
            content="```mermaid\ngraph LR\n    A[Input] --> B[Process] --> C[Output]\n```",
            caption="Process flow diagram",
        )

        app_config = get_app_config()
        model, _ = build_model(app_config, node="generate_media_items")
        model.responses = [mock_response.model_dump_json()]

        generator = MermaidDiagramGenerator(model=model)

        description = "A flowchart showing data processing flow"
        section_title = "Data Processing"

        result = await generator.ainvoke(description, section_title)

        assert hasattr(result, "content")
        assert hasattr(result, "caption")
        assert "graph LR" in result.content
        assert "Process flow diagram" in result.caption

    @pytest.mark.asyncio
    async def test_mermaid_diagram_generator_error_handling(self) -> None:
        """Test Mermaid diagram generator error handling."""
        # Test with invalid response that should trigger error handling
        from pydantic import BaseModel

        class InvalidResponse(BaseModel):
            invalid: str

        mock_response = InvalidResponse(invalid="response")

        app_config = get_app_config()
        model, _ = build_model(app_config, node="generate_media_items")
        model.responses = [mock_response.model_dump_json()]

        generator = MermaidDiagramGenerator(model=model)

        description = "Test diagram"
        section_title = "Test Section"

        # Should handle error gracefully
        result = await generator.ainvoke(description, section_title)

        assert hasattr(result, "content")
        assert hasattr(result, "caption")
        # Should contain error information
        assert "Error" in result.content or "Failed" in result.content

    def test_mermaid_diagram_generator_requires_mocked_response_for_fake_model(self) -> None:
        """Test that fake model works without mocked response (uses default)."""
        app_config = get_app_config()
        model, _ = build_model(app_config, node="generate_media_items")
        # Don't set responses - should use default

        # Should not raise an error - fake model uses default response when mocked_response is None
        generator = MermaidDiagramGenerator(model=model)

        # Verify it was created successfully
        assert generator.model == model


class TestMediaGeneratorOrchestrator:
    """Test the MediaGeneratorOrchestrator class."""

    def test_media_generator_orchestrator_initialization(self) -> None:
        """Test creating orchestrator."""
        from pydantic import BaseModel

        from brown.entities.guidelines import ArticleGuideline
        from brown.entities.research import Research
        from brown.nodes.base import ToolCall

        class MockToolCallResponse(BaseModel):
            tool_calls: list[ToolCall]

        mock_response = MockToolCallResponse(
            tool_calls=[
                ToolCall(
                    name="generate_mermaid_diagram",
                    args={"description_of_the_diagram": "Test diagram description", "section_title": "Test Section"},
                    id="test_call_1",
                    type="tool_call",
                )
            ]
        )

        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="generate_media_items")
        model.responses = [mock_response.model_dump_json()]

        article_guideline = ArticleGuideline(content="Create a test diagram")
        research = Research(content="Test research content")

        orchestrator = MediaGeneratorOrchestrator(
            article_guideline=article_guideline,
            research=research,
            model=model,
            toolkit=toolkit,
        )

        assert orchestrator.model == model
        assert orchestrator.toolkit == toolkit

    @pytest.mark.asyncio
    async def test_media_generator_orchestrator_ainvoke(self) -> None:
        """Test orchestrator execution."""
        from pydantic import BaseModel

        class MockResponse(BaseModel):
            content: str

        mock_response = MockResponse(content="Mock response")

        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="generate_media_items")
        model.responses = [mock_response.model_dump_json()]

        from brown.entities.guidelines import ArticleGuideline
        from brown.entities.research import Research

        article_guideline = ArticleGuideline(content="Create a test diagram")
        research = Research(content="Test research content")

        orchestrator = MediaGeneratorOrchestrator(
            article_guideline=article_guideline,
            research=research,
            model=model,
            toolkit=toolkit,
        )

        result = await orchestrator.ainvoke()

        assert isinstance(result, list)
        # The orchestrator returns a list of ToolCall objects
        # Since we're using a simple mock response, it should return an empty list
        assert len(result) == 0

    def test_media_generator_orchestrator_requires_mocked_response_for_fake_model(self) -> None:
        """Test that fake model works without mocked response (uses default)."""
        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="generate_media_items")
        # Don't set responses - should use default

        from brown.entities.guidelines import ArticleGuideline
        from brown.entities.research import Research

        article_guideline = ArticleGuideline(content="Create a test diagram")
        research = Research(content="Test research content")

        # Should not raise an error - fake model uses default response when mocked_response is None
        orchestrator = MediaGeneratorOrchestrator(
            article_guideline=article_guideline,
            research=research,
            model=model,
            toolkit=toolkit,
        )

        # Verify it was created successfully
        assert orchestrator.model == model
        assert orchestrator.toolkit == toolkit
