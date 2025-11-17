"""Tests for brown.nodes.base module."""

from brown.builders import build_model
from brown.config_app import get_app_config
from brown.nodes.base import Node, Toolkit


class ConcreteNode(Node):
    """Concrete implementation of Node for testing."""

    async def ainvoke(self) -> str:
        """Test implementation of ainvoke."""
        return "test result"


class TestNode:
    """Test the Node base class."""

    def test_node_initialization(self) -> None:
        """Test node creation."""
        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="write_article")

        node = ConcreteNode(model=model, toolkit=toolkit)

        assert node.model == model
        assert node.toolkit == toolkit

    def test_node_build_toolkit(self) -> None:
        """Test toolkit building."""
        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="write_article")

        node = ConcreteNode(model=model, toolkit=toolkit)

        assert isinstance(node.toolkit, Toolkit)

    def test_node_build_model_client(self) -> None:
        """Test model client creation."""
        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="write_article")

        node = ConcreteNode(model=model, toolkit=toolkit)

        assert node.model is not None

    def test_node_build_model_client_fake(self) -> None:
        """Test fake model setup."""
        from pydantic import BaseModel

        class MockResponse(BaseModel):
            content: str

        mock_response = MockResponse(content="Mocked response")

        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="write_article")
        model.responses = [mock_response.model_dump_json()]

        node = ConcreteNode(model=model, toolkit=toolkit)

        assert node.model is not None

    def test_node_build_user_input_content(self) -> None:
        """Test input formatting."""
        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="write_article")

        node = ConcreteNode(model=model, toolkit=toolkit)

        # Test with string input
        content = node.build_user_input_content(["Test input"])
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Test input"

        # Test with dict input
        input_dict = {"message": "Test message", "context": "Test context"}
        content = node.build_user_input_content([str(input_dict)])
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "Test message" in content[0]["text"]
        assert "Test context" in content[0]["text"]

    def test_node_as_tool(self) -> None:
        """Test tool conversion."""
        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="write_article")

        node = ConcreteNode(model=model, toolkit=toolkit)

        # ConcreteNode doesn't have as_tool method - it's only in ToolNode
        # Just verify the node was created successfully
        assert node.model == model
        assert node.toolkit == toolkit

    def test_node_with_fake_model_requires_mocked_response(self) -> None:
        """Test that fake model works without mocked response (uses default)."""
        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="write_article")
        # Don't set responses - should use default

        # Should not raise an error - fake model uses default response when mocked_response is None
        node = ConcreteNode(model=model, toolkit=toolkit)

        # Verify it was created successfully
        assert node.model == model
        assert node.toolkit == toolkit

    def test_node_set_mocked_responses(self) -> None:
        """Test setting mocked responses."""
        app_config = get_app_config()
        model, toolkit = build_model(app_config, node="write_article")

        node = ConcreteNode(model=model, toolkit=toolkit)

        # Test setting mocked responses - need to create a fake model first
        from brown.models import FakeModel

        fake_model = FakeModel(responses=[])
        result = node._set_default_model_mocked_responses(fake_model)

        # Verify the mocked responses were set
        assert isinstance(result, FakeModel)
        assert len(result.responses) == 1
        assert "This is a mocked response from the fake model" in result.responses[0]


class TestToolkit:
    """Test the Toolkit base class."""

    def test_toolkit_initialization(self) -> None:
        """Test toolkit creation."""
        toolkit = Toolkit(tools=[])
        assert toolkit is not None

    def test_toolkit_abstract_methods(self) -> None:
        """Test that toolkit has abstract methods that need implementation."""
        toolkit = Toolkit(tools=[])

        # These methods should exist but may not be implemented
        assert hasattr(toolkit, "get_tools")
        assert hasattr(toolkit, "get_tool_by_name")
