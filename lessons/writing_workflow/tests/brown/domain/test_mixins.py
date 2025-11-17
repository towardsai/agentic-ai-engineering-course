"""Tests for brown.entities.mixins module."""

from brown.entities.mixins import ContextMixin, MarkdownMixin


class TestContextMixin:
    """Test the ContextMixin class."""

    def test_context_mixin_xml_tag(self) -> None:
        """Test XML tag generation from class name."""

        class TestClass(ContextMixin):
            def __init__(self):
                self.content = "test content"

        test_instance = TestClass()
        assert test_instance.xml_tag == "test_class"

    def test_context_mixin_xml_tag_with_profile(self) -> None:
        """Test XML tag generation for profile classes."""

        class TestProfile(ContextMixin):
            def __init__(self):
                self.content = "profile content"

        test_instance = TestProfile()
        assert test_instance.xml_tag == "test_profile"

    def test_context_mixin_xml_tag_with_suffix(self) -> None:
        """Test XML tag generation with specific suffixes."""

        class TestItem(ContextMixin):
            def __init__(self):
                self.content = "item content"

        test_instance = TestItem()
        assert test_instance.xml_tag == "test_item"


class TestMarkdownMixin:
    """Test the MarkdownMixin class."""

    def test_markdown_mixin_abstract(self) -> None:
        """Test that MarkdownMixin can be instantiated but to_markdown is abstract."""
        # MarkdownMixin is not an ABC, so it can be instantiated
        mixin = MarkdownMixin()
        assert mixin is not None

    def test_markdown_mixin_implementation(self) -> None:
        """Test implementing MarkdownMixin."""

        class TestMarkdown(MarkdownMixin):
            def __init__(self, content: str):
                self.content = content

            def to_markdown(self) -> str:
                return self.content

        test_instance = TestMarkdown("test content")
        assert test_instance.to_markdown() == "test content"

    def test_markdown_mixin_missing_implementation(self) -> None:
        """Test that missing to_markdown implementation returns None when called."""

        class IncompleteMarkdown(MarkdownMixin):
            def __init__(self, content: str):
                self.content = content

            # Missing to_markdown implementation

        instance = IncompleteMarkdown("test content")
        # Since MarkdownMixin is not an ABC, the method returns None
        result = instance.to_markdown()
        assert result is None
