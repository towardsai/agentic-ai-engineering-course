"""Tests for brown.entities.media_items module."""

from brown.entities.media_items import MediaItem, MediaItems, MermaidDiagram


class TestMediaItem:
    """Test the MediaItem class."""

    def test_media_item_creation(self) -> None:
        """Test creating a media item."""
        media_item = MediaItem(location="section1", content="graph TD\n    A[Start] --> B[End]", caption="Flow diagram")

        assert media_item.location == "section1"
        assert media_item.content == "graph TD\n    A[Start] --> B[End]"
        assert media_item.caption == "Flow diagram"

    def test_media_item_to_context(self) -> None:
        """Test media item context generation."""
        media_item = MediaItem(location="introduction", content="graph LR\n    A --> B", caption="Simple flowchart")
        context = media_item.to_context()

        assert "<media_item>" in context
        assert "</media_item>" in context
        assert "<location>introduction</location>" in context
        assert "<content>graph LR\n    A --> B</content>" in context
        assert "<caption>Simple flowchart</caption>" in context

    def test_media_item_different_types(self) -> None:
        """Test media item with different content."""
        # Test diagram
        diagram_item = MediaItem(location="section1", content="graph TD\n    A --> B", caption="Process flow")
        assert diagram_item.content == "graph TD\n    A --> B"

        # Test code
        code_item = MediaItem(location="code_example", content="def hello():\n    print('Hello, World!')", caption="Python function")
        assert code_item.content == "def hello():\n    print('Hello, World!')"

    def test_media_item_str_representation(self) -> None:
        """Test string representation."""
        media_item = MediaItem(location="test", content="content", caption="test caption")
        str_repr = str(media_item)

        # MediaItem doesn't have a custom __str__ method, so it uses the default
        assert "location='test'" in str_repr
        assert "content='content'" in str_repr


class TestMediaItems:
    """Test the MediaItems class."""

    def test_media_items_creation(self) -> None:
        """Test creating media items collection."""
        items = [
            MediaItem(location="section1", content="graph TD\n    A --> B", caption="Diagram 1"),
            MediaItem(location="section2", content="print('hello')", caption="Code example"),
        ]
        media_items = MediaItems(media_items=items)

        assert len(media_items.media_items) == 2
        assert media_items.media_items[0].location == "section1"
        assert media_items.media_items[1].location == "section2"

    def test_media_items_to_context(self) -> None:
        """Test media items context generation."""
        items = [
            MediaItem(location="diagram", content="graph LR\n    A --> B", caption="Flow chart"),
            MediaItem(location="example", content="def test():\n    pass", caption="Code snippet"),
        ]
        media_items = MediaItems(media_items=items)
        context = media_items.to_context()

        assert "<media_items>" in context
        assert "</media_items>" in context
        assert "<location>diagram</location>" in context
        assert "<location>example</location>" in context
        assert "graph LR\n    A --> B" in context
        assert "def test():\n    pass" in context

    def test_media_items_empty(self) -> None:
        """Test empty media items collection."""
        media_items = MediaItems(media_items=[])
        context = media_items.to_context()

        assert "<media_items>" in context
        assert "</media_items>" in context
        # Should not contain any item content
        assert "<location>" not in context

    def test_media_items_single(self) -> None:
        """Test single media item."""
        item = MediaItem(location="single", content="single content", caption="Single item")
        media_items = MediaItems(media_items=[item])
        context = media_items.to_context()

        assert "<location>single</location>" in context
        assert "single content" in context

    def test_media_items_str_representation(self) -> None:
        """Test string representation."""
        items = [
            MediaItem(location="item1", content="content1", caption="Item 1"),
            MediaItem(location="item2", content="content2", caption="Item 2"),
        ]
        media_items = MediaItems(media_items=items)
        str_repr = str(media_items)

        # MediaItems doesn't have a custom __str__ method, so it uses the default
        assert "media_items=" in str_repr

    def test_media_items_build_classmethod(self) -> None:
        """Test the build class method."""
        media_items = MediaItems.build()
        assert media_items.media_items is None

        items = [MediaItem(location="test", content="content", caption="test")]
        media_items = MediaItems.build(media_items=items)
        assert media_items.media_items == items

    def test_are_available_only_in_source_property(self) -> None:
        """Test the are_available_only_in_source property."""
        media_items = MediaItems(media_items=None)
        assert media_items.are_available_only_in_source is True

        media_items = MediaItems(media_items=[])
        assert media_items.are_available_only_in_source is False


class TestMermaidDiagram:
    """Test the MermaidDiagram class."""

    def test_mermaid_diagram_creation(self) -> None:
        """Test creating a MermaidDiagram."""
        diagram = MermaidDiagram(location="section1", content="graph TD\n    A --> B", caption="Mermaid diagram")

        assert diagram.location == "section1"
        assert diagram.content == "graph TD\n    A --> B"
        assert diagram.caption == "Mermaid diagram"
        assert isinstance(diagram, MediaItem)
