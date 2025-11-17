"""Tests for brown.utils.s module."""

import pytest

from brown.utils.s import camel_to_snake, clean_markdown_links, normalize_any_to_str


class TestCamelToSnake:
    """Test the camel_to_snake function."""

    def test_camel_to_snake_simple(self) -> None:
        """Test simple CamelCase conversion."""
        assert camel_to_snake("CamelCase") == "camel_case"
        assert camel_to_snake("XMLHttpRequest") == "xml_http_request"
        assert camel_to_snake("HTTPServer") == "http_server"

    def test_camel_to_snake_with_numbers(self) -> None:
        """Test CamelCase with numbers."""
        assert camel_to_snake("HTTP2Server") == "http2_server"
        assert camel_to_snake("APIv1") == "ap_iv1"

    def test_camel_to_snake_single_word(self) -> None:
        """Test single word input."""
        assert camel_to_snake("word") == "word"
        assert camel_to_snake("Word") == "word"

    def test_camel_to_snake_empty(self) -> None:
        """Test empty string input."""
        assert camel_to_snake("") == ""

    def test_camel_to_snake_already_snake(self) -> None:
        """Test already snake_case input."""
        assert camel_to_snake("snake_case") == "snake_case"
        assert camel_to_snake("already_snake") == "already_snake"


class TestNormalizeAnyToStr:
    """Test the normalize_any_to_str function."""

    def test_normalize_any_to_str_with_string(self) -> None:
        """Test string passthrough."""
        assert normalize_any_to_str("hello world") == "hello world"
        assert normalize_any_to_str("") == ""

    def test_normalize_any_to_str_with_list(self) -> None:
        """Test list joining."""
        assert normalize_any_to_str(["line1", "line2", "line3"]) == "line1\nline2\nline3"
        assert normalize_any_to_str(["single"]) == "single"
        with pytest.raises(ValueError, match="Expected a str, list or dict, got a `<class 'list'>`"):
            normalize_any_to_str([])

    def test_normalize_any_to_str_with_dict(self) -> None:
        """Test dict content extraction."""
        assert normalize_any_to_str({"content": "hello world"}) == "hello world"
        assert normalize_any_to_str({"content": "test", "other": "ignored"}) == "test"

    def test_normalize_any_to_str_invalid_list(self) -> None:
        """Test invalid list content."""
        with pytest.raises(ValueError, match="Expected a list of strings"):
            normalize_any_to_str([1, 2, 3])

    def test_normalize_any_to_str_invalid_dict(self) -> None:
        """Test invalid dict content."""
        with pytest.raises(ValueError, match="Expected a dict with a `content` key"):
            normalize_any_to_str({"other": "value"})

    def test_normalize_any_to_str_invalid_type(self) -> None:
        """Test invalid input type."""
        with pytest.raises(ValueError, match="Expected a str, list or dict"):
            normalize_any_to_str(123)
        with pytest.raises(ValueError, match="Expected a str, list or dict"):
            normalize_any_to_str(None)


class TestCleanMarkdownLinks:
    """Test the clean_markdown_links function."""

    def test_clean_markdown_links_simple(self) -> None:
        """Test simple markdown link cleaning."""
        text = "Check out [this link](https://example.com) for more info."
        expected = "Check out  https://example.com  for more info."
        assert clean_markdown_links(text) == expected

    def test_clean_markdown_links_multiple(self) -> None:
        """Test multiple markdown links."""
        text = "See [link1](url1) and [link2](url2) for details."
        expected = "See  url1  and  url2  for details."
        assert clean_markdown_links(text) == expected

    def test_clean_markdown_links_with_images(self) -> None:
        """Test image link cleaning."""
        text = "![alt text](image.jpg) and [text link](url)"
        expected = " image.jpg  and  url "
        assert clean_markdown_links(text) == expected

    def test_clean_markdown_links_malformed(self) -> None:
        """Test malformed markdown links."""
        text = "Check [unclosed link and [proper link](url) here."
        expected = "Check  url  here."
        assert clean_markdown_links(text) == expected

    def test_clean_markdown_links_no_links(self) -> None:
        """Test text without markdown links."""
        text = "This is plain text without any links."
        assert clean_markdown_links(text) == text

    def test_clean_markdown_links_empty(self) -> None:
        """Test empty string."""
        assert clean_markdown_links("") == ""

    def test_clean_markdown_links_complex(self) -> None:
        """Test complex markdown with various link types."""
        text = """
        # Title
        
        This is a [reference link](https://example.com) with some text.
        
        ![Image](https://example.com/image.jpg) shows the diagram.
        
        Multiple [links](url1) and [more links](url2) in one paragraph.
        """
        expected = """
        # Title
        
        This is a  https://example.com  with some text.
        
         https://example.com/image.jpg  shows the diagram.
        
        Multiple  url1  and  url2  in one paragraph.
        """
        assert clean_markdown_links(text) == expected

    def test_clean_markdown_links_with_text(self) -> None:
        """Test markdown links with descriptive text."""
        text = "Read the [official documentation](https://docs.example.com) for details."
        expected = "Read the  https://docs.example.com  for details."
        assert clean_markdown_links(text) == expected

    def test_clean_markdown_links_nested_brackets(self) -> None:
        """Test markdown links with nested brackets."""
        text = "Check [this [nested] link](url) for info."
        expected = "Check [this [nested] link](url) for info."
        assert clean_markdown_links(text) == expected
