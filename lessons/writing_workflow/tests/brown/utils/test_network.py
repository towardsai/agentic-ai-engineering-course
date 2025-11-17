"""Tests for brown.utils.network module."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from brown.utils.network import is_image_domain_accepted, is_image_url_valid, ping


class TestIsImageDomainAccepted:
    """Test the is_image_domain_accepted function."""

    def test_valid_domains(self) -> None:
        """Test valid image domains."""
        assert is_image_domain_accepted("https://example.com/image.jpg") is True
        assert is_image_domain_accepted("https://cdn.example.com/pic.png") is True
        assert is_image_domain_accepted("https://images.google.com/photo.jpg") is True

    def test_invalid_domains(self) -> None:
        """Test invalid image domains."""
        assert is_image_domain_accepted("https://github.com/user/repo/image.jpg") is False
        assert is_image_domain_accepted("https://githubusercontent.com/image.png") is False
        assert is_image_domain_accepted("https://api.github.com/image.gif") is False

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        assert is_image_domain_accepted("") is True
        assert is_image_domain_accepted("not-a-url") is True
        assert is_image_domain_accepted("https://subdomain.github.com/image.jpg") is False


class TestPing:
    """Test the ping function."""

    @pytest.mark.asyncio
    async def test_ping_success(self) -> None:
        """Test successful ping."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "image/jpeg"}

        with patch("brown.utils.network.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head.return_value = mock_response
            result = await ping("https://example.com/image.jpg")
            assert result is True

    @pytest.mark.asyncio
    async def test_ping_head_fallback_to_get(self) -> None:
        """Test ping with HEAD fallback to GET."""
        # Mock HEAD response that triggers fallback
        mock_head_response = AsyncMock()
        mock_head_response.status_code = 405

        # Mock GET response
        mock_get_response = AsyncMock()
        mock_get_response.status_code = 200
        mock_get_response.headers = {"Content-Type": "image/png"}

        with patch("brown.utils.network.httpx.AsyncClient") as mock_client:
            mock_client_instance = mock_client.return_value.__aenter__.return_value
            mock_client_instance.head.return_value = mock_head_response
            mock_client_instance.get.return_value = mock_get_response

            result = await ping("https://example.com/image.png")
            assert result is True

    @pytest.mark.asyncio
    async def test_ping_invalid_content_type(self) -> None:
        """Test ping with invalid content type."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html"}

        with patch("brown.utils.network.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head.return_value = mock_response
            result = await ping("https://example.com/page.html")
            assert result is False

    @pytest.mark.asyncio
    async def test_ping_non_200_status(self) -> None:
        """Test ping with non-200 status code."""
        mock_response = AsyncMock()
        mock_response.status_code = 404

        with patch("brown.utils.network.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head.return_value = mock_response
            result = await ping("https://example.com/notfound.jpg")
            assert result is False

    @pytest.mark.asyncio
    async def test_ping_request_error(self) -> None:
        """Test ping with request error."""
        with patch("brown.utils.network.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head.side_effect = httpx.RequestError("Network error")
            result = await ping("https://example.com/image.jpg")
            assert result is False

    @pytest.mark.asyncio
    async def test_ping_timeout(self) -> None:
        """Test ping with timeout."""
        with patch("brown.utils.network.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.head.side_effect = httpx.TimeoutException("Timeout")
            result = await ping("https://example.com/image.jpg", timeout=0.1)
            assert result is False

    @pytest.mark.asyncio
    async def test_ping_invalid_url(self) -> None:
        """Test ping with invalid URL."""
        result = await ping("not-a-valid-url")
        assert result is False


class TestIsImageUrlValid:
    """Test the is_image_url_valid function."""

    @pytest.mark.asyncio
    async def test_valid_image_url(self) -> None:
        """Test valid image URL."""
        with patch("brown.utils.network.ping", return_value=True) as mock_ping:
            result = await is_image_url_valid("https://example.com/image.jpg")
            assert result is True
            mock_ping.assert_called_once_with("https://example.com/image.jpg", 5.0)

    @pytest.mark.asyncio
    async def test_invalid_domain(self) -> None:
        """Test invalid domain."""
        result = await is_image_url_valid("https://github.com/user/repo/image.jpg")
        assert result is False

    @pytest.mark.asyncio
    async def test_valid_domain_but_unreachable(self) -> None:
        """Test valid domain but unreachable URL."""
        with patch("brown.utils.network.ping", return_value=False) as mock_ping:
            result = await is_image_url_valid("https://example.com/image.jpg")
            assert result is False
            mock_ping.assert_called_once_with("https://example.com/image.jpg", 5.0)

    @pytest.mark.asyncio
    async def test_custom_timeout(self) -> None:
        """Test custom timeout."""
        with patch("brown.utils.network.ping", return_value=True) as mock_ping:
            result = await is_image_url_valid("https://example.com/image.jpg", timeout=10.0)
            assert result is True
            mock_ping.assert_called_once_with("https://example.com/image.jpg", 10.0)

    @pytest.mark.asyncio
    async def test_empty_url(self) -> None:
        """Test empty URL."""
        result = await is_image_url_valid("")
        assert result is False  # Empty URL fails ping check

    @pytest.mark.asyncio
    async def test_malformed_url(self) -> None:
        """Test malformed URL."""
        result = await is_image_url_valid("not-a-url")
        assert result is False  # Malformed URL fails ping check
