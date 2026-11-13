# Copyright (c) 2026 MNKT
"""Tests for the checks on model-supplied URLs (`marvin/tools/urls.py`) and the tools that use them."""

import asyncio
import io
import socket
import threading
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.request import Request

import pytest

from marvin.tools import urls
from marvin.tools.base import ToolStatus
from marvin.tools.builtin import BrowserTool
from marvin.tools.coding import BrowserAutomationTool, ImageTool, WebFetchTool
from marvin.tools.urls import UnsafeURLError, check_url, safe_urlopen

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

IMDS_URL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
PUBLIC_ADDRESS = "93.184.215.14"


def _resolving_to(*addresses: str) -> Callable[..., list[tuple[Any, ...]]]:
    """A `socket.getaddrinfo` stand-in that resolves every host to `addresses`, without DNS."""

    def getaddrinfo(*_args: Any, **_kwargs: Any) -> list[tuple[Any, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0)) for address in addresses]

    return getaddrinfo


class TestCheckUrl:
    @pytest.mark.parametrize(
        "url",
        ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/", "data:text/plain,hi", "http:///path"],
    )
    def test_rejects_other_schemes_and_missing_hosts(self, url: str) -> None:
        with pytest.raises(UnsafeURLError, match="Only http and https URLs"):
            check_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            IMDS_URL,
            "http://169.254.170.2/v2/credentials/abc",  # ECS task metadata and credentials
            "http://127.0.0.1/",
            "http://localhost:8080/",
            "http://[::1]/",
            "http://[fe80::1]/",
            "http://[fd00:ec2::254]/latest/meta-data/",  # EC2 IMDS over IPv6
            "http://0.0.0.0/",
            "http://[::ffff:169.254.169.254]/",  # IPv4-mapped IMDS
        ],
    )
    def test_rejects_loopback_link_local_and_metadata_addresses(self, url: str) -> None:
        with pytest.raises(UnsafeURLError, match="Blocked URL"):
            check_url(url)

    def test_rejects_a_host_with_any_blocked_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _resolving_to(PUBLIC_ADDRESS, "169.254.169.254"))

        with pytest.raises(UnsafeURLError, match=r"\(169\.254\.169\.254\)"):
            check_url("https://metadata.example.com/")

    def test_accepts_a_public_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _resolving_to(PUBLIC_ADDRESS))

        check_url("https://example.com/page?q=1")

    def test_accepts_a_private_address(self) -> None:
        check_url("http://10.0.0.5:8080/health")  # RFC 1918 stays reachable, for in-VPC services


class _RedirectToImds(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(302)
        self.send_header("Location", IMDS_URL)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        """Silence the default stderr access log."""


@pytest.fixture
def redirect_server() -> Iterator[str]:
    """A localhost server that redirects every request to EC2 IMDS; yields its URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectToImds)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    server.server_close()


class TestRedirects:
    def test_redirect_to_a_metadata_address_is_refused(self) -> None:
        handler = urls._CheckedRedirectHandler()

        with pytest.raises(UnsafeURLError, match="Blocked URL"):
            handler.redirect_request(Request("http://example.com/"), io.BytesIO(), 302, "Found", Message(), IMDS_URL)

    def test_redirect_to_a_public_host_is_followed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "getaddrinfo", _resolving_to(PUBLIC_ADDRESS))
        handler = urls._CheckedRedirectHandler()

        new = handler.redirect_request(
            Request("http://example.com/"), io.BytesIO(), 302, "Found", Message(), "https://example.org/next"
        )

        assert new is not None
        assert new.full_url == "https://example.org/next"

    def test_safe_urlopen_checks_redirects(self, monkeypatch: pytest.MonkeyPatch, redirect_server: str) -> None:
        # Let the localhost test server pass as a public host; the redirect target is resolved for real.
        resolve = urls._resolve
        monkeypatch.setattr(urls, "_resolve", lambda host: {PUBLIC_ADDRESS} if host == "127.0.0.1" else resolve(host))

        with pytest.raises(UnsafeURLError, match=r"Blocked URL http://169\.254\.169\.254/"):
            safe_urlopen(redirect_server, timeout=5)


class TestToolsRefuseBlockedUrls:
    def test_web_fetch(self) -> None:
        result = asyncio.run(WebFetchTool().execute(url=IMDS_URL, timeout=1))

        assert result.status == ToolStatus.ERROR
        assert "Blocked URL" in (result.error or "")

    def test_browser_fetch(self) -> None:
        result = asyncio.run(BrowserTool().execute(url="file:///etc/passwd"))

        assert result.status == ToolStatus.ERROR
        assert "Only http and https URLs" in (result.error or "")

    def test_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        result = asyncio.run(ImageTool().execute(source="http://localhost:9/cat.png"))

        assert result.status == ToolStatus.ERROR
        assert "Blocked URL" in (result.error or "")

    def test_browser_does_not_navigate(self) -> None:
        page = AsyncMock()
        browser = AsyncMock()
        browser.new_page.return_value = page
        playwright = MagicMock()
        playwright.chromium.launch = AsyncMock(return_value=browser)
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=playwright)
        session.__aexit__ = AsyncMock(return_value=None)

        with patch("marvin.tools.coding.async_playwright", return_value=session):
            result = asyncio.run(BrowserAutomationTool().execute(action="content", url=IMDS_URL))

        assert result.status == ToolStatus.ERROR
        assert "Blocked URL" in (result.error or "")
        page.goto.assert_not_awaited()
