# Copyright (c) 2026 MNKT
"""Checks for URLs that tools fetch on the model's behalf."""

import ipaddress
import socket
from typing import IO, TYPE_CHECKING, override
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

if TYPE_CHECKING:
    from http.client import HTTPMessage, HTTPResponse

# EC2 instance metadata over IPv6. Over IPv4, IMDS (169.254.169.254) and ECS task metadata
# (169.254.170.2) are link-local, so the link-local check covers them.
_IMDS_IPV6 = ipaddress.IPv6Address("fd00:ec2::254")


class UnsafeURLError(ValueError):
    """A URL that a tool must not fetch."""


def _resolve(host: str) -> set[str]:
    """Return every address `host` resolves to."""
    return {str(info[4][0]) for info in socket.getaddrinfo(host, None)}


def _is_blocked(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip == _IMDS_IPV6


def check_url(url: str) -> None:
    """Raise `UnsafeURLError` unless `url` is http(s) and none of its host's addresses is blocked.

    Blocked: loopback, link-local (169.254.0.0/16, which holds EC2 IMDS and ECS task metadata;
    fe80::/10), unspecified (0.0.0.0, ::) and fd00:ec2::254 (IMDS over IPv6). Private ranges
    (RFC 1918) are allowed. The HTTP client resolves the host again to connect, so DNS
    rebinding can pass this check.
    """
    # Browsers read a backslash in an http(s) URL as a slash: `http://169.254.169.254\@example.com/`
    # goes to 169.254.169.254. Check the host a browser would use.
    parts = urlsplit(url.replace("\\", "/"))
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UnsafeURLError(f"Only http and https URLs with a host can be fetched: {url}")
    blocked = sorted(address for address in _resolve(parts.hostname) if _is_blocked(address))
    if blocked:
        raise UnsafeURLError(
            f"Blocked URL {url}: {parts.hostname} resolves to a loopback, link-local or metadata address "
            f"({', '.join(blocked)})"
        )


class _CheckedRedirectHandler(HTTPRedirectHandler):
    """Runs `check_url` on each redirect target before following it."""

    @override
    def redirect_request(
        self, req: Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> Request | None:
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = build_opener(_CheckedRedirectHandler)


def safe_urlopen(url: str | Request, *, timeout: float) -> HTTPResponse:
    """Open a model-supplied http(s) URL, checking it and each redirect target with `check_url`."""
    check_url(url.full_url if isinstance(url, Request) else url)
    return _OPENER.open(url, timeout=timeout)
