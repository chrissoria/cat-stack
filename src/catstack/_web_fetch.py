"""
Web content fetching utilities for URL input type.

Provides URL detection, HTML text extraction, and batch URL fetching
for use as a preprocessing step before text classification/extraction/summarization.
"""

import html as html_lib
import ipaddress
import re
import socket
from urllib.parse import urlsplit

import requests

__all__ = [
    "is_url",
    "fetch_url_text",
    "fetch_urls",
    "detect_url_input",
    "strip_html_tags",
]

_DEFAULT_TIMEOUT = 30

_MAX_CONTENT_CHARS = 50000

# Hard cap on bytes pulled from the response before bailing — guards against
# OOM on a hostile or accidentally-huge URL. 5x slack over the char cap so
# HTML markup that gets stripped later still leaves real payload room.
_MAX_RESPONSE_BYTES = 5 * _MAX_CONTENT_CHARS

# Schemes fetch_url_text will follow. Anything else (file://, ftp://, data:,
# javascript:, ...) is rejected at validation time.
_ALLOWED_SCHEMES = frozenset({"http", "https"})

_USER_AGENT = (
    "Mozilla/5.0 (compatible; CatStack/1.0; "
    "+https://github.com/chrissoria/cat-stack)"
)


def is_url(s) -> bool:
    """
    Check whether a string is a well-formed http(s) URL.

    Structural check only — no DNS resolution, no network call. Rejects
    strings with embedded control characters, non-http(s) schemes, and
    missing netloc.
    """
    if not isinstance(s, str):
        return False
    s = s.strip()
    if any(c in s for c in ("\r", "\n", "\x00")):
        return False
    try:
        parts = urlsplit(s)
    except Exception:
        return False
    return parts.scheme in _ALLOWED_SCHEMES and bool(parts.netloc)


def detect_url_input(items) -> bool:
    """
    Check whether input data is a collection of URLs.

    Inspects the first non-null item in the iterable. Returns True if it
    looks like a URL.
    """
    import pandas as pd

    if isinstance(items, str):
        return is_url(items)

    if hasattr(items, "__iter__"):
        for item in items:
            if item is not None:
                try:
                    if pd.isna(item):
                        continue
                except (TypeError, ValueError):
                    pass
                return is_url(str(item))

    return False


def _validate_url_safe(url):
    """
    Validate a URL for safe fetching: structure + SSRF host guard.

    Returns (cleaned_url, error_message). error_message is None on success.

    The SSRF guard resolves the hostname via socket.getaddrinfo and rejects
    if ANY returned address is private, loopback, link-local, reserved,
    multicast, or unspecified. Catches AWS metadata (169.254.169.254),
    localhost (127.0.0.1, ::1), RFC1918, GCP metadata host, and similar
    internal targets before any HTTP request goes out.

    Does NOT defend against DNS rebinding (resolve-once-then-reconnect to
    a different IP); that requires a custom HTTPAdapter and is out of
    scope here.
    """
    if not isinstance(url, str):
        return "", "url must be a string"
    url = url.strip()
    if any(c in url for c in ("\r", "\n", "\x00")):
        return "", "url contains control characters"
    try:
        parts = urlsplit(url)
    except Exception as e:
        return "", f"could not parse url: {e}"
    if parts.scheme not in _ALLOWED_SCHEMES:
        return "", f"scheme must be http or https; got {parts.scheme!r}"
    if not parts.netloc:
        return "", "url has empty netloc"
    hostname = parts.hostname
    if not hostname:
        return "", "url has empty hostname"

    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return "", f"could not resolve {hostname!r}: {e}"

    for info in addrinfo:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return "", f"resolved address {ip_str!r} is not a valid IP"
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return "", (
                f"{hostname!r} resolves to {ip_str} (private/internal); "
                f"refusing to fetch as an SSRF guard"
            )

    return url, None


def strip_html_tags(html: str) -> str:
    """
    Extract readable text from an HTML string.

    Removes non-content elements (navigation, headers, footers, sidebars,
    forms, scripts, styles), strips remaining tags, collapses whitespace,
    and decodes HTML entities.
    """
    text = html

    _JUNK_TAGS = (
        "script", "style", "nav", "header", "footer", "aside",
        "noscript", "iframe", "form", "svg",
    )
    for tag in _JUNK_TAGS:
        text = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

    for tag in ("input", "meta", "link", "img"):
        text = re.sub(rf"<{tag}[^>]*/?\s*>", "", text, flags=re.IGNORECASE)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = html_lib.unescape(text)
    return text


def fetch_url_text(url: str, timeout: int = _DEFAULT_TIMEOUT):
    """
    Fetch a single URL and extract its text content.

    Pre-flight: the URL's scheme and hostname are validated, and the
    hostname is resolved; if it points at a private/internal IP, the
    fetch is refused (SSRF guard). The response body is streamed and
    capped to prevent OOM on very large pages. TLS errors are surfaced —
    there is no silent verify=False fallback.

    Returns (text, error). error is None on success.
    """
    cleaned_url, validation_error = _validate_url_safe(url)
    if validation_error:
        return "", f"Error fetching {url}: {validation_error}"

    headers = {"User-Agent": _USER_AGENT}
    try:
        with requests.get(
            cleaned_url,
            headers=headers,
            timeout=timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            encoding = response.encoding

            chunks = []
            bytes_read = 0
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                chunks.append(chunk)
                bytes_read += len(chunk)
                if bytes_read > _MAX_RESPONSE_BYTES:
                    break
            raw = b"".join(chunks)

        encoding = encoding or "utf-8"
        try:
            body = raw.decode(encoding, errors="replace")
        except (LookupError, TypeError):
            body = raw.decode("utf-8", errors="replace")

        if (
            "text/html" in content_type
            or "text/plain" in content_type
            or not content_type
        ):
            text = strip_html_tags(body)
        else:
            text = body

        if len(text) > _MAX_CONTENT_CHARS:
            text = text[:_MAX_CONTENT_CHARS] + (
                f"\n\n[Content truncated at {_MAX_CONTENT_CHARS} characters]"
            )

        return text, None

    except requests.exceptions.Timeout:
        return "", f"Timeout after {timeout}s fetching {url}"
    except requests.exceptions.SSLError as e:
        return "", f"SSL/TLS error fetching {url}: {e}"
    except requests.exceptions.HTTPError as e:
        return "", f"HTTP {e.response.status_code} fetching {url}"
    except Exception as e:
        return "", f"Error fetching {url}: {e}"


def fetch_urls(urls, timeout: int = _DEFAULT_TIMEOUT):
    """
    Fetch content from a list of URLs.

    Returns list of (original_url, fetched_text, error) tuples. On success
    error is None; on failure fetched_text is "".
    """
    results = []
    for url in urls:
        url_str = str(url).strip()
        if not is_url(url_str):
            results.append((url_str, "", f"Not a valid URL: {url_str}"))
            continue
        text, error = fetch_url_text(url_str, timeout=timeout)
        results.append((url_str, text, error))
    return results
