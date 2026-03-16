"""
Web content fetching utilities for URL input type.

Provides URL detection, HTML text extraction, and batch URL fetching
for use as a preprocessing step before text classification/extraction/summarization.
"""

import re

import requests

__all__ = [
    "is_url",
    "fetch_url_text",
    "fetch_urls",
    "detect_url_input",
    "strip_html_tags",
]

# Timeout for individual URL fetches (seconds)
_DEFAULT_TIMEOUT = 30

# Maximum characters to keep from fetched content
_MAX_CONTENT_CHARS = 50000

# User-Agent header for polite web scraping
_USER_AGENT = (
    "Mozilla/5.0 (compatible; CatStack/1.0; "
    "+https://github.com/chrissoria/cat-stack)"
)


def is_url(s) -> bool:
    """
    Check if a string looks like a URL (starts with http:// or https://).

    Args:
        s: Value to check.

    Returns:
        True if the value is a string starting with http:// or https://.
    """
    if not isinstance(s, str):
        return False
    return bool(re.match(r"https?://", s.strip()))


def detect_url_input(items) -> bool:
    """
    Check whether input data is a collection of URLs.

    Inspects the first non-null item in the iterable.  Returns True if
    it looks like a URL.

    Args:
        items: A single string, list, pandas Series, or other iterable.

    Returns:
        True if the input appears to be URL data.
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


def strip_html_tags(html: str) -> str:
    """
    Remove HTML tags and clean up whitespace from an HTML string.

    Strips ``<script>`` and ``<style>`` blocks, removes all remaining tags,
    collapses whitespace, and decodes common HTML entities.

    Args:
        html: Raw HTML string.

    Returns:
        Plain-text string.
    """
    # Remove script and style elements entirely
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        "",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Decode common HTML entities
    entity_map = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
    }
    for entity, char in entity_map.items():
        text = text.replace(entity, char)
    return text


def fetch_url_text(url: str, timeout: int = _DEFAULT_TIMEOUT):
    """
    Fetch a single URL and extract its text content.

    HTML responses are stripped of tags; other content types are returned
    as-is.  Very long pages are truncated to ``_MAX_CONTENT_CHARS``.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        tuple: ``(text, error)`` where *text* is the extracted content and
        *error* is ``None`` on success or an error message string.
    """
    try:
        headers = {"User-Agent": _USER_AGENT}
        try:
            response = requests.get(url.strip(), headers=headers, timeout=timeout)
        except requests.exceptions.SSLError:
            # Retry without SSL verification as fallback
            response = requests.get(
                url.strip(), headers=headers, timeout=timeout, verify=False
            )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if (
            "text/html" in content_type
            or "text/plain" in content_type
            or not content_type
        ):
            text = strip_html_tags(response.text)
        else:
            text = response.text

        # Truncate very long content
        if len(text) > _MAX_CONTENT_CHARS:
            text = text[:_MAX_CONTENT_CHARS] + (
                f"\n\n[Content truncated at {_MAX_CONTENT_CHARS} characters]"
            )

        return text, None

    except requests.exceptions.Timeout:
        return "", f"Timeout after {timeout}s fetching {url}"
    except requests.exceptions.HTTPError as e:
        return "", f"HTTP {e.response.status_code} fetching {url}"
    except Exception as e:
        return "", f"Error fetching {url}: {e}"


def fetch_urls(urls, timeout: int = _DEFAULT_TIMEOUT):
    """
    Fetch content from a list of URLs.

    Args:
        urls: Iterable of URL strings.
        timeout: Per-URL request timeout in seconds.

    Returns:
        list of ``(original_url, fetched_text, error)`` tuples.  On success
        *error* is ``None``; on failure *fetched_text* is ``""``.
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
