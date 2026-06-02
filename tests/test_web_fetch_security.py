"""
Tests for C13 (SSRF) + C14 (silent SSL bypass) and the two adjacent
bugs in catstack/_web_fetch.py — loose URL parsing and unbounded
response body reads.

Pre-fix behavior:
  - is_url used re.match (unanchored) so it accepted strings like
    "http://evil\\nhttp://victim".
  - fetch_url_text passed any url straight to requests.get with no
    host/IP validation; AWS metadata, localhost, RFC1918, etc. were all
    reachable.
  - On any SSLError, fetch_url_text silently retried with verify=False,
    converting MITM-detectable conditions into silent compromises.
  - requests.get(...) was not streaming; the entire response body landed
    in memory before _MAX_CONTENT_CHARS truncation. A multi-GB URL would
    OOM.
"""

from unittest.mock import patch, MagicMock

import pytest

from cat_stack._web_fetch import (
    is_url,
    fetch_url_text,
    fetch_urls,
    _validate_url_safe,
)


class TestIsUrl:
    def test_accepts_valid_http(self):
        assert is_url("http://example.com")

    def test_accepts_valid_https(self):
        assert is_url("https://example.com/path?x=1")

    def test_rejects_non_string(self):
        assert not is_url(None)
        assert not is_url(123)
        assert not is_url(["http://x.com"])

    def test_rejects_wrong_schemes(self):
        for url in (
            "ftp://example.com",
            "file:///etc/passwd",
            "data:text/plain,hello",
            "javascript:alert(1)",
        ):
            assert not is_url(url), f"should reject {url!r}"

    def test_rejects_empty_netloc(self):
        assert not is_url("http://")
        assert not is_url("https:///path")

    def test_rejects_crlf_injection(self):
        """Pre-fix re.match was unanchored — accepted 'http://evil\\nhttp://victim'.
        The urlsplit-based check rejects any control char."""
        assert not is_url("http://evil.com\nhttp://victim.com")
        assert not is_url("http://evil.com\r\nMalicious-Header: yes")

    def test_rejects_nul_byte(self):
        assert not is_url("http://x.com\x00")


class TestValidateUrlSafeStructure:
    def test_rejects_non_string(self):
        cleaned, err = _validate_url_safe(None)
        assert err is not None
        assert "string" in err

    def test_rejects_embedded_control_chars(self):
        """Trailing whitespace is stripped harmlessly; embedded \\r\\n
        (CRLF injection) is rejected."""
        cleaned, err = _validate_url_safe("http://x.com\nMalicious-Header: yes")
        assert err is not None
        assert "control" in err

    def test_rejects_wrong_scheme(self):
        cleaned, err = _validate_url_safe("file:///etc/passwd")
        assert err is not None
        assert "scheme" in err

    def test_rejects_empty_netloc(self):
        cleaned, err = _validate_url_safe("http:///path")
        assert err is not None


class TestValidateUrlSafeSsrf:
    """The SSRF guard — host resolution + IP family check. We mock
    socket.getaddrinfo so the tests don't depend on real DNS."""

    @staticmethod
    def _addrinfo(ip):
        # getaddrinfo returns a list of 5-tuples: (family, type, proto, canonname, sockaddr)
        # sockaddr for IPv4 is (host, port); for IPv6 is (host, port, flowinfo, scopeid)
        return [(2, 1, 6, "", (ip, 0))]

    @patch("socket.getaddrinfo")
    def test_accepts_public_ipv4(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = self._addrinfo("93.184.216.34")  # example.com
        cleaned, err = _validate_url_safe("https://example.com/")
        assert err is None, f"unexpected rejection: {err}"
        assert cleaned == "https://example.com/"

    @patch("socket.getaddrinfo")
    def test_rejects_localhost(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = self._addrinfo("127.0.0.1")
        cleaned, err = _validate_url_safe("http://localhost/")
        assert err is not None
        assert "SSRF" in err or "private" in err or "internal" in err

    @patch("socket.getaddrinfo")
    def test_rejects_aws_metadata_ip(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = self._addrinfo("169.254.169.254")
        cleaned, err = _validate_url_safe("http://169.254.169.254/latest/meta-data/")
        assert err is not None
        assert "169.254.169.254" in err

    @patch("socket.getaddrinfo")
    def test_rejects_rfc1918(self, mock_getaddrinfo):
        for ip in ("10.0.0.1", "172.16.0.1", "192.168.1.1"):
            mock_getaddrinfo.return_value = self._addrinfo(ip)
            cleaned, err = _validate_url_safe(f"http://example.com/")
            assert err is not None, f"failed to reject {ip}"

    @patch("socket.getaddrinfo")
    def test_rejects_unspecified_address(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = self._addrinfo("0.0.0.0")
        cleaned, err = _validate_url_safe("http://0.0.0.0/")
        assert err is not None

    @patch("socket.getaddrinfo")
    def test_rejects_ipv6_loopback(self, mock_getaddrinfo):
        # IPv6 sockaddr is a 4-tuple
        mock_getaddrinfo.return_value = [(30, 1, 6, "", ("::1", 0, 0, 0))]
        cleaned, err = _validate_url_safe("http://[::1]/")
        assert err is not None

    @patch("socket.getaddrinfo")
    def test_rejects_when_any_resolved_address_is_private(self, mock_getaddrinfo):
        """If hostname resolves to multiple IPs and any of them is private,
        reject. (Defense against an attacker controlling DNS to return one
        public + one private.)"""
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 0)),  # public
            (2, 1, 6, "", ("10.0.0.1", 0)),         # private
        ]
        cleaned, err = _validate_url_safe("http://shifty.example.com/")
        assert err is not None


class TestFetchUrlTextValidation:
    @patch("socket.getaddrinfo")
    def test_localhost_blocked_pre_network(self, mock_getaddrinfo):
        """fetch_url_text must reject SSRF targets before making any HTTP
        call — verify requests.get is never invoked."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 0))]
        with patch("requests.get") as mock_get:
            text, err = fetch_url_text("http://localhost/")
            assert err is not None
            assert text == ""
            assert not mock_get.called, "HTTP request should never go out for blocked SSRF target"

    @patch("socket.getaddrinfo")
    def test_bad_scheme_blocked_pre_network(self, mock_getaddrinfo):
        with patch("requests.get") as mock_get:
            text, err = fetch_url_text("file:///etc/passwd")
            assert err is not None
            assert not mock_get.called


class TestNoSilentSslBypass:
    """C14 regression: an SSLError surfaces as an error to the caller
    instead of being papered over with verify=False."""

    @patch("socket.getaddrinfo")
    @patch("requests.get")
    def test_ssl_error_does_not_retry_with_verify_false(
        self, mock_get, mock_getaddrinfo
    ):
        import requests as r
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
        mock_get.side_effect = r.exceptions.SSLError("certificate verify failed")

        text, err = fetch_url_text("https://example.com/")

        # The error is surfaced
        assert text == ""
        assert err is not None
        assert "SSL" in err or "TLS" in err
        # There was exactly ONE call to requests.get — no retry with verify=False
        assert mock_get.call_count == 1
        # That single call did NOT set verify=False
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs.get("verify") is not False


class TestStreamingCap:
    @patch("socket.getaddrinfo")
    @patch("requests.get")
    def test_iter_content_used_not_full_body(self, mock_get, mock_getaddrinfo):
        """The body is consumed via iter_content (streaming), not via
        response.text/.content (which loads everything up-front)."""
        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]

        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content = MagicMock(return_value=[b"hello world"])
        mock_get.return_value = mock_response

        text, err = fetch_url_text("https://example.com/")
        assert err is None
        assert "hello world" in text
        assert mock_response.iter_content.called

    @patch("socket.getaddrinfo")
    @patch("requests.get")
    def test_stops_reading_after_max_bytes(self, mock_get, mock_getaddrinfo):
        """If the response body is larger than _MAX_RESPONSE_BYTES, the
        reader bails out instead of accumulating the whole thing."""
        from cat_stack._web_fetch import _MAX_RESPONSE_BYTES

        mock_getaddrinfo.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]

        # Emit chunks that together exceed the cap by 10x; tracker counts
        # how many we yield.
        chunk = b"x" * 8192
        n_chunks_to_exceed_cap = (_MAX_RESPONSE_BYTES // 8192) + 5
        chunks_yielded = []

        def chunked():
            for i in range(n_chunks_to_exceed_cap * 10):
                chunks_yielded.append(i)
                yield chunk

        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_response.encoding = "utf-8"
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content = MagicMock(side_effect=lambda chunk_size: chunked())
        mock_get.return_value = mock_response

        text, err = fetch_url_text("https://example.com/")
        assert err is None
        # We should have read enough to exceed the cap but stopped soon
        # after, NOT consumed all 10× the cap.
        assert len(chunks_yielded) <= n_chunks_to_exceed_cap + 2, (
            f"reader did not bail out at the byte cap; consumed {len(chunks_yielded)} chunks"
        )


class TestFetchUrlsPropagation:
    """fetch_urls is a thin wrapper — make sure the structural rejection
    still happens via is_url (so we don't even hit DNS for clearly-bad URLs)."""

    def test_invalid_url_rejected_before_dns(self):
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            results = fetch_urls(["not-a-url", "file:///etc/passwd"])
            assert all(err is not None for (_, _, err) in results)
            assert not mock_getaddrinfo.called
