"""
Tests for PDF-VALIDATION (#48): _load_pdf_files now refuses files that
don't have the PDF magic bytes (b"%PDF-") in the first 1024 bytes.

Pre-fix: a webpage saved with .pdf extension would pass _load_pdf_files
unchecked, get fed to PyMuPDF (extremely permissive), render as a
near-blank page, and the VLM would happily classify the blank page and
return processing_status: "success" with bogus category columns. The
user got a clean DataFrame of garbage with no signal that the input was
malformed.

Post-fix:
  - single bogus path → ValueError with a clear explanation
  - list with bogus path → ValueError naming the offending path
  - directory glob → bogus files are warned-and-skipped, real PDFs pass
"""

import os

import pytest

from cat_stack.pdf_functions import _is_likely_pdf, _load_pdf_files


HTML_BODY = (
    b"<!DOCTYPE html>\n<html><head><title>Not a PDF</title></head>"
    b"<body><h1>Hello</h1></body></html>\n"
)

REAL_PDF_HEADER = b"%PDF-1.7\n%\x80\x80\x80\x80\n"  # minimal valid PDF preamble


def _write(path, content: bytes):
    with open(path, "wb") as f:
        f.write(content)


class TestIsLikelyPdf:
    def test_html_with_pdf_extension_fails(self, tmp_path):
        p = tmp_path / "fake.pdf"
        _write(p, HTML_BODY)
        assert _is_likely_pdf(str(p)) is False

    def test_real_pdf_header_passes(self, tmp_path):
        p = tmp_path / "real.pdf"
        _write(p, REAL_PDF_HEADER + b"% rest of PDF body...")
        assert _is_likely_pdf(str(p)) is True

    def test_pdf_with_leading_bytes_still_passes(self, tmp_path):
        """Real PDFs occasionally have leading bytes (e.g. MIME-wrapped).
        The 1024-byte scan window catches those."""
        p = tmp_path / "wrapped.pdf"
        _write(p, b"Content-Type: application/pdf\n\n" + REAL_PDF_HEADER)
        assert _is_likely_pdf(str(p)) is True

    def test_empty_file_fails(self, tmp_path):
        p = tmp_path / "empty.pdf"
        _write(p, b"")
        assert _is_likely_pdf(str(p)) is False

    def test_nonexistent_path_fails_gracefully(self, tmp_path):
        # No raise — just returns False
        assert _is_likely_pdf(str(tmp_path / "nope.pdf")) is False


class TestLoadPdfFilesRejectsBogus:
    def test_single_bogus_pdf_raises(self, tmp_path):
        p = tmp_path / "fake.pdf"
        _write(p, HTML_BODY)
        with pytest.raises(ValueError) as exc:
            _load_pdf_files(str(p))
        msg = str(exc.value)
        assert "fake.pdf" in msg
        assert "PDF header" in msg
        # Bug-prevention rationale should appear so user understands why
        # we're refusing rather than just letting PyMuPDF stumble through.
        assert "PyMuPDF" in msg or "junk" in msg

    def test_real_single_pdf_passes(self, tmp_path):
        p = tmp_path / "real.pdf"
        _write(p, REAL_PDF_HEADER)
        result = _load_pdf_files(str(p))
        assert result == [str(p)]

    def test_list_containing_bogus_raises(self, tmp_path):
        good = tmp_path / "good.pdf"
        bad = tmp_path / "bad.pdf"
        _write(good, REAL_PDF_HEADER)
        _write(bad, HTML_BODY)
        with pytest.raises(ValueError) as exc:
            _load_pdf_files([str(good), str(bad)])
        assert "bad.pdf" in str(exc.value)

    def test_directory_with_mixed_files_filters_bogus(self, tmp_path, capsys):
        good = tmp_path / "real.pdf"
        bad = tmp_path / "fake.pdf"
        _write(good, REAL_PDF_HEADER)
        _write(bad, HTML_BODY)

        result = _load_pdf_files(str(tmp_path))
        # Real PDF kept, bogus dropped
        assert result == [str(good)]

        # Warning printed about the skip so the user knows
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "fake.pdf" in out

    def test_directory_with_only_bogus_pdfs_returns_empty(self, tmp_path):
        bad1 = tmp_path / "bad1.pdf"
        bad2 = tmp_path / "bad2.pdf"
        _write(bad1, HTML_BODY)
        _write(bad2, b"random text, no PDF header at all")
        result = _load_pdf_files(str(tmp_path))
        assert result == []
