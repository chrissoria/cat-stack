"""
Tests for task #29 (H-IMG): image-loader hardening.

Three sub-fixes:
  1. Case-insensitive directory matching — `glob.glob('*.jpg')` is
     case-sensitive on every platform, so `IMG.JPG` (uppercase, common
     from phone cameras) silently dropped. Switched to
     pathlib + suffix.lower().
  2. image_features and image_score_drawing now call _load_image_files
     instead of their own inline glob — fixes the additional bug that
     those two functions returned [] when passed a single file path
     instead of a directory.
  3. Size guard in _encode_image: print a one-time warning per path
     when a file exceeds 20 MB.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from cat_stack.image_functions import _load_image_files, _encode_image


def _touch(p: Path, size_bytes: int = 8):
    """Create a small file with the given size."""
    p.write_bytes(b"\x00" * size_bytes)


class TestLoadImageFilesCaseInsensitive:
    def test_directory_picks_up_mixed_case_extensions(self, tmp_path):
        """Each of these would have been missed by at least one of the
        old globs (`*.jpg`, `*.JPG`, etc.) because no single-case glob
        matched them all."""
        files = {
            "lowercase.jpg",
            "UPPERCASE.JPG",
            "MixedCase.Jpg",
            "Trail.JpEg",
            "lower.png",
            "UPPER.PNG",
        }
        for name in files:
            _touch(tmp_path / name)

        loaded = _load_image_files(str(tmp_path))
        loaded_names = {os.path.basename(p) for p in loaded}
        assert loaded_names == files, (
            f"missing: {files - loaded_names}; extras: {loaded_names - files}"
        )

    def test_directory_excludes_non_image_files(self, tmp_path):
        """Non-image files (text, etc.) should be filtered out."""
        _touch(tmp_path / "real.jpg")
        _touch(tmp_path / "doc.txt")
        _touch(tmp_path / "data.csv")
        _touch(tmp_path / ".DS_Store")  # macOS metadata

        loaded = _load_image_files(str(tmp_path))
        assert [os.path.basename(p) for p in loaded] == ["real.jpg"]

    def test_directory_returns_sorted(self, tmp_path):
        """Output should be sorted for reproducible iteration order."""
        for name in ["c.jpg", "a.JPG", "b.png"]:
            _touch(tmp_path / name)
        loaded = _load_image_files(str(tmp_path))
        assert loaded == sorted(loaded), "expected sorted output"

    def test_single_file_path_returns_list_of_one(self, tmp_path):
        """Single file → wraps it in a list."""
        p = tmp_path / "only.png"
        _touch(p)
        loaded = _load_image_files(str(p))
        assert loaded == [str(p)]

    def test_list_input_returned_as_is(self):
        """List → returned unchanged (so callers can pass pre-validated lists)."""
        inputs = ["a.jpg", "b.png"]
        assert _load_image_files(inputs) == inputs

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_image_files(str(tmp_path / "does-not-exist"))


class TestSingleFileFlowsViaSharedLoader:
    """image_features and image_score_drawing previously had their own
    inline glob that ONLY handled directory inputs — passing a single
    file path returned []. After consolidating to _load_image_files,
    both correctly handle the single-file case."""

    def test_image_features_consolidation_call_site_exists(self):
        """Audit: confirm image_features no longer has its own
        image_extensions inline list."""
        import inspect
        from cat_stack.image_functions import image_features
        src = inspect.getsource(image_features)
        assert "image_extensions" not in src, (
            "image_features should use _load_image_files, not maintain "
            "its own image_extensions list"
        )
        assert "_load_image_files" in src

    def test_image_score_drawing_consolidation_call_site_exists(self):
        """Same audit for image_score_drawing."""
        import inspect
        from cat_stack.image_functions import image_score_drawing
        src = inspect.getsource(image_score_drawing)
        assert "image_extensions" not in src
        assert "_load_image_files" in src


class TestSizeGuardWarning:
    """_encode_image warns once per path when file > 20 MB."""

    def test_small_file_no_warning(self, tmp_path, capsys):
        p = tmp_path / "small.png"
        _touch(p, size_bytes=1024)
        _encode_image(str(p))
        out = capsys.readouterr().out
        assert "Warning" not in out

    def test_large_file_triggers_warning(self, tmp_path, capsys):
        from cat_stack import image_functions as f
        f._warned_large_images.clear()

        p = tmp_path / "huge.png"
        _touch(p, size_bytes=21 * 1024 * 1024)  # 21 MB
        _encode_image(str(p))

        out = capsys.readouterr().out
        assert "Warning" in out
        assert "21.0 MB" in out or "21." in out
        assert str(p) in out

    def test_repeat_warnings_are_suppressed(self, tmp_path, capsys):
        """Same path → warning fires once, not on every call."""
        from cat_stack import image_functions as f
        f._warned_large_images.clear()

        p = tmp_path / "huge2.png"
        _touch(p, size_bytes=25 * 1024 * 1024)

        _encode_image(str(p))
        first_out = capsys.readouterr().out
        assert "Warning" in first_out

        _encode_image(str(p))
        second_out = capsys.readouterr().out
        assert "Warning" not in second_out

    def test_encoding_succeeds_alongside_warning(self, tmp_path):
        """The warning is informational — the file still gets encoded."""
        from cat_stack import image_functions as f
        f._warned_large_images.clear()

        p = tmp_path / "huge3.png"
        _touch(p, size_bytes=21 * 1024 * 1024)
        encoded, ext, valid = _encode_image(str(p))
        assert valid
        assert ext == "png"
        assert encoded is not None
