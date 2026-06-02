"""
Tests for C6+C8 + task #42: replace inline base64 encoding in
image_score_drawing and image_features with the existing _encode_image
helper, and skip invalid inputs via `continue` instead of letting them
fall through into the provider dispatch.

The fix eliminates three concrete bugs at once:

- **C8** (image_score_drawing reference encoding): used raw uppercase
  file extensions, no jpg→jpeg normalization, broke on extensionless
  paths, and reassigned the `reference_image` parameter to a data URI
  in-place.
- **C6** (image_score_drawing per-iteration encoding): an "Error: …"
  string for invalid inputs was unconditionally rewrapped into a fake
  data URI on the line right after the validity check, then shipped to
  the provider.
- **#42** (image_features inline encoding): same hand-rolled pattern as
  image_score_drawing, lacking jpg→jpeg normalization. No duplicate-wrap
  bug there, but the same regression risk.

After the fix, both functions delegate to `_encode_image`, which returns
`(encoded, ext, is_valid)`. Invalid inputs trigger `continue` before the
dispatch ever runs.
"""

import inspect
from unittest.mock import patch

import pytest

from cat_stack.image_functions import (
    image_score_drawing,
    image_features,
)


def _src(fn):
    return inspect.getsource(fn)


class TestStaticPatterns:
    def test_image_score_drawing_uses_encode_image(self):
        assert "_encode_image" in _src(image_score_drawing)

    def test_image_features_uses_encode_image(self):
        assert "_encode_image" in _src(image_features)

    def test_image_score_drawing_no_inline_base64(self):
        """No hand-rolled base64.b64encode inside the function body —
        delegating to _encode_image is now the contract."""
        assert "base64.b64encode" not in _src(image_score_drawing)

    def test_image_features_no_inline_base64(self):
        assert "base64.b64encode" not in _src(image_features)

    def test_no_duplicate_encoded_image_assignment_in_score_drawing(self):
        """C6 regression: exactly one `encoded_image = ` assignment inside
        the per-iteration loop. The pre-fix code had two — the second one
        unconditionally rewrapped the error string from the first."""
        src = _src(image_score_drawing)
        assignments = [
            line for line in src.splitlines()
            if "encoded_image =" in line and "==" not in line
        ]
        assert len(assignments) == 1, (
            f"expected 1 encoded_image assignment, got {len(assignments)}: "
            f"{[a.strip() for a in assignments]}"
        )

    def test_no_dead_valid_image_branch_in_score_drawing(self):
        """The unreachable `elif valid_image == False` branch and the
        `valid_image = True/False` flag assignments must be gone."""
        src = _src(image_score_drawing)
        assert "valid_image == False" not in src
        assert "valid_image = False" not in src
        assert "valid_image = True" not in src

    def test_no_dead_valid_image_branch_in_features(self):
        src = _src(image_features)
        assert "valid_image == False" not in src
        assert "valid_image = False" not in src
        assert "valid_image = True" not in src


class TestReferenceEncoding:
    @patch("cat_stack.image_functions._encode_image")
    def test_invalid_reference_path_raises_filenotfound(self, mock_encode):
        """C8: a broken reference path must raise eagerly. The pre-fix
        code silently produced a malformed data URI that the provider
        would later reject."""
        mock_encode.return_value = (None, None, False)
        with pytest.raises(FileNotFoundError, match="reference_image"):
            image_score_drawing(
                reference_image_description="circle",
                image_input=[],
                reference_image="/nonexistent.png",
                api_key="fake",
                model_source="openai",
            )

    @patch("cat_stack.image_functions._encode_image")
    def test_reference_path_passed_to_encode_image(self, mock_encode):
        """The reference path goes through _encode_image, which handles
        jpg→jpeg normalization and extensionless-path edge cases. The
        hand-rolled `.split('.')[-1]` approach is gone."""
        mock_encode.return_value = ("DATA", "jpeg", True)
        try:
            image_score_drawing(
                reference_image_description="circle",
                image_input=[],
                reference_image="/path/ref.JPG",
                api_key="fake",
                model_source="openai",
            )
        except ValueError:
            # Empty image_input causes pd.concat on an empty list to raise
            # near the end of the function; doesn't affect what we're testing.
            pass
        # The first call to _encode_image is the reference resolution,
        # which must happen before any iteration.
        first_call_args = mock_encode.call_args_list[0]
        assert first_call_args.args[0] == "/path/ref.JPG"
