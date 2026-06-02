"""
Tests for C9: `reply` is initialized at the top of every per-image loop
iteration in image_score_drawing and image_features.

Bug shape (verified empirically before the fix): on any non-success
path (401, 403, network error, Anthropic's empty-content response),
`reply` was never assigned, but the post-dispatch `if reply is not None:`
check still referenced it. Two failure modes:

1. First-iteration failure → UnboundLocalError, whole call crashes
   instead of recording a graceful error row.
2. Later-iteration failure → `reply` retains the value from the previous
   *successful* iteration → wrong JSON gets attached to the failing row
   (silent data corruption).

Fix: `reply = None` at the very top of each loop iteration. One-line
change per function; eliminates both modes.
"""

import inspect
from unittest.mock import patch, MagicMock

from cat_stack.image_functions import image_score_drawing, image_features


def _src(fn):
    return inspect.getsource(fn)


class TestStaticPatterns:
    def test_image_score_drawing_initializes_reply(self):
        src = _src(image_score_drawing)
        # The pattern is `reply = None` on its own line at the top of the
        # for-loop iteration; not just any occurrence of `reply = None`.
        loop_marker = "for i, img_path in enumerate"
        loop_idx = src.find(loop_marker)
        assert loop_idx != -1
        # Within the next 200 chars after the for-line, reply = None
        # should appear (before any encoding / dispatch).
        following = src[loop_idx : loop_idx + 200]
        assert "reply = None" in following, (
            f"Expected `reply = None` right after the for-loop header. "
            f"First 200 chars after for: {following!r}"
        )

    def test_image_features_initializes_reply(self):
        src = _src(image_features)
        loop_marker = "for i, img_path in enumerate"
        loop_idx = src.find(loop_marker)
        assert loop_idx != -1
        following = src[loop_idx : loop_idx + 200]
        assert "reply = None" in following


class TestRuntimeBehavior:
    @patch("requests.post")
    def test_first_iteration_http_401_does_not_crash(self, mock_post):
        """Regression: pre-fix code raised UnboundLocalError on a 401 in
        the first iteration. After the fix, the error must be captured
        in the row instead, with the {"1":"e"} sentinel."""
        import requests as r

        mock_response = MagicMock()
        mock_response.status_code = 401
        http_error = r.exceptions.HTTPError(
            "401 Client Error: Unauthorized", response=mock_response
        )
        mock_response.raise_for_status.side_effect = http_error
        mock_post.return_value = mock_response

        # _encode_image needs a real file; use a tiny one we know exists.
        # Any PNG will do; the API call gets mocked out before any image
        # bytes are inspected.
        import os
        png = "/Users/chrissoria/Documents/Research/improving_catllm_classification/plots/a19f_Llama_4_1_confusion_matrix.png"
        assert os.path.exists(png)

        try:
            result = image_features(
                image_description="test",
                image_input=[png],
                features_to_extract=["test feature"],
                api_key="fake",
                user_model="gpt-4o-mini",
                model_source="openai",
            )
        except UnboundLocalError as e:
            raise AssertionError(
                f"C9 regression: UnboundLocalError on first-iteration failure: {e}"
            )

        assert len(result) == 1
        # The row's model_response (or link1) should contain the error,
        # and the json column should be the error sentinel.
        responses = result.get("model_response", result.get("link1"))
        assert "401" in str(responses.iloc[0]) or "Error" in str(responses.iloc[0])
        # extracted_jsons should have {"1":"e"} → after normalization,
        # the column "1" has value "e"
        assert "1" in result.columns and str(result["1"].iloc[0]) == "e"
