"""
Tests for C7: Anthropic media_type is derived from the actual file
extension instead of hardcoded.

Smoke-tested against the real Anthropic API: it returns
  400 invalid_request_error "The image was specified using the
  image/jpeg media type, but the image appears to be a image/png image"
when the declared media_type doesn't match the actual bytes. So the
pre-fix code (hardcoded "image/jpeg" for user images, "image/png" for
the reference) reliably failed every PNG-vs-Anthropic call.

After the fix, all six Anthropic dispatch sites in image_functions.py
follow the same `f"image/{ext}" if ext else "image/jpeg"` pattern that
the newer image_multi_class paths already use.
"""

import inspect

from cat_stack.image_functions import image_score_drawing, image_features


def _src(fn):
    return inspect.getsource(fn)


def test_image_score_drawing_anthropic_uses_ext():
    """No hardcoded media_type literals — both reference and user image
    derive their media_type from the encoded extension."""
    src = _src(image_score_drawing)
    # We expect 0 hardcoded literals in this function
    assert '"media_type": "image/png"' not in src
    assert '"media_type": "image/jpeg"' not in src
    # And the derived-from-ext pattern should be present (for both
    # ref_ext and ext)
    assert 'f"image/{ref_ext}"' in src, "reference media_type should derive from ref_ext"
    assert 'f"image/{ext}"' in src, "user-image media_type should derive from ext"


def test_image_features_anthropic_uses_ext():
    src = _src(image_features)
    assert '"media_type": "image/jpeg"' not in src
    assert '"media_type": "image/png"' not in src
    assert 'f"image/{ext}"' in src, "media_type should derive from ext"


def test_full_module_has_no_hardcoded_anthropic_media_type():
    """Sweep the whole module — no place should construct a media_type
    by hand. Use _encode_image's returned ext."""
    with open(
        "/Users/chrissoria/Documents/Research/cat-stack/src/catstack/image_functions.py"
    ) as f:
        src = f.read()
    # zero hardcoded image/png or image/jpeg as media_type values
    import re
    matches = re.findall(r'"media_type"\s*:\s*"image/(png|jpeg|jpg|gif|webp|heic|heif)"', src)
    assert not matches, (
        f"found {len(matches)} hardcoded media_type literal(s): {matches}. "
        f"Anthropic rejects mismatched media_type with HTTP 400 — derive "
        f"from the actual file extension instead."
    )
