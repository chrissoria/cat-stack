"""
Test for C16: is_image_mode is assigned alongside is_pdf_mode at the top
of summarize_ensemble, before any closure that references it.

Before the fix, is_image_mode was set ~300 lines after the
summarize_single_item closure had already referenced it. The closure
worked by Python's late binding — its body resolves names at call time
rather than def time — but a future refactor that ran the closure before
the assignment would NameError. C16 hoists the assignment to sit next to
is_pdf_mode so the two flags travel together.
"""

import inspect

from cat_stack.text_functions_ensemble import summarize_ensemble


def test_is_image_mode_assigned_before_summarize_single_item_closure():
    """The is_image_mode assignment must come before the closure is defined."""
    src = inspect.getsource(summarize_ensemble)
    lines = src.splitlines()

    assign_lines = [
        i for i, line in enumerate(lines)
        if "is_image_mode = (file_type == 'image')" in line
    ]
    closure_def_line = next(
        (i for i, line in enumerate(lines) if "def summarize_single_item" in line),
        None,
    )

    assert len(assign_lines) == 1, (
        f"Expected exactly one is_image_mode assignment; found {len(assign_lines)} "
        f"at relative lines {assign_lines}. The C16 fix collapsed two assignments "
        f"into one — a re-introduced second assignment suggests partial regression."
    )
    assert closure_def_line is not None, "summarize_single_item closure not found"
    assert assign_lines[0] < closure_def_line, (
        f"is_image_mode assigned at relative line {assign_lines[0]} but the closure "
        f"that uses it is defined at relative line {closure_def_line} — the late "
        f"binding is fragile and a future refactor could trigger NameError."
    )


def test_is_image_mode_assigned_next_to_is_pdf_mode():
    """The two mode flags should be set together so they stay in sync
    under future refactors (e.g., adding a new file_type branch)."""
    src = inspect.getsource(summarize_ensemble)
    lines = src.splitlines()

    pdf_assigns = [
        i for i, line in enumerate(lines)
        if "is_pdf_mode = (file_type == 'pdf')" in line
    ]
    image_assigns = [
        i for i, line in enumerate(lines)
        if "is_image_mode = (file_type == 'image')" in line
    ]

    assert pdf_assigns and image_assigns
    delta = abs(pdf_assigns[0] - image_assigns[0])
    assert delta <= 2, (
        f"is_pdf_mode at relative line {pdf_assigns[0]}, is_image_mode at "
        f"relative line {image_assigns[0]} — these should sit together "
        f"(within 2 lines) for clarity and to keep DOCX-reset logic in sync."
    )


def test_docx_branch_resets_both_mode_flags():
    """The DOCX-to-text conversion path resets is_pdf_mode = False; it
    must also reset is_image_mode = False so the post-conversion code
    doesn't accidentally treat the converted text as image input."""
    src = inspect.getsource(summarize_ensemble)
    # Both resets should appear inside the file_type == 'docx' block
    assert "is_pdf_mode = False" in src
    assert "is_image_mode = False" in src, (
        "DOCX branch must reset is_image_mode = False alongside is_pdf_mode = False"
    )
