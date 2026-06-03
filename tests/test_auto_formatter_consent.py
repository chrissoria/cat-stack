"""
Tests for AUTO-FORMATTER (task #45): json_formatter=None default now
prompts the user on the first malformed-JSON row instead of silently
producing broken output, and the existing auto-install path now requires
explicit consent.

Behavior matrix:
  json_formatter=True  → explicit opt-in. Eager dep check + load.
                         _formatter_state["_consent"] == "approved".
  json_formatter=False → explicit opt-out. _formatter_state = None.
  json_formatter=None  → "auto" mode. Placeholder state with
                         _formatter_state["_consent"] == "auto" and
                         _loader = None. First malformed row triggers
                         _prompt_formatter_consent().

Non-TTY contexts (CI, batch) decline silently with a one-time suggestion.
"""

import inspect
import io
import sys
from unittest.mock import patch, MagicMock

import pytest

from cat_stack._formatter import (
    _check_dependencies_installed,
    _install_dependencies,
    _prompt_formatter_consent,
    _ensure_dependencies,
)


# ── _check_dependencies_installed ──────────────────────────────────────

class TestCheckDependenciesInstalled:
    def test_no_side_effects_on_test_env(self):
        """Returns a bool; doesn't raise, doesn't install, doesn't prompt."""
        result = _check_dependencies_installed()
        assert isinstance(result, bool)

    @patch.dict("sys.modules", {"torch": None, "transformers": None, "accelerate": None})
    def test_returns_false_when_modules_absent(self):
        # Patching to None makes the import raise; pure check should return False.
        result = _check_dependencies_installed()
        assert result is False


# ── _prompt_formatter_consent ───────────────────────────────────────────

class TestPromptFormatterConsent:
    @patch("sys.stdin.isatty", return_value=False)
    def test_non_tty_returns_declined_without_blocking(self, _mock_isatty):
        """CI / batch scripts must NOT block on input(). Print a suggestion
        and continue."""
        out = io.StringIO()
        with patch("sys.stdout", out), patch("builtins.input") as mock_input:
            result = _prompt_formatter_consent("test-model")
        assert result == "declined"
        assert not mock_input.called, "input() must not be called in non-TTY contexts"
        assert "test-model" in out.getvalue() or "Malformed JSON" in out.getvalue()

    @patch("sys.stdin.isatty", return_value=True)
    @patch("cat_stack._formatter._check_dependencies_installed", return_value=True)
    @patch("builtins.input", return_value="y")
    def test_tty_yes_approves(self, _input, _deps, _tty):
        assert _prompt_formatter_consent("a-model") == "approved"

    @patch("sys.stdin.isatty", return_value=True)
    @patch("cat_stack._formatter._check_dependencies_installed", return_value=True)
    @patch("builtins.input", return_value="")
    def test_tty_empty_defaults_to_yes(self, _input, _deps, _tty):
        """Default is Y (capital Y); pressing Enter should approve."""
        assert _prompt_formatter_consent("a-model") == "approved"

    @patch("sys.stdin.isatty", return_value=True)
    @patch("cat_stack._formatter._check_dependencies_installed", return_value=True)
    @patch("builtins.input", return_value="n")
    def test_tty_no_declines(self, _input, _deps, _tty):
        assert _prompt_formatter_consent("a-model") == "declined"

    @patch("sys.stdin.isatty", return_value=True)
    @patch("cat_stack._formatter._check_dependencies_installed", return_value=True)
    @patch("builtins.input", side_effect=EOFError)
    def test_tty_eof_declines_gracefully(self, _input, _deps, _tty):
        """EOFError (Ctrl-D, piped non-TTY that still triggered isatty=True)
        must NOT raise; treat as decline."""
        assert _prompt_formatter_consent("a-model") == "declined"

    @patch("sys.stdin.isatty", return_value=True)
    @patch("cat_stack._formatter._check_dependencies_installed", return_value=False)
    @patch("cat_stack._formatter._install_dependencies", return_value=True)
    @patch("builtins.input", return_value="y")
    def test_tty_yes_installs_when_deps_missing(self, _input, mock_install, _deps_check, _tty):
        """Path B: deps not installed; user says yes; we install then return
        approved."""
        result = _prompt_formatter_consent("a-model")
        assert result == "approved"
        assert mock_install.called

    @patch("sys.stdin.isatty", return_value=True)
    @patch("cat_stack._formatter._check_dependencies_installed", return_value=False)
    @patch("cat_stack._formatter._install_dependencies", return_value=False)
    @patch("builtins.input", return_value="y")
    def test_tty_yes_but_install_fails_declines(self, _input, _install, _deps_check, _tty):
        """If consent is given but the install fails, we degrade to declined
        (caller continues without the formatter)."""
        assert _prompt_formatter_consent("a-model") == "declined"


# ── classify() builds the right formatter_state per json_formatter value ──

class TestClassifyFormatterStateConstruction:
    """Inspect the source of classify() to verify the three states result
    in the right `_formatter_state` shape. We can't easily run classify()
    end-to-end without setting up models, so static-check the structure."""

    @classmethod
    def setup_class(cls):
        import cat_stack
        import os
        cls.src = open(
            os.path.join(os.path.dirname(cat_stack.__file__), "classify.py")
        ).read()

    def test_explicit_true_uses_approved_consent(self):
        assert '"_consent": "approved"' in self.src, (
            "json_formatter=True path must set _consent='approved' so the "
            "fallback skips the prompt."
        )

    def test_none_uses_auto_consent_with_lazy_loader(self):
        assert '"_consent": "auto"' in self.src
        # In auto mode, _loader is None until consent is granted
        assert '"_loader": None' in self.src

    def test_handles_three_states_distinctly(self):
        """Source should branch on the three states (True / False / None)
        rather than just truthy."""
        assert "json_formatter is True" in self.src
        assert "json_formatter is None" in self.src


# ── _try_formatter_fallback fires the consent prompt correctly ──────────

class TestFallbackConsentFlow:
    """Static-check the closure in classify_ensemble. End-to-end test would
    require classify_ensemble's whole orchestration which is too heavy."""

    @classmethod
    def setup_class(cls):
        from cat_stack.text_functions_ensemble import classify_ensemble
        cls.src = inspect.getsource(classify_ensemble)

    def test_imports_prompt_function(self):
        assert "_prompt_formatter_consent" in self.src

    def test_caches_consent_decision(self):
        """The closure must write the consent decision back to
        formatter_state so subsequent rows don't re-prompt."""
        assert 'formatter_state["_consent"] = consent' in self.src

    def test_skips_when_declined_without_acquiring_lock(self):
        """The 'declined' fast-skip should appear before the `with lock:`
        block — declining a 50-row run shouldn't acquire the lock 49 times."""
        skip_idx = self.src.find('"_consent") == "declined"')
        lock_idx = self.src.find("with lock:")
        assert skip_idx > 0 and lock_idx > 0
        assert skip_idx < lock_idx, (
            "the declined-skip should come before the with-lock block"
        )

    def test_returns_early_if_not_approved(self):
        """After the prompt, if consent is anything other than 'approved'
        (declined OR install-failed OR EOFError), return early without
        trying to load the model."""
        # The check `formatter_state.get("_consent") != "approved"` followed
        # by `return json_result` should appear inside the locked block.
        assert '"_consent") != "approved"' in self.src

    def test_loader_set_only_after_approval(self):
        """The lazy loader (cat_stack._formatter.load_formatter) should be
        assigned to formatter_state["_loader"] only inside the 'approved'
        branch — not at construction time in auto mode."""
        # We expect something like: if consent == "approved": ... load_formatter
        assert "from ._formatter import load_formatter" in self.src
        assert 'formatter_state["_loader"] = load_formatter' in self.src
