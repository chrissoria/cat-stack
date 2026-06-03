"""
Tests for H-FORMATTER: thread-safety of the JSON formatter fallback path.

Two races existed in `_try_formatter_fallback` (a closure inside
`text_functions_ensemble.classify_ensemble`):

  1. Concurrent lazy-load. Multiple ThreadPoolExecutor workers could all
     see `formatter_state["_loaded"] == False`, all invoke the ~10s,
     ~1 GB `_loader()` call, all race-write to the shared dict.

  2. Concurrent inference. HuggingFace transformer models maintain
     internal KV-cache state and are NOT thread-safe for concurrent
     `model.generate()` calls — outputs can corrupt silently.

Fix: a single `threading.Lock` stored in `formatter_state["_lock"]`
wraps the lazy-load AND the `run_formatter()` call. Pre-initialized at
`classify.py:613` where the dict is constructed; defensively
`setdefault`-ed inside the helper for robustness if a future caller
forgets.

The helper itself is a closure (not directly importable), so this test
file uses:
  - Static source checks against the live code to assert the lock
    pattern is in place.
  - A behavioral simulation that mirrors the exact pre-fix vs. post-fix
    locking shape, run under a real ThreadPoolExecutor, to verify the
    `setdefault + with lock` pattern actually serializes work.
"""

import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait

import pytest

import cat_stack.classify
from cat_stack.text_functions_ensemble import classify_ensemble


# ── Static checks: the live code has the right shape ───────────────────

class TestStaticPatterns:
    def test_classify_py_initializes_lock(self):
        """classify.py constructs _formatter_state with a `_lock` key."""
        src = inspect.getsource(cat_stack.classify)
        # The dict literal should include `"_lock": threading.Lock()`
        assert '"_lock": threading.Lock()' in src, (
            "classify.py:_formatter_state must pre-initialize a threading.Lock "
            "so we don't race on lock creation itself."
        )

    def test_classify_py_imports_threading(self):
        import os
        classify_path = os.path.join(os.path.dirname(cat_stack.__file__), "classify.py")
        with open(classify_path) as f:
            src = f.read()
        assert "import threading" in src

    def test_try_formatter_fallback_acquires_lock(self):
        """The closure source must wrap the load + run_formatter calls in
        `with lock:` (acquired via setdefault for back-compat)."""
        src = inspect.getsource(classify_ensemble)
        assert 'setdefault("_lock"' in src, (
            "_try_formatter_fallback must setdefault the lock so any "
            "caller-provided formatter_state without a pre-init lock still works."
        )
        assert "with lock:" in src, (
            "The lazy-load + run_formatter critical section must be guarded by "
            "`with lock:`."
        )

    def test_run_formatter_call_is_inside_lock(self):
        """run_formatter must be inside the locked block, not after it —
        otherwise Race 2 (concurrent inference) isn't fixed."""
        src = inspect.getsource(classify_ensemble)
        # Crude but effective: find the locked block and confirm
        # run_formatter is mentioned inside it.
        lock_idx = src.find("with lock:")
        assert lock_idx >= 0
        # The block ends at the next non-indented line, but we can just
        # check that `run_formatter(` appears between the lock and the
        # next obvious lock-exit marker (`extract_json` runs after).
        post_lock_segment = src[lock_idx:lock_idx + 2000]
        run_idx = post_lock_segment.find("run_formatter(")
        extract_idx = post_lock_segment.find("extract_json(fixed_output)")
        assert run_idx > 0, "run_formatter not found near the lock"
        assert extract_idx > run_idx, (
            "extract_json should appear after run_formatter; if reversed, "
            "the lock structure has been disturbed."
        )

    def test_lazy_load_assignments_inside_lock(self):
        """All `formatter_state[...] = ...` writes that set up the loaded
        model must be inside the locked block."""
        src = inspect.getsource(classify_ensemble)
        lock_idx = src.find("with lock:")
        loaded_assign_idx = src.find('formatter_state["_loaded"] = True')
        assert lock_idx >= 0 and loaded_assign_idx > lock_idx, (
            "The `_loaded = True` assignment must come after `with lock:` — "
            "otherwise concurrent loaders can both pass the loaded check."
        )


# ── Behavioral simulation: verify the locking pattern actually works ───

class TestLockingPatternBehavior:
    """Mirror the exact post-fix shape and run it under a real
    ThreadPoolExecutor. Confirms that the `setdefault + with lock` idiom
    does what we expect under concurrent load."""

    @staticmethod
    def _simulate(formatter_state, load_delay=0.05, run_delay=0.02):
        """A faithful stand-in for _try_formatter_fallback's critical
        section. If the locking is correct, _loader runs exactly once
        and run_formatter calls are serialized."""
        lock = formatter_state.setdefault("_lock", threading.Lock())
        with lock:
            if not formatter_state.get("_loaded"):
                time.sleep(load_delay)              # simulate slow load
                formatter_state["_loader_calls"] = formatter_state.get("_loader_calls", 0) + 1
                formatter_state["_loaded"] = True
            # Track concurrent entry into the inference section
            formatter_state["_in_run_count"] = formatter_state.get("_in_run_count", 0) + 1
            formatter_state["_max_concurrent"] = max(
                formatter_state["_max_concurrent"],
                formatter_state["_in_run_count"],
            )
            time.sleep(run_delay)                   # simulate inference
            formatter_state["_in_run_count"] -= 1
            formatter_state["_total_runs"] = formatter_state.get("_total_runs", 0) + 1
        return "ok"

    def test_concurrent_load_invokes_loader_exactly_once(self):
        """Race 1: 20 concurrent workers all hit the fallback; the
        ~10s loader runs exactly once thanks to the lock."""
        state = {"_loaded": False, "_max_concurrent": 0}
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(self._simulate, state) for _ in range(20)]
            wait(futures)
        assert state["_loader_calls"] == 1, (
            f"loader should run exactly once; ran {state['_loader_calls']} times"
        )
        assert state["_total_runs"] == 20, "all workers should complete"

    def test_concurrent_inference_is_serialized(self):
        """Race 2: even when many workers reach the run_formatter call,
        only one is in the critical section at a time."""
        state = {"_loaded": False, "_max_concurrent": 0}
        with ThreadPoolExecutor(max_workers=20) as ex:
            futures = [ex.submit(self._simulate, state) for _ in range(20)]
            wait(futures)
        assert state["_max_concurrent"] == 1, (
            f"run_formatter calls must be serialized; observed max "
            f"concurrent = {state['_max_concurrent']}"
        )

    def test_setdefault_defends_against_missing_lock(self):
        """If a caller constructed formatter_state without `_lock`, the
        helper must still serialize correctly via setdefault."""
        state = {"_loaded": False, "_max_concurrent": 0}  # no `_lock`!
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(self._simulate, state) for _ in range(10)]
            wait(futures)
        assert state["_loader_calls"] == 1
        assert state["_max_concurrent"] == 1
        # And a `_lock` should now exist on the state
        assert isinstance(state["_lock"], type(threading.Lock()))

    def test_pre_initialized_lock_is_used(self):
        """If the caller pre-initialized a lock, it should be used, not
        replaced by a fresh one."""
        original_lock = threading.Lock()
        state = {"_loaded": False, "_max_concurrent": 0, "_lock": original_lock}
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(self._simulate, state) for _ in range(10)]
            wait(futures)
        assert state["_lock"] is original_lock, (
            "setdefault should preserve the caller-provided lock, not replace it"
        )
