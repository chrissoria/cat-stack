"""v1.6.8 reasoning-control payload behavior.

Unit-level: assert what _build_payload puts on the wire at thinking_budget=0
for the providers fixed in v1.6.8. No network calls.
"""
import catstack._providers as _prov
from catstack._providers import UnifiedLLMClient


def _client(model, provider):
    return UnifiedLLMClient(model=model, api_key="test-key", provider=provider)


class TestGoogleExplicitZeroBudget:
    def test_budget_zero_sends_explicit_thinking_config(self):
        c = _client("gemini-3.5-flash", "google")
        p = c._build_payload([{"role": "user", "content": "x"}],
                             None, 0.0, thinking_budget=0)
        # v1.6.7 sent nothing here; v1.6.8 sends an explicit 0.
        assert p["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0

    def test_positive_budget_unchanged(self):
        c = _client("gemini-3.5-flash", "google")
        p = c._build_payload([{"role": "user", "content": "x"}],
                             None, 0.0, thinking_budget=512)
        assert p["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 512

    def test_cached_floor_used_after_400(self):
        c = _client("gemini-3.1-pro-preview", "google")
        c._google_thinking_floor = 128  # simulate a prior 400-fallback
        p = c._build_payload([{"role": "user", "content": "x"}],
                             None, 0.0, thinking_budget=0)
        assert p["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 128


class TestXaiReasoningEffort:
    def test_hybrid_gets_low_at_budget_zero(self):
        c = _client("grok-4.3", "xai")
        p = c._build_payload([{"role": "user", "content": "x"}],
                             None, 0.0, thinking_budget=0)
        assert p.get("reasoning_effort") == "low"

    def test_non_reasoning_variant_gets_no_field(self):
        # Sending reasoning_effort to a *-non-reasoning variant paradoxically
        # turns reasoning ON (verified 2026-06-13), so the field is withheld.
        c = _client("grok-4-1-fast-non-reasoning", "xai")
        p = c._build_payload([{"role": "user", "content": "x"}],
                             None, 0.0, thinking_budget=0)
        assert "reasoning_effort" not in p

    def test_cached_rejection_skips_field(self):
        c = _client("grok-some-variant", "xai")
        c._xai_no_reasoning_effort = True  # simulate a prior 400-fallback
        p = c._build_payload([{"role": "user", "content": "x"}],
                             None, 0.0, thinking_budget=0)
        assert "reasoning_effort" not in p


class TestHfUncontrolledWarning:
    def test_gpt_oss_warns_once_per_process(self, capsys):
        # Warning state is module-level (once per process), and a fresh
        # client is built per row in real runs — so two clients still warn
        # only once. Reset the module set for test isolation.
        _prov._WARNED_UNCONTROLLED_REASONING.discard("openai/gpt-oss-120b")
        for _ in range(2):
            c = _client("openai/gpt-oss-120b", "huggingface")
            c._build_payload([{"role": "user", "content": "x"}],
                             None, 0.0, thinking_budget=0)
        out = capsys.readouterr().out
        assert out.count("no effective reasoning control delivered") == 1

    def test_qwen_gets_enable_thinking_false_not_warning(self, capsys):
        c = _client("Qwen/Qwen3.5-397B-A17B", "huggingface")
        p = c._build_payload([{"role": "user", "content": "x"}],
                             None, 0.0, thinking_budget=0)
        out = capsys.readouterr().out
        assert p["chat_template_kwargs"] == {"enable_thinking": False}
        assert "no effective reasoning control" not in out
