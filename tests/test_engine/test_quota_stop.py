from src.engine.llm_adapter import LLMAdapter, QuotaExhaustedError, _is_quota_exhausted, _is_transient_api_error
from src.engine.causal.llm_trace import LLMTrace, TracingAdapter, prompt_key


def test_quota_errors_are_not_retried():
    assert _is_quota_exhausted("Error code: 402 - Insufficient Balance")
    assert _is_quota_exhausted("余额不足，请充值")
    assert not _is_transient_api_error("Insufficient Balance")
    assert _is_transient_api_error("429 rate_limit")
    err = QuotaExhaustedError("stop")
    assert isinstance(err, Exception)


def test_tracing_adapter_does_not_cache_quota_errors():
    class Boom(LLMAdapter):
        def complete_json(self, system: str, user: str) -> dict:
            raise QuotaExhaustedError("Insufficient Balance")

    trace = LLMTrace()
    adapter = TracingAdapter(Boom(), trace)
    try:
        adapter.complete_json("sys", '{"task":"x"}')
        raise AssertionError("expected QuotaExhaustedError")
    except QuotaExhaustedError:
        pass
    assert prompt_key("sys", '{"task":"x"}') not in trace.errors
