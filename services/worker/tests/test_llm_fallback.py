"""Regression: transient endpoint failures must not trigger sticky deterministic
fallback. A single late timeout previously poisoned every remaining cell in a
run and marked the whole batch non-scientific (Paper 1 slices 0/2, 2026-05-28)."""

import subprocess

import pytest
import requests

from app.llm import (
    DeterministicFakeClient,
    LiveOrFallbackClient,
    ModelDescriptor,
    _is_transient_llm_error,
)


class _RaisingClient:
    scientific = True
    degradation_reason = None

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.calls = 0

    def generate(self, prompt: str, *, seed: int) -> str:
        self.calls += 1
        raise self._exc

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(name="raising", digest="x")


@pytest.mark.parametrize(
    "exc",
    [
        requests.exceptions.Timeout("read timed out"),
        requests.exceptions.ConnectionError("connection aborted"),
        subprocess.TimeoutExpired(cmd="codex", timeout=600),
        TimeoutError("timed out"),
        RuntimeError("OpenAICompat generate failed after 4 retries: retryable 503"),
    ],
)
def test_transient_error_reraises_and_stays_scientific(exc):
    client = LiveOrFallbackClient(_RaisingClient(exc), DeterministicFakeClient())
    with pytest.raises(Exception):
        client.generate("p", seed=1)
    # Must NOT have flipped to sticky fallback — next cell may succeed live.
    assert client.scientific is True
    assert client.degradation_reason is None
    assert _is_transient_llm_error(exc) is True


def test_persistent_error_degrades_sticky_to_fake():
    persistent = RuntimeError("404 model 'foo' not found")
    live = _RaisingClient(persistent)
    client = LiveOrFallbackClient(live, DeterministicFakeClient())
    # First call degrades to fake instead of crashing.
    out = client.generate("p", seed=1)
    assert isinstance(out, str) and out
    assert client.scientific is False
    assert client.degradation_reason is not None
    # Sticky: subsequent call goes straight to fake, live not retried.
    calls_before = live.calls
    client.generate("p2", seed=2)
    assert live.calls == calls_before
    assert _is_transient_llm_error(persistent) is False
