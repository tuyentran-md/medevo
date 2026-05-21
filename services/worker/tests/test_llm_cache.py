from __future__ import annotations

import pytest

from app.llm import CachedLLMClient, ModelDescriptor, llm_cache_stats


class CountingClient:
    scientific = True
    degradation_reason = None

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, *, seed: int) -> str:
        self.calls += 1
        return f"response-{self.calls}-{seed}-{prompt}"

    def describe(self) -> ModelDescriptor:
        return ModelDescriptor(name="counting", digest="test")


def test_cached_llm_client_reuses_prompt_seed_response(tmp_path) -> None:
    live = CountingClient()
    cached = CachedLLMClient(live, namespace="test:model", cache_dir=tmp_path)

    first = cached.generate("prompt", seed=7)
    second = cached.generate("prompt", seed=7)

    assert first == second
    assert live.calls == 1
    assert llm_cache_stats(cached) == {
        "enabled": True,
        "cache_only": False,
        "hits": 1,
        "misses": 1,
        "writes": 1,
    }


def test_cached_llm_client_cache_only_refuses_miss(tmp_path) -> None:
    cached = CachedLLMClient(
        CountingClient(),
        namespace="test:model",
        cache_dir=tmp_path,
        cache_only=True,
    )

    with pytest.raises(RuntimeError, match="cache miss"):
        cached.generate("prompt", seed=8)
