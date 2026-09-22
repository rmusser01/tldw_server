"""Regression guard for TASK-13305.

`load_parakeet_mlx_model` cached the loaded model in a single module global that was
not keyed by model identity:

    if _mlx_model_cache and not force_reload:
        logger.debug("Using cached Parakeet MLX model")
        return _mlx_model_cache

So the first model loaded in a process won for every later request. Asking for
`parakeet-tdt-1.1b` after `0.6b` had been cached returned the **0.6b** model while
the log said "Using cached model" and the requested name was reported upstream —
transcription quality silently differed from what the caller selected and from what
the response claimed, with no error at any layer.

The cache stays a single slot deliberately: these models are 1-3 GB, and holding
several resident would trade this bug for a memory regression (the same review
flagged exactly that for the keyed-but-unlocked Nemo and ONNX caches). A request for
a different model now evicts rather than silently mis-serves.
"""

import sys
import types

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
    Audio_Transcription_Parakeet_MLX as mlx_mod,
)


class _FakeModel:
    """Carries the id it was built from so tests can tell models apart."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


@pytest.fixture
def mlx_stubs(monkeypatch: pytest.MonkeyPatch):
    """Make the loader runnable off-Apple-Silicon with a fake parakeet_mlx."""
    loaded: list[str] = []

    def _from_pretrained(model_id: str, **_kwargs):
        loaded.append(model_id)
        return _FakeModel(model_id)

    fake = types.ModuleType("parakeet_mlx")
    fake.from_pretrained = _from_pretrained
    monkeypatch.setitem(sys.modules, "parakeet_mlx", fake)

    monkeypatch.setattr(mlx_mod, "IS_MACOS", True, raising=False)
    monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)
    monkeypatch.setattr(mlx_mod, "check_parakeet_mlx_installed", lambda: True)

    # Start from a cold cache and restore module state afterwards.
    monkeypatch.setattr(mlx_mod, "_mlx_model_cache", None, raising=False)
    monkeypatch.setattr(mlx_mod, "_mlx_model_cache_key", None, raising=False)

    yield loaded

    mlx_mod._mlx_model_cache = None
    if hasattr(mlx_mod, "_mlx_model_cache_key"):
        mlx_mod._mlx_model_cache_key = None


def test_requesting_a_second_model_does_not_return_the_first(mlx_stubs) -> None:
    first = mlx_mod.load_parakeet_mlx_model(model_path="mlx-community/parakeet-tdt-0.6b-v3")
    assert first is not None and first.model_id.endswith("0.6b-v3")

    second = mlx_mod.load_parakeet_mlx_model(model_path="mlx-community/parakeet-tdt-1.1b")

    assert second is not None, "the second load returned nothing"
    assert second.model_id.endswith("1.1b"), (
        f"requested parakeet-tdt-1.1b but received {second.model_id!r} -- the model "
        "cache is not keyed by model identity, so the first model loaded in the "
        "process is returned for every later request"
    )
    assert mlx_stubs == [
        "mlx-community/parakeet-tdt-0.6b-v3",
        "mlx-community/parakeet-tdt-1.1b",
    ], "the second model was never actually loaded"


def test_same_model_is_served_from_cache(mlx_stubs) -> None:
    """Control: caching must still work for a repeat request."""
    first = mlx_mod.load_parakeet_mlx_model(model_path="mlx-community/parakeet-tdt-0.6b-v3")
    again = mlx_mod.load_parakeet_mlx_model(model_path="mlx-community/parakeet-tdt-0.6b-v3")

    assert again is first, "a repeat request for the same model should hit the cache"
    assert mlx_stubs == ["mlx-community/parakeet-tdt-0.6b-v3"], (
        "the same model was loaded twice; caching regressed"
    )


def test_force_reload_still_reloads(mlx_stubs) -> None:
    """Control: force_reload must bypass the cache regardless of keying."""
    mlx_mod.load_parakeet_mlx_model(model_path="m")
    mlx_mod.load_parakeet_mlx_model(model_path="m", force_reload=True)

    assert mlx_stubs == ["m", "m"]


def test_externally_seeded_cache_is_still_honoured(monkeypatch, mlx_stubs) -> None:
    """Back-compat: several existing tests set _mlx_model_cache directly.

    tests/Audio/test_stt_execution_plan_local.py:1116 assigns the global and asserts
    identity, so an entry seeded without a key must still be served rather than
    silently reloaded.
    """
    sentinel = _FakeModel("externally-seeded")
    monkeypatch.setattr(mlx_mod, "_mlx_model_cache", sentinel, raising=False)
    monkeypatch.setattr(mlx_mod, "_mlx_model_cache_key", None, raising=False)

    got = mlx_mod.load_parakeet_mlx_model(model_path="anything")

    assert got is sentinel
    assert mlx_stubs == [], "an externally-seeded cache entry was ignored and reloaded"
