import pytest

from tldw_Server_API.tests.ChromaDB import conftest as chromadb_conftest


pytestmark = pytest.mark.unit


class _FakeConfig:
    def __init__(self, run_models: bool = False) -> None:
        self.run_models = run_models

    def getoption(self, name: str) -> bool:
        assert name == "--run-model-tests"
        return self.run_models


def test_hf_fixture_uses_deterministic_embeddings_by_default_without_connectivity_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUN_MODEL_TESTS", raising=False)

    def fail_if_called() -> bool:
        raise AssertionError("Hugging Face connectivity should be opt-in")

    monkeypatch.setattr(chromadb_conftest, "_hf_connectivity_ok", fail_if_called)

    embed, used_real_model, dim = chromadb_conftest.hf_or_deterministic_embeddings.__wrapped__(
        _FakeConfig(run_models=False)
    )

    vectors = embed(["probe"])
    assert used_real_model is False
    assert dim == 384
    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_hf_fixture_checks_connectivity_when_model_tests_are_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RUN_MODEL_TESTS", raising=False)
    calls = 0

    def offline() -> bool:
        nonlocal calls
        calls += 1
        return False

    monkeypatch.setattr(chromadb_conftest, "_hf_connectivity_ok", offline)

    _embed, used_real_model, dim = chromadb_conftest.hf_or_deterministic_embeddings.__wrapped__(
        _FakeConfig(run_models=True)
    )

    assert calls == 1
    assert used_real_model is False
    assert dim == 384
