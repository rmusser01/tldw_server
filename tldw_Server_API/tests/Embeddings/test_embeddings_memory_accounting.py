import pytest

from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create as EC


class _FakeTorch:
    class cuda:  # noqa: N801 - mimic torch.cuda namespace
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    @staticmethod
    def device(name):
        return name


class _FakeGauge:
    def labels(self, **_kwargs):
        class _Label:
            def inc(self):
                return None

            def dec(self):
                return None

            def set(self, _value):
                return None

        return _Label()


@pytest.mark.unit
def test_hf_unload_clears_memory_usage(monkeypatch, tmp_path):
    monkeypatch.setattr(EC, "_import_torch", lambda: _FakeTorch)
    monkeypatch.setattr(EC, "ACTIVE_EMBEDDERS", _FakeGauge())
    monkeypatch.setattr(EC, "log_counter", lambda *args, **kwargs: None)

    monkeypatch.setattr(EC, "model_memory_usage", {}, raising=False)
    monkeypatch.setattr(EC, "model_last_used", {}, raising=False)
    monkeypatch.setattr(EC, "model_in_use_counts", {}, raising=False)

    cfg = EC.HFModelCfg(provider="huggingface", model_name_or_path="dummy")
    embedder = EC.HuggingFaceEmbedder("huggingface:dummy", cfg, str(tmp_path))

    embedder.model = object()
    embedder.tokenizer = object()
    EC.model_memory_usage["huggingface:dummy"] = 1.25

    embedder.unload_model()

    assert EC.model_memory_usage["huggingface:dummy"] == 0.0


@pytest.mark.unit
def test_hf_embedder_del_tolerates_partial_initialization():
    embedder = object.__new__(EC.HuggingFaceEmbedder)
    embedder.model_identifier = "huggingface:partial"

    embedder.__del__()
