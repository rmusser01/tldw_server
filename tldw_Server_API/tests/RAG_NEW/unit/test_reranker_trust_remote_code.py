import sys
import types

import pytest

from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
    RerankingConfig,
    TransformersCrossEncoderReranker,
)


class _DummyCrossEncoder:
    def __init__(
        self,
        model_id: str,
        device=None,
        trust_remote_code: bool = False,
        revision=None,
        local_files_only: bool = False,
    ):  # noqa: D401
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.local_files_only = local_files_only

    def predict(self, pairs, batch_size=32):  # pragma: no cover - unused in this test
        return [0.0 for _ in pairs]


@pytest.mark.unit
def test_mxbai_reranker_enables_trust_remote_code(monkeypatch):
    # Stub sentence_transformers to avoid heavyweight downloads.
    stub_module = types.SimpleNamespace(CrossEncoder=_DummyCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub_module)

    # Keep config lookup deterministic.
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.load_and_log_configs",
        lambda: {"TRUSTED_HF_REMOTE_CODE_MODELS": []},
    )

    cfg = RerankingConfig(
        model_name="mixedbread-ai/mxbai-rerank-large-v2",
        transformers_trust_remote_code=False,
    )
    reranker = TransformersCrossEncoderReranker(cfg)

    assert reranker._trust_remote_code is True
    assert getattr(reranker._ce, "trust_remote_code", False) is True
    assert getattr(reranker._ce, "local_files_only", True) is False
