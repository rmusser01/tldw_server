"""Token bounds must preserve the complete canonical source, never decode a substitute."""

from threading import RLock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Embeddings import input_limits
from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create as embedding


class CharacterTokenizer:
    model_max_length = 16

    def encode(self, text, add_special_tokens=True, truncation=False):
        assert truncation is False
        return list(range(len(text) + (2 if add_special_tokens else 0)))


def chunk(text, start=7):
    return {"text": text, "start": start, "end": start + len(text), "index": 0, "metadata": {"page_number": 3}}


@pytest.mark.parametrize("text", ["https://example.org/" + "abc%20" * 30, "x" * 91, "😀界é\n" * 25])
def test_refinement_preserves_all_original_characters_and_exact_offsets(text):
    result = input_limits.split_chunks_to_token_budget([chunk(text)], CharacterTokenizer(), 16)
    assert "".join(row["text"] for row in result) == text
    assert [row["index"] for row in result] == list(range(len(result)))
    for row in result:
        assert row["text"] == text[row["start"] - 7 : row["end"] - 7]
        assert len(CharacterTokenizer().encode(row["text"])) <= 16
        assert row["metadata"]["page_number"] == 3
    assert all(left["end"] == right["start"] for left, right in zip(result, result[1:]))


@given(st.text(min_size=1, max_size=150), st.integers(min_value=3, max_value=40))
def test_partition_has_no_missing_characters_and_each_part_fits(text, limit):
    result = input_limits.split_chunks_to_token_budget([chunk(text)], CharacterTokenizer(), limit)
    assert "".join(row["text"] for row in result) == text
    assert all(len(CharacterTokenizer().encode(row["text"])) <= limit for row in result)


def test_prompt_overhead_counts_and_impossible_single_character_fails():
    result = input_limits.split_chunks_to_token_budget(
        [chunk("abcdefghijk")], CharacterTokenizer(), 12, format_text=lambda text: "prefix:" + text
    )
    assert all(len(CharacterTokenizer().encode("prefix:" + row["text"])) <= 12 for row in result)
    with pytest.raises(ValueError, match="token budget"):
        input_limits.split_chunks_to_token_budget([chunk("x")], CharacterTokenizer(), 2)


def test_invalid_canonical_offsets_fail_before_embedding():
    with pytest.raises(ValueError, match="offset"):
        input_limits.split_chunks_to_token_budget([{**chunk("abc"), "end": 8}], CharacterTokenizer(), 16)


def test_raw_hf_embedding_rejects_over_limit_before_inference(monkeypatch):
    import torch

    class Tokenizer(CharacterTokenizer):
        def __call__(self, texts, **options):
            pytest.fail("Oversized input reached padded tensor allocation")

    instance = embedding.HuggingFaceEmbedder.__new__(embedding.HuggingFaceEmbedder)
    instance._lock = RLock()
    instance.config = embedding.HFModelCfg(model_name_or_path="unit-test", max_length=16)
    instance.model_identifier = "unit-test"
    instance.tokenizer = Tokenizer()
    instance.device = torch.device("cpu")
    instance.load_model = lambda: None
    instance.model = lambda **kwargs: pytest.fail("Oversized input reached inference")
    from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingInputError

    with pytest.raises(EmbeddingInputError, match="token"):
        instance.create_embeddings(["x" * 30])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["direct", "jobs"])
@pytest.mark.parametrize("fallback", [False, True])
async def test_media_callers_store_refined_exact_chunks_and_rebudget_fallback(monkeypatch, tmp_path, path, fallback):
    import json

    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as api
    from tldw_Server_API.app.core.Embeddings import media_embeddings as media
    from tldw_Server_API.app.core.Embeddings.services import jobs_worker as worker

    source = "https://example.org/" + "abc%20" * 30 + "😀界"
    captured = {}
    monkeypatch.setattr(embedding, "_load_hf_input_tokenizer", lambda *args: CharacterTokenizer())

    def embed(texts, user_app_config, model_id_override=None, **kwargs):
        if fallback and model_id_override == "huggingface:unit/model":
            raise RuntimeError("Primary unavailable")
        assert all(len(CharacterTokenizer().encode(text)) <= 16 for text in texts)
        captured["texts"] = texts
        return [[0.1, 0.2] for _ in texts]

    await api.embedding_cache.clear()
    monkeypatch.setattr(embedding, "create_embeddings_batch", embed)
    if path == "direct":

        class Manager:
            def __init__(self, **kwargs):
                pass

            def store_in_chroma(self, **kwargs):
                captured["stored"] = kwargs

        monkeypatch.setattr(media, "ChromaDBManager", Manager)
        result = await media.generate_embeddings_for_media(
            1, {"content": source, "media_item": {}}, "unit/model", "huggingface", 1000, 200
        )
        assert result["status"] == "success"
        saved = captured["stored"]
        assert "".join(saved["texts"]) == source
        for text, metadata in zip(saved["texts"], saved["metadatas"]):
            assert text == source[metadata["chunk_start"] : metadata["chunk_end"]]
    else:
        monkeypatch.setattr(worker, "_artifact_dir", lambda *args: tmp_path)
        monkeypatch.setattr(worker, "_update_root_result", lambda *args, **kwargs: None)
        monkeypatch.setattr(worker, "_update_root_progress", lambda *args, **kwargs: None)
        chunks_path = worker._chunk_artifact_path(tmp_path)
        worker._write_json(chunks_path, [chunk(source, 0)])
        result = await worker._handle_embedding_job(
            {"id": 1, "uuid": "test"},
            {},
            media_id=1,
            user_id="1",
            embedding_model="unit/model",
            embedding_provider="huggingface",
            chunk_size=1000,
            chunk_overlap=200,
            root_uuid=None,
        )
        refined = json.loads(chunks_path.read_text())
        assert "".join(row["text"] for row in refined) == source
        assert result["total_chunks"] == len(refined) == len(captured["texts"])
        assert all(row["text"] == source[row["start"] : row["end"]] for row in refined)


@pytest.mark.asyncio
async def test_legacy_hf_embedding_checkpoint_is_recomputed_not_reused(monkeypatch, tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as api
    from tldw_Server_API.app.core.Embeddings.services import jobs_worker as worker

    monkeypatch.setattr(worker, "_artifact_dir", lambda *args: tmp_path)
    monkeypatch.setattr(worker, "_update_root_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(embedding, "_load_hf_input_tokenizer", lambda *args: CharacterTokenizer())
    worker._write_json(worker._chunk_artifact_path(tmp_path), [chunk("x" * 40, 0)])
    worker._write_json(
        worker._embedding_artifact_path(tmp_path),
        {"embeddings": [[0.1]], "embedding_model": "unit/model", "embedding_provider": "huggingface"},
    )
    calls = []

    def embed(texts, user_app_config, model_id_override=None, **kwargs):
        calls.append(texts)
        return [[0.2] for text in texts]

    await api.embedding_cache.clear()
    monkeypatch.setattr(embedding, "create_embeddings_batch", embed)
    result = await worker._handle_embedding_job(
        {"id": 1},
        {},
        media_id=1,
        user_id="1",
        embedding_model="unit/model",
        embedding_provider="huggingface",
        chunk_size=1000,
        chunk_overlap=200,
        root_uuid=None,
    )
    assert len(calls) == 1 and len(calls[0]) == result["embedding_count"] > 1


@pytest.mark.asyncio
async def test_storage_refuses_unverified_legacy_hf_embedding_checkpoint(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Embeddings.services import jobs_worker as worker

    monkeypatch.setattr(worker, "_artifact_dir", lambda *args: tmp_path)
    worker._write_json(worker._chunk_artifact_path(tmp_path), [chunk("x" * 40, 0)])
    worker._write_json(
        worker._embedding_artifact_path(tmp_path), {"embeddings": [[0.1]], "embedding_provider": "huggingface"}
    )
    with pytest.raises(worker.EmbeddingsJobError, match="coverage"):
        await worker._handle_storage_job(
            {"id": 1},
            {},
            media_id=1,
            user_id="1",
            embedding_model="unit/model",
            embedding_provider="huggingface",
            root_uuid=None,
        )


@pytest.mark.asyncio
async def test_raw_api_length_error_is_actionable_before_breaker_or_provider(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as api
    from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingInputError
    from tldw_Server_API.app.core.Infrastructure.circuit_breaker import registry

    monkeypatch.setattr(embedding, "_load_hf_input_tokenizer", lambda *args: CharacterTokenizer())
    monkeypatch.setattr(
        registry, "get", lambda *args: pytest.fail("Invalid input reached circuit breaker")
    )
    with pytest.raises(EmbeddingInputError) as raised:
        await api.create_embeddings_with_circuit_breaker(
            ["x" * 40], "huggingface", "unit/model", {"model_name_or_path": "unit/model"}
        )
    response = api._embedding_domain_error_to_http(raised.value)
    assert response.status_code == 400 and raised.value.code == "input_too_long" and not raised.value.retryable
