# Embeddings Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the confirmed embeddings and media-embeddings defects, normalize the approved API semantics, and add focused regression coverage without widening scope.

**Architecture:** Keep the fixes local to the API/orchestration layer. Tighten token-array validation and backend-sensitive cache identity in `embeddings_v5_production_enhanced.py`, then split generation failures from storage failures and make batch enqueue outcomes truthful in `media_embeddings.py`.

**Tech Stack:** FastAPI, Pydantic v2 models, pytest, `unittest.mock`, ChromaDB manager wrapper, Jobs adapter

---

## File Map

- Modify: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
  - Reject token-array decode failures instead of converting them into empty strings.
  - Derive stable backend-sensitive cache identity before cache lookup.
  - Use the same cache key inputs for cache read and cache write.
- Modify: `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`
  - Split primary generation from storage so only generation failures can trigger fallback.
  - Expand `BatchMediaEmbeddingsResponse` to represent both full acceptance and partial acceptance.
  - Return truthful `202` partial responses while preserving true failure status codes when nothing is queued.
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py`
  - Cover decode failure at helper level and endpoint level.
  - Prove the endpoint does not call downstream embedding generation after decode failure.
- Create: `tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py`
  - Prove `local_api` requests with different `api_url` values do not alias the same endpoint cache entry.
- Create: `tldw_Server_API/tests/Embeddings/test_media_embeddings_failure_classification.py`
  - Prove storage failures do not trigger fallback and surface storage-classified errors.
  - Prove true generation failures can still fall back.
- Modify: `tldw_Server_API/tests/Embeddings/test_media_embeddings_submission_semantics.py`
  - Update direct-call assertions from `HTTPException(500)` on partial enqueue to structured partial success.
- Modify: `tldw_Server_API/tests/Embeddings/test_media_embedding_jobs.py`
  - Update HTTP-level assertions from `500` on partial enqueue to `202` with explicit partial body.

### Task 1: Reject Invalid Token Arrays

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.unit
def test_single_token_array_decode_failure_returns_400_and_skips_embedding_creation(client, monkeypatch):
    async def override_user():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=False)

    app.dependency_overrides[get_request_user] = override_user

    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod

    class _BadEncoder:
        def decode(self, _tokens):
            raise ValueError("boom")

    downstream = AsyncMock(side_effect=AssertionError("embedding creation should not be called"))
    monkeypatch.setattr(emb_mod, "get_tokenizer", lambda _model: _BadEncoder())
    monkeypatch.setattr(emb_mod, "create_embeddings_batch_async", downstream)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": [101, 102]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid token array input"
    downstream.assert_not_awaited()


@pytest.mark.unit
def test_tokens_to_texts_raises_on_decode_failure(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod

    class _BadEncoder:
        def decode(self, _tokens):
            raise ValueError("boom")

    monkeypatch.setattr(emb_mod, "get_tokenizer", lambda _model: _BadEncoder())

    with pytest.raises(ValueError, match="Invalid token array input"):
        emb_mod.tokens_to_texts([1, 2], "text-embedding-3-small")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py -k "decode_failure" -v
```

Expected: FAIL because the helper currently returns `[""]` and the endpoint currently proceeds to a `200` success path instead of failing fast.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
def tokens_to_texts(
    tokens_input: list[int] | list[list[int]],
    model_name: str,
) -> tuple[list[str], int, list[int]]:
    try:
        enc = get_tokenizer(model_name)
    except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        enc = tiktoken.get_encoding("cl100k_base")

    texts: list[str] = []
    total_tokens = 0
    token_counts: list[int] = []

    if tokens_input and isinstance(tokens_input, list) and isinstance(tokens_input[0], int):
        arr = tokens_input
        total_tokens += len(arr)
        token_counts.append(len(arr))
        try:
            texts.append(enc.decode(arr))
        except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Failed to decode token array for model '{}' (index 0, tokens={}): {}",
                model_name,
                len(arr),
                exc,
            )
            raise ValueError("Invalid token array input") from exc
        return texts, total_tokens, token_counts

    if tokens_input and isinstance(tokens_input, list):
        for idx, arr in enumerate(tokens_input):
            if not isinstance(arr, list) or not all(isinstance(x, int) for x in arr):
                raise ValueError("Invalid token array format")
            total_tokens += len(arr)
            token_counts.append(len(arr))
            try:
                texts.append(enc.decode(arr))
            except _EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    "Failed to decode token array for model '{}' (index {}, tokens={}): {}",
                    model_name,
                    idx,
                    len(arr),
                    exc,
                )
                raise ValueError("Invalid token array input") from exc
        return texts, total_tokens, token_counts

    raise ValueError("Invalid token array input")
```

```python
# tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py
from unittest.mock import AsyncMock, Mock
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py -k "decode_failure" -v
```

Expected: PASS with one endpoint-level `400` assertion and one helper-level `ValueError` assertion.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py
git commit -m "fix: reject invalid embeddings token arrays"
```

### Task 2: Make Endpoint Cache Identity Backend-Sensitive

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Create: `tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest


class RecordingCache:
    def __init__(self) -> None:
        self.data = {}
        self.get_keys = []
        self.set_keys = []

    async def get(self, key):
        self.get_keys.append(key)
        return self.data.get(key)

    async def set(self, key, value):
        self.set_keys.append(key)
        self.data[key] = value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_api_url_participates_in_cache_identity(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as mod

    cache = RecordingCache()
    provider_calls = []

    async def fake_create_embeddings_with_circuit_breaker(
        texts,
        provider,
        model_id,
        config,
        metadata=None,
        dimensions=None,
    ):
        _ = (texts, provider, model_id, metadata, dimensions)
        provider_calls.append(config["api_url"])
        marker = 1.0 if config["api_url"].endswith("/one") else 2.0
        return [[marker, marker]]

    monkeypatch.setattr(mod, "embedding_cache", cache)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_create_embeddings_with_circuit_breaker)

    first = await mod.create_embeddings_batch_async(
        ["hello"],
        provider="local_api",
        model_id="embed-model",
        api_url="https://embed.example/one",
    )
    second = await mod.create_embeddings_batch_async(
        ["hello"],
        provider="local_api",
        model_id="embed-model",
        api_url="https://embed.example/two",
    )

    assert first == [[1.0, 1.0]]
    assert second == [[2.0, 2.0]]
    assert provider_calls == ["https://embed.example/one", "https://embed.example/two"]
    assert cache.get_keys[0] != cache.get_keys[1]
    assert cache.set_keys[0] != cache.set_keys[1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py -v
```

Expected: FAIL because the current cache key ignores `api_url`, so the second call reuses the first cached vector and the provider is not called twice with distinct backends.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
def get_cache_key(
    text: str,
    provider: str,
    model: str,
    dimensions: int | None = None,
    backend_identity: str | None = None,
) -> str:
    key_parts = [text, provider, model]
    if dimensions is not None:
        key_parts.append(str(dimensions))
    if backend_identity:
        key_parts.append(str(backend_identity).rstrip("/"))
    return hashlib.sha256("|".join(key_parts).encode()).hexdigest()


def _cache_backend_identity(provider: str, config: dict[str, Any]) -> str | None:
    if provider == "local_api":
        api_url = config.get("api_url") or settings.get("LOCAL_API_URL")
        if api_url:
            return str(api_url).rstrip("/")
    return None
```

```python
# tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
async def create_embeddings_batch_async(
    texts: list[str],
    provider: str,
    model_id: str | None = None,
    dimensions: int | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[list[float]]:
    provider = (provider or "").strip().lower()
    try:
        provider_enum = EmbeddingProvider(provider)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown provider: {provider}") from None

    try:
        _validate_dimensions_request(provider, model_id or "", dimensions)
        config = build_provider_config(provider_enum, model_id, api_key, api_url, dimensions)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    backend_identity = _cache_backend_identity(provider, config)

    embeddings: list[list[float] | None] = []
    uncached_texts: list[str] = []
    uncached_indices: list[int] = []

    for i, text in enumerate(texts):
        cache_key = get_cache_key(text, provider, model_id or "default", dimensions, backend_identity)
        cached = await embedding_cache.get(cache_key)
        if cached:
            embeddings.append(cached)
        else:
            embeddings.append(None)
            uncached_texts.append(text)
            uncached_indices.append(i)

    # Generate embeddings for uncached texts via the provider
    all_new_embeddings = await _call_provider(config, uncached_texts)

    for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
        embedding = all_new_embeddings[i]
        embeddings[idx] = embedding
        cache_key = get_cache_key(text, provider, model_id or "default", dimensions, backend_identity)
        await embedding_cache.set(cache_key, embedding)

    return [embedding for embedding in embeddings if embedding is not None]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_production.py -k "cache" -v
```

Expected: PASS with the new backend-sensitive cache identity test and the existing cache-performance smoke tests still green.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py
git commit -m "fix: scope embeddings cache by backend identity"
```

### Task 3: Split Generation Failures From Storage Failures

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`
- Create: `tldw_Server_API/tests/Embeddings/test_media_embeddings_failure_classification.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from tldw_Server_API.app.api.v1.endpoints import (
    embeddings_v5_production_enhanced,
    media_embeddings,
)


def _media_content():
    return {
        "media_item": {"title": "Doc", "author": "Author", "metadata": {}},
        "content": {"content": "hello world"},
    }


@pytest.mark.asyncio
async def test_storage_failure_returns_storage_error_without_fallback(monkeypatch):
    calls = []

    async def fake_create_embeddings_batch_async(*, texts, provider, model_id, metadata):
        _ = (texts, metadata)
        calls.append((provider, model_id))
        return [[0.1, 0.2, 0.3]]

    class FailingChromaDBManager:
        def __init__(self, *, user_id, user_embedding_config):
            _ = (user_id, user_embedding_config)

        def store_in_chroma(self, **_kwargs):
            raise RuntimeError("disk full")

    monkeypatch.setattr(
        embeddings_v5_production_enhanced,
        "create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
    )
    monkeypatch.setattr(
        media_embeddings,
        "chunk_media_content",
        lambda *_args, **_kwargs: [{"text": "hello world", "index": 0, "start": 0, "end": 11}],
    )
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {"USER_DB_BASE_DIR": "/tmp/test"})
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", FailingChromaDBManager)

    result = await media_embeddings.generate_embeddings_for_media(
        media_id=42,
        media_content=_media_content(),
        embedding_model="text-embedding-3-small",
        embedding_provider="openai",
        chunk_size=1000,
        chunk_overlap=200,
        user_id="tenant-7",
    )

    assert result["status"] == "error"
    assert "store" in result["message"].lower()
    assert "storage" in result["error"].lower()
    assert calls == [("openai", "text-embedding-3-small")]


@pytest.mark.asyncio
async def test_generation_failure_can_still_fall_back(monkeypatch):
    calls = []

    async def fake_create_embeddings_batch_async(*, texts, provider, model_id, metadata):
        _ = (texts, metadata)
        calls.append((provider, model_id))
        if provider == "openai":
            raise RuntimeError("provider down")
        return [[0.1, 0.2, 0.3]]

    class RecordingChromaDBManager:
        def __init__(self, *, user_id, user_embedding_config):
            _ = (user_id, user_embedding_config)

        def store_in_chroma(self, **_kwargs):
            return None

    monkeypatch.setattr(
        embeddings_v5_production_enhanced,
        "create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
    )
    monkeypatch.setattr(
        media_embeddings,
        "chunk_media_content",
        lambda *_args, **_kwargs: [{"text": "hello world", "index": 0, "start": 0, "end": 11}],
    )
    monkeypatch.setattr(media_embeddings, "_user_embedding_config", lambda: {"USER_DB_BASE_DIR": "/tmp/test"})
    monkeypatch.setattr(media_embeddings, "ChromaDBManager", RecordingChromaDBManager)

    result = await media_embeddings.generate_embeddings_for_media(
        media_id=42,
        media_content=_media_content(),
        embedding_model="text-embedding-3-small",
        embedding_provider="openai",
        chunk_size=1000,
        chunk_overlap=200,
        user_id="tenant-7",
    )

    assert result["status"] == "success"
    assert calls == [
        ("openai", "text-embedding-3-small"),
        ("huggingface", media_embeddings.FALLBACK_EMBEDDING_MODEL),
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Embeddings/test_media_embeddings_failure_classification.py -v
```

Expected: FAIL because the current broad `except Exception:` path treats storage failures like generation failures and triggers fallback for the first test.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/api/v1/endpoints/media_embeddings.py
def _storage_failure_result(exc: Exception, chunk_count: int) -> dict[str, Any]:
    message = f"Failed to store embeddings: {exc}"
    return {
        "status": "error",
        "message": message,
        "error": f"storage_failure: {type(exc).__name__}: {exc}",
        "embedding_count": 0,
        "chunks_processed": chunk_count,
    }
```

```python
# tldw_Server_API/app/api/v1/endpoints/media_embeddings.py
def _store_embeddings_for_media(
    *,
    media_id: int,
    media_content: dict[str, Any],
    chunks: list[dict[str, Any]],
    chunk_texts: list[str],
    embeddings: list[Any],
    embedding_model: str,
    embedding_provider: str,
    user_id: str,
) -> None:
    collection_name = f"user_{user_id}_media_embeddings"
    extra_metadata = {}
    media_item_meta = media_content.get("media_item", {})
    if isinstance(media_item_meta, dict):
        extra_metadata = media_item_meta.get("metadata") or {}

    metadatas = []
    for chunk in chunks:
        metadata = {
            "media_id": str(media_id),
            "chunk_index": chunk["index"],
            "chunk_start": chunk["start"],
            "chunk_end": chunk["end"],
            "chunk_type": chunk.get("chunk_type", "text"),
            "title": media_content["media_item"].get("title", ""),
            "author": media_content["media_item"].get("author", ""),
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
        }
        if isinstance(extra_metadata, dict) and extra_metadata:
            metadata["extra"] = dict(extra_metadata)
        metadatas.append(metadata)

    ids = [f"media_{media_id}_chunk_{i}" for i in range(len(chunks))]
    embeddings_list = [emb.tolist() for emb in embeddings] if embeddings and hasattr(embeddings[0], "tolist") else embeddings
    manager = ChromaDBManager(user_id=str(user_id), user_embedding_config=_user_embedding_config())
    manager.store_in_chroma(
        collection_name=collection_name,
        texts=chunk_texts,
        embeddings=embeddings_list,
        ids=ids,
        metadatas=metadatas,
        embedding_model_id_for_dim_check=embedding_model,
    )


async def generate_embeddings_for_media(
    media_id: int,
    media_content: dict[str, Any],
    embedding_model: str,
    embedding_provider: str,
    chunk_size: int,
    chunk_overlap: int,
    user_id: str = "1",
) -> dict[str, Any]:
    try:
        embeddings = await create_embeddings_batch_async(
            texts=chunk_texts,
            provider=embedding_provider,
            model_id=embedding_model,
            metadata=request_metadata,
        )
    except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS:
        embeddings = None
    else:
        validation_error = _validate_embeddings_result(embeddings, len(chunk_texts))
        if validation_error:
            return {
                "status": "error",
                "message": validation_error,
                "error": validation_error,
                "embedding_count": len(embeddings) if embeddings else 0,
                "chunks_processed": len(chunks),
            }
        try:
            _store_embeddings_for_media(
                media_id=media_id,
                media_content=media_content,
                chunks=chunks,
                chunk_texts=chunk_texts,
                embeddings=embeddings,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                user_id=str(user_id),
            )
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
            return _storage_failure_result(exc, len(chunks))
        return {
            "status": "success",
            "message": f"Successfully generated {len(embeddings)} embeddings",
            "embedding_count": len(embeddings),
            "chunks_processed": len(chunks),
        }

    if embedding_model != FALLBACK_EMBEDDING_MODEL:
        try:
            fallback_embeddings = await create_embeddings_batch_async(
                texts=chunk_texts,
                provider="huggingface",
                model_id=FALLBACK_EMBEDDING_MODEL,
                metadata=request_metadata,
            )
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
            return {
                "status": "error",
                "message": f"Fallback embedding generation also failed: {exc}",
                "error": str(exc),
                "embedding_count": 0,
                "chunks_processed": len(chunks),
            }
        validation_error = _validate_embeddings_result(fallback_embeddings, len(chunk_texts))
        if validation_error:
            return {
                "status": "error",
                "message": validation_error,
                "error": validation_error,
                "embedding_count": len(fallback_embeddings) if fallback_embeddings else 0,
                "chunks_processed": len(chunks),
            }
        try:
            _store_embeddings_for_media(
                media_id=media_id,
                media_content=media_content,
                chunks=chunks,
                chunk_texts=chunk_texts,
                embeddings=fallback_embeddings,
                embedding_model=FALLBACK_EMBEDDING_MODEL,
                embedding_provider="huggingface",
                user_id=str(user_id),
            )
        except _MEDIA_EMBEDDINGS_NONCRITICAL_EXCEPTIONS as exc:
            return _storage_failure_result(exc, len(chunks))
        return {
            "status": "success",
            "message": f"Generated embeddings using fallback model {FALLBACK_EMBEDDING_MODEL}",
            "embedding_count": len(fallback_embeddings),
            "chunks_processed": len(chunks),
        }
```

Implementation note: keep fallback eligibility tied to generation exceptions only. Do not wrap `_store_embeddings_for_media()` in the same `try` block as provider generation.

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_failure_classification.py \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_storage_scope.py -v
```

Expected: PASS with storage failures classified as storage failures and the existing user-scoped Chroma manager test still green.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/media_embeddings.py \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_failure_classification.py
git commit -m "fix: separate media embedding storage from fallback"
```

### Task 4: Return Truthful Partial Success for Batch Enqueue

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_media_embeddings_submission_semantics.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_media_embedding_jobs.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_generate_embeddings_batch_returns_partial_response(monkeypatch):
    class _PartialAdapter:
        def create_job(self, **kwargs):
            media_id = int(kwargs["media_id"])
            if media_id == 456:
                raise RuntimeError("enqueue failed")
            return {"uuid": f"job-{media_id}"}

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "EmbeddingsJobsAdapter", _PartialAdapter)

    result = await media_embeddings.generate_embeddings_batch(
        request=media_embeddings.BatchMediaEmbeddingsRequest(media_ids=[123, 456]),
        db=_FakeMediaDB([123, 456]),
        current_user=_user(),
    )

    assert result.status == "partial"
    assert result.job_ids == ["job-123"]
    assert result.submitted == 1
    assert result.failed_media_ids == [456]
    assert result.failure_reasons == ["media_id=456: RuntimeError"]


@pytest.mark.asyncio
async def test_generate_embeddings_batch_raises_if_nothing_was_queued(monkeypatch):
    class _FailingAdapter:
        def create_job(self, **kwargs):
            raise RuntimeError(f"enqueue failed for {kwargs['media_id']}")

    monkeypatch.setattr(media_embeddings, "_embeddings_jobs_backend", lambda: "jobs")
    monkeypatch.setattr(media_embeddings, "_resolve_model_provider", lambda *_: ("model-a", "provider-a"))
    monkeypatch.setattr(media_embeddings, "EmbeddingsJobsAdapter", _FailingAdapter)

    with pytest.raises(HTTPException) as excinfo:
        await media_embeddings.generate_embeddings_batch(
            request=media_embeddings.BatchMediaEmbeddingsRequest(media_ids=[123]),
            db=_FakeMediaDB([123]),
            current_user=_user(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail["submitted"] == 0
```

```python
def test_media_embedding_batch_returns_202_on_partial_enqueue_failure(monkeypatch):
    os.environ["TESTING"] = "true"
    response = None
    try:
        from tldw_Server_API.app.api.v1.endpoints import media_embeddings as media_embeddings_endpoint

        class _PartialAdapter:
            def create_job(self, **kwargs):
                media_id = int(kwargs["media_id"])
                if media_id == 456:
                    raise RuntimeError("enqueue failed")
                return {"uuid": f"job-{media_id}"}

        app.dependency_overrides[get_media_db_for_user] = lambda: _FakeMediaDB(media_ids=[123, 456])
        monkeypatch.setattr(media_embeddings_endpoint, "EmbeddingsJobsAdapter", _PartialAdapter)

        with _client() as client:
            api_key = get_settings().SINGLE_USER_API_KEY
            response = client.post(
                "/api/v1/media/embeddings/batch",
                json={"media_ids": [123, 456]},
                headers={"X-API-KEY": api_key},
            )
    finally:
        os.environ.pop("TESTING", None)
        app.dependency_overrides.pop(get_media_db_for_user, None)

    assert response is not None
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "partial"
    assert body["job_ids"] == ["job-123"]
    assert body["submitted"] == 1
    assert body["failed_media_ids"] == [456]
    assert body["failure_reasons"] == ["media_id=456: RuntimeError"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_submission_semantics.py \
  tldw_Server_API/tests/Embeddings/test_media_embedding_jobs.py -k "partial or nothing_was_queued" -v
```

Expected: FAIL because the current endpoint raises `HTTPException(500)` whenever any enqueue fails, even after one or more jobs were already queued.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/api/v1/endpoints/media_embeddings.py
class BatchMediaEmbeddingsResponse(BaseModel):
    status: Literal["accepted", "partial"]
    job_ids: list[str]
    submitted: int
    failed_media_ids: list[int] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
```

```python
# tldw_Server_API/app/api/v1/endpoints/media_embeddings.py
if failed_media_ids and not job_ids:
    detail = {
        "error": "batch_enqueue_failed",
        "message": "Failed to queue one or more embedding jobs",
        "submitted": 0,
        "failed_media_ids": failed_media_ids,
        "failure_reasons": failure_reasons,
    }
    raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

if failed_media_ids:
    return BatchMediaEmbeddingsResponse(
        status="partial",
        job_ids=job_ids,
        submitted=len(job_ids),
        failed_media_ids=failed_media_ids,
        failure_reasons=failure_reasons,
    )

return BatchMediaEmbeddingsResponse(
    status="accepted",
    job_ids=job_ids,
    submitted=len(job_ids),
    failed_media_ids=[],
    failure_reasons=[],
)
```

```python
# tldw_Server_API/app/api/v1/endpoints/media_embeddings.py
from typing import Annotated, Any, Literal, Optional
```

- [ ] **Step 4: Run the full focused verification and security checks**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_failure_classification.py \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_storage_scope.py \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_submission_semantics.py \
  tldw_Server_API/tests/Embeddings/test_media_embedding_jobs.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py \
  tldw_Server_API/tests/Embeddings/test_l2_normalization_policy.py -v
```

Expected: PASS for all focused regression tests covering the corrected contracts.

Run:
```bash
source .venv/bin/activate && python -m bandit \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  tldw_Server_API/app/api/v1/endpoints/media_embeddings.py \
  -f json -o /tmp/bandit_embeddings_remediation.json
```

Expected: exit `0` and `/tmp/bandit_embeddings_remediation.json` created with no new unresolved findings in the touched endpoint code.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/media_embeddings.py \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_submission_semantics.py \
  tldw_Server_API/tests/Embeddings/test_media_embedding_jobs.py
git commit -m "fix: make media embeddings batch acceptance truthful"
```
