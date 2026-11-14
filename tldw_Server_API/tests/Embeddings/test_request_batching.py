import asyncio
import time
from collections import deque
from contextlib import suppress
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.core.Embeddings.request_batching import (
    BatchRequest,
    RequestBatcher,
    create_embeddings_batch_async,
)
from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import HFModelCfg
from tldw_Server_API.app.core.Embeddings.simplified_config import (
    BatchingConfig,
    EmbeddingsConfig,
    ProviderConfig,
    SecurityConfig,
)


@pytest.mark.asyncio
async def test_batched_requests_include_provider_credentials(monkeypatch):
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )
    batcher = RequestBatcher(config=config)

    captured = {}

    async def fake_create_embeddings_batch_async(texts, user_app_config, model_id_override=None):
        captured["config"] = user_app_config
        return [[0.1] for _ in texts]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    request = BatchRequest(
        request_id="req-1",
        text="hello",
        model="text-embedding-3-small",
        provider="openai",
        metadata={},
        future=future,
        timestamp=time.time(),
    )

    await batcher._process_batch([request], "openai", "text-embedding-3-small")

    assert future.done()
    assert future.result() == [0.1]

    user_app_config = captured["config"]
    assert user_app_config["openai_api"]["api_key"] == "sk-test"
    model_entry = user_app_config["embedding_config"]["models"]["openai:text-embedding-3-small"]
    assert model_entry["api_key"] == "sk-test"
    assert model_entry["model_name_or_path"] == "text-embedding-3-small"

    await batcher.shutdown()


@pytest.mark.asyncio
async def test_process_batch_skips_cancelled_futures(monkeypatch):
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )
    batcher = RequestBatcher(config=config)

    async def fake_create_embeddings_batch_async(texts, user_app_config, model_id_override=None):
        return [[0.1] for _ in texts]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )

    loop = asyncio.get_running_loop()
    cancelled_future = loop.create_future()
    cancelled_future.cancel()
    active_future = loop.create_future()

    batch = [
        BatchRequest(
            request_id="req-cancelled",
            text="cancelled",
            model="text-embedding-3-small",
            provider="openai",
            metadata={},
            future=cancelled_future,
            timestamp=time.time(),
        ),
        BatchRequest(
            request_id="req-active",
            text="active",
            model="text-embedding-3-small",
            provider="openai",
            metadata={},
            future=active_future,
            timestamp=time.time(),
        ),
    ]

    await batcher._process_batch(batch, "openai", "text-embedding-3-small")

    assert cancelled_future.cancelled()
    assert active_future.result() == [0.1]

    await batcher.shutdown()


@pytest.mark.asyncio
async def test_override_config_passthrough(monkeypatch, tmp_path):
    """Override configs supplied at submit time should reach the embedder."""
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="huggingface",
                models=["sentence-transformers/all-MiniLM-L6-v2"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="huggingface",
        default_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    batcher = RequestBatcher(config=config)

    override_config = {
        "embedding_config": {
            "default_model_id": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
            "model_storage_base_dir": str(tmp_path),
            "models": {
                "huggingface:sentence-transformers/all-MiniLM-L6-v2": HFModelCfg(
                    provider="huggingface",
                    model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                    trust_remote_code=True,
                    hf_cache_dir_subpath="custom_cache",
                ),
            },
        }
    }

    captured = {}

    async def fake_create_embeddings_batch_async(texts, user_app_config, model_id_override=None):
        captured["config"] = user_app_config
        return [[0.42] for _ in texts]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )

    request = BatchRequest(
        request_id="req-override",
        text="hello override",
        model="sentence-transformers/all-MiniLM-L6-v2",
        provider="huggingface",
        metadata={},
        config_override=override_config,
        future=asyncio.get_running_loop().create_future(),
        timestamp=time.time(),
    )

    await batcher._process_batch([request], "huggingface", "sentence-transformers/all-MiniLM-L6-v2")

    assert request.future.done()
    assert request.future.result() == [0.42]
    assert captured["config"] is override_config
    model_cfg = captured["config"]["embedding_config"]["models"]["huggingface:sentence-transformers/all-MiniLM-L6-v2"]
    assert isinstance(model_cfg, HFModelCfg)
    assert model_cfg.trust_remote_code is True

    await batcher.shutdown()


@pytest.mark.asyncio
async def test_local_alias_provider_resolution():
    """Local provider aliases (local -> local_api) should resolve correctly."""
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="local",
                api_url="http://localhost:8080/v1/embeddings",
                models=["my-local-model"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="local",
        default_model="my-local-model",
    )
    batcher = RequestBatcher(config=config)

    user_config = batcher._build_user_app_config("local_api", "my-local-model")

    model_entry = user_config["embedding_config"]["models"]["local_api:my-local-model"]
    assert model_entry["provider"] == "local_api"
    assert user_config["local_api"]["api_url"] == "http://localhost:8080/v1/embeddings"

    await batcher.shutdown()


@pytest.mark.asyncio
async def test_local_provider_with_api_url_routes_to_local_api():
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="local",
                api_url="http://localhost:8080/v1/embeddings",
                models=["my-local-model"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="local",
        default_model="my-local-model",
    )
    batcher = RequestBatcher(config=config)
    batcher.metrics = SimpleNamespace(log_batch_size=lambda *args, **kwargs: None, log_error=lambda *args, **kwargs: None)

    stop_event = asyncio.Event()
    captured = {}

    async def stub_process_queue(self, queue_key):
        await stop_event.wait()
        queue = self.queues[queue_key]
        batch = list(queue)
        queue.clear()
        captured["provider"] = queue_key[0]
        for req in batch:
            if not req.future.done():
                req.future.set_result([0.0])

    batcher._process_queue = MethodType(stub_process_queue, batcher)

    task = asyncio.create_task(
        batcher.submit_request(
            text="alpha",
            model="my-local-model",
            provider="local",
        )
    )

    await asyncio.sleep(0.05)
    assert {key[0] for key in batcher.queues.keys()} == {"local_api"}

    stop_event.set()
    await task
    assert captured["provider"] == "local_api"
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_local_provider_without_api_url_routes_in_process():
    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="local",
                api_url=None,
                models=["my-local-model"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="local",
        default_model="my-local-model",
    )
    batcher = RequestBatcher(config=config)
    batcher.metrics = SimpleNamespace(log_batch_size=lambda *args, **kwargs: None, log_error=lambda *args, **kwargs: None)

    stop_event = asyncio.Event()
    captured = {}

    async def stub_process_queue(self, queue_key):
        await stop_event.wait()
        queue = self.queues[queue_key]
        batch = list(queue)
        queue.clear()
        captured["provider"] = queue_key[0]
        for req in batch:
            if not req.future.done():
                req.future.set_result([0.0])

    batcher._process_queue = MethodType(stub_process_queue, batcher)

    task = asyncio.create_task(
        batcher.submit_request(
            text="bravo",
            model="my-local-model",
            provider="local",
        )
    )

    await asyncio.sleep(0.05)
    assert {key[0] for key in batcher.queues.keys()} == {"local"}

    stop_event.set()
    await task
    assert captured["provider"] == "local"
    await batcher.shutdown()


@pytest.mark.asyncio
async def test_submit_request_enforces_rate_limits(monkeypatch):
    limiter_calls = []

    class StubLimiter:
        def __init__(self):
            self.rate_limiter = SimpleNamespace(user_tiers={"user-1": "free"})

        async def check_rate_limit_async(self, user_id, cost=1, ip_address=None, tokens_units=0):
            limiter_calls.append((user_id, cost))
            return False, 7

    stub_limiter = StubLimiter()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.request_batching.get_async_rate_limiter",
        lambda: stub_limiter,
        raising=True,
    )

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=True),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    batcher = RequestBatcher(config=config)
    batcher.metrics = SimpleNamespace(
        log_rate_limit_hit=Mock(),
        log_batch_size=lambda *args, **kwargs: None,
        log_error=lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="Retry after 7s"):
        await batcher.submit_request(
            text="hello",
            model="text-embedding-3-small",
            provider="openai",
            metadata={"user_id": "user-1"},
        )

    assert limiter_calls == [("user-1", 1)]
    batcher.metrics.log_rate_limit_hit.assert_called_once_with("user-1", "free")

    await batcher.shutdown()


@pytest.mark.asyncio
async def test_submit_request_respects_rate_limit_when_batching_disabled(monkeypatch):
    calls = []

    class StubLimiter:
        def __init__(self):
            self.rate_limiter = SimpleNamespace(user_tiers={"user-1": "free"})

        async def check_rate_limit_async(self, user_id, cost=1, ip_address=None, tokens_units=0):
            calls.append(user_id)
            if len(calls) == 1:
                return True, None
            return False, 42

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.request_batching.get_async_rate_limiter",
        lambda: StubLimiter(),
        raising=True,
    )

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=True),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    batcher = RequestBatcher(config=config)
    batcher.enabled = False
    batcher.metrics = SimpleNamespace(
        log_rate_limit_hit=Mock(),
    )

    async def fake_process_single(
        self,
        text,
        model,
        provider,
        metadata=None,
        config_override=None,
        rate_limit_applied=False,
    ):
        return [0.5]

    batcher._process_single = MethodType(fake_process_single, batcher)

    result = await batcher.submit_request(
        text="hello",
        model="text-embedding-3-small",
        provider="openai",
        metadata={"user_id": "user-1"},
    )
    assert result == [0.5]

    with pytest.raises(RuntimeError, match="Retry after 42s"):
        await batcher.submit_request(
            text="world",
            model="text-embedding-3-small",
            provider="openai",
            metadata={"user_id": "user-1"},
        )

    assert calls == ["user-1", "user-1"]
    batcher.metrics.log_rate_limit_hit.assert_called_once_with("user-1", "free")

    await batcher.shutdown()


@pytest.mark.asyncio
async def test_process_single_skips_provider_rate_limit_after_submit(monkeypatch):
    captured = {}

    class StubLimiter:
        async def check_rate_limit_async(self, user_id, cost=1, ip_address=None, tokens_units=0):
            return True, None

    class StubService:
        async def create_embedding(self, text, model=None, provider=None, user_id=None, use_batching=False):
            captured["user_id"] = user_id
            return [0.11]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.request_batching.get_async_rate_limiter",
        lambda: StubLimiter(),
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.async_embeddings.get_async_embedding_service",
        lambda: StubService(),
        raising=True,
    )

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=False,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=True),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    batcher = RequestBatcher(config=config)
    batcher.enabled = False

    result = await batcher.submit_request(
        text="hello",
        model="text-embedding-3-small",
        provider="openai",
        metadata={"user_id": "user-1"},
    )

    assert result == [0.11]
    assert captured["user_id"] is None

    await batcher.shutdown()


@pytest.mark.asyncio
async def test_process_single_preserves_user_id_without_rate_limit(monkeypatch):
    captured = {}

    class StubLimiter:
        async def check_rate_limit_async(self, user_id, cost=1, ip_address=None, tokens_units=0):
            return True, None

    class StubService:
        async def create_embedding(self, text, model=None, provider=None, user_id=None, use_batching=False):
            captured["user_id"] = user_id
            return [0.22]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.request_batching.get_async_rate_limiter",
        lambda: StubLimiter(),
        raising=True,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.async_embeddings.get_async_embedding_service",
        lambda: StubService(),
        raising=True,
    )

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=False,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    batcher = RequestBatcher(config=config)
    batcher.enabled = False

    result = await batcher.submit_request(
        text="hello",
        model="text-embedding-3-small",
        provider="openai",
        metadata={"user_id": "user-2"},
    )

    assert result == [0.22]
    assert captured["user_id"] == "user-2"

    await batcher.shutdown()


@pytest.mark.asyncio
async def test_global_batch_helper_passes_metadata(monkeypatch):
    captured = {}

    class StubBatcher:
        enabled = True

        async def submit_request(self, text, model, provider, metadata=None, config_override=None):
            captured["metadata"] = metadata
            captured["config_override"] = config_override
            return [0.42]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.request_batching.get_batcher",
        lambda: StubBatcher(),
        raising=True,
    )

    result = await create_embeddings_batch_async(
        texts=["hello"],
        config={},
        model_id_override="openai:text-embedding-3-small",
        metadata={"user_id": "user-xyz"},
    )

    assert result == [[0.42]]
    assert captured["metadata"] == {"user_id": "user-xyz"}
    assert captured["config_override"] == {}


@pytest.mark.asyncio
async def test_global_batch_helper_uses_default_model_id(monkeypatch):
    captured = {}

    class StubBatcher:
        enabled = True

        async def submit_request(self, text, model, provider, metadata=None, config_override=None):
            captured["model"] = model
            captured["provider"] = provider
            captured["config_override"] = config_override
            return [0.11]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.request_batching.get_batcher",
        lambda: StubBatcher(),
        raising=True,
    )

    config = {
        "embedding_config": {
            "default_model_id": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
            "models": {
                "huggingface:sentence-transformers/all-MiniLM-L6-v2": {
                    "provider": "huggingface",
                    "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
                }
            },
        }
    }

    result = await create_embeddings_batch_async(
        texts=["hello"],
        config=config,
    )

    assert result == [[0.11]]
    assert captured["provider"] == "huggingface"
    assert captured["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["config_override"] == config


@pytest.mark.asyncio
async def test_config_override_creates_distinct_queues(monkeypatch):
    async def fake_create_embeddings_batch_async(texts, user_app_config, model_id_override=None):
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    batcher = RequestBatcher(config=config)
    batcher.metrics = SimpleNamespace(
        log_batch_size=lambda *args, **kwargs: None,
        log_error=lambda *args, **kwargs: None,
    )

    stop_event = asyncio.Event()
    original_process_batch = batcher._process_batch

    async def stub_process_queue(self, queue_key):
        await stop_event.wait()
        queue = self.queues[queue_key]
        batch = list(queue)
        queue.clear()
        await original_process_batch(batch, queue_key[0], queue_key[1])

    batcher._process_queue = MethodType(stub_process_queue, batcher)

    override1 = {"embedding_config": {"default_model_id": "openai:text-embedding-3-small"}}
    override2 = {"embedding_config": {"default_model_id": "openai:text-embedding-3-small", "variant": "alt"}}

    task1 = asyncio.create_task(
        batcher.submit_request(
            text="alpha",
            model="text-embedding-3-small",
            provider="openai",
            config_override=override1,
        )
    )
    task2 = asyncio.create_task(
        batcher.submit_request(
            text="bravo",
            model="text-embedding-3-small",
            provider="openai",
            config_override=override2,
        )
    )

    await asyncio.sleep(0.05)

    assert len(batcher.queues) == 2
    queue_labels = {batcher._queue_label(key) for key in batcher.queues.keys()}
    assert len(queue_labels) == 2

    stop_event.set()
    await asyncio.gather(task1, task2)
    await batcher.shutdown()


def test_identical_override_configs_share_queue(tmp_path):


    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="huggingface",
                models=["sentence-transformers/all-MiniLM-L6-v2"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="huggingface",
        default_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    batcher = RequestBatcher(config=config)

    def _make_override():

        return {
            "embedding_config": {
                "default_model_id": "huggingface:sentence-transformers/all-MiniLM-L6-v2",
                "model_storage_base_dir": str(tmp_path),
                "models": {
                    "huggingface:sentence-transformers/all-MiniLM-L6-v2": HFModelCfg(
                        provider="huggingface",
                        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
                        trust_remote_code=False,
                        hf_cache_dir_subpath="hf_cache",
                    )
                },
            }
        }

    override_one = _make_override()
    override_two = _make_override()

    key_one = batcher._queue_key(
        "huggingface", "sentence-transformers/all-MiniLM-L6-v2", override_one
    )
    key_two = batcher._queue_key(
        "huggingface", "sentence-transformers/all-MiniLM-L6-v2", override_two
    )

    assert key_one == key_two


@pytest.mark.asyncio
async def test_shutdown_cancels_processing_tasks(monkeypatch):
    async def fake_create_embeddings_batch_async(texts, user_app_config, model_id_override=None):
        await asyncio.sleep(0)
        return [[0.1] for _ in texts]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    batcher = RequestBatcher(config=config)

    result = await batcher.submit_request(
        text="hello",
        model="text-embedding-3-small",
        provider="openai",
    )
    assert result == [0.1]

    await asyncio.wait_for(batcher.shutdown(), timeout=1.0)
    assert all(task.cancelled() or task.done() for task in batcher.processing_tasks.values())


@pytest.mark.asyncio
async def test_submit_request_restarts_cancelled_processing_task(monkeypatch):
    call_counter = {"count": 0}

    async def fake_create_embeddings_batch_async(texts, user_app_config, model_id_override=None):
        call_counter["count"] += 1
        await asyncio.sleep(0)
        return [[float(call_counter["count"])] for _ in texts]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch_async",
        fake_create_embeddings_batch_async,
        raising=True,
    )

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=4,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    batcher = RequestBatcher(config=config)

    result_one = await asyncio.wait_for(
        batcher.submit_request(
            text="first",
            model="text-embedding-3-small",
            provider="openai",
        ),
        timeout=1.0,
    )
    assert result_one == [1.0]

    queue_key = batcher._queue_key("openai", "text-embedding-3-small", None)
    task = batcher.processing_tasks.get(queue_key)
    assert task is not None

    task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)

    assert queue_key not in batcher.processing_tasks

    result_two = await asyncio.wait_for(
        batcher.submit_request(
            text="second",
            model="text-embedding-3-small",
            provider="openai",
        ),
        timeout=1.0,
    )
    assert result_two == [2.0]

    await asyncio.wait_for(batcher.shutdown(), timeout=1.0)


def test_build_user_app_config_normalizes_local_provider():


    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="local",
                api_key="dummy",
                api_url="http://localhost:8080/v1/embeddings",
                models=["mini"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            adaptive_batching=False,
            max_batch_size=2,
            batch_timeout_ms=10,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="local",
        default_model="mini",
    )

    batcher = RequestBatcher(config=config)
    app_config = batcher._build_user_app_config("local", "mini")

    model_entry = app_config["embedding_config"]["models"]["local:mini"]
    assert model_entry["provider"] == "local_api"
    assert app_config["local_api"]["api_url"] == "http://localhost:8080/v1/embeddings"


@pytest.mark.asyncio
async def test_processing_task_restarts_when_queue_has_pending_items(monkeypatch):
    keepalive = asyncio.Event()

    async def stub_process_queue(self, queue_key):
        await keepalive.wait()

    monkeypatch.setattr(
        RequestBatcher,
        "_process_queue",
        stub_process_queue,
        raising=True,
    )

    config = EmbeddingsConfig(
        providers=[
            ProviderConfig(
                name="openai",
                api_key="sk-test",
                models=["text-embedding-3-small"],
            )
        ],
        batching=BatchingConfig(
            enabled=True,
            max_batch_size=2,
            batch_timeout_ms=10,
            adaptive_batching=False,
        ),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="openai",
        default_model="text-embedding-3-small",
    )

    batcher = RequestBatcher(config=config)
    queue_key = batcher._queue_key("openai", "text-embedding-3-small", None)
    batcher.queues[queue_key] = deque()
    batcher.queues[queue_key].append(object())

    completed_task = asyncio.create_task(asyncio.sleep(0))
    batcher.processing_tasks[queue_key] = completed_task
    await completed_task

    batcher._handle_task_done(queue_key, completed_task)

    restarted_task = batcher.processing_tasks.get(queue_key)
    assert restarted_task is not None
    assert not restarted_task.done()

    batcher.queues[queue_key].clear()
    restarted_task.cancel()
    with suppress(asyncio.CancelledError):
        await restarted_task

    keepalive.set()
    await asyncio.wait_for(batcher.shutdown(), timeout=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_type", [OSError, RuntimeError], ids=["offline-model-oserror", "existing-runtime-error"]
)
async def test_provider_failure_settles_all_submitters_and_keeps_queue_usable(monkeypatch, failure_type):
    """External provider failure must not orphan the real batch's request futures."""
    config = EmbeddingsConfig(
        providers=[ProviderConfig(name="huggingface", models=["test-model"])],
        batching=BatchingConfig(enabled=True, max_batch_size=2, batch_timeout_ms=10, adaptive_batching=False),
        security=SecurityConfig(enable_rate_limiting=False),
        default_provider="huggingface",
        default_model="test-model",
    )
    batcher = RequestBatcher(config=config)
    failed = asyncio.Event()
    owners = []
    failure = failure_type("controlled offline provider failure")
    calls = 0

    async def provider(texts, user_app_config, model_id_override=None):
        nonlocal calls
        calls += 1
        owners.append(asyncio.current_task())
        if calls == 1:
            failed.set()
            raise failure
        return [[0.25, 0.5] for _ in texts]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create.create_embeddings_batch_async",
        provider,
    )
    submitters = [
        asyncio.create_task(batcher.submit_request(text, "test-model", "huggingface")) for text in ("first", "second")
    ]
    try:
        await asyncio.wait_for(failed.wait(), timeout=2)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert all(task.done() for task in submitters)  # nosec B101 # Pytest behavior assertion; no runtime security boundary.
        outcomes = await asyncio.gather(*submitters, return_exceptions=True)
        assert all(outcome is failure for outcome in outcomes)  # nosec B101 # Pytest behavior assertion; no runtime security boundary.
        recovered = await asyncio.wait_for(batcher.submit_request("recovered", "test-model", "huggingface"), timeout=2)
        assert recovered == [0.25, 0.5]  # nosec B101 # Pytest behavior assertion; no runtime security boundary.
        assert calls == 2  # nosec B101 # Pytest behavior assertion; no runtime security boundary.
    finally:
        for task in submitters:
            if not task.done():
                task.cancel()
        await asyncio.gather(*submitters, return_exceptions=True)
        await batcher.shutdown()
        await asyncio.gather(*owners, return_exceptions=True)
