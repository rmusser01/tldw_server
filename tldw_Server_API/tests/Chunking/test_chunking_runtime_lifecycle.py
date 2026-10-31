from __future__ import annotations

import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import UploadFile
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import chunking as chunking_module
from tldw_Server_API.app.api.v1.schemas.chunking_schema import (
    ChunkingOptionsRequest,
    ChunkingTextRequest,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ServerFallbackCredentials
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    dumps_envelope,
    encrypt_byok_payload,
)
from tldw_Server_API.app.core.Chat import bounded_daemon
from tldw_Server_API.app.core.Chunking.strategies.rolling_summarize import (
    LLM_USAGE_SUCCEEDED_KEY,
    LLM_USAGE_TRACKER_KEY,
)
from tldw_Server_API.app.core.Chunking.templates import TemplateProcessor

pytestmark = pytest.mark.unit

_PROVIDER_CAPACITY_DETAIL = {
    "error_code": "provider_capacity_exhausted",
    "message": "The chunking provider is temporarily busy.",
}


def _encrypted_provider_row(api_key: str) -> dict[str, object]:
    return {
        "encrypted_blob": dumps_envelope(
            encrypt_byok_payload(build_secret_payload(api_key))
        ),
        "revoked_at": None,
        "last_used_at": None,
    }


def _record_provider_success(llm_config: dict) -> None:
    tracker = llm_config[LLM_USAGE_TRACKER_KEY]
    tracker[LLM_USAGE_SUCCEEDED_KEY] = True


def _install_owned_worker_cancellation_ack(monkeypatch) -> asyncio.Event:
    cancellation_ack = asyncio.Event()
    original_drain = bounded_daemon._drain_owned_task

    async def acknowledging_drain(task):
        cancellation_ack.set()
        return await original_drain(task)

    monkeypatch.setattr(bounded_daemon, "_drain_owned_task", acknowledging_drain)
    return cancellation_ack


async def _wait_for_thread_event(
    event: threading.Event,
    *,
    timeout: float = 1.0,
) -> bool:
    """Wait for a thread event without consuming the default executor."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.001)
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_key",
    ["key-a", None],
    ids=["A-to-B", "absent-to-B"],
)
async def test_chunking_resolver_freezes_credentials_with_provider_model_snapshot(
    monkeypatch,
    initial_key,
):
    """A rotation after config capture cannot mix B credentials with A's model."""
    initial_snapshot = {
        "llm_api_settings": {"default_api": "openai"},
        "openai_api": {
            "api_key": initial_key,
            "api_base_url": "https://a.example/v1",
            "model_for_summarization": "model-a",
        },
    }
    live_config = {
        "openai_api": {
            "api_key": initial_key,
            "api_base_url": "https://a.example/v1",
            "model_for_summarization": "model-a",
        }
    }
    resolve_entered = asyncio.Event()
    allow_resolve = asyncio.Event()
    fallback_snapshots = []

    class Runtime:
        def __init__(self, **kwargs):
            assert "fallback_resolver" not in kwargs
            self.server_config_snapshot = kwargs["server_config_snapshot"]

        async def resolve(self, provider, *, model=None):
            assert provider == "openai"
            assert model == "model-a"
            resolve_entered.set()
            await allow_resolve.wait()
            fallback = snapshot_fallback(provider, self.server_config_snapshot)
            return SimpleNamespace(
                provider=provider,
                api_key=fallback.api_key,
                app_config=fallback.app_config,
                credentials_resolved=True,
            )

        async def close(self):
            return None

    def snapshot_fallback(provider, snapshot):
        fallback_snapshots.append(snapshot)
        section = snapshot[f"{provider}_api"]
        return ServerFallbackCredentials(
            api_key=section.get("api_key"),
            credential_fields={},
            app_config={f"{provider}_api": dict(section)},
        )

    monkeypatch.setattr(
        chunking_module,
        "derive_trusted_credential_scope",
        lambda *_args: (1, (), (), None),
    )
    monkeypatch.setattr(chunking_module, "ProviderCredentialRuntime", Runtime)
    task = asyncio.create_task(
        chunking_module._resolve_chunking_credentials(
            "openai",
            model="model-a",
            app_config_snapshot=initial_snapshot,
            current_user=SimpleNamespace(id=1),
            request=_http_request(),
        )
    )
    await asyncio.wait_for(resolve_entered.wait(), 1.0)
    live_config["openai_api"] = {
        "api_key": "key-b",
        "api_base_url": "https://b.example/v1",
        "model_for_summarization": "model-b",
    }
    allow_resolve.set()
    runtime, handle = await task

    assert runtime is not None
    assert handle.api_key == initial_key
    assert handle.app_config["openai_api"]["api_base_url"] == "https://a.example/v1"
    assert fallback_snapshots == [initial_snapshot]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["user", "team", "org", "absent"])
async def test_chunking_real_byok_resolution_cannot_mix_config_generations(
    monkeypatch,
    source,
):
    """Chunking receives one A-generation key/endpoint/model tuple after A→B."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv(
        "BYOK_ENCRYPTION_KEY",
        base64.b64encode(b"k" * 32).decode("ascii"),
    )
    reset_settings()
    row = _encrypted_provider_row("sk-byok-generation-a")
    initial_snapshot = {
        "openai_api": {
            "api_key": "sk-static-generation-a",
            "api_base_url": "https://generation-a.example/v1",
            "model": "model-a",
        }
    }
    live_config = {"openai_api": dict(initial_snapshot["openai_api"])}
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()

    class UserRepo:
        calls = 0

        async def fetch_secret_for_active_user(
            self,
            _user_id,
            _provider,
            *,
            include_revoked=False,
        ):
            assert include_revoked is True
            self.calls += 1
            if self.calls == 1:
                lookup_started.set()
                await release_lookup.wait()
            return row if source == "user" else None

        async def fetch_secret_for_user(
            self,
            _user_id,
            _provider,
            *,
            include_revoked=False,
        ):
            assert include_revoked is True
            return None

    class SharedRepo:
        async def fetch_secret(
            self,
            scope_type,
            _scope_id,
            _provider,
            *,
            include_revoked=False,
        ):
            assert include_revoked is True
            return row if scope_type == source else None

    async def get_user_repo():
        return UserRepo()

    async def get_shared_repo():
        return SharedRepo()

    class EmptyOverrideSnapshot:
        def enforce(self, _model):
            return None

        def ensure_healthy(self):
            return None

        def server_fallback(self, base_fallback=None):
            return base_fallback

    monkeypatch.setattr(byok_runtime, "_get_user_repo", get_user_repo)
    monkeypatch.setattr(byok_runtime, "_get_org_repo", get_shared_repo)
    monkeypatch.setattr(byok_runtime, "is_byok_enabled", lambda: True)
    monkeypatch.setattr(byok_runtime, "is_provider_allowlisted", lambda _provider: True)
    monkeypatch.setattr(byok_runtime, "loaded_config_data", live_config)
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: {"openai_api": dict(live_config["openai_api"])},
    )
    monkeypatch.setattr(
        chunking_module,
        "derive_trusted_credential_scope",
        lambda *_args: (
            7,
            [11] if source == "team" else [],
            [13] if source == "org" else [],
            True,
        ),
    )
    monkeypatch.setattr(
        chunking_module,
        "capture_provider_override_call_snapshot",
        lambda _provider: EmptyOverrideSnapshot(),
    )

    task = asyncio.create_task(
        chunking_module._resolve_chunking_credentials(
            "openai",
            model="model-a",
            app_config_snapshot=initial_snapshot,
            current_user=SimpleNamespace(id=7),
            request=_http_request(),
        )
    )
    runtime = None
    try:
        await asyncio.wait_for(lookup_started.wait(), timeout=1.0)
        live_config["openai_api"] = {
            "api_key": "sk-static-generation-b",
            "api_base_url": "https://generation-b.example/v1",
            "model": "model-b",
        }
        release_lookup.set()
        runtime, handle = await task
        expected_key = (
            "sk-static-generation-a"
            if source == "absent"
            else "sk-byok-generation-a"
        )
        assert handle.api_key == expected_key
        assert handle.app_config == {
            "openai_api": {
                "api_base_url": "https://generation-a.example/v1",
                "model": "model-a",
            }
        }
    finally:
        release_lookup.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if runtime is not None:
            await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file", "template"])
async def test_rolling_config_snapshot_capture_runs_off_event_loop(
    monkeypatch,
    kind,
):
    event_loop_thread = threading.get_ident()
    capture_threads = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "summary-model"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle

        async def close(self):
            return None

    def load_snapshot():
        capture_threads.append(threading.get_ident())
        return _rolling_config()

    async def resolve_credentials(*_args, **_kwargs):
        return Runtime(), handle

    def process(_text, _options, _tokenizer, _llm_func, llm_config):
        _record_provider_success(llm_config)
        return [
            {
                "text": "summary chunk",
                "metadata": {
                    "method": "rolling_summarize",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    def process_template(self, text, template, **options):
        del text, template, options
        _record_provider_success(self._chunker.llm_config)
        return ["template chunk"]

    monkeypatch.setattr(chunking_module, "load_server_configs", load_snapshot)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)

    if kind == "template":
        result = await _invoke_rolling_template("off-loop-config-template")
    else:
        result = await _invoke_rolling_endpoint(kind, f"off-loop-config-{kind}")

    assert result.chunks
    assert capture_threads
    assert capture_threads[0] != event_loop_thread


def _http_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/chunking",
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 1),
            "scheme": "http",
        }
    )


def _rolling_config(provider: str = "openai") -> dict:
    return {
        "llm_api_settings": {
            "default_api": provider,
            "default_api_for_tasks": provider,
        },
        f"{provider}_api": {
            "model": "summary-model",
            "model_for_summarization": "summary-model",
            "max_tokens_for_summarization_step": 128,
        },
    }


async def _invoke_rolling_endpoint(kind: str, text: str):
    request = _http_request()
    user = SimpleNamespace(id=1)
    if kind == "json":
        return await chunking_module.process_text_for_chunking_json(
            ChunkingTextRequest(
                text_content=text,
                file_name=f"{text}.txt",
                options=ChunkingOptionsRequest(
                    method="rolling_summarize",
                    max_size=256,
                    overlap=0,
                ),
            ),
            http_request=request,
            current_user=user,
            media_db=None,
        )

    upload = UploadFile(file=io.BytesIO(text.encode()), filename=f"{text}.txt")
    return await chunking_module.process_file_for_chunking(
        http_request=request,
        file=upload,
        method="rolling_summarize",
        max_size=256,
        overlap=0,
        language="en",
        tokenizer_name_or_path="gpt2",
        code_mode=None,
        adaptive=False,
        multi_level=False,
        custom_chapter_pattern=None,
        semantic_similarity_threshold=0.7,
        semantic_overlap_sentences=2,
        json_chunkable_data_key="data",
        summarization_detail=0.5,
        llm_step_temperature=None,
        llm_step_system_prompt=None,
        llm_step_max_tokens=None,
        current_user=user,
    )


class _RollingTemplateDatabase:
    def get_chunking_template(self, *, name):
        assert name == "rolling-template"
        return {
            "name": name,
            "description": "rolling template",
            "template_json": (
                '{"chunking":{"method":"rolling_summarize",'
                '"config":{"max_size":256,"overlap":0}}}'
            ),
            "tags": [],
            "version": 1,
        }


async def _invoke_rolling_template(text: str):
    return await chunking_module.process_text_for_chunking_json(
        ChunkingTextRequest(
            text_content=text,
            file_name=f"{text}.txt",
            options=ChunkingOptionsRequest(template_name="rolling-template"),
        ),
        http_request=_http_request(),
        current_user=SimpleNamespace(id=1),
        media_db=_RollingTemplateDatabase(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file"])
async def test_rolling_endpoint_preserves_provider_generated_summary(
    monkeypatch,
    kind,
):
    events = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "summary-model"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            events.append("mark")

        async def close(self):
            events.append("close")

    async def resolve_credentials(*_args, **_kwargs):
        return Runtime(), handle

    def analyze(*_args, **_kwargs):
        return "provider-generated summary"

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "general_llm_analyzer", analyze)

    result = await _invoke_rolling_endpoint(
        kind,
        "Original source sentence that must not replace the semantic summary.",
    )

    assert result.chunks[0].text == "provider-generated summary"
    assert events == ["mark", "close"]


@pytest.mark.asyncio
async def test_rolling_template_accepts_real_processor_result_contract(monkeypatch):
    events = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "summary-model"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            events.append("mark")

        async def close(self):
            events.append("close")

    async def resolve_credentials(*_args, **_kwargs):
        return Runtime(), handle

    def analyze(*_args, **_kwargs):
        return "provider-generated template summary"

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "general_llm_analyzer", analyze)

    result = await _invoke_rolling_template(
        "Original template source sentence for the real processor.",
    )

    assert result.chunks[0].text == "provider-generated template summary"
    assert events == ["mark", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file"])
@pytest.mark.parametrize("failure_site", ["config", "credentials"])
async def test_rolling_setup_failure_is_bounded_and_detached(
    monkeypatch,
    kind,
    failure_site,
):
    secret = f"raw-{kind}-{failure_site}-secret"

    def fail_config():
        raise RuntimeError(secret)

    async def fail_credentials(*_args, **_kwargs):
        raise RuntimeError(secret)

    if failure_site == "config":
        monkeypatch.setattr(chunking_module, "load_server_configs", fail_config)
    else:
        monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
        monkeypatch.setattr(
            chunking_module,
            "_resolve_chunking_credentials",
            fail_credentials,
        )

    with pytest.raises(chunking_module.HTTPException) as exc_info:
        await _invoke_rolling_endpoint(kind, f"setup-failure-{kind}-{failure_site}")

    assert exc_info.value.status_code == 500
    expected_detail = (
        "An internal error occurred during text chunking"
        if kind == "json"
        else "Internal error during file chunking"
    )
    assert exc_info.value.detail == expected_detail
    assert secret not in str(exc_info.value.detail)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file"])
async def test_rolling_endpoint_cancellation_drains_worker_and_mark_before_close(
    monkeypatch,
    kind,
):
    cancellation_ack = _install_owned_worker_cancellation_ack(monkeypatch)
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    events = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "summary-model"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            assert "close" not in events
            events.append("mark")

        async def close(self):
            events.append("close")

    runtime = Runtime()

    async def resolve_credentials(
        provider,
        *,
        model,
        app_config_snapshot,
        current_user,
        request,
    ):
        assert provider == "openai"
        assert model == "summary-model"
        assert app_config_snapshot == _rolling_config()
        assert current_user.id == 1
        assert request is not None
        return runtime, handle

    def process(_text, _options, _tokenizer, _llm_func, llm_config):
        entered.set()
        release.wait()
        _record_provider_success(llm_config)
        exited.set()
        return [
            {
                "text": "summary chunk",
                "metadata": {
                    "method": "rolling_summarize",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)
    monkeypatch.setattr(
        chunking_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    task = asyncio.create_task(_invoke_rolling_endpoint(kind, f"cancel-{kind}"))
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        pool_was_active = pool.active_count == 1
        task.cancel()
        await asyncio.wait_for(cancellation_ack.wait(), 1.0)
        assert events == []
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert exited.is_set()
    assert events == ["mark", "close"]
    assert pool_was_active
    assert pool.active_count == 0


@pytest.mark.asyncio
async def test_rolling_template_processor_runs_off_event_loop(monkeypatch):
    main_thread = threading.get_ident()
    process_threads = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "summary-model"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle

        async def close(self):
            return None

    async def resolve_credentials(*_args, **_kwargs):
        return Runtime(), handle

    def process_template(self, text, template, **options):
        del text, template, options
        process_threads.append(threading.get_ident())
        _record_provider_success(self._chunker.llm_config)
        return ["template chunk"]

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)

    result = await _invoke_rolling_template("off-loop-template")

    assert result.chunks[0].text == "template chunk"
    assert process_threads
    assert process_threads[0] != main_thread


@pytest.mark.asyncio
async def test_rolling_template_cancellation_drains_worker_and_mark_before_close(
    monkeypatch,
):
    cancellation_ack = _install_owned_worker_cancellation_ack(monkeypatch)
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    events = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "summary-model"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            assert "close" not in events
            events.append("mark")

        async def close(self):
            events.append("close")

    runtime = Runtime()

    async def resolve_credentials(*_args, **_kwargs):
        return runtime, handle

    def process_template(self, text, template, **options):
        del text, template, options
        entered.set()
        release.wait()
        _record_provider_success(self._chunker.llm_config)
        exited.set()
        return ["template chunk"]

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)
    monkeypatch.setattr(
        chunking_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    task = asyncio.create_task(_invoke_rolling_template("cancel-template"))
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        pool_was_active = pool.active_count == 1
        task.cancel()
        await asyncio.wait_for(cancellation_ack.wait(), 1.0)
        assert events == []
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert exited.is_set()
    assert events == ["mark", "close"]
    assert pool_was_active
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file", "template"])
async def test_rolling_endpoint_fails_closed_when_pipeline_skips_provider(
    monkeypatch,
    kind,
):
    events = []
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "summary-model"}},
        credentials_resolved=True,
    )

    class Runtime:
        async def mark_used(self, _resolved_handle):
            pytest.fail("a skipped provider call must not be marked as used")

        async def close(self):
            events.append("close")

    async def resolve_credentials(*_args, **_kwargs):
        return Runtime(), handle

    def process(*_args, **_kwargs):
        return [
            {
                "text": "unverified raw chunk",
                "metadata": {
                    "method": "rolling_summarize",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    def process_template(self, text, template, **options):
        del self, text, template, options
        return ["unverified raw chunk"]

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)

    with pytest.raises(chunking_module.HTTPException) as exc_info:
        if kind == "template":
            await _invoke_rolling_template("no-provider-call-template")
        else:
            await _invoke_rolling_endpoint(kind, f"no-provider-call-{kind}")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Chunking input or options are invalid."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert events == ["close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file", "template"])
async def test_rolling_endpoints_accept_runtime_certified_bedrock_default_chain(
    monkeypatch,
    kind,
):
    app_config = {
        "bedrock_api": {
            "model": "summary-model",
            "_runtime_auth_source": "aws_default_chain",
        }
    }
    handle = SimpleNamespace(
        provider="bedrock",
        api_key=None,
        app_config=app_config,
        credentials_resolved=True,
    )
    events = []
    boundary_configs = []

    class Runtime:
        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            events.append("mark")

        async def close(self):
            events.append("close")

    async def resolve_credentials(provider, *, model, **_kwargs):
        assert provider == "bedrock"
        assert model == "summary-model"
        return Runtime(), handle

    def process(_text, _options, _tokenizer, _llm_func, llm_config):
        boundary_configs.append(llm_config)
        _record_provider_success(llm_config)
        return [
            {
                "text": "bedrock chunk",
                "metadata": {
                    "method": "rolling_summarize",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    def process_template(self, text, template, **options):
        del text, template, options
        boundary_configs.append(self._chunker.llm_config)
        _record_provider_success(self._chunker.llm_config)
        return ["bedrock template chunk"]

    monkeypatch.setattr(
        chunking_module,
        "load_server_configs",
        lambda: _rolling_config("bedrock"),
    )
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)

    if kind == "template":
        result = await _invoke_rolling_template("bedrock-template")
    else:
        result = await _invoke_rolling_endpoint(kind, f"bedrock-{kind}")

    assert result.chunks
    assert len(boundary_configs) == 1
    assert boundary_configs[0]["api_name"] == "bedrock"
    assert boundary_configs[0]["api_key"] is None
    assert boundary_configs[0]["app_config"] == app_config
    assert boundary_configs[0]["credentials_resolved"] is True
    assert events == ["mark", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file", "template"])
async def test_rolling_endpoints_reject_uncertified_bedrock_default_chain(
    monkeypatch,
    kind,
):
    handle = SimpleNamespace(
        provider="bedrock",
        api_key=None,
        app_config={
            "bedrock_api": {
                "model": "summary-model",
                "_runtime_auth_source": "aws_default_chain",
            }
        },
        credentials_resolved=False,
    )
    events = []

    class Runtime:
        async def mark_used(self, _resolved_handle):
            pytest.fail("rejected credentials must not be marked as used")

        async def close(self):
            events.append("close")

    async def resolve_credentials(*_args, **_kwargs):
        return Runtime(), handle

    def forbidden_process(*_args, **_kwargs):
        pytest.fail("chunking must not run with uncertified Bedrock credentials")

    monkeypatch.setattr(
        chunking_module,
        "load_server_configs",
        lambda: _rolling_config("bedrock"),
    )
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", forbidden_process)
    monkeypatch.setattr(TemplateProcessor, "process_template", forbidden_process)

    with pytest.raises(chunking_module.HTTPException) as exc_info:
        if kind == "template":
            await _invoke_rolling_template("uncertified-bedrock-template")
        else:
            await _invoke_rolling_endpoint(kind, f"uncertified-bedrock-{kind}")

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["error_code"] == "missing_provider_credentials"
    assert events == ["close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["file", "template"])
async def test_rolling_setup_failure_closes_every_resolved_runtime(
    monkeypatch,
    kind,
):
    events = []
    resolve_count = 0

    class Runtime:
        def __init__(self, label):
            self.label = label

        async def mark_used(self, resolved_handle):
            assert resolved_handle.label == self.label
            events.append(f"mark-{self.label}")

        async def close(self):
            events.append(f"close-{self.label}")

    async def resolve_credentials(*_args, **_kwargs):
        nonlocal resolve_count
        label = resolve_count
        resolve_count += 1
        return Runtime(label), SimpleNamespace(
            provider="openai",
            api_key=f"runtime-key-{label}",
            app_config={"openai_api": {"model": "summary-model"}},
            credentials_resolved=True,
            label=label,
        )

    config = _rolling_config()
    config["openai_api"]["max_tokens_for_summarization_step"] = "invalid"

    def process(*_args):
        return [
            {
                "text": "fallback chunk",
                "metadata": {
                    "method": "rolling_summarize",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    monkeypatch.setattr(chunking_module, "load_server_configs", lambda: config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)

    if kind == "file":
        with pytest.raises(chunking_module.HTTPException) as exc_info:
            await _invoke_rolling_endpoint("file", "invalid-setup-file")
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Chunking input or options are invalid."
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert resolve_count == 1
        assert events == ["close-0"]
        return

    with pytest.raises(chunking_module.HTTPException) as exc_info:
        await _invoke_rolling_template("invalid-setup-template")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Chunking input or options are invalid."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert resolve_count == 1
    assert events == ["close-0"]


@pytest.mark.asyncio
async def test_rolling_template_processor_setup_failure_closes_runtime_once(
    monkeypatch,
):
    events = []
    resolve_count = 0

    class Runtime:
        async def mark_used(self, _resolved_handle):
            pytest.fail("failed template setup must not mark credentials as used")

        async def close(self):
            events.append("close")

    async def resolve_credentials(*_args, **_kwargs):
        nonlocal resolve_count
        resolve_count += 1
        return Runtime(), SimpleNamespace(
            provider="openai",
            api_key="runtime-key",
            app_config={"openai_api": {"model": "summary-model"}},
            credentials_resolved=True,
        )

    def fail_processor_setup(*_args, **_kwargs):
        raise RuntimeError("sensitive processor setup failure")

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(TemplateProcessor, "__init__", fail_processor_setup)

    with pytest.raises(chunking_module.HTTPException) as exc_info:
        await _invoke_rolling_template("processor-setup-failure-template")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "An internal error occurred during text chunking"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert resolve_count == 1
    assert events == ["close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file", "template"])
async def test_concurrent_rolling_endpoints_keep_runtime_snapshots_isolated(
    monkeypatch,
    kind,
):
    entered = {label: threading.Event() for label in ("a", "b")}
    release = {label: threading.Event() for label in ("a", "b")}
    events = {label: [] for label in ("a", "b")}
    boundaries = {}
    expected_handles = {
        label: SimpleNamespace(
            provider="openai",
            api_key=f"key-{label}",
            app_config={
                "openai_api": {
                    "model": "summary-model",
                    "api_base_url": f"https://{label}.example/v1",
                }
            },
            credentials_resolved=True,
            label=label,
        )
        for label in ("a", "b")
    }
    handles = list(expected_handles.values())

    class Runtime:
        def __init__(self, label):
            self.label = label

        async def mark_used(self, resolved_handle):
            assert resolved_handle.label == self.label
            events[self.label].append("mark")

        async def close(self):
            events[self.label].append("close")

    async def resolve_credentials(*_args, **_kwargs):
        handle = handles.pop(0)
        return Runtime(handle.label), handle

    def record_boundary(text, llm_config):
        label = text.rsplit("-", 1)[-1]
        assert llm_config["api_key"] == f"key-{label}"
        assert llm_config["app_config"]["openai_api"]["api_base_url"] == (
            f"https://{label}.example/v1"
        )
        assert llm_config["credentials_resolved"] is True
        assert llm_config["provider_credentials"] is expected_handles[label]
        _record_provider_success(llm_config)
        boundaries[label] = llm_config
        entered[label].set()
        release[label].wait()
        return label

    def process(text, _options, _tokenizer, _llm_func, llm_config):
        label = record_boundary(text, llm_config)
        return [
            {
                "text": f"chunk-{label}",
                "metadata": {
                    "method": "rolling_summarize",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    def process_template(self, text, template, **options):
        del template, options
        label = record_boundary(text, self._chunker.llm_config)
        return [f"chunk-{label}"]

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(chunking_module, "_resolve_chunking_credentials", resolve_credentials)
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)

    async def invoke(label):
        text = f"concurrent-{label}"
        if kind == "template":
            return await _invoke_rolling_template(text)
        return await _invoke_rolling_endpoint(kind, text)

    first = asyncio.create_task(invoke("a"))
    second = None
    try:
        assert await asyncio.to_thread(entered["a"].wait, 1.0)
        second = asyncio.create_task(invoke("b"))
        assert await asyncio.to_thread(entered["b"].wait, 1.0)

        release["b"].set()
        second_result = await second
        assert second_result.chunks[0].text == "chunk-b"
        assert events["b"] == ["mark", "close"]
        assert events["a"] == []

        release["a"].set()
        first_result = await first
        assert first_result.chunks[0].text == "chunk-a"
    finally:
        for gate in release.values():
            gate.set()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )

    assert set(boundaries) == {"a", "b"}
    assert events == {
        "a": ["mark", "close"],
        "b": ["mark", "close"],
    }


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("kind", ["json", "file", "template"])
async def test_rolling_credential_dispatch_bypasses_saturated_default_executor(
    monkeypatch,
    kind,
):
    """Credential-bearing processing starts outside the default-executor queue."""
    loop = asyncio.get_running_loop()
    previous_default_executor = getattr(loop, "_default_executor", None)
    default_executor = ThreadPoolExecutor(max_workers=1)
    default_entered = threading.Event()
    default_release = threading.Event()
    provider_entered = threading.Event()
    provider_release = threading.Event()
    events: list[str] = []
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)
    handle = SimpleNamespace(
        provider="openai",
        api_key="runtime-key",
        app_config={"openai_api": {"model": "summary-model"}},
        credentials_resolved=True,
    )
    default_blocker = None
    task = None

    def block_default_executor() -> None:
        default_entered.set()
        assert default_release.wait(timeout=2.0)

    class Runtime:
        async def mark_used(self, resolved_handle):
            assert resolved_handle is handle
            events.append("mark")

        async def close(self):
            events.append("close")

    async def resolve_credentials(*_args, **_kwargs):
        nonlocal default_blocker
        default_blocker = loop.run_in_executor(None, block_default_executor)
        assert await _wait_for_thread_event(default_entered)
        return Runtime(), handle

    def run_provider(llm_config):
        events.append("provider-start")
        provider_entered.set()
        assert provider_release.wait(timeout=2.0)
        _record_provider_success(llm_config)
        events.append("provider-exit")

    def process(_text, _options, _tokenizer, _llm_func, llm_config):
        run_provider(llm_config)
        return [
            {
                "text": "summary chunk",
                "metadata": {
                    "method": "rolling_summarize",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    def process_template(self, text, template, **options):
        del text, template, options
        run_provider(self._chunker.llm_config)
        return ["template chunk"]

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(
        chunking_module,
        "_resolve_chunking_credentials",
        resolve_credentials,
    )
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)
    monkeypatch.setattr(
        chunking_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    loop.set_default_executor(default_executor)
    try:
        if kind == "template":
            task = asyncio.create_task(
                _invoke_rolling_template(f"default-saturated-{kind}")
            )
        else:
            task = asyncio.create_task(
                _invoke_rolling_endpoint(kind, f"default-saturated-{kind}")
            )

        assert await _wait_for_thread_event(default_entered)
        assert await _wait_for_thread_event(provider_entered, timeout=0.5)
        assert not default_release.is_set()
        assert pool.active_count == 1

        provider_release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result.chunks
    finally:
        provider_release.set()
        default_release.set()
        if default_blocker is not None:
            await asyncio.gather(default_blocker, return_exceptions=True)
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        replacement_executor = previous_default_executor or ThreadPoolExecutor()
        loop.set_default_executor(replacement_executor)
        default_executor.shutdown(wait=True, cancel_futures=True)

    assert events == ["provider-start", "provider-exit", "mark", "close"]
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("kind", ["json", "file", "template"])
async def test_rolling_pool_saturation_fails_closed_before_dispatch(
    monkeypatch,
    kind,
):
    """Excess rolling work never starts later or marks rejected credentials."""
    entered = threading.Event()
    release = threading.Event()
    starts: list[str] = []
    runtimes = []
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)

    class Runtime:
        def __init__(self, handle):
            self.handle = handle
            self.events: list[str] = []

        async def mark_used(self, resolved_handle):
            assert resolved_handle is self.handle
            self.events.append("mark")

        async def close(self):
            self.events.append("close")

    async def resolve_credentials(*_args, **_kwargs):
        index = len(runtimes)
        handle = SimpleNamespace(
            provider="openai",
            api_key=f"runtime-key-{index}",
            app_config={"openai_api": {"model": "summary-model"}},
            credentials_resolved=True,
        )
        runtime = Runtime(handle)
        runtimes.append(runtime)
        return runtime, handle

    def run_provider(text, llm_config):
        starts.append(text)
        if "admitted" in text:
            entered.set()
            assert release.wait(timeout=2.0)
        _record_provider_success(llm_config)

    def process(text, _options, _tokenizer, _llm_func, llm_config):
        run_provider(text, llm_config)
        return [
            {
                "text": "summary chunk",
                "metadata": {
                    "method": "rolling_summarize",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    def process_template(self, text, template, **options):
        del template, options
        run_provider(text, self._chunker.llm_config)
        return ["template chunk"]

    async def invoke(label):
        if kind == "template":
            return await _invoke_rolling_template(label)
        return await _invoke_rolling_endpoint(kind, label)

    monkeypatch.setattr(chunking_module, "load_server_configs", _rolling_config)
    monkeypatch.setattr(
        chunking_module,
        "_resolve_chunking_credentials",
        resolve_credentials,
    )
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)
    monkeypatch.setattr(
        chunking_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    admitted_label = f"admitted-{kind}"
    rejected_label = f"rejected-{kind}"
    admitted = asyncio.create_task(invoke(admitted_label))
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        pool_was_active = pool.active_count == 1
        with pytest.raises(chunking_module.HTTPException) as exc_info:
            await invoke(rejected_label)
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == _PROVIDER_CAPACITY_DETAIL
        assert "runtime-key-1" not in str(exc_info.value.detail)
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None
        assert starts == [admitted_label]
        assert len(runtimes) == 2
        assert runtimes[1].events == ["close"]
        assert pool_was_active
    finally:
        release.set()
        await asyncio.wait_for(admitted, timeout=1.0)

    await asyncio.sleep(0)
    assert starts == [admitted_label]
    assert runtimes[0].events == ["mark", "close"]
    assert runtimes[1].events == ["close"]
    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["json", "file", "template"])
async def test_non_llm_chunking_ignores_saturated_provider_pool(
    monkeypatch,
    kind,
):
    """Ordinary chunking remains isolated from provider adapter capacity."""

    blocker_entered = threading.Event()
    blocker_release = threading.Event()
    pool_released = threading.Event()
    starts: list[str] = []
    pool = bounded_daemon.BoundedDaemonPool(capacity=1)

    def occupy_provider_pool() -> None:
        blocker_entered.set()
        blocker_release.wait(timeout=2.0)

    async def reject_credential_resolution(*_args, **_kwargs):
        pytest.fail("non-LLM chunking must not resolve provider credentials")

    def process(text, *_args):
        starts.append(text)
        return [
            {
                "text": "ordinary chunk",
                "metadata": {
                    "method": "words",
                    "chunk_index": 1,
                    "total_chunks": 1,
                },
            }
        ]

    def process_template(self, text, template, **options):
        del self, template, options
        starts.append(text)
        return ["ordinary template chunk"]

    class NonLlmTemplateDatabase:
        def get_chunking_template(self, *, name):
            assert name == "ordinary-template"
            return {
                "name": name,
                "description": "ordinary template",
                "template_json": (
                    '{"chunking":{"method":"words",'
                    '"config":{"max_size":256,"overlap":0}}}'
                ),
                "tags": [],
                "version": 1,
            }

    async def invoke():
        text = f"ordinary-{kind}"
        if kind == "template":
            return await chunking_module.process_text_for_chunking_json(
                ChunkingTextRequest(
                    text_content=text,
                    file_name=f"{text}.txt",
                    options=ChunkingOptionsRequest(
                        template_name="ordinary-template"
                    ),
                ),
                http_request=_http_request(),
                current_user=SimpleNamespace(id=1),
                media_db=NonLlmTemplateDatabase(),
            )
        if kind == "json":
            return await chunking_module.process_text_for_chunking_json(
                ChunkingTextRequest(
                    text_content=text,
                    file_name=f"{text}.txt",
                    options=ChunkingOptionsRequest(
                        method="words",
                        max_size=256,
                        overlap=0,
                    ),
                ),
                http_request=_http_request(),
                current_user=SimpleNamespace(id=1),
                media_db=None,
            )

        upload = UploadFile(
            file=io.BytesIO(text.encode()),
            filename=f"{text}.txt",
        )
        return await chunking_module.process_file_for_chunking(
            http_request=_http_request(),
            file=upload,
            method="words",
            max_size=256,
            overlap=0,
            language="en",
            tokenizer_name_or_path="gpt2",
            code_mode=None,
            adaptive=False,
            multi_level=False,
            custom_chapter_pattern=None,
            semantic_similarity_threshold=0.7,
            semantic_overlap_sentences=2,
            json_chunkable_data_key="data",
            summarization_detail=0.5,
            llm_step_temperature=None,
            llm_step_system_prompt=None,
            llm_step_max_tokens=None,
            current_user=SimpleNamespace(id=1),
        )

    monkeypatch.setattr(
        chunking_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
    )
    monkeypatch.setattr(
        chunking_module,
        "_resolve_chunking_credentials",
        reject_credential_resolution,
    )
    monkeypatch.setattr(chunking_module, "improved_chunking_process", process)
    monkeypatch.setattr(TemplateProcessor, "process_template", process_template)

    pool.start(
        occupy_provider_pool,
        name="test-provider-pool-blocker",
        released_event=pool_released,
    )
    try:
        assert await _wait_for_thread_event(blocker_entered)
        assert pool.active_count == 1
        result = await asyncio.wait_for(invoke(), timeout=1.0)
        assert result.chunks
        assert pool.active_count == 1
    finally:
        blocker_release.set()
        assert await _wait_for_thread_event(pool_released)

    assert starts == [f"ordinary-{kind}"]
    assert pool.active_count == 0


@pytest.mark.parametrize("path", ["/chunk_text", "/chunk_file"])
def test_chunking_routes_document_provider_capacity_response(path):
    """Both public endpoint shapes advertise their retryable overload response."""

    route = next(
        route
        for route in chunking_module.chunking_router.routes
        if route.path == path
    )

    assert route.responses[503] == {
        "description": "Provider adapter capacity is temporarily exhausted."
    }
