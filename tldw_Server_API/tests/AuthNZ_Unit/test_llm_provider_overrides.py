import asyncio
import copy
import importlib
import multiprocessing
import threading
from concurrent.futures import Future

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ServerFallbackCredentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverride,
    LLMProviderOverridesRefreshError,
    apply_llm_provider_overrides_to_listing,
    capture_provider_override_call_snapshot,
    get_llm_provider_override,
    get_llm_provider_overrides_snapshot,
    get_override_credentials,
    get_override_model_priority,
    get_override_server_fallback,
    set_llm_provider_overrides_cache_for_tests,
    validate_provider_override,
)


class _BackgroundEventLoop:
    """Own a real event loop on another thread for reset-boundary tests."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.owner_thread_id: int | None = None
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._started.wait(timeout=5)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.owner_thread_id = threading.get_ident()
        self._started.set()
        try:
            self.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            if pending:
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self.loop.close()

    def call(self, callback) -> Future:
        result: Future = Future()

        def invoke() -> None:
            try:
                result.set_result(callback())
            except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
                result.set_exception(exc)

        self.loop.call_soon_threadsafe(invoke)
        return result

    def barrier(self, *, turns: int = 0) -> Future:
        result: Future = Future()

        def advance(remaining: int) -> None:
            if remaining:
                self.loop.call_soon(advance, remaining - 1)
            else:
                result.set_result(None)

        self.loop.call_soon_threadsafe(advance, turns)
        return result

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        assert not self._thread.is_alive()


def _provider_override_worker(connection, shared_store) -> None:
    """Run one isolated override-cache process for the multi-worker regression."""

    async def _run() -> None:
        module = importlib.import_module(
            "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
        )

        class FakeRepo:
            def __init__(self, _pool) -> None:
                pass

            async def ensure_tables(self) -> None:
                return None

            async def list_overrides(self):
                return [{"provider": "openai", "api_key": shared_store.api_key}]

        async def fake_get_pool():
            return object()

        module.AuthnzLLMProviderOverridesRepo = FakeRepo
        module.get_db_pool = fake_get_pool
        module._parse_override_row = lambda row: module.LLMProviderOverride(
            provider="openai",
            api_key=row["api_key"],
        )
        await module.refresh_llm_provider_overrides(pool=object())
        connection.send(("ready", module.get_override_server_fallback("openai").api_key))

        while True:
            command = await asyncio.to_thread(connection.recv)
            if command == "stop":
                await module.shutdown_llm_provider_override_recovery()
                connection.send(("stopped", None))
                return
            if command == "refresh":
                await module.refresh_llm_provider_overrides(pool=object())
                connection.send(
                    ("refreshed", module.get_override_server_fallback("openai").api_key)
                )
                continue
            if command == "get":
                connection.send(
                    ("value", module.get_override_server_fallback("openai").api_key)
                )
                continue
            if command == "expire-and-recover":
                with module._OVERRIDE_LOCK:
                    module._OVERRIDE_CACHE_REFRESHED_AT -= (
                        module._OVERRIDE_REFRESH_INTERVAL_SECONDS + 0.1
                    )
                stale = module.get_override_server_fallback("openai").api_key
                for _ in range(500):
                    await asyncio.sleep(0.01)
                    current = module.get_override_server_fallback("openai")
                    if current and current.api_key == shared_store.api_key:
                        connection.send(("recovered", (stale, current.api_key)))
                        break
                else:
                    connection.send(("error", "worker cache did not converge"))

    try:
        asyncio.run(_run())
    except BaseException as exc:
        try:
            connection.send(("error", type(exc).__name__))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise


def _fresh_override_cache_worker(connection) -> None:
    """Report whether a fresh process permits an unbootstrapped override read."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    try:
        module.get_llm_provider_overrides_snapshot()
    except ByokResolutionError as exc:
        connection.send(("blocked", exc.code))
    else:
        connection.send(("allowed", None))
    finally:
        connection.close()


@pytest.fixture(autouse=True)
def healthy_override_cache_between_tests():
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    with module._OVERRIDE_LOCK:
        original = copy.deepcopy(module._OVERRIDE_CACHE)
        original_healthy = module._OVERRIDE_CACHE_HEALTHY
        original_ttl_enabled = not module._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
    set_llm_provider_overrides_cache_for_tests({})
    try:
        yield
    finally:
        set_llm_provider_overrides_cache_for_tests(
            original,
            healthy=original_healthy,
            ttl_enabled=original_ttl_enabled,
        )


def test_fresh_process_fails_closed_before_override_bootstrap() -> None:
    """An unbootstrapped worker cannot treat an empty policy cache as valid."""
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_fresh_override_cache_worker,
        args=(child_connection,),
    )
    process.start()
    child_connection.close()
    try:
        assert parent_connection.poll(10)
        assert parent_connection.recv() == (
            "blocked",
            "credential_store_unavailable",
        )
    finally:
        parent_connection.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_provider_override_repr_redacts_decrypted_credentials() -> None:
    override = LLMProviderOverride(
        provider="openai",
        api_key="repr-secret-api-key",
        credential_fields={
            "refresh_token": "repr-secret-refresh-token",
            "base_url": "https://private-provider.example/v1",
        },
    )

    rendered = repr(override)

    assert rendered == "LLMProviderOverride(provider='openai', credentials=[REDACTED])"
    assert "repr-secret" not in rendered
    assert "private-provider.example" not in rendered


def _nested_override() -> LLMProviderOverride:
    return LLMProviderOverride(
        provider="openai",
        allowed_models=["gpt-a"],
        config={
            "default_model": "gpt-a",
            "routing": {
                "model_rankings": {"highest_quality": ["gpt-a"]},
            },
        },
        api_key="snapshot-key",
        credential_fields={
            "base_url": "https://snapshot.example/v1",
            "metadata": {"labels": ["snapshot-a"]},
        },
    )


def _assert_pristine_nested_override(override: LLMProviderOverride) -> None:
    assert override.allowed_models == ["gpt-a"]
    assert override.config["routing"]["model_rankings"]["highest_quality"] == [
        "gpt-a"
    ]
    assert override.credential_fields["metadata"]["labels"] == ["snapshot-a"]


def test_override_cache_publish_defensively_copies_nested_input() -> None:
    source = _nested_override()
    set_llm_provider_overrides_cache_for_tests({"openai": source})

    source.allowed_models.append("source-mutation")
    source.config["routing"]["model_rankings"]["highest_quality"].append(
        "source-mutation"
    )
    source.credential_fields["metadata"]["labels"].append("source-mutation")

    cached = get_llm_provider_override("openai")
    assert cached is not None
    _assert_pristine_nested_override(cached)


@pytest.mark.parametrize(
    "accessor",
    [
        lambda: get_llm_provider_override("openai"),
        lambda: get_llm_provider_overrides_snapshot()["openai"],
        lambda: capture_provider_override_call_snapshot("openai")._override,
    ],
    ids=("single", "mapping-snapshot", "call-snapshot"),
)
def test_override_accessors_return_transitively_isolated_snapshots(accessor) -> None:
    set_llm_provider_overrides_cache_for_tests({"openai": _nested_override()})
    exposed = accessor()
    assert exposed is not None

    exposed.allowed_models.append("accessor-mutation")
    exposed.config["routing"]["model_rankings"]["highest_quality"].append(
        "accessor-mutation"
    )
    exposed.credential_fields["metadata"]["labels"].append("accessor-mutation")

    fresh = get_llm_provider_override("openai")
    assert fresh is not None
    _assert_pristine_nested_override(fresh)


def test_apply_overrides_filters_models_and_status() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=False,
                allowed_models=["gpt-4o"],
                api_key_hint="abcd",
            )
        }
    )

    payload = {
        "providers": [
            {
                "name": "openai",
                "enabled": True,
                "models": ["gpt-4o", "gpt-3.5-turbo"],
                "models_info": [
                    {"name": "gpt-4o", "notes": "ok"},
                    {"name": "gpt-3.5-turbo", "notes": "legacy"},
                ],
            }
        ]
    }

    updated = apply_llm_provider_overrides_to_listing(payload)
    provider = updated["providers"][0]
    assert provider["enabled"] is False
    assert provider["models"] == ["gpt-4o"]
    assert provider["models_info"] == [{"name": "gpt-4o", "notes": "ok"}]

    set_llm_provider_overrides_cache_for_tests({})


def test_apply_overrides_does_not_expose_private_override_envelope() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=False,
                allowed_models=["gpt-4o"],
                config={
                    "default_model": "gpt-4o",
                    "private_canary": "hidden-config-value",
                },
                api_key="hidden-api-key",
                api_key_hint="hidden-api-hint",
                credential_fields={"org_id": "hidden-org-id"},
            )
        }
    )

    updated = apply_llm_provider_overrides_to_listing(
        {
            "providers": [
                {
                    "name": "openai",
                    "enabled": True,
                    "models": ["gpt-4o", "gpt-3.5-turbo"],
                }
            ]
        }
    )

    provider = updated["providers"][0]
    assert provider["enabled"] is False
    assert provider["models"] == ["gpt-4o"]
    assert provider["default_model"] == "gpt-4o"
    assert "override" not in provider
    rendered = repr(provider)
    for hidden in (
        "hidden-config-value",
        "hidden-api-key",
        "hidden-api-hint",
        "hidden-org-id",
    ):
        assert hidden not in rendered


def _capture_provider_override_warnings(module):
    messages: list[str] = []
    sink_id = module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    return messages, sink_id


def test_parse_override_row_sanitizes_secret_decrypt_warning(monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")

    def fail_decrypt(_payload):
        raise RuntimeError("decrypt failed at /private/provider-secret.key")

    monkeypatch.setattr(module, "loads_envelope", lambda _blob: {"ciphertext": "blob"})
    monkeypatch.setattr(module, "decrypt_byok_payload", fail_decrypt)
    opaque_payload = "opaque-" + "provider-payload"
    messages, sink_id = _capture_provider_override_warnings(module)

    try:
        override = module._parse_override_row(
            {"provider": "OpenAI", "secret_blob": opaque_payload}
        )
    finally:
        module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert override.api_key is None
    assert "Provider override decrypt failed" in joined
    assert "decrypt failed at" not in joined
    assert "/private/provider-secret.key" not in joined


def test_parse_override_row_uses_runtime_canonical_provider_identity() -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")

    override = module._parse_override_row({"provider": " OAI "})

    assert override.provider == "openai"


def test_override_accessor_resolves_runtime_provider_alias() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="canonical-key")}
    )

    fallback = get_override_server_fallback("OAI")

    assert fallback is not None
    assert fallback.api_key == "canonical-key"


async def test_refresh_provider_overrides_sanitizes_load_warning(monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    last_good = LLMProviderOverride(provider="openai", api_key="last-good-key")
    set_llm_provider_overrides_cache_for_tests({"openai": last_good})

    async def fail_get_pool():
        raise RuntimeError("provider override DB failed at /private/provider-overrides.db")

    monkeypatch.setattr(module, "get_db_pool", fail_get_pool)
    messages, sink_id = _capture_provider_override_warnings(module)

    try:
        with pytest.raises(LLMProviderOverridesRefreshError) as exc_info:
            await module.refresh_llm_provider_overrides()
    finally:
        module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert str(exc_info.value) == "Provider credential storage is temporarily unavailable."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    with module._OVERRIDE_LOCK:
        assert dict(module._OVERRIDE_CACHE) == {"openai": last_good}
    with pytest.raises(ByokResolutionError) as exc_info:
        get_override_server_fallback("openai")
    assert exc_info.value.code == "credential_store_unavailable"
    assert "Failed to load provider overrides" in joined
    assert "provider override DB failed" not in joined
    assert "/private/provider-overrides.db" not in joined
    set_llm_provider_overrides_cache_for_tests({})


async def test_refresh_provider_overrides_sanitizes_row_parse_warning(monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    last_good = LLMProviderOverride(provider="openai", api_key="last-good-key")
    set_llm_provider_overrides_cache_for_tests({"openai": last_good})

    class FakeRepo:
        def __init__(self, _pool):
            pass

        async def ensure_tables(self):
            return None

        async def list_overrides(self):
            return [{"provider": "openai"}]

    def fail_parse(_row):
        raise RuntimeError("provider override row failed at /private/provider-row.json")

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(module, "_parse_override_row", fail_parse)
    messages, sink_id = _capture_provider_override_warnings(module)

    try:
        with pytest.raises(LLMProviderOverridesRefreshError) as exc_info:
            await module.refresh_llm_provider_overrides(pool=object())
    finally:
        module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert str(exc_info.value) == "Provider credential storage is temporarily unavailable."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    with module._OVERRIDE_LOCK:
        assert dict(module._OVERRIDE_CACHE) == {"openai": last_good}
    with pytest.raises(ByokResolutionError) as exc_info:
        get_override_server_fallback("openai")
    assert exc_info.value.code == "credential_store_unavailable"
    assert "Failed to parse provider override row" in joined
    assert "provider override row failed" not in joined
    assert "/private/provider-row.json" not in joined
    set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.concurrent
async def test_concurrent_steady_refreshes_never_run_schema_ddl(monkeypatch) -> None:
    """Periodic/select-only refreshes must not take PostgreSQL DDL locks."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    ddl_calls = 0
    select_calls = 0

    class FakePostgresPool:
        pool = object()

    class FakeRepo:
        def __init__(self, pool) -> None:
            assert isinstance(pool, FakePostgresPool)

        async def ensure_tables(self) -> None:
            nonlocal ddl_calls
            ddl_calls += 1

        async def list_overrides(self):
            nonlocal select_calls
            select_calls += 1
            await asyncio.sleep(0)
            return []

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    pool = FakePostgresPool()

    await asyncio.gather(
        *(module.refresh_llm_provider_overrides(pool=pool, force=True) for _ in range(4))
    )

    assert select_calls == 4
    assert ddl_calls == 0


async def test_refresh_prefers_canonical_row_over_legacy_alias_deterministically(
    monkeypatch,
) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")

    class FakeRepo:
        def __init__(self, _pool):
            pass

        async def ensure_tables(self):
            return None

        async def list_overrides(self):
            return [
                {
                    "provider": "oai",
                    "config_json": {"default_model": "legacy-model"},
                },
                {
                    "provider": "openai",
                    "config_json": {"default_model": "canonical-model"},
                },
            ]

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)

    refreshed = await module.refresh_llm_provider_overrides(pool=object())

    assert list(refreshed) == ["openai"]
    assert refreshed["openai"].config["default_model"] == "canonical-model"


async def test_refresh_fails_closed_for_ambiguous_legacy_alias_rows(monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")

    class FakeRepo:
        def __init__(self, _pool):
            pass

        async def ensure_tables(self):
            return None

        async def list_overrides(self):
            return [
                {"provider": "aws-bedrock"},
                {"provider": "amazon-bedrock"},
            ]

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    set_llm_provider_overrides_cache_for_tests(
        {"bedrock": LLMProviderOverride(provider="bedrock", api_key="last-good")}
    )

    with pytest.raises(LLMProviderOverridesRefreshError):
        await module.refresh_llm_provider_overrides(pool=object())
    with pytest.raises(ByokResolutionError) as exc_info:
        get_override_server_fallback("bedrock")
    assert exc_info.value.code == "credential_store_unavailable"


async def test_successful_refresh_recovers_override_fallback_after_store_failure(monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    rows_or_error: list[dict[str, object]] | Exception = RuntimeError("store unavailable")

    class FakeRepo:
        def __init__(self, _pool):
            pass

        async def ensure_tables(self):
            return None

        async def list_overrides(self):
            if isinstance(rows_or_error, Exception):
                raise rows_or_error
            return rows_or_error

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")}
    )

    with pytest.raises(LLMProviderOverridesRefreshError):
        await module.refresh_llm_provider_overrides(pool=object())
    with pytest.raises(ByokResolutionError) as exc_info:
        get_override_server_fallback("openai")
    assert exc_info.value.code == "credential_store_unavailable"

    rows_or_error = [
        {
            "provider": "openai",
            "config_json": {"auth_source": "api_key"},
        }
    ]
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda _row: LLMProviderOverride(provider="openai", api_key="recovered-key"),
    )

    refreshed = await module.refresh_llm_provider_overrides(pool=object())

    assert refreshed["openai"].api_key == "recovered-key"
    fallback = get_override_server_fallback("openai")
    assert fallback is not None
    assert fallback.api_key == "recovered-key"
    set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.parametrize("older_fails", [False, True], ids=("stale_rows", "late_failure"))
async def test_overlapping_refreshes_are_serialized_before_publication(
    monkeypatch,
    older_fails,
) -> None:
    """Overlapping reads publish in lock order and finish with the newest snapshot."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    started = {name: asyncio.Event() for name in ("older", "newer")}
    release = {name: asyncio.Event() for name in ("older", "newer")}

    class Pool:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeRepo:
        def __init__(self, pool: Pool) -> None:
            self.name = pool.name

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            started[self.name].set()
            await asyncio.wait_for(release[self.name].wait(), timeout=5)
            if self.name == "older" and older_fails:
                raise RuntimeError("late stale failure at /private/old.db")
            return [{"provider": "openai", "api_key": f"{self.name}-key"}]

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda row: LLMProviderOverride(provider="openai", api_key=row["api_key"]),
    )
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")}
    )

    older_task = asyncio.create_task(module.refresh_llm_provider_overrides(Pool("older")))
    await asyncio.wait_for(started["older"].wait(), timeout=5)
    newer_task = asyncio.create_task(module.refresh_llm_provider_overrides(Pool("newer")))
    await asyncio.sleep(0)
    assert started["newer"].is_set() is False

    release["older"].set()
    if older_fails:
        with pytest.raises(LLMProviderOverridesRefreshError):
            await older_task
    else:
        older_result = await older_task
        assert older_result["openai"].api_key == "older-key"

    await asyncio.wait_for(started["newer"].wait(), timeout=5)
    release["newer"].set()
    newer_result = await newer_task
    assert newer_result["openai"].api_key == "newer-key"
    fallback = get_override_server_fallback("openai")
    assert fallback is not None
    assert fallback.api_key == "newer-key"


async def test_older_healthy_refresh_does_not_fail_while_newer_refresh_is_pending(
    monkeypatch,
) -> None:
    """Normal overlap must not turn a successful store read into a false outage."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    older_started = asyncio.Event()
    older_release = asyncio.Event()
    newer_started = asyncio.Event()
    newer_release = asyncio.Event()

    class Pool:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeRepo:
        def __init__(self, pool: Pool) -> None:
            self.name = pool.name

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            if self.name == "older":
                older_started.set()
                await asyncio.wait_for(older_release.wait(), timeout=5)
            else:
                newer_started.set()
                await asyncio.wait_for(newer_release.wait(), timeout=5)
            return [{"provider": "openai", "api_key": f"{self.name}-key"}]

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda row: LLMProviderOverride(provider="openai", api_key=row["api_key"]),
    )

    older_task = asyncio.create_task(module.refresh_llm_provider_overrides(Pool("older")))
    await asyncio.wait_for(older_started.wait(), timeout=5)
    newer_task = asyncio.create_task(module.refresh_llm_provider_overrides(Pool("newer")))

    # The old implementation reserves a generation before I/O. Wait until the
    # overlap is observable without requiring the fixed implementation to start
    # the serialized second database read prematurely.
    for _ in range(100):
        if module._OVERRIDE_REFRESH_GENERATION >= module._OVERRIDE_COMPLETED_GENERATION + 2:
            break
        await asyncio.sleep(0)

    older_release.set()
    older_result = await asyncio.wait_for(older_task, timeout=5)
    assert older_result["openai"].api_key == "older-key"

    await asyncio.wait_for(newer_started.wait(), timeout=5)
    newer_release.set()
    newer_result = await asyncio.wait_for(newer_task, timeout=5)
    assert newer_result["openai"].api_key == "newer-key"
    assert get_override_server_fallback("openai").api_key == "newer-key"


async def test_newer_serialized_refresh_failure_keeps_last_good_snapshot_closed(
    monkeypatch,
) -> None:
    """A later failed read preserves prior data internally but makes it unusable."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    older_started = asyncio.Event()
    older_release = asyncio.Event()

    class Pool:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeRepo:
        def __init__(self, pool: Pool) -> None:
            self.name = pool.name

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            if self.name == "newer":
                raise RuntimeError("newest store failure at /private/new.db")
            older_started.set()
            await asyncio.wait_for(older_release.wait(), timeout=5)
            return [{"provider": "openai", "api_key": "stale-key"}]

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda row: LLMProviderOverride(provider="openai", api_key=row["api_key"]),
    )
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")}
    )

    older_task = asyncio.create_task(module.refresh_llm_provider_overrides(Pool("older")))
    await asyncio.wait_for(older_started.wait(), timeout=5)
    older_release.set()
    older_result = await older_task
    assert older_result["openai"].api_key == "stale-key"

    with pytest.raises(LLMProviderOverridesRefreshError):
        await module.refresh_llm_provider_overrides(Pool("newer"))

    with module._OVERRIDE_LOCK:
        assert module._OVERRIDE_CACHE["openai"].api_key == "stale-key"
    with pytest.raises(ByokResolutionError) as exc_info:
        get_override_server_fallback("openai")
    assert exc_info.value.code == "credential_store_unavailable"


async def test_unhealthy_public_resolutions_schedule_one_recovery_refresh(
    monkeypatch,
) -> None:
    """Concurrent fail-closed reads trigger one demand-driven recovery attempt."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    mode = "failed"
    recovery_loaded = asyncio.Event()
    list_calls = 0

    class FakeRepo:
        def __init__(self, _pool) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            nonlocal list_calls
            list_calls += 1
            if mode == "failed":
                raise RuntimeError("transient store failure at /private/store.db")
            recovery_loaded.set()
            return [{"provider": "openai", "api_key": "recovered-key"}]

    async def fake_get_pool():
        return object()

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(module, "get_db_pool", fake_get_pool)
    monkeypatch.setattr(module, "_OVERRIDE_RECOVERY_BACKOFF_INITIAL_SECONDS", 0.0)
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda row: LLMProviderOverride(provider="openai", api_key=row["api_key"]),
    )
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")}
    )

    with pytest.raises(LLMProviderOverridesRefreshError):
        await module.refresh_llm_provider_overrides(pool=object())
    mode = "healthy"
    for _ in range(8):
        with pytest.raises(ByokResolutionError):
            get_override_server_fallback("openai")

    await asyncio.wait_for(recovery_loaded.wait(), timeout=5)
    for _ in range(20):
        await asyncio.sleep(0)
        try:
            fallback = get_override_server_fallback("openai")
        except ByokResolutionError:
            continue
        break
    else:
        pytest.fail("automatic provider override recovery did not publish its snapshot")

    assert list_calls == 2
    assert fallback is not None
    assert fallback.api_key == "recovered-key"


async def test_soft_stale_snapshot_refreshes_in_background_without_serving_gap(
    monkeypatch,
) -> None:
    """Healthy traffic refreshes worker-local caches before the hard deadline."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    refreshed = asyncio.Event()

    class FakeRepo:
        def __init__(self, _pool) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            refreshed.set()
            return [{"provider": "openai", "api_key": "rotated-key"}]

    async def fake_get_pool():
        return object()

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(module, "get_db_pool", fake_get_pool)
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda row: LLMProviderOverride(provider="openai", api_key=row["api_key"]),
    )
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="old-key")},
        ttl_enabled=True,
    )
    with module._OVERRIDE_LOCK:
        module._OVERRIDE_CACHE_REFRESHED_AT = (
            module.time.monotonic() - module._OVERRIDE_REFRESH_INTERVAL_SECONDS - 0.1
        )

    first = get_override_server_fallback("openai")
    assert first is not None
    assert first.api_key == "old-key"
    await asyncio.wait_for(refreshed.wait(), timeout=5)

    for _ in range(100):
        await asyncio.sleep(0)
        current = get_override_server_fallback("openai")
        if current and current.api_key == "rotated-key":
            break
    else:
        pytest.fail("soft-stale provider override cache did not refresh")


async def test_periodic_worker_refresh_propagates_rotation_without_request_traffic(
    monkeypatch,
) -> None:
    """Each production worker converges even when no request touches its cache."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    api_key = "key-a"
    rotated_loaded = asyncio.Event()

    class FakeRepo:
        def __init__(self, _pool) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            if api_key == "key-b":
                rotated_loaded.set()
            return [{"provider": "openai", "api_key": api_key}]

    async def fake_get_pool():
        return object()

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(module, "get_db_pool", fake_get_pool)
    monkeypatch.setattr(module, "_OVERRIDE_REFRESH_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda row: LLMProviderOverride(provider="openai", api_key=row["api_key"]),
    )

    await module.refresh_llm_provider_overrides(pool=object())
    module.start_llm_provider_override_refresh_service()
    api_key = "key-b"
    try:
        await asyncio.wait_for(rotated_loaded.wait(), timeout=5)
        for _ in range(100):
            await asyncio.sleep(0)
            current = get_override_server_fallback("openai")
            if current and current.api_key == "key-b":
                break
        else:
            pytest.fail("periodic worker refresh did not publish the rotated key")
    finally:
        await module.shutdown_llm_provider_override_recovery()

    assert module._OVERRIDE_REFRESH_SERVICE_TASK is None


async def test_periodic_worker_refresh_respects_store_failure_backoff(monkeypatch) -> None:
    """A durable store outage must not trigger a warning/query storm per worker."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    refreshed = asyncio.Event()
    calls = 0

    async def fake_refresh(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        refreshed.set()
        return {}

    monkeypatch.setattr(module, "refresh_llm_provider_overrides", fake_refresh)
    monkeypatch.setattr(module, "_OVERRIDE_REFRESH_INTERVAL_SECONDS", 0.01)
    with module._OVERRIDE_LOCK:
        module._OVERRIDE_RECOVERY_NEXT_RETRY_AT = module.time.monotonic() + 1.0

    module.start_llm_provider_override_refresh_service()
    try:
        await asyncio.sleep(0.05)
        assert calls == 0

        with module._OVERRIDE_LOCK:
            module._OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0
        await asyncio.wait_for(refreshed.wait(), timeout=1)
    finally:
        await module.shutdown_llm_provider_override_recovery()

    assert calls == 1


async def test_hard_stale_snapshot_fails_closed_until_reverified(monkeypatch) -> None:
    """A worker cannot serve credentials beyond the hard verification deadline."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    release = asyncio.Event()

    class FakeRepo:
        def __init__(self, _pool) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            await release.wait()
            return [{"provider": "openai", "api_key": "rotated-key"}]

    async def fake_get_pool():
        return object()

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(module, "get_db_pool", fake_get_pool)
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda row: LLMProviderOverride(provider="openai", api_key=row["api_key"]),
    )
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="old-key")},
        ttl_enabled=True,
    )
    with module._OVERRIDE_LOCK:
        module._OVERRIDE_CACHE_REFRESHED_AT = (
            module.time.monotonic() - module._OVERRIDE_MAX_STALE_SECONDS - 0.1
        )

    with pytest.raises(ByokResolutionError) as exc_info:
        get_override_server_fallback("openai")
    assert exc_info.value.code == "credential_store_unavailable"

    release.set()
    for _ in range(100):
        await asyncio.sleep(0)
        try:
            current = get_override_server_fallback("openai")
        except ByokResolutionError:
            continue
        assert current is not None
        assert current.api_key == "rotated-key"
        break
    else:
        pytest.fail("hard-stale provider override cache did not recover")


def test_two_worker_caches_converge_after_shared_store_rotation() -> None:
    """Independent production workers re-read the durable store after soft TTL."""
    context = multiprocessing.get_context("spawn")
    manager = context.Manager()
    shared_store = manager.Namespace()
    shared_store.api_key = "key-a"
    workers = []
    connections = []

    try:
        for _ in range(2):
            parent_connection, child_connection = context.Pipe()
            process = context.Process(
                target=_provider_override_worker,
                args=(child_connection, shared_store),
            )
            process.start()
            workers.append(process)
            connections.append(parent_connection)

        assert [connection.poll(20) for connection in connections] == [True, True]
        assert [connection.recv() for connection in connections] == [
            ("ready", "key-a"),
            ("ready", "key-a"),
        ]

        shared_store.api_key = "key-b"
        connections[0].send("refresh")
        assert connections[0].poll(10)
        assert connections[0].recv() == ("refreshed", "key-b")

        connections[1].send("get")
        assert connections[1].poll(10)
        assert connections[1].recv() == ("value", "key-a")

        connections[1].send("expire-and-recover")
        assert connections[1].poll(10)
        assert connections[1].recv() == ("recovered", ("key-a", "key-b"))
    finally:
        for connection in connections:
            try:
                connection.send("stop")
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        manager.shutdown()


async def test_cancelling_active_refresh_marks_last_good_snapshot_unhealthy(
    monkeypatch,
) -> None:
    """Cancellation cannot leave stale credentials marked safe to serve."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    started = asyncio.Event()

    class FakeRepo:
        def __init__(self, _pool) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")}
    )

    task = asyncio.create_task(module.refresh_llm_provider_overrides(pool=object()))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(ByokResolutionError) as exc_info:
        get_override_server_fallback("openai")
    assert exc_info.value.code == "credential_store_unavailable"


async def test_timed_out_recovery_releases_singleflight_and_allows_retry(
    monkeypatch,
) -> None:
    """A hung recovery attempt must time out so a later healthy read can recover."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    mode = "hung"
    first_started = asyncio.Event()
    recovered = asyncio.Event()

    class FakeRepo:
        def __init__(self, _pool) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            if mode == "hung":
                first_started.set()
                await asyncio.Event().wait()
            recovered.set()
            return [{"provider": "openai", "api_key": "recovered-key"}]

    async def fake_get_pool():
        return object()

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(module, "get_db_pool", fake_get_pool)
    monkeypatch.setattr(module, "_OVERRIDE_REFRESH_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(module, "_OVERRIDE_RECOVERY_BACKOFF_INITIAL_SECONDS", 0.0)
    monkeypatch.setattr(
        module,
        "_parse_override_row",
        lambda row: LLMProviderOverride(provider="openai", api_key=row["api_key"]),
    )
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="last-good-key")},
        healthy=False,
    )
    module._OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0

    with pytest.raises(ByokResolutionError):
        get_override_server_fallback("openai")
    await asyncio.wait_for(first_started.wait(), timeout=5)

    for _ in range(100):
        await asyncio.sleep(0.01)
        if not module._OVERRIDE_RECOVERY_IN_FLIGHT:
            break
    assert module._OVERRIDE_RECOVERY_IN_FLIGHT is False

    mode = "healthy"
    module._OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0
    with pytest.raises(ByokResolutionError):
        get_override_server_fallback("openai")
    await asyncio.wait_for(recovered.wait(), timeout=5)

    for _ in range(100):
        await asyncio.sleep(0)
        try:
            fallback = get_override_server_fallback("openai")
        except ByokResolutionError:
            continue
        assert fallback is not None
        assert fallback.api_key == "recovered-key"
        break
    else:
        pytest.fail("provider override recovery never published the healthy retry")


async def test_shutdown_cancels_owned_recovery_task(monkeypatch) -> None:
    """The process lifecycle owns and drains the recovery task."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")
    started = asyncio.Event()

    async def hung_refresh(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(module, "refresh_llm_provider_overrides", hung_refresh)
    set_llm_provider_overrides_cache_for_tests({}, healthy=False)
    module._OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0

    with pytest.raises(ByokResolutionError):
        get_override_server_fallback("openai")
    await asyncio.wait_for(started.wait(), timeout=5)
    await module.shutdown_llm_provider_override_recovery()

    assert module._OVERRIDE_RECOVERY_IN_FLIGHT is False
    assert module._OVERRIDE_RECOVERY_TASK is None


async def test_shutdown_drains_same_loop_retired_task() -> None:
    """Shutdown includes delayed retired work, not only active worker slots."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    cancellation_observed = asyncio.Event()
    release_cleanup = asyncio.Event()
    task_started = asyncio.Event()

    async def retired_cleanup() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await release_cleanup.wait()
            raise

    retired_task = asyncio.create_task(
        retired_cleanup(),
        name="shutdown-retired-same-loop",
    )
    await task_started.wait()
    with module._OVERRIDE_LOCK:
        module._OVERRIDE_RETIRED_TASKS.add(retired_task)

    shutdown_task = asyncio.create_task(
        module.shutdown_llm_provider_override_recovery()
    )
    try:
        await asyncio.wait_for(cancellation_observed.wait(), timeout=5)
        assert not shutdown_task.done()

        release_cleanup.set()
        await asyncio.wait_for(shutdown_task, timeout=5)
        assert retired_task.done()
        assert retired_task.cancelled()
        with module._OVERRIDE_LOCK:
            assert retired_task not in module._OVERRIDE_RETIRED_TASKS
    finally:
        release_cleanup.set()
        if not retired_task.done():
            retired_task.cancel()
        await asyncio.gather(retired_task, shutdown_task, return_exceptions=True)
        with module._OVERRIDE_LOCK:
            module._OVERRIDE_RETIRED_TASKS.discard(retired_task)


async def test_shutdown_times_out_without_abandoning_same_loop_retired_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation-suppressing cleanup remains owned after bounded shutdown."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    cancellation_observed = asyncio.Event()
    release_cleanup = asyncio.Event()
    task_started = asyncio.Event()

    async def non_cooperative_cleanup() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await release_cleanup.wait()
            raise

    retired_task = asyncio.create_task(
        non_cooperative_cleanup(),
        name="shutdown-timeout-retired-same-loop",
    )
    await task_started.wait()
    monkeypatch.setattr(module, "_OVERRIDE_REFRESH_TIMEOUT_SECONDS", 0.05)
    with module._OVERRIDE_LOCK:
        module._OVERRIDE_RETIRED_TASKS.add(retired_task)

    shutdown_task = asyncio.create_task(
        module.shutdown_llm_provider_override_recovery()
    )
    try:
        done, pending = await asyncio.wait({shutdown_task}, timeout=0.25)
        assert done == {shutdown_task}
        assert pending == set()
        with pytest.raises(RuntimeError, match="Timed out draining"):
            await shutdown_task
        assert cancellation_observed.is_set()
        assert not retired_task.done()
        with module._OVERRIDE_LOCK:
            assert retired_task in module._OVERRIDE_RETIRED_TASKS

        release_cleanup.set()
        await asyncio.gather(retired_task, return_exceptions=True)
        await asyncio.sleep(0)
        with module._OVERRIDE_LOCK:
            assert retired_task not in module._OVERRIDE_RETIRED_TASKS
    finally:
        release_cleanup.set()
        if not retired_task.done():
            retired_task.cancel()
        await asyncio.gather(retired_task, shutdown_task, return_exceptions=True)
        with module._OVERRIDE_LOCK:
            module._OVERRIDE_RETIRED_TASKS.discard(retired_task)


async def test_shutdown_cancels_foreign_active_task_on_owner_loop() -> None:
    """Shutdown routes foreign cancellation to the task's owning event loop."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    background = _BackgroundEventLoop()
    cancel_called = threading.Event()
    cancellation_observed = threading.Event()
    cancel_thread_ids: list[int] = []
    release_cleanup = background.call(asyncio.Event).result(timeout=5)
    task_started = threading.Event()

    class RecordingTask(asyncio.Task):
        def cancel(self, msg=None) -> bool:
            caller_thread_id = threading.get_ident()
            cancel_thread_ids.append(caller_thread_id)
            cancel_called.set()
            if caller_thread_id != background.owner_thread_id:
                return False
            return super().cancel(msg)

    async def foreign_cleanup() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await release_cleanup.wait()
            raise

    foreign_task = background.call(
        lambda: RecordingTask(
            foreign_cleanup(),
            loop=background.loop,
            name="shutdown-active-foreign-loop",
        )
    ).result(timeout=5)
    assert task_started.wait(timeout=5)
    with module._OVERRIDE_LOCK:
        module._OVERRIDE_REFRESH_SERVICE_TASK = foreign_task

    shutdown_task = asyncio.create_task(
        module.shutdown_llm_provider_override_recovery()
    )
    try:
        assert await asyncio.to_thread(cancel_called.wait, 5)
        assert cancel_thread_ids == [background.owner_thread_id]
        assert await asyncio.to_thread(cancellation_observed.wait, 5)
        assert not shutdown_task.done()

        background.loop.call_soon_threadsafe(release_cleanup.set)
        await asyncio.wait_for(shutdown_task, timeout=5)
        assert foreign_task.done()
        assert foreign_task.cancelled()
        with module._OVERRIDE_LOCK:
            assert module._OVERRIDE_REFRESH_SERVICE_TASK is None
            assert foreign_task not in module._OVERRIDE_RETIRED_TASKS
    finally:
        background.loop.call_soon_threadsafe(release_cleanup.set)
        background.loop.call_soon_threadsafe(foreign_task.cancel)
        await asyncio.gather(shutdown_task, return_exceptions=True)
        background.barrier(turns=3).result(timeout=5)
        with module._OVERRIDE_LOCK:
            if module._OVERRIDE_REFRESH_SERVICE_TASK is foreign_task:
                module._OVERRIDE_REFRESH_SERVICE_TASK = None
            module._OVERRIDE_RETIRED_TASKS.discard(foreign_task)
        background.close()


@pytest.mark.parametrize("worker", ["recovery", "periodic"])
async def test_worker_that_loses_registration_is_retired_until_completion(
    worker: str,
) -> None:
    """A task rejected after creation remains owned until cancellation completes."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    loop = asyncio.get_running_loop()
    original_create_task = loop.create_task
    created_tasks: list[asyncio.Task] = []

    def create_task_across_reset(coro, *, name=None, **kwargs):
        task = original_create_task(coro, name=name, **kwargs)
        created_tasks.append(task)
        set_llm_provider_overrides_cache_for_tests({})
        return task

    loop.create_task = create_task_across_reset
    try:
        if worker == "recovery":
            with module._OVERRIDE_LOCK:
                module._OVERRIDE_CACHE_HEALTHY = False
                module._OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0
            module._schedule_override_recovery()
        else:
            module.start_llm_provider_override_refresh_service()
    finally:
        loop.create_task = original_create_task

    assert len(created_tasks) == 1
    losing_task = created_tasks[0]
    try:
        with module._OVERRIDE_LOCK:
            assert module._OVERRIDE_RECOVERY_TASK is None
            assert module._OVERRIDE_REFRESH_SERVICE_TASK is None
            assert losing_task in module._OVERRIDE_RETIRED_TASKS

        await asyncio.gather(losing_task, return_exceptions=True)
        await asyncio.sleep(0)
        with module._OVERRIDE_LOCK:
            assert losing_task not in module._OVERRIDE_RETIRED_TASKS
    finally:
        if not losing_task.done():
            losing_task.cancel()
        await asyncio.gather(losing_task, return_exceptions=True)
        with module._OVERRIDE_LOCK:
            module._OVERRIDE_RETIRED_TASKS.discard(losing_task)


@pytest.mark.parametrize(
    "task_attribute",
    ["_OVERRIDE_RECOVERY_TASK", "_OVERRIDE_REFRESH_SERVICE_TASK"],
    ids=("recovery", "periodic"),
)
def test_test_cache_reset_schedules_task_cancellation_on_its_owning_loop(
    task_attribute: str,
) -> None:
    """A foreign sync reset waits until its owner loop has drained the task."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    background = _BackgroundEventLoop()
    cancel_called = threading.Event()
    cancellation_observed = threading.Event()
    cancel_thread_ids: list[int] = []
    release_drain = background.call(asyncio.Event).result(timeout=5)
    reset_errors: list[RuntimeError] = []
    reset_finished = threading.Event()
    task_started = threading.Event()

    class RecordingTask(asyncio.Task):
        def cancel(self, msg=None) -> bool:
            caller_thread_id = threading.get_ident()
            cancel_thread_ids.append(caller_thread_id)
            cancel_called.set()
            if caller_thread_id != background.owner_thread_id:
                return False
            return super().cancel(msg)

    async def wait_forever() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await release_drain.wait()
            raise

    task = background.call(
        lambda: RecordingTask(
            wait_forever(),
            loop=background.loop,
            name=f"provider-overrides-{task_attribute}",
        )
    ).result(timeout=5)
    assert task_started.wait(timeout=5)
    with module._OVERRIDE_LOCK:
        setattr(module, task_attribute, task)
        if task_attribute == "_OVERRIDE_RECOVERY_TASK":
            module._OVERRIDE_RECOVERY_IN_FLIGHT = True

    def reset_cache() -> None:
        try:
            set_llm_provider_overrides_cache_for_tests({})
        except RuntimeError as exc:
            reset_errors.append(exc)
        finally:
            reset_finished.set()

    reset_thread = threading.Thread(target=reset_cache)
    try:
        caller_thread_id = threading.get_ident()
        reset_thread.start()
        assert cancel_called.wait(timeout=5)
        assert cancellation_observed.wait(timeout=5)

        assert background.owner_thread_id != caller_thread_id
        assert cancel_thread_ids == [background.owner_thread_id]
        assert reset_thread.is_alive()
        assert not reset_finished.is_set()

        background.loop.call_soon_threadsafe(release_drain.set)
        reset_thread.join(timeout=5)
        assert not reset_thread.is_alive()
        assert reset_errors == []
        assert task.done()
        assert task.cancelled()
    finally:
        background.loop.call_soon_threadsafe(release_drain.set)
        reset_thread.join(timeout=5)
        with module._OVERRIDE_LOCK:
            if getattr(module, task_attribute) is task:
                setattr(module, task_attribute, None)
            module._OVERRIDE_RECOVERY_IN_FLIGHT = False
        background.loop.call_soon_threadsafe(task.cancel)
        background.barrier().result(timeout=5)
        background.close()


@pytest.mark.parametrize(
    "task_attribute",
    ["_OVERRIDE_RECOVERY_TASK", "_OVERRIDE_REFRESH_SERVICE_TASK"],
    ids=("recovery", "periodic"),
)
async def test_test_cache_reset_on_owner_loop_retains_delayed_task_without_wrapper(
    task_attribute: str,
) -> None:
    """Same-loop reset retains the original task until cancellation cleanup ends."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    cancellation_observed = asyncio.Event()
    release_cleanup = asyncio.Event()
    task_started = asyncio.Event()

    async def delayed_cancellation_cleanup() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await release_cleanup.wait()
            raise

    task = asyncio.create_task(
        delayed_cancellation_cleanup(),
        name=f"same-loop-{task_attribute}",
    )
    await task_started.wait()
    with module._OVERRIDE_LOCK:
        setattr(module, task_attribute, task)
        if task_attribute == "_OVERRIDE_RECOVERY_TASK":
            module._OVERRIDE_RECOVERY_IN_FLIGHT = True

    try:
        set_llm_provider_overrides_cache_for_tests({})

        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is None
            assert task in module._OVERRIDE_RETIRED_TASKS
        assert not task.done()
        assert not any(
            candidate.get_name() == "llm-provider-overrides-reset-drain"
            for candidate in asyncio.all_tasks()
        )

        await cancellation_observed.wait()
        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is None
            assert task in module._OVERRIDE_RETIRED_TASKS

        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is None
            assert task not in module._OVERRIDE_RETIRED_TASKS
    finally:
        release_cleanup.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        reset_drains = [
            candidate
            for candidate in asyncio.all_tasks()
            if candidate is not asyncio.current_task()
            and candidate.get_name() == "llm-provider-overrides-reset-drain"
        ]
        if reset_drains:
            await asyncio.gather(*reset_drains, return_exceptions=True)
        with module._OVERRIDE_LOCK:
            if getattr(module, task_attribute) is task:
                setattr(module, task_attribute, None)
            retired_tasks = getattr(module, "_OVERRIDE_RETIRED_TASKS", None)
            if retired_tasks is not None:
                retired_tasks.discard(task)
            module._OVERRIDE_RECOVERY_IN_FLIGHT = False


async def test_same_loop_reset_allows_immediate_periodic_service_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retired old service cannot occupy or later clear the active slot."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    cancellation_observed = asyncio.Event()
    release_cleanup = asyncio.Event()
    task_started = asyncio.Event()

    async def delayed_cancellation_cleanup() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await release_cleanup.wait()
            raise

    old_task = asyncio.create_task(
        delayed_cancellation_cleanup(),
        name="same-loop-retired-periodic",
    )
    await task_started.wait()
    monkeypatch.setattr(module, "_OVERRIDE_REFRESH_INTERVAL_SECONDS", 3600.0)
    with module._OVERRIDE_LOCK:
        module._OVERRIDE_REFRESH_SERVICE_TASK = old_task

    replacement_task = None
    try:
        set_llm_provider_overrides_cache_for_tests({})
        module.start_llm_provider_override_refresh_service()
        await asyncio.sleep(0)

        with module._OVERRIDE_LOCK:
            replacement_task = module._OVERRIDE_REFRESH_SERVICE_TASK
            assert replacement_task is not None
            assert replacement_task is not old_task
            assert old_task in module._OVERRIDE_RETIRED_TASKS

        await cancellation_observed.wait()
        release_cleanup.set()
        await asyncio.gather(old_task, return_exceptions=True)
        await asyncio.sleep(0)

        with module._OVERRIDE_LOCK:
            assert module._OVERRIDE_REFRESH_SERVICE_TASK is replacement_task
            assert old_task not in module._OVERRIDE_RETIRED_TASKS
    finally:
        release_cleanup.set()
        tasks = [old_task]
        if replacement_task is not None:
            replacement_task.cancel()
            tasks.append(replacement_task)
        if not old_task.done():
            old_task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with module._OVERRIDE_LOCK:
            if module._OVERRIDE_REFRESH_SERVICE_TASK in tasks:
                module._OVERRIDE_REFRESH_SERVICE_TASK = None
            retired_tasks = getattr(module, "_OVERRIDE_RETIRED_TASKS", None)
            if retired_tasks is not None:
                retired_tasks.discard(old_task)


@pytest.mark.parametrize(
    "task_attribute",
    ["_OVERRIDE_RECOVERY_TASK", "_OVERRIDE_REFRESH_SERVICE_TASK"],
    ids=("recovery", "periodic"),
)
def test_test_cache_reset_never_drives_task_owned_by_stopped_open_loop(
    task_attribute: str,
) -> None:
    """Reset fails promptly instead of running blocking cleanup on a stopped loop."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    loop = asyncio.new_event_loop()
    cancellation_observed = threading.Event()
    release_blocking_cleanup = threading.Event()
    task_started = threading.Event()

    async def wait_forever() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            release_blocking_cleanup.wait(timeout=5)
            raise

    task = loop.create_task(wait_forever(), name=f"stopped-{task_attribute}")

    def run_until_stopped() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    owner_thread = threading.Thread(target=run_until_stopped)
    owner_thread.start()
    assert task_started.wait(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    owner_thread.join(timeout=5)
    assert not owner_thread.is_alive()
    assert not loop.is_running()
    assert not loop.is_closed()
    assert not task.done()

    with module._OVERRIDE_LOCK:
        setattr(module, task_attribute, task)
        if task_attribute == "_OVERRIDE_RECOVERY_TASK":
            module._OVERRIDE_RECOVERY_IN_FLIGHT = True

    reset_errors: list[RuntimeError] = []
    reset_finished = threading.Event()

    def reset_cache() -> None:
        try:
            set_llm_provider_overrides_cache_for_tests({})
        except RuntimeError as exc:
            reset_errors.append(exc)
        finally:
            reset_finished.set()

    reset_thread = threading.Thread(target=reset_cache)
    try:
        reset_thread.start()
        assert reset_finished.wait(timeout=1)
        reset_thread.join(timeout=1)
        assert not reset_thread.is_alive()
        assert len(reset_errors) == 1
        assert "owner loop is stopped" in str(reset_errors[0])
        assert not cancellation_observed.is_set()
        assert not task.done()
        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is None
            assert task in module._OVERRIDE_RETIRED_TASKS
        assert asyncio.all_tasks(loop) == {task}
        assert not any(
            thread.name == "llm-provider-overrides-reset-loop-driver"
            and thread.is_alive()
            for thread in threading.enumerate()
        )
    finally:
        release_blocking_cleanup.set()
        reset_thread.join(timeout=5)
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        with module._OVERRIDE_LOCK:
            if getattr(module, task_attribute) is task:
                setattr(module, task_attribute, None)
            retired_tasks = getattr(module, "_OVERRIDE_RETIRED_TASKS", None)
            if retired_tasks is not None:
                retired_tasks.discard(task)
            module._OVERRIDE_RECOVERY_IN_FLIGHT = False
        loop.close()


@pytest.mark.parametrize(
    "task_attribute",
    ["_OVERRIDE_RECOVERY_TASK", "_OVERRIDE_REFRESH_SERVICE_TASK"],
    ids=("recovery", "periodic"),
)
def test_test_cache_reset_retains_pending_task_when_owner_loop_is_closed(
    task_attribute: str,
) -> None:
    """An impossible drain fails loudly without discarding pending-task evidence."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )

    class ClosedLoop:
        @staticmethod
        def is_closed() -> bool:
            return True

    class PendingTask:
        @staticmethod
        def done() -> bool:
            return False

        @staticmethod
        def get_loop() -> ClosedLoop:
            return ClosedLoop()

    task = PendingTask()
    with module._OVERRIDE_LOCK:
        setattr(module, task_attribute, task)
        if task_attribute == "_OVERRIDE_RECOVERY_TASK":
            module._OVERRIDE_RECOVERY_IN_FLIGHT = True

    try:
        with pytest.raises(RuntimeError, match="owner loop is closed"):
            set_llm_provider_overrides_cache_for_tests({})

        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is None
            assert task in module._OVERRIDE_RETIRED_TASKS
            if task_attribute == "_OVERRIDE_RECOVERY_TASK":
                assert module._OVERRIDE_RECOVERY_IN_FLIGHT is False
    finally:
        with module._OVERRIDE_LOCK:
            if getattr(module, task_attribute) is task:
                setattr(module, task_attribute, None)
            retired_tasks = getattr(module, "_OVERRIDE_RETIRED_TASKS", None)
            if retired_tasks is not None:
                retired_tasks.discard(task)
            module._OVERRIDE_RECOVERY_IN_FLIGHT = False


@pytest.mark.parametrize(
    "task_attribute",
    ["_OVERRIDE_RECOVERY_TASK", "_OVERRIDE_REFRESH_SERVICE_TASK"],
    ids=("recovery", "periodic"),
)
def test_test_cache_reset_foreign_timeout_leaves_only_original_task(
    monkeypatch: pytest.MonkeyPatch,
    task_attribute: str,
) -> None:
    """A foreign-loop timeout cannot orphan a wrapper drain task."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    background = _BackgroundEventLoop()
    cancellation_observed = threading.Event()
    release_cleanup = background.call(asyncio.Event).result(timeout=5)
    task_started = threading.Event()

    async def non_cooperative_cleanup() -> None:
        task_started.set()
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_observed.set()
                try:
                    await release_cleanup.wait()
                except asyncio.CancelledError:
                    continue
                return

    task = background.call(
        lambda: background.loop.create_task(
            non_cooperative_cleanup(),
            name=f"foreign-timeout-{task_attribute}",
        )
    ).result(timeout=5)
    assert task_started.wait(timeout=5)
    monkeypatch.setattr(
        module,
        "_OVERRIDE_REFRESH_TIMEOUT_SECONDS",
        0.05,
    )
    with module._OVERRIDE_LOCK:
        setattr(module, task_attribute, task)
        if task_attribute == "_OVERRIDE_RECOVERY_TASK":
            module._OVERRIDE_RECOVERY_IN_FLIGHT = True

    try:
        with pytest.raises(RuntimeError, match="Timed out draining"):
            set_llm_provider_overrides_cache_for_tests({})
        assert cancellation_observed.wait(timeout=5)

        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is None
            assert task in module._OVERRIDE_RETIRED_TASKS
        owner_tasks = background.call(
            lambda: asyncio.all_tasks(background.loop)
        ).result(timeout=5)
        assert owner_tasks == {task}
    finally:
        background.loop.call_soon_threadsafe(release_cleanup.set)
        background.barrier(turns=3).result(timeout=5)
        with module._OVERRIDE_LOCK:
            assert task not in module._OVERRIDE_RETIRED_TASKS
            if getattr(module, task_attribute) is task:
                setattr(module, task_attribute, None)
            retired_tasks = getattr(module, "_OVERRIDE_RETIRED_TASKS", None)
            if retired_tasks is not None:
                retired_tasks.discard(task)
            module._OVERRIDE_RECOVERY_IN_FLIGHT = False
        background.close()


@pytest.mark.parametrize(
    "task_attribute",
    ["_OVERRIDE_RECOVERY_TASK", "_OVERRIDE_REFRESH_SERVICE_TASK"],
    ids=("recovery", "periodic"),
)
def test_foreign_timeout_keeps_retired_task_when_replacement_claims_active_slot(
    monkeypatch: pytest.MonkeyPatch,
    task_attribute: str,
) -> None:
    """A replacement worker cannot make a timed-out retired task unowned."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    background = _BackgroundEventLoop()
    cancellation_observed = threading.Event()
    old_started = threading.Event()
    replacement_started = threading.Event()
    release_old = background.call(asyncio.Event).result(timeout=5)
    release_replacement = background.call(asyncio.Event).result(timeout=5)

    async def old_worker() -> None:
        old_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await release_old.wait()
            raise

    async def replacement_worker() -> None:
        replacement_started.set()
        await release_replacement.wait()

    old_task = background.call(
        lambda: background.loop.create_task(
            old_worker(),
            name=f"retired-old-{task_attribute}",
        )
    ).result(timeout=5)
    assert old_started.wait(timeout=5)
    monkeypatch.setattr(
        module,
        "_OVERRIDE_REFRESH_TIMEOUT_SECONDS",
        0.5,
    )
    with module._OVERRIDE_LOCK:
        setattr(module, task_attribute, old_task)
        if task_attribute == "_OVERRIDE_RECOVERY_TASK":
            module._OVERRIDE_RECOVERY_IN_FLIGHT = True

    reset_errors: list[RuntimeError] = []
    reset_finished = threading.Event()

    def reset_cache() -> None:
        try:
            set_llm_provider_overrides_cache_for_tests({})
        except RuntimeError as exc:
            reset_errors.append(exc)
        finally:
            reset_finished.set()

    reset_thread = threading.Thread(target=reset_cache)
    replacement_task = None
    try:
        reset_thread.start()
        assert cancellation_observed.wait(timeout=5)
        replacement_task = background.call(
            lambda: background.loop.create_task(
                replacement_worker(),
                name=f"active-replacement-{task_attribute}",
            )
        ).result(timeout=5)
        assert replacement_started.wait(timeout=5)
        with module._OVERRIDE_LOCK:
            setattr(module, task_attribute, replacement_task)
            if task_attribute == "_OVERRIDE_RECOVERY_TASK":
                module._OVERRIDE_RECOVERY_IN_FLIGHT = True

        assert reset_finished.wait(timeout=5)
        reset_thread.join(timeout=5)
        assert len(reset_errors) == 1
        assert "Timed out draining" in str(reset_errors[0])
        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is replacement_task
            assert old_task in module._OVERRIDE_RETIRED_TASKS

        background.loop.call_soon_threadsafe(release_old.set)
        background.barrier(turns=3).result(timeout=5)
        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is replacement_task
            assert old_task not in module._OVERRIDE_RETIRED_TASKS
    finally:
        background.loop.call_soon_threadsafe(release_old.set)
        background.loop.call_soon_threadsafe(release_replacement.set)
        reset_thread.join(timeout=5)
        background.barrier(turns=3).result(timeout=5)
        with module._OVERRIDE_LOCK:
            if getattr(module, task_attribute) in {old_task, replacement_task}:
                setattr(module, task_attribute, None)
            retired_tasks = getattr(module, "_OVERRIDE_RETIRED_TASKS", None)
            if retired_tasks is not None:
                retired_tasks.discard(old_task)
            module._OVERRIDE_RECOVERY_IN_FLIGHT = False
        background.close()


@pytest.mark.parametrize(
    "task_attribute",
    ["_OVERRIDE_RECOVERY_TASK", "_OVERRIDE_REFRESH_SERVICE_TASK"],
    ids=("recovery", "periodic"),
)
def test_test_cache_reset_stopped_timeout_leaves_no_wrapper_or_driver(
    monkeypatch: pytest.MonkeyPatch,
    task_attribute: str,
) -> None:
    """A stopped-loop timeout retains only the original pending task."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    loop = asyncio.new_event_loop()
    cancellation_observed = threading.Event()
    release_cleanup = asyncio.Event()
    task_started = threading.Event()

    async def non_cooperative_cleanup() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_observed.set()
            await release_cleanup.wait()

    task = loop.create_task(
        non_cooperative_cleanup(),
        name=f"stopped-timeout-{task_attribute}",
    )

    def run_until_stopped() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    owner_thread = threading.Thread(target=run_until_stopped)
    owner_thread.start()
    assert task_started.wait(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    owner_thread.join(timeout=5)
    assert not owner_thread.is_alive()
    with module._OVERRIDE_LOCK:
        setattr(module, task_attribute, task)
        if task_attribute == "_OVERRIDE_RECOVERY_TASK":
            module._OVERRIDE_RECOVERY_IN_FLIGHT = True

    try:
        with pytest.raises(RuntimeError, match="owner loop is stopped"):
            set_llm_provider_overrides_cache_for_tests({})
        assert not cancellation_observed.is_set()

        with module._OVERRIDE_LOCK:
            assert getattr(module, task_attribute) is None
            assert task in module._OVERRIDE_RETIRED_TASKS
        assert asyncio.all_tasks(loop) == {task}
        assert not any(
            thread.name == "llm-provider-overrides-reset-loop-driver"
            and thread.is_alive()
            for thread in threading.enumerate()
        )
    finally:
        release_cleanup.set()
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        with module._OVERRIDE_LOCK:
            assert task not in module._OVERRIDE_RETIRED_TASKS
            if getattr(module, task_attribute) is task:
                setattr(module, task_attribute, None)
            retired_tasks = getattr(module, "_OVERRIDE_RETIRED_TASKS", None)
            if retired_tasks is not None:
                retired_tasks.discard(task)
            module._OVERRIDE_RECOVERY_IN_FLIGHT = False
        loop.close()


@pytest.mark.parametrize("worker", ["recovery", "periodic"])
def test_test_cache_reset_invalidates_task_created_before_reset(
    monkeypatch: pytest.MonkeyPatch,
    worker: str,
) -> None:
    """A task registered after reset cannot publish its obsolete snapshot."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    background = _BackgroundEventLoop()
    task_created = threading.Event()
    allow_create_to_return = threading.Event()
    stale_refresh_started = threading.Event()
    created_tasks: list[asyncio.Task] = []

    async def stale_refresh(*_args, **_kwargs):
        with module._OVERRIDE_LOCK:
            module._OVERRIDE_CACHE.clear()
            module._OVERRIDE_CACHE["openai"] = LLMProviderOverride(
                provider="openai",
                api_key="stale-background-key",
            )
        stale_refresh_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(module, "refresh_llm_provider_overrides", stale_refresh)
    monkeypatch.setattr(module, "_OVERRIDE_REFRESH_INTERVAL_SECONDS", 0.0)

    def start_worker_across_reset() -> None:
        original_create_task = background.loop.create_task

        def blocked_create_task(coro, *, name=None, **_kwargs):
            task = original_create_task(coro, name=name)
            created_tasks.append(task)
            task_created.set()
            assert allow_create_to_return.wait(timeout=5)
            return task

        background.loop.create_task = blocked_create_task
        try:
            if worker == "recovery":
                with module._OVERRIDE_LOCK:
                    module._OVERRIDE_CACHE_HEALTHY = False
                    module._OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0
                module._schedule_override_recovery()
            else:
                module.start_llm_provider_override_refresh_service()
        finally:
            background.loop.create_task = original_create_task

    start_result = background.call(start_worker_across_reset)
    assert task_created.wait(timeout=5)
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="seeded-after-reset-key",
            )
        }
    )
    allow_create_to_return.set()

    try:
        start_result.result(timeout=5)
        background.barrier(turns=2).result(timeout=5)

        assert not stale_refresh_started.is_set()
        fallback = get_override_server_fallback("openai")
        assert fallback is not None
        assert fallback.api_key == "seeded-after-reset-key"
    finally:
        task = created_tasks[0]
        with module._OVERRIDE_LOCK:
            if module._OVERRIDE_RECOVERY_TASK is task:
                module._OVERRIDE_RECOVERY_TASK = None
            if module._OVERRIDE_REFRESH_SERVICE_TASK is task:
                module._OVERRIDE_REFRESH_SERVICE_TASK = None
            module._OVERRIDE_RECOVERY_IN_FLIGHT = False
        background.loop.call_soon_threadsafe(task.cancel)
        background.barrier().result(timeout=5)
        background.close()
        set_llm_provider_overrides_cache_for_tests({})


@pytest.mark.parametrize(
    "row",
    [
        {"provider": "openai", "allowed_models": "{broken-json"},
        {"provider": "openai", "allowed_models": '{"not": "a-list"}'},
        {"provider": "openai", "config_json": '["not", "an", "object"]'},
        {"provider": "openai", "is_enabled": "sometimes"},
        {"provider": "voyage"},
    ],
    ids=(
        "invalid_models_json",
        "models_object",
        "config_list",
        "invalid_boolean",
        "unsupported_provider",
    ),
)
async def test_real_malformed_stored_rows_fail_refresh_closed(monkeypatch, row) -> None:
    """Corrupt durable rows cannot silently weaken provider restrictions."""
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")

    class FakeRepo:
        def __init__(self, _pool) -> None:
            pass

        async def ensure_tables(self) -> None:
            return None

        async def list_overrides(self):
            return [row]

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                allowed_models=["gpt-4o"],
                api_key="last-good-key",
            )
        }
    )

    with pytest.raises(LLMProviderOverridesRefreshError):
        await module.refresh_llm_provider_overrides(pool=object())
    with pytest.raises(ByokResolutionError) as exc_info:
        validate_provider_override("openai", "forbidden-model")
    assert exc_info.value.code == "credential_store_unavailable"


def test_all_override_accessors_fail_closed_when_snapshot_is_unhealthy() -> None:
    """Legacy consumers cannot bypass the atomic health-aware boundary."""
    set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="stale-key")},
        healthy=False,
    )

    for accessor in (
        lambda: get_llm_provider_override("openai"),
        get_llm_provider_overrides_snapshot,
        lambda: get_override_credentials("openai"),
        lambda: validate_provider_override("openai", "gpt-4o"),
    ):
        with pytest.raises(ByokResolutionError) as exc_info:
            accessor()
        assert exc_info.value.code == "credential_store_unavailable"


def test_validate_provider_override_blocks_disallowed_model() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=True,
                allowed_models=["gpt-4o"],
            )
        }
    )

    blocked = validate_provider_override("openai", "gpt-3.5-turbo")
    assert blocked is not None
    assert blocked["error_code"] == "model_not_allowed"

    allowed = validate_provider_override("openai", "gpt-4o")
    assert allowed is None

    set_llm_provider_overrides_cache_for_tests({})


def test_get_override_model_priority_reads_routing_rankings() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                config={
                    "routing": {
                        "model_rankings": {
                            "highest_quality": ["gpt-4.1", "gpt-4.1-mini"],
                        }
                    }
                },
            )
        }
    )

    assert get_override_model_priority("openai", "highest_quality") == [
        "gpt-4.1",
        "gpt-4.1-mini",
    ]

    updated = apply_llm_provider_overrides_to_listing(
        {
            "providers": [
                {
                    "name": "openai",
                    "models": ["gpt-4.1-mini", "gpt-4.1"],
                    "models_info": [
                        {"name": "gpt-4.1-mini"},
                        {"name": "gpt-4.1"},
                    ],
                }
            ]
        }
    )
    assert updated["providers"][0]["models"] == ["gpt-4.1", "gpt-4.1-mini"]
    assert [
        model["name"] for model in updated["providers"][0]["models_info"]
    ] == ["gpt-4.1", "gpt-4.1-mini"]

    set_llm_provider_overrides_cache_for_tests({})


def test_apply_overrides_sorts_models_info_without_crashing_on_non_dict_entries() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                config={
                    "routing": {
                        "model_rankings": {
                            "highest_quality": ["gpt-4.1", "gpt-4.1-mini"],
                        }
                    }
                },
            )
        }
    )

    updated = apply_llm_provider_overrides_to_listing(
        {
            "providers": [
                {
                    "name": "openai",
                    "models_info": [
                        None,
                        {"name": "gpt-4.1-mini"},
                        "broken",
                        {"name": "gpt-4.1"},
                    ],
                }
            ]
        }
    )

    assert [
        model["name"] for model in updated["providers"][0]["models_info"]
    ] == ["gpt-4.1", "gpt-4.1-mini"]

    set_llm_provider_overrides_cache_for_tests({})


def test_apply_overrides_uses_one_snapshot_for_models_and_priority(monkeypatch) -> None:
    module = importlib.import_module(
        "tldw_Server_API.app.core.AuthNZ.llm_provider_overrides"
    )
    reads = 0
    snapshot_a = {
        "openai": LLMProviderOverride(
            provider="openai",
            config={
                "routing": {
                    "model_rankings": {
                        "highest_quality": ["model-a", "model-b"],
                    }
                }
            },
        )
    }
    snapshot_b = {
        "openai": LLMProviderOverride(
            provider="openai",
            config={
                "routing": {
                    "model_rankings": {
                        "highest_quality": ["model-b", "model-a"],
                    }
                }
            },
        )
    }

    def sequenced_snapshot(_provider: str = "provider-overrides"):
        nonlocal reads
        reads += 1
        return snapshot_a if reads == 1 else snapshot_b

    monkeypatch.setattr(module, "_get_healthy_override_snapshot", sequenced_snapshot)

    updated = module.apply_llm_provider_overrides_to_listing(
        {
            "providers": [
                {
                    "name": "openai",
                    "models": ["model-b", "model-a"],
                    "models_info": [{"name": "model-b"}, {"name": "model-a"}],
                }
            ]
        }
    )

    assert reads == 1
    assert updated["providers"][0]["models"] == ["model-a", "model-b"]
    assert [
        model["name"] for model in updated["providers"][0]["models_info"]
    ] == ["model-a", "model-b"]


def test_override_fallback_projects_flat_config_over_one_static_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override credentials retain static adapter options in one provider section."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime

    static_reads = 0

    def load_static_snapshot() -> dict[str, object]:
        nonlocal static_reads
        static_reads += 1
        return {
            "openai_api": {
                "api_key": "static-key-must-not-win",
                "api_base_url": "https://static.example/v1",
                "model": "gpt-static",
                "timeout": 17,
            },
            "anthropic_api": {
                "api_key": "unrelated-secret-must-not-project",
                "model": "claude-unrelated",
            },
            "HTTP": {"connect_timeout": 3},
        }

    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        load_static_snapshot,
    )
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                config={
                    "default_model": "gpt-override",
                    "api_base_url": "https://override.example/v1",
                    "retry_attempts": 4,
                    "private_canary": "must-not-project",
                },
                api_key="override-key",
                credential_fields={
                    "org_id": "org-override",
                    "project_id": "project-override",
                },
            )
        }
    )

    static_fallback = byok_runtime.resolve_static_server_fallback("openai")
    fallback = capture_provider_override_call_snapshot("openai").server_fallback(
        static_fallback
    )

    assert fallback is not None
    assert static_reads == 1
    assert fallback.api_key == "override-key"
    assert fallback.credential_fields == {
        "org_id": "org-override",
        "project_id": "project-override",
    }
    assert fallback.auth_source is None
    assert fallback.app_config == {
        "openai_api": {
            "api_base_url": "https://override.example/v1",
            "model": "gpt-override",
            "timeout": 17,
            "retry_attempts": 4,
            "org_id": "org-override",
            "project_id": "project-override",
        },
        "HTTP": {"connect_timeout": 3},
    }
    assert "unrelated-secret" not in repr(fallback.app_config)
    assert "private_canary" not in repr(fallback.app_config)


def test_override_fallback_accepts_authoritative_empty_static_config() -> None:
    """An optional empty config snapshot must not discard valid static auth."""
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                config={"default_model": "gpt-override"},
            )
        }
    )
    static_fallback = ServerFallbackCredentials(
        api_key="static-key",
        credential_fields={},
        app_config=None,
    )

    fallback = capture_provider_override_call_snapshot("openai").server_fallback(
        static_fallback
    )

    assert fallback is not None
    assert fallback.api_key == "static-key"
    assert fallback.app_config == {"openai_api": {"model": "gpt-override"}}


def test_override_key_replacement_drops_static_openai_credential_metadata() -> None:
    """An admin key override cannot inherit org/project fields from the server key."""

    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                api_key="override-key",
            )
        }
    )
    static_fallback = ServerFallbackCredentials(
        api_key="static-key",
        credential_fields={
            "org_id": "static-org-short",
            "project_id": "static-project-id",
        },
        app_config={
            "openai_api": {
                "model": "gpt-static",
                "organization": "static-org",
                "organization_id": "static-org-id",
                "org_id": "static-org-short",
                "project": "static-project",
                "project_id": "static-project-id",
            }
        },
    )

    fallback = capture_provider_override_call_snapshot("openai").server_fallback(
        static_fallback
    )

    assert fallback is not None
    assert fallback.api_key == "override-key"
    assert fallback.credential_fields == {}
    assert fallback.app_config == {"openai_api": {"model": "gpt-static"}}
