"""The moves that cut the foundation layer's upward imports (TASK-13950.12)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_database_error_is_defined_in_core_exceptions_and_reexported_by_authnz() -> None:
    from tldw_Server_API.app.core import exceptions as core_exc
    from tldw_Server_API.app.core.AuthNZ import exceptions as authnz_exc

    assert core_exc.DatabaseError.__module__ == "tldw_Server_API.app.core.exceptions"
    assert authnz_exc.DatabaseError is core_exc.DatabaseError
    assert authnz_exc.UserRegistrationException is core_exc.UserRegistrationException
    assert issubclass(core_exc.DatabaseError, core_exc.UserRegistrationException)
    assert issubclass(authnz_exc.TransactionError, core_exc.DatabaseError)
    assert issubclass(core_exc.AudioQuotaStoreUnavailable, authnz_exc.DatabaseError)


def test_provider_identity_lives_in_utils_and_the_llm_calls_path_reexports_it() -> None:
    from tldw_Server_API.app.core.LLM_Calls import provider_identity as old
    from tldw_Server_API.app.core.Utils import provider_identity as new

    assert new.__name__ == "tldw_Server_API.app.core.Utils.provider_identity"
    assert old.canonical_provider_name is new.canonical_provider_name
    assert old.provider_lookup_names is new.provider_lookup_names
    assert old.PROVIDER_ALIASES is new.PROVIDER_ALIASES
    assert new.canonical_provider_name("OpenAI-Compatible") == "custom-openai-api"


def test_llamacpp_handler_settings_are_plain_data(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from tldw_Server_API.app.core import config

    monkeypatch.setenv("LLAMACPP_ENABLED", "1")
    monkeypatch.setenv("LLAMACPP_PORT", "9001")
    monkeypatch.setenv("LLAMACPP_ALLOWED_PATHS", "/models/a, /models/b")

    settings = config.get_llamacpp_handler_settings()

    assert settings is not None
    assert settings["enabled"] is True and settings["default_port"] == 9001
    assert settings["allowed_paths"] == [Path("/models/a"), Path("/models/b")]


def test_llamacpp_handler_settings_are_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core import config

    monkeypatch.setenv("LLAMACPP_ENABLED", "0")
    assert config.get_llamacpp_handler_settings() is None


def test_request_id_helpers_live_in_logging() -> None:
    from tldw_Server_API.app.core.Logging import log_context
    from tldw_Server_API.app.core.Security import request_id_middleware

    assert log_context._clean_request_id.__module__ == "tldw_Server_API.app.core.Logging.log_context"
    assert request_id_middleware._clean_request_id is log_context._clean_request_id
    assert log_context._clean_request_id("req-1") == "req-1"
    assert log_context._clean_request_id("bad id!") != "bad id!"


def test_stt_observability_context_is_one_contextvar() -> None:
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core import stt_observability_context as old
    from tldw_Server_API.app.core.Logging import stt_observability_context as new

    assert old.OPAQUE_STT_ENDPOINT_ID is new.OPAQUE_STT_ENDPOINT_ID
    endpoint_id = "sha256:" + "a" * 64
    with http_client.opaque_stt_http_observability(endpoint_id):
        assert old.get_opaque_stt_endpoint_id() == endpoint_id
    assert new.get_opaque_stt_endpoint_id() is None


def test_prompt_loader_lives_in_context_integrity_with_no_utils_shim() -> None:
    import importlib

    from tldw_Server_API.app.core.Context_Integrity import prompt_loader

    assert callable(prompt_loader.load_prompt)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tldw_Server_API.app.core.Utils.prompt_loader")


def test_utils_no_longer_owns_db_code() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import safe_metadata_write_ops
    from tldw_Server_API.app.core.Utils import Utils, metadata_utils

    assert callable(safe_metadata_write_ops.update_version_safe_metadata_in_transaction)
    assert not hasattr(metadata_utils, "update_version_safe_metadata_in_transaction")
    assert not hasattr(Utils, "get_db_config")


def test_email_metric_families_are_defined_in_metrics() -> None:
    from tldw_Server_API.app.core.Metrics.email_metrics import iter_email_metric_definitions
    from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry

    names = {definition["name"] for definition in iter_email_metric_definitions()}
    assert names == {
        "email_ingestion_parse_total", "email_ingestion_parse_seconds",
        "email_ingestion_persist_total", "email_ingestion_persist_seconds",
        "email_native_persist_seconds", "email_ingestion_dedupe_total",
    }
    assert {"email_ingestion_parse_total", "email_ingestion_dedupe_total"} <= set(MetricsRegistry().metrics)


def test_old_security_egress_path_is_the_moved_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from unittest import mock

    import tldw_Server_API.app.core.Security.egress as old
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress as via_package
    from tldw_Server_API.app.core.Utils import egress as new

    # tests/conftest.py defaults this to "false"; the policy reads it per call.
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    assert old is new and via_package is new
    with mock.patch(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        return_value=SimpleNamespace(allowed=True),
    ):
        assert http_client._is_url_allowed("http://169.254.169.254/") is True
    assert http_client._is_url_allowed("http://169.254.169.254/") is False


def test_http_client_denies_metadata_ip_without_core_security() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import sys\n"
        "from tldw_Server_API.app.core import http_client\n"
        "from tldw_Server_API.app.core.exceptions import EgressPolicyError\n"
        "try:\n"
        "    http_client._validate_egress_or_raise('http://169.254.169.254/latest/meta-data/')\n"
        "except EgressPolicyError:\n"
        "    print('denied')\n"
        "print(sorted(m for m in sys.modules if m.startswith('tldw_Server_API.app.core.Security')))\n"
    )
    out = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "WORKFLOWS_EGRESS_BLOCK_PRIVATE": "true"},
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout.splitlines()
    assert out[-2:] == ["denied", "[]"]


def test_old_egress_path_aliases_when_imported_first() -> None:
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import tldw_Server_API.app.core.Security.egress as old\n"
        "from tldw_Server_API.app.core.Utils import egress as new\n"
        "from tldw_Server_API.app.core.Security import egress as via_package\n"
        "assert old is new and via_package is new\n"
    )
    result = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_http_client_observability_is_noop_until_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from tldw_Server_API.app.core import http_client

    monkeypatch.setattr(http_client, "_metrics_registry_getter", None)
    monkeypatch.setattr(http_client, "_tracing_manager_getter", None)
    assert http_client.get_metrics_registry().increment("http_client_requests_total", 1, labels={}) is None
    assert "X-Request-Id" not in http_client._inject_trace_headers({})
    client = http_client.create_client(transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request)))
    try:  # the request path opens a span on the no-op tracing manager
        assert http_client.fetch(method="GET", url="http://93.184.216.34/ok", client=client).status_code == 200
    finally:
        client.close()


def test_a_process_without_core_metrics_still_fetches() -> None:
    import subprocess
    import sys
    from pathlib import Path

    # tests/conftest.py imports core.Metrics, so only a fresh interpreter has no installed hooks.
    script = (
        "import asyncio, sys\n"
        "import httpx\n"
        "from tldw_Server_API.app.core import http_client\n"
        "def handler(request):\n"
        "    return httpx.Response(200, request=request, text='ok')\n"
        "client = http_client.create_client(transport=httpx.MockTransport(handler))\n"
        "print(http_client.fetch(method='GET', url='http://93.184.216.34/ok', client=client).status_code)\n"
        "async def main():\n"
        "    aclient = http_client.create_async_client(transport=httpx.MockTransport(handler))\n"
        "    try:\n"
        "        return (await http_client.afetch(method='GET', url='http://93.184.216.34/ok', client=aclient)).status_code\n"
        "    finally:\n"
        "        await aclient.aclose()\n"
        "print(asyncio.run(main()))\n"
        "print('tldw_Server_API.app.core.Metrics' in sys.modules)\n"
    )
    result = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.splitlines()[-3:] == ["200", "200", "False"]


def test_core_metrics_installs_http_client_observability() -> None:
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Metrics import get_metrics_registry, get_tracing_manager

    assert http_client._metrics_registry_getter is get_metrics_registry
    assert http_client._tracing_manager_getter is get_tracing_manager


def test_installing_observability_survives_a_failing_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Metrics import http_client_metrics, traces

    def broken_registry() -> None:
        raise RuntimeError("metrics registry unavailable")

    monkeypatch.setattr(http_client_metrics, "get_metrics_registry", broken_registry)
    monkeypatch.setattr(http_client, "_metrics_registry_getter", None)
    monkeypatch.setattr(http_client, "_tracing_manager_getter", None)
    http_client_metrics.install_http_client_observability()  # must not raise
    assert http_client._metrics_registry_getter is broken_registry
    assert http_client._tracing_manager_getter is traces.get_tracing_manager


def test_installed_tracing_manager_feeds_x_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core import http_client

    class _Tracing:
        def get_baggage(self, key: str) -> str | None:
            return "rid-1" if key == "request_id" else None

    monkeypatch.setattr(http_client, "_tracing_manager_getter", _Tracing)
    assert http_client._inject_trace_headers({})["X-Request-Id"] == "rid-1"


def test_http_client_metric_families_register_on_a_fresh_registry() -> None:
    from tldw_Server_API.app.core.Metrics.http_client_metrics import register_http_client_metrics
    from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry

    registry = MetricsRegistry()
    register_http_client_metrics(registry)
    assert {
        "http_client_requests_total", "http_client_request_duration_seconds",
        "http_client_retries_total", "http_client_egress_denials_total",
    } <= set(registry.metrics)
