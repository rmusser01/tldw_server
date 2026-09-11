from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

import tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter as or_mod
from tldw_Server_API.app.core.AuthNZ import byok_helpers, byok_runtime
from tldw_Server_API.app.core.LLM_Calls.providers.openrouter_adapter import (
    OpenRouterAdapter,
)
from tldw_Server_API.tests.provider_credential_test_helpers import (
    resolved_request_fields,
)


class _CaptureClient:
    def __init__(self):
        self.last_headers: dict[str, str] | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url: str, headers: dict[str, str], json: dict[str, Any]):
        self.last_headers = dict(headers)
        class R:
            status_code = 200
            def raise_for_status(self):
                return None
            def json(self):
                return {"object": "chat.completion", "choices": [{"message": {"content": "ok"}}]}
        return R()

    def stream(self, *a, **k):  # pragma: no cover - not used here
        raise RuntimeError("not used")


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENROUTER", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    yield


def test_openrouter_headers_include_site_meta(monkeypatch):
    cap = _CaptureClient()
    monkeypatch.setattr(or_mod, "http_client_factory", lambda *a, **k: cap)

    a = OpenRouterAdapter()
    req = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "meta-llama/llama-3-8b",
        "api_key": "k",
        "app_config": {
            "openrouter_api": {
                "site_url": "https://example.com",
                "site_name": "TLDW-Test",
            }
        },
    }
    out = a.chat(req)
    assert out["object"] == "chat.completion"
    assert cap.last_headers is not None
    # Verify OpenRouter-specific header quirks
    assert cap.last_headers.get("HTTP-Referer") == "https://example.com"
    assert cap.last_headers.get("X-Title") == "TLDW-Test"
    # Authorization preserved
    assert cap.last_headers.get("Authorization", "").startswith("Bearer ")


@pytest.mark.unit
def test_resolved_openrouter_empty_snapshot_ignores_late_site_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://late-site.example")
    monkeypatch.setenv("OPENROUTER_SITE_NAME", "Late Site")

    headers = OpenRouterAdapter()._headers(
        "snapshot-key",
        {
            "app_config": {},
            "credentials_resolved": True,
        },
    )

    assert headers["HTTP-Referer"] == "https://openrouter.ai"
    assert headers["X-Title"] == "TLDW-API"


@pytest.mark.unit
def test_unmarked_openrouter_headers_preserve_legacy_site_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://legacy-site.example")
    monkeypatch.setenv("OPENROUTER_SITE_NAME", "Legacy Site")

    headers = OpenRouterAdapter()._headers(
        "legacy-key",
        {"app_config": {}},
    )

    assert headers["HTTP-Referer"] == "https://legacy-site.example"
    assert headers["X-Title"] == "Legacy Site"


@pytest.mark.unit
def test_openrouter_static_snapshot_freezes_site_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(byok_helpers, "load_and_log_configs", lambda **_kwargs: {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "snapshot-key-a")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://snapshot-a.example")
    monkeypatch.setenv("OPENROUTER_SITE_NAME", "Snapshot A")
    snapshot = byok_helpers.load_server_config_snapshot()

    monkeypatch.setenv("OPENROUTER_API_KEY", "late-key-b")
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://late-b.example")
    monkeypatch.setenv("OPENROUTER_SITE_NAME", "Late B")
    monkeypatch.setattr(
        byok_runtime,
        "load_server_config_snapshot",
        lambda: snapshot,
    )
    fallback = byok_runtime.resolve_static_server_fallback("openrouter")
    cap = _CaptureClient()
    monkeypatch.setattr(or_mod, "http_client_factory", lambda *a, **k: cap)

    OpenRouterAdapter().chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "snapshot-model",
            **resolved_request_fields(
                "openrouter",
                api_key=fallback.api_key,
                app_config=dict(fallback.app_config or {}),
                model="snapshot-model",
            ),
        }
    )

    assert fallback.api_key == "snapshot-key-a"
    assert cap.last_headers is not None
    assert cap.last_headers["HTTP-Referer"] == "https://snapshot-a.example"
    assert cap.last_headers["X-Title"] == "Snapshot A"


class _ConcurrentCaptureClient:
    def __init__(
        self,
        captured: list[tuple[str, str]],
        captured_guard: threading.Lock,
        gate: threading.Barrier,
    ) -> None:
        self._captured = captured
        self._captured_guard = captured_guard
        self._gate = gate

    def __enter__(self) -> _ConcurrentCaptureClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> Any:
        del url, json
        self._gate.wait(timeout=5)
        with self._captured_guard:
            self._captured.append(
                (headers["HTTP-Referer"], headers["X-Title"])
            )

        class Response:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return {
                    "object": "chat.completion",
                    "choices": [{"message": {"content": "ok"}}],
                }

        return Response()


@pytest.mark.unit
@pytest.mark.concurrent
def test_concurrent_openrouter_full_dispatch_keeps_site_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://late-site.example")
    monkeypatch.setenv("OPENROUTER_SITE_NAME", "Late Site")
    captured: list[tuple[str, str]] = []
    captured_guard = threading.Lock()
    gate = threading.Barrier(2)
    monkeypatch.setattr(
        or_mod,
        "http_client_factory",
        lambda *a, **k: _ConcurrentCaptureClient(
            captured,
            captured_guard,
            gate,
        ),
    )
    adapter = OpenRouterAdapter()

    def request(site_url: str, site_name: str) -> dict[str, Any]:
        return {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "snapshot-model",
            **resolved_request_fields(
                "openrouter",
                api_key="snapshot-key",
                app_config={
                    "openrouter_api": {
                        "site_url": site_url,
                        "site_name": site_name,
                    }
                },
                model="snapshot-model",
            ),
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            adapter.chat,
            request("https://snapshot-a.example", "Snapshot A"),
        )
        second = executor.submit(
            adapter.chat,
            request("https://snapshot-b.example", "Snapshot B"),
        )
        first.result(timeout=10)
        second.result(timeout=10)

    assert set(captured) == {
        ("https://snapshot-a.example", "Snapshot A"),
        ("https://snapshot-b.example", "Snapshot B"),
    }
