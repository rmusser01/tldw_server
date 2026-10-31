"""Characterization test pinning ADR-021's final-cleanup shutdown order."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Utils.app_services import app_services
from tldw_Server_API.app.services import (
    shutdown_cleanup_timed_segments as timed,
)
from tldw_Server_API.app.services import (
    shutdown_resource_cleanup as cleanup,
)

pytestmark = pytest.mark.unit

ORDER = [
    "db_pool", "session_manager", "mcp_server", "mcp_server.wait",
    "mcp_rate_limiter", "tts_service", "tts_resource_manager", "http_client",
    "chacha", "prompts", "chat_workflows", "provider_manager", "request_queue", "local_llm_manager",
    "evaluations", "unified_audit", "executors", "cpu_pools", "telemetry",
]


class _Service:
    def __init__(self, name: str, events: list[str]) -> None:
        self._name = name
        self._events = events

    async def close(self) -> None:
        self._events.append(self._name)

    async def shutdown(self) -> None:
        self._events.append(self._name)

    async def aclose(self, wait_timeout: float = 5.0) -> None:
        self._events.append(self._name)

    async def wait_for_shutdown_completion(self) -> bool:
        self._events.append(f"{self._name}.wait")
        return True


def _step(name: str, events: list[str]):
    async def run(**_kwargs) -> None:
        events.append(name)

    return run


@contextmanager
def _untimed(_app, _name):
    yield


async def test_final_cleanup_closes_the_container_services_in_adr_021_order(monkeypatch) -> None:
    events: list[str] = []
    for helper, label in (
        ("_shutdown_mcp_rate_limiter", "mcp_rate_limiter"),
        ("_shutdown_tts_service", "tts_service"),
        ("_shutdown_tts_resource_manager", "tts_resource_manager"),
        ("_shutdown_http_client", "http_client"),
        ("_shutdown_prompts_resources", "prompts"),
        ("_shutdown_chat_workflows_resources", "chat_workflows"),
        ("_shutdown_provider_manager", "provider_manager"),
        ("_shutdown_request_queue", "request_queue"),
        ("_shutdown_local_llm_manager", "local_llm_manager"),
    ):
        monkeypatch.setattr(cleanup, helper, _step(label, events))
    for helper, label in (
        ("_shutdown_evaluations_resources", "evaluations"),
        ("_shutdown_unified_audit_services", "unified_audit"),
        ("_shutdown_executor_resources", "executors"),
        ("_shutdown_cpu_pools", "cpu_pools"),
        ("_shutdown_telemetry_services", "telemetry"),
    ):
        monkeypatch.setattr(timed, helper, _step(label, events))
    monkeypatch.setattr(app_services(), "chacha_runtime", _Service("chacha", events))

    handles = await timed.shutdown_cleanup_timed_segments(
        app=SimpleNamespace(state=SimpleNamespace()),
        db_pool=_Service("db_pool", events),
        session_manager=_Service("session_manager", events),
        heavy_startup_handles=SimpleNamespace(
            mcp_server=_Service("mcp_server", events), provider_manager=None, request_queue=None
        ),
        in_pytest_for_db_pool_shutdown=False,
        in_pytest_for_tts_shutdown=False,
        import_exceptions=(ImportError,),
        startup_guard_exceptions=(RuntimeError,),
        timed_shutdown_segment=_untimed,
    )
    assert handles == timed.CleanupTimedShutdownHandles()
    assert events == ORDER
