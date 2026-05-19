from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.router_groups.content import API_V1_PREFIX, iter_content_router_specs
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ
from tldw_Server_API.app.core.Research_Studio.capabilities import build_research_studio_capabilities


def _closure_values(callable_obj: Any) -> list[Any]:
    closure = getattr(callable_obj, "__closure__", None) or ()
    return [cell.cell_contents for cell in closure]


def test_content_router_specs_include_research_studio_capabilities_router():
    specs = list(iter_content_router_specs())

    matching = [spec for spec in specs if spec.route_key == "research-studio"]

    assert len(matching) == 1
    assert matching[0].prefix == API_V1_PREFIX
    assert matching[0].tags == ("research-studio",)


def test_research_studio_capabilities_route_is_permission_gated_and_rate_limited():
    from tldw_Server_API.app.api.v1.endpoints import research_studio

    route = next(
        route
        for route in research_studio.router.routes
        if getattr(route, "path", None) == "/research-studio/capabilities"
    )
    dependency_calls = [dependency.call for dependency in route.dependant.dependencies]

    assert any(
        MEDIA_READ in value
        for call in dependency_calls
        for value in _closure_values(call)
        if isinstance(value, list)
    )
    assert any(
        getattr(call, "_tldw_rate_limit_resource", None) == "research_studio.capabilities"
        for call in dependency_calls
    )


@pytest.mark.asyncio
async def test_research_studio_capabilities_endpoint_returns_user_scoped_payload(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import research_studio

    seen: dict[str, Any] = {}

    async def fake_collect_research_studio_capabilities(*, user_id: int | str | None = None):
        seen["user_id"] = user_id
        return build_research_studio_capabilities(
            aggregate_health={
                "status": "ok",
                "checks": {"database": {"status": "healthy"}, "chacha_notes": {"status": "healthy"}},
            },
            rag_health={"status": "degraded"},
            llm_health={"status": "healthy", "components": {"providers": {"initialized": True, "count": 1}}},
            slides_health={"status": "ok"},
            tts_health={"status": "healthy", "providers": {"available": 1}},
        )

    monkeypatch.setattr(
        research_studio,
        "collect_research_studio_capabilities",
        fake_collect_research_studio_capabilities,
    )

    result = await research_studio.research_studio_capabilities(current_user=SimpleNamespace(id=42))

    assert seen == {"user_id": 42}
    assert result.capabilities["chat"].mode == "warn"
    assert result.capabilities["chat"].reason_code == "rag_degraded"
    assert "api_key" not in result.model_dump_json()


def test_openapi_contains_research_studio_capabilities_path():
    from tldw_Server_API.app.main import app

    schema = app.openapi()

    assert "/api/v1/research-studio/capabilities" in schema["paths"]
