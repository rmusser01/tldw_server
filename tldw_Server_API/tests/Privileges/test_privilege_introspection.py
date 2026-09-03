from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import APIRouter, Depends, FastAPI

from tldw_Server_API.app.core.AuthNZ.privilege_catalog import PrivilegeCatalog, load_catalog
from tldw_Server_API.app.core.PrivilegeMaps import startup as privilege_startup
from tldw_Server_API.app.core.PrivilegeMaps.introspection import (
    collect_privilege_route_registry,
    serialize_route_registry,
)
from tldw_Server_API.app.main import app as fastapi_app


def _build_test_catalog() -> PrivilegeCatalog:
    """Construct a minimal privilege catalog for introspection unit tests."""
    payload: dict[str, object] = {
        "version": "test-1.0",
        "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
        "scopes": [
            {
                "id": "media.ingest",
                "description": "Ingest media assets.",
                "resource_tags": ["media", "ingest"],
                "sensitivity_tier": "high",
                "rate_limit_class": "standard",
                "default_roles": ["admin"],
                "feature_flag_id": None,
                "ownership_predicates": [],
                "doc_url": None,
            }
        ],
        "feature_flags": [],
        "rate_limit_classes": [
            {
                "id": "standard",
                "requests_per_min": 60,
                "burst": 10,
                "notes": "Test tier",
            }
        ],
        "ownership_predicates": [],
    }
    return PrivilegeCatalog.model_validate(payload)


def _make_async_dependency(
    name: str,
    *,
    scope: str | None = None,
    endpoint: str | None = None,
    rate_resource: str | None = None,
    rate_resources: tuple[str, ...] | None = None,
):
    async def _dependency():
        return None

    _dependency.__name__ = name
    _dependency.__qualname__ = name
    if scope:
        _dependency._tldw_scope_name = scope
    if endpoint:
        _dependency._tldw_endpoint_id = endpoint
    if rate_resource:
        _dependency._tldw_rate_limit_resource = rate_resource
    if rate_resources:
        _dependency._tldw_rate_limit_resources = rate_resources
    return _dependency


def test_collect_privilege_route_registry_supports_dynamic_rate_resources():
    catalog = _build_test_catalog()
    app = FastAPI()
    scope_dep = _make_async_dependency("scope", scope="media.ingest")
    rate_dep = _make_async_dependency(
        "dynamic_rate",
        rate_resources=("media.ingest", "rag.search"),
    )

    @app.get("/dynamic", dependencies=[Depends(scope_dep), Depends(rate_dep)])
    async def dynamic_route():
        return {"status": "ok"}

    registry = collect_privilege_route_registry(app, catalog)

    assert registry["media.ingest"][0].rate_limit_resources == (
        "media.ingest",
        "rag.search",
    )


def test_collect_privilege_route_registry_captures_metadata():

    catalog = _build_test_catalog()
    app = FastAPI()
    router = APIRouter()

    scope_dep = _make_async_dependency(
        "media_scope_guard",
        scope="media.ingest",
    )
    rate_dep = _make_async_dependency(
        "media_rate_limiter",
        rate_resource="media.ingest",
    )

    @router.post(
        "/api/v1/media/process",
        tags=["media", "catalog"],
        summary="Process media",
        description="Upload or process media payloads.",
        dependencies=[Depends(scope_dep), Depends(rate_dep)],
    )
    async def media_process():
        return {"status": "ok"}

    app.include_router(router)

    registry = collect_privilege_route_registry(app, catalog, strict=True)

    assert "media.ingest" in registry
    assert len(registry["media.ingest"]) == 1
    metadata = registry["media.ingest"][0]

    assert metadata.path == "/api/v1/media/process"
    assert metadata.methods == ("POST",)
    assert metadata.tags == ("media", "catalog")
    assert metadata.summary == "Process media"
    assert metadata.description == "Upload or process media payloads."
    assert metadata.endpoint.endswith("media_process")

    # Dependencies are sorted alphabetically
    dependency_ids = [dep.id for dep in metadata.dependencies]
    assert dependency_ids == [
        "test_privilege_introspection.media_scope_guard",
        "test_privilege_introspection.media_rate_limiter",
    ]
    assert metadata.rate_limit_resources == ("media.ingest",)
    assert {dep.module for dep in metadata.dependencies} == {
        scope_dep.__module__,
        rate_dep.__module__,
    }
    assert set(metadata.dependency_sources) == {
        f"{scope_dep.__module__}.{scope_dep.__qualname__}",
        f"{rate_dep.__module__}.{rate_dep.__qualname__}",
    }


def test_collect_privilege_route_registry_strict_unknown_scope():

    catalog = _build_test_catalog()
    app = FastAPI()
    router = APIRouter()

    unknown_scope_dep = _make_async_dependency(
        "unknown_scope_guard",
        scope="media.unknown",
    )

    @router.get("/api/v1/claims/debug", dependencies=[Depends(unknown_scope_dep)])
    async def debug_endpoint():
        return {"status": "ok"}

    app.include_router(router)

    with pytest.raises(ValueError) as exc_info:
        collect_privilege_route_registry(app, catalog, strict=True)

    assert "media.unknown" in str(exc_info.value)


def test_validate_privilege_metadata_on_startup_invokes_strict_mode(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    catalog = _build_test_catalog()
    sample_registry: dict[str, list[object]] = {"media.ingest": []}
    calls: dict[str, object] = {}
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "1")

    def fake_load_catalog() -> PrivilegeCatalog:

        calls["load_catalog"] = True
        return catalog

    def fake_collect_registry(
        app_arg: FastAPI,
        catalog_arg: PrivilegeCatalog,
        *,
        strict: bool,
    ):
        calls["collect_called"] = strict
        assert strict is True
        assert app_arg is app
        assert catalog_arg is catalog
        return sample_registry

    monkeypatch.setattr(privilege_startup, "load_catalog", fake_load_catalog)
    monkeypatch.setattr(privilege_startup, "collect_privilege_route_registry", fake_collect_registry)

    registry = privilege_startup.validate_privilege_metadata_on_startup(app)
    assert registry is sample_registry
    assert calls["load_catalog"] is True
    assert calls["collect_called"] is True


def test_serialize_route_registry_outputs_deterministic_structure():

    catalog = _build_test_catalog()
    app = FastAPI()
    router = APIRouter()

    scope_dep = _make_async_dependency(
        "media_scope_guard",
        scope="media.ingest",
    )
    rate_dep = _make_async_dependency(
        "media_rate_limiter",
        rate_resource="media.ingest",
    )

    @router.post(
        "/api/v1/media/process",
        summary="Process media",
        description="Upload media assets.",
        tags=["media", "ingest"],
        dependencies=[Depends(scope_dep), Depends(rate_dep)],
    )
    async def media_process():
        return {"status": "ok"}

    app.include_router(router)

    registry = collect_privilege_route_registry(app, catalog, strict=True)
    serialized_first = serialize_route_registry(registry)
    serialized_second = serialize_route_registry(registry)

    assert serialized_first == serialized_second
    assert list(serialized_first.keys()) == ["media.ingest"]
    entry = serialized_first["media.ingest"][0]
    assert entry["path"] == "/api/v1/media/process"
    assert entry["methods"] == ["POST"]
    assert entry["tags"] == ["media", "ingest"]
    assert entry["name"] == "media_process"
    assert entry["endpoint"].endswith("media_process")
    assert entry["summary"] == "Process media"
    assert entry["description"] == "Upload media assets."
    assert entry["rate_limit_resources"] == ["media.ingest"]
    dependency_ids = {dep["id"] for dep in entry["dependencies"]}
    assert dependency_ids == {
        "test_privilege_introspection.media_scope_guard",
        "test_privilege_introspection.media_rate_limiter",
    }
    for dep in entry["dependencies"]:
        assert dep["type"] == "dependency"
        assert dep["module"].endswith("test_privilege_introspection")


def test_privilege_registry_snapshot_matches_live_app():

    catalog = load_catalog()
    registry = collect_privilege_route_registry(fastapi_app, catalog, strict=True)
    serialized = serialize_route_registry(registry)

    snapshot_path = Path("tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json")
    assert snapshot_path.exists(), "Missing privilege registry snapshot fixture."
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert serialized == expected, (
        "Privilege route registry snapshot is stale. "
        "Run Helper_Scripts/update_privilege_registry_snapshot.py to regenerate and commit the updated snapshot."
    )
