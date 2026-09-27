from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

import pytest
from fastapi import Depends, FastAPI
from pydantic import ValidationError

from tldw_Server_API.app.core.AuthNZ.privilege_catalog import load_catalog
from tldw_Server_API.app.core.PrivilegeMaps.introspection import (
    collect_privilege_route_registry,
    serialize_route_registry,
)


def _write_catalog(tmp_path: Path, rate_limit_class: str = "standard") -> Path:
    payload = f"""
version: 1.0.0
updated_at: 2025-01-01T00:00:00Z
scopes:
  - id: test.scope
    description: Test scope
    resource_tags:
      - test
    sensitivity_tier: low
    rate_limit_class: {rate_limit_class}
    default_roles:
      - admin
    feature_flag_id: null
    ownership_predicates: []
    doc_url: null
feature_flags: []
rate_limit_classes:
  - id: standard
    requests_per_min: 100
    burst: 200
    notes: test quota
ownership_predicates: []
""".strip()
    path = tmp_path / "privilege_catalog.yaml"
    path.write_text(payload, encoding="utf-8")
    return path


def _write_catalog_payload(tmp_path: Path, payload: str) -> Path:
    path = tmp_path / "privilege_catalog.yaml"
    path.write_text(payload.strip(), encoding="utf-8")
    return path


def _assign_scope(scope_name: str) -> Callable[[], None]:
    def _dependency() -> None:
        return None

    setattr(_dependency, "_tldw_scope_name", scope_name)
    return _dependency


def test_load_catalog_success(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    catalog = load_catalog(catalog_path)
    assert catalog.version == "1.0.0"
    assert [scope.id for scope in catalog.scopes] == ["test.scope"]


def test_vn_preflight_catalog_has_finite_rate_policy() -> None:
    """Preflight must not silently lose its catalog-backed rate limit."""
    catalog = load_catalog()
    scope = next(scope for scope in catalog.scopes if scope.id == "vn_assets.preflight")
    policy = next(policy for policy in catalog.rate_limit_classes if policy.id == scope.rate_limit_class)

    assert policy.requests_per_min > 0
    assert policy.burst > 0


def test_load_catalog_invalid_rate_limit(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path, rate_limit_class="unknown")
    with pytest.raises(ValidationError):
        load_catalog(catalog_path)


def test_load_catalog_rejects_duplicate_feature_flag_ids(tmp_path: Path) -> None:
    catalog_path = _write_catalog_payload(
        tmp_path,
        """
version: 1.0.0
updated_at: 2025-01-01T00:00:00Z
scopes:
  - id: test.scope
    description: Test scope
    resource_tags: [test]
    sensitivity_tier: low
    rate_limit_class: standard
    default_roles: [admin]
    feature_flag_id: ff.dup
    ownership_predicates: []
    doc_url: null
feature_flags:
  - id: ff.dup
    description: First
    default_state: enabled
    allowed_roles: []
    expires_at: null
  - id: ff.dup
    description: Duplicate
    default_state: disabled
    allowed_roles: []
    expires_at: null
rate_limit_classes:
  - id: standard
    requests_per_min: 100
    burst: 200
    notes: test
ownership_predicates: []
        """,
    )
    with pytest.raises(ValidationError, match="Duplicate feature flag id detected"):
        load_catalog(catalog_path)


def test_load_catalog_rejects_duplicate_rate_limit_class_ids(tmp_path: Path) -> None:
    catalog_path = _write_catalog_payload(
        tmp_path,
        """
version: 1.0.0
updated_at: 2025-01-01T00:00:00Z
scopes:
  - id: test.scope
    description: Test scope
    resource_tags: [test]
    sensitivity_tier: low
    rate_limit_class: standard
    default_roles: [admin]
    feature_flag_id: null
    ownership_predicates: []
    doc_url: null
feature_flags: []
rate_limit_classes:
  - id: standard
    requests_per_min: 100
    burst: 200
    notes: first
  - id: standard
    requests_per_min: 50
    burst: 75
    notes: duplicate
ownership_predicates: []
        """,
    )
    with pytest.raises(ValidationError, match="Duplicate rate limit class id detected"):
        load_catalog(catalog_path)


def test_load_catalog_rejects_duplicate_ownership_predicate_ids(tmp_path: Path) -> None:
    catalog_path = _write_catalog_payload(
        tmp_path,
        """
version: 1.0.0
updated_at: 2025-01-01T00:00:00Z
scopes:
  - id: test.scope
    description: Test scope
    resource_tags: [test]
    sensitivity_tier: low
    rate_limit_class: standard
    default_roles: [admin]
    feature_flag_id: null
    ownership_predicates: [owner.match]
    doc_url: null
feature_flags: []
rate_limit_classes:
  - id: standard
    requests_per_min: 100
    burst: 200
    notes: test
ownership_predicates:
  - id: owner.match
    evaluator: owner_match
    description: first
  - id: owner.match
    evaluator: owner_match_alt
    description: duplicate
        """,
    )
    with pytest.raises(ValidationError, match="Duplicate ownership predicate id detected"):
        load_catalog(catalog_path)


def test_import_smoke_privilege_modules() -> None:
    startup = importlib.import_module("tldw_Server_API.app.core.PrivilegeMaps.startup")
    service = importlib.import_module("tldw_Server_API.app.core.PrivilegeMaps.service")
    assert hasattr(startup, "validate_privilege_metadata_on_startup")
    assert hasattr(service, "PrivilegeMapService")


def test_collect_privilege_registry_strict_unknown_scope(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    catalog = load_catalog(catalog_path)

    app = FastAPI()

    @app.get("/known", dependencies=[Depends(_assign_scope("test.scope"))])
    def _known():
        return {"ok": True}

    @app.get("/unknown", dependencies=[Depends(_assign_scope("unknown.scope"))])
    def _unknown():
        return {"ok": False}

    # Non-strict mode should not raise, but strict mode must catch the unknown scope.
    collect_privilege_route_registry(app, catalog, strict=False)

    with pytest.raises(ValueError) as excinfo:
        collect_privilege_route_registry(app, catalog, strict=True)

    assert "unknown.scope" in str(excinfo.value)
    assert "/unknown" in str(excinfo.value)


def test_collect_privilege_registry_strict_success(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    catalog = load_catalog(catalog_path)

    app = FastAPI()

    @app.get("/known", dependencies=[Depends(_assign_scope("test.scope"))])
    def _known():
        return {"ok": True}

    registry = collect_privilege_route_registry(app, catalog, strict=True)
    assert "test.scope" in registry
    assert registry["test.scope"][0].path == "/known"


def test_serialize_route_registry_orders_entries(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path)
    catalog = load_catalog(catalog_path)
    app = FastAPI()

    @app.get("/alpha", dependencies=[Depends(_assign_scope("test.scope"))])
    def _alpha():
        return {"ok": True}

    registry = collect_privilege_route_registry(app, catalog, strict=True)
    serialized = serialize_route_registry(registry)
    assert list(serialized.keys()) == ["test.scope"]
    entry = serialized["test.scope"][0]
    assert entry["path"] == "/alpha"
    assert entry["dependencies"][0]["type"] == "dependency"
