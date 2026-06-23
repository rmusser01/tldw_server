from __future__ import annotations

from datetime import datetime, timezone

from tldw_Server_API.app.core.AuthNZ.privilege_catalog import PrivilegeCatalog
from tldw_Server_API.app.core.PrivilegeMaps.service import PrivilegeMapService


def _build_catalog() -> PrivilegeCatalog:

    payload = {
        "version": "test-roles-1.0",
        "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
        "scopes": [
            {
                "id": "rag.search",
                "description": "Run RAG search queries.",
                "resource_tags": ["rag"],
                "sensitivity_tier": "moderate",
                "rate_limit_class": "standard",
                "default_roles": ["analyst"],
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
                "notes": "Unit test tier",
            }
        ],
        "ownership_predicates": [],
    }
    return PrivilegeCatalog.model_validate(payload)


def test_scope_resolution_normalizes_role_case():

    catalog = _build_catalog()
    service = PrivilegeMapService(route_registry={}, catalog=catalog)

    scopes = service._resolve_scopes_for_user(["Analyst"], [])

    assert "rag.search" in scopes


def test_group_by_role_normalizes_case():

    catalog = _build_catalog()
    service = PrivilegeMapService(route_registry={}, catalog=catalog)

    users = [
        {
            "id": "1",
            "username": "Casey",
            "primary_role": "Admin",
            "allowed_scopes": {"rag.search"},
        },
        {
            "id": "2",
            "username": "Chris",
            "primary_role": "admin",
            "allowed_scopes": {"rag.search"},
        },
    ]

    buckets = service._group_by_role(users)
    bucket_map = {bucket["key"]: bucket for bucket in buckets}

    assert set(bucket_map.keys()) == {"admin"}
    assert bucket_map["admin"]["users"] == 2


def test_detail_role_filter_is_case_insensitive():

    catalog = _build_catalog()
    service = PrivilegeMapService(route_registry={}, catalog=catalog)

    users = [
        {
            "id": "1",
            "username": "Casey",
            "primary_role": "Admin",
            "roles": ["Admin"],
            "permissions": [],
            "feature_flags": set(),
            "allowed_scopes": {"rag.search"},
        },
        {
            "id": "2",
            "username": "Chris",
            "primary_role": "Analyst",
            "roles": ["Analyst"],
            "permissions": [],
            "feature_flags": set(),
            "allowed_scopes": {"rag.search"},
        }
    ]

    items = service._build_detail_items(
        users,
        resource_filter=None,
        role_filter="admin",
    )

    assert items
    assert {item["user_id"] for item in items} == {"1"}
    assert {item["role"] for item in items} == {"Admin"}
