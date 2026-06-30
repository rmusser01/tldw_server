from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import HTTPException
from loguru import logger

import pytest

from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileBulkUpdateRequest,
    UserProfileUpdateEntry,
)

pytestmark = pytest.mark.unit

_LEAK = "admin backend exploded at /tmp/admin-secret-token"


def _assert_safe_log(rendered: str) -> None:
    assert "admin backend exploded" not in rendered
    assert "/tmp/admin-secret-token" not in rendered
    assert "exc_info" not in rendered


def _capture_logs():
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    return records, sink_id


def test_bundle_size_estimate_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_bundle_service as bundle_service

    def _fail_resolve_dataset_path(*_args, **_kwargs):
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(bundle_service, "_resolve_dataset_db_path", _fail_resolve_dataset_path)

    records, sink_id = _capture_logs()
    try:
        assert bundle_service._estimate_total_db_size(["authnz"], user_id=None) == 1024
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))


def test_retention_floor_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_data_ops_service as data_ops_service

    class _FailingSettings:
        def __getattr__(self, _name):
            raise RuntimeError(_LEAK)

    monkeypatch.setattr(data_ops_service, "get_settings", lambda: _FailingSettings())

    records, sink_id = _capture_logs()
    try:
        assert data_ops_service._effective_retention_days("privilege_snapshots_weekly", 7) == 7
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))


def test_bulk_confirm_threshold_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    monkeypatch.delenv("BULK_UPDATE_CONFIRM_THRESHOLD", raising=False)

    def _fail_load_config():
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(profiles_service, "load_comprehensive_config", _fail_load_config)

    records, sink_id = _capture_logs()
    try:
        assert profiles_service._get_bulk_confirm_threshold() == 1000
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))


def test_bulk_candidate_user_id_log_omits_raw_exception() -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    class _ExplodingUser:
        def get(self, _key):
            raise RuntimeError(_LEAK)

    records, sink_id = _capture_logs()
    try:
        assert profiles_service._coerce_bulk_candidate_user_id(_ExplodingUser()) is None
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))


@pytest.mark.asyncio
async def test_list_user_profiles_includes_canonical_page_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    class _ProfileService:
        def __init__(self, _db_pool):
            pass

        def parse_sections(self, _sections):
            return {"identity"}

        def _get_metrics_registry(self):
            return None

    async def _empty_candidates(**_kwargs):
        return []

    async def _db_pool():
        return object()

    async def _api_key_manager():
        return object()

    async def _repo_from_pool():
        return object()

    monkeypatch.setattr(profiles_service, "_load_bulk_user_candidates", _empty_candidates)
    monkeypatch.setattr(profiles_service, "get_db_pool", _db_pool)
    monkeypatch.setattr(profiles_service, "get_api_key_manager", _api_key_manager)
    monkeypatch.setattr(profiles_service, "UserProfileService", _ProfileService)
    monkeypatch.setattr(profiles_service, "AuthnzUsersRepo", SimpleNamespace(from_pool=_repo_from_pool))

    response, _audit = await profiles_service.list_user_profiles(
        principal=object(),
        sections=None,
        include_sources=False,
        include_raw=False,
        mask_secrets=True,
        user_ids=None,
        org_id=None,
        team_id=None,
        role=None,
        is_active=None,
        search=None,
        page=1,
        limit=10,
        session_manager=object(),
    )

    assert response.pagination.model_dump(mode="json") == {
        "mode": "page",
        "page": 1,
        "per_page": 10,
        "total": 0,
        "total_pages": 0,
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_profile_batch_telemetry_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    class _FailingRegistry:
        def observe(self, *_args, **_kwargs):
            raise RuntimeError(_LEAK)

    class _ProfileService:
        def __init__(self, _db_pool):
            pass

        def parse_sections(self, _sections):
            return {"identity"}

        def _get_metrics_registry(self):
            return _FailingRegistry()

    async def _empty_candidates(**_kwargs):
        return []

    async def _db_pool():
        return object()

    async def _api_key_manager():
        return object()

    async def _repo_from_pool():
        return object()

    monkeypatch.setattr(profiles_service, "_load_bulk_user_candidates", _empty_candidates)
    monkeypatch.setattr(profiles_service, "get_db_pool", _db_pool)
    monkeypatch.setattr(profiles_service, "get_api_key_manager", _api_key_manager)
    monkeypatch.setattr(profiles_service, "UserProfileService", _ProfileService)
    monkeypatch.setattr(profiles_service, "AuthnzUsersRepo", SimpleNamespace(from_pool=_repo_from_pool))

    records, sink_id = _capture_logs()
    try:
        response, _audit = await profiles_service.list_user_profiles(
            principal=object(),
            sections=None,
            include_sources=False,
            include_raw=False,
            mask_secrets=True,
            user_ids=None,
            org_id=None,
            team_id=None,
            role=None,
            is_active=None,
            search=None,
            page=1,
            limit=10,
            session_manager=object(),
        )
    finally:
        logger.remove(sink_id)

    assert response.total == 0
    _assert_safe_log("\n".join(records))


@pytest.mark.asyncio
async def test_get_user_profile_failure_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    async def _allow_scope(*_args, **_kwargs):
        return None

    async def _repo_from_pool():
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(profiles_service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(profiles_service, "AuthnzUsersRepo", SimpleNamespace(from_pool=_repo_from_pool))

    records, sink_id = _capture_logs()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await profiles_service.get_user_profile(
                user_id=42,
                principal=object(),
                sections=None,
                include_sources=False,
                include_raw=False,
                mask_secrets=True,
                session_manager=object(),
            )
    finally:
        logger.remove(sink_id)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to retrieve user profile"
    _assert_safe_log("\n".join(records))


@pytest.mark.asyncio
async def test_bulk_profile_update_user_failure_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    class _ProfileService:
        def __init__(self, _db_pool):
            pass

        async def get_profile_version(self, *, user_id: int):
            return datetime.now(timezone.utc)

        def _get_metrics_registry(self):
            return None

    class _UpdateService:
        def __init__(self, _db_pool):
            pass

        async def apply_updates(self, **_kwargs):
            raise RuntimeError(_LEAK)

    async def _allow_scope(*_args, **_kwargs):
        return None

    async def _candidates(**_kwargs):
        return [42]

    async def _before_values(**_kwargs):
        return {}

    async def _repo_from_pool():
        return object()

    async def _db_pool():
        return object()

    monkeypatch.setattr(profiles_service.admin_scope_service, "enforce_admin_user_scope", _allow_scope)
    monkeypatch.setattr(profiles_service, "_load_bulk_user_candidates", _candidates)
    monkeypatch.setattr(profiles_service, "_get_bulk_confirm_threshold", lambda: 1000)
    monkeypatch.setattr(profiles_service, "get_db_pool", _db_pool)
    monkeypatch.setattr(profiles_service, "UserProfileService", _ProfileService)
    monkeypatch.setattr(profiles_service, "UserProfileUpdateService", _UpdateService)
    monkeypatch.setattr(profiles_service, "_build_bulk_update_before_values", _before_values)
    monkeypatch.setattr(profiles_service, "load_user_profile_catalog", lambda: SimpleNamespace(entries=[]))
    monkeypatch.setattr(profiles_service, "AuthnzUsersRepo", SimpleNamespace(from_pool=_repo_from_pool))

    records, sink_id = _capture_logs()
    try:
        response, _audit = await profiles_service.bulk_update_user_profiles(
            payload=UserProfileBulkUpdateRequest(
                updates=[UserProfileUpdateEntry(key="identity.role", value="user")],
                dry_run=True,
            ),
            principal=SimpleNamespace(
                user_id=7,
                active_org_id=None,
                active_team_id=None,
                roles=["admin"],
                permissions=["*"],
            ),
        )
    finally:
        logger.remove(sink_id)

    assert response.failed == 1
    assert response.results[0].error == "update_failed"
    _assert_safe_log("\n".join(records))


@pytest.mark.asyncio
async def test_bulk_profile_update_metrics_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    class _FailingRegistry:
        def increment(self, *_args, **_kwargs):
            raise RuntimeError(_LEAK)

    class _ProfileService:
        def __init__(self, _db_pool):
            pass

        def _get_metrics_registry(self):
            return _FailingRegistry()

    class _UpdateService:
        def __init__(self, _db_pool):
            pass

    async def _empty_candidates(**_kwargs):
        return []

    async def _repo_from_pool():
        return object()

    async def _db_pool():
        return object()

    monkeypatch.setattr(profiles_service, "_load_bulk_user_candidates", _empty_candidates)
    monkeypatch.setattr(profiles_service, "_get_bulk_confirm_threshold", lambda: 1000)
    monkeypatch.setattr(profiles_service, "get_db_pool", _db_pool)
    monkeypatch.setattr(profiles_service, "UserProfileService", _ProfileService)
    monkeypatch.setattr(profiles_service, "UserProfileUpdateService", _UpdateService)
    monkeypatch.setattr(profiles_service, "load_user_profile_catalog", lambda: SimpleNamespace(entries=[]))
    monkeypatch.setattr(profiles_service, "AuthnzUsersRepo", SimpleNamespace(from_pool=_repo_from_pool))

    records, sink_id = _capture_logs()
    try:
        response, _audit = await profiles_service.bulk_update_user_profiles(
            payload=UserProfileBulkUpdateRequest(
                updates=[UserProfileUpdateEntry(key="identity.role", value="user")],
                dry_run=True,
            ),
            principal=SimpleNamespace(
                user_id=7,
                active_org_id=None,
                active_team_id=None,
                roles=["admin"],
                permissions=["*"],
            ),
        )
    finally:
        logger.remove(sink_id)

    assert response.total_targets == 0
    _assert_safe_log("\n".join(records))
