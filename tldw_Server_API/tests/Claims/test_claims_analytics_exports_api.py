from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, Response

from tldw_Server_API.app.api.v1.endpoints import claims as claims_endpoint
from tldw_Server_API.app.api.v1.schemas.claims_schemas import (
    ClaimsAnalyticsExportFilters,
    ClaimsAnalyticsExportRequest,
)
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_analytics_exports, claims_jobs, claims_service
from tldw_Server_API.app.core.Claims_Extraction.claims_analytics_exports import ClaimsAnalyticsExportError
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

pytestmark = pytest.mark.unit

_SNAPSHOT = "2026-08-08T12:00:00.000Z"
_EXPORT_ID = "a" * 32


class _FakeDb:
    def __init__(self, backend_type: BackendType = BackendType.SQLITE) -> None:
        self.backend_type = backend_type
        self.attach_calls: list[dict[str, Any]] = []
        self.transition_calls: list[dict[str, Any]] = []
        self.attach_error: Exception | None = None

    def attach_claims_analytics_export_job(self, **kwargs: Any) -> bool:
        self.attach_calls.append(kwargs)
        if self.attach_error is not None:
            raise self.attach_error
        return True

    def transition_claims_analytics_export_status(self, **kwargs: Any) -> bool:
        self.transition_calls.append(kwargs)
        return True


def _principal(*, platform_admin: bool = True) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        subject="claims-admin",
        roles=["admin"] if platform_admin else ["claims_admin"],
        permissions=[CLAIMS_ADMIN],
        is_admin=platform_admin,
    )


def _user(user_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(id=user_id, username=f"user-{user_id}")


def _normalized(owner: str, *, format: str = "json") -> dict[str, Any]:
    return {
        "owner_user_id": owner,
        "format": format,
        "filters": {"event_type": "unsupported_ratio", "end_time": _SNAPSHOT},
        "pagination": {"limit": 10, "offset": 0},
        "snapshot_at": _SNAPSHOT,
    }


def _row(owner: str, *, status: str, format: str = "json") -> dict[str, Any]:
    return {
        "export_id": _EXPORT_ID,
        "user_id": owner,
        "format": format,
        "status": status,
        "job_id": None,
        "error_code": None,
        "snapshot_at": _SNAPSHOT,
        "created_at": "2026-08-08T12:00:01.000Z",
    }


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool,
    expected_owner: str = "1",
) -> list[dict[str, Any]]:
    normalized_calls: list[dict[str, Any]] = []

    def _normalize(payload: dict[str, Any], *, owner_user_id: str) -> dict[str, Any]:
        normalized_calls.append({"payload": payload, "owner_user_id": owner_user_id})
        assert owner_user_id == expected_owner
        assert "workspace_id" not in (payload.get("filters") or {})
        return _normalized(owner_user_id, format=str(payload.get("format") or "json"))

    monkeypatch.setattr(claims_jobs, "claims_analytics_export_jobs_enabled", lambda: enabled)
    monkeypatch.setattr(claims_analytics_exports, "normalize_export_request", _normalize)
    monkeypatch.setattr(claims_analytics_exports, "reconcile_export_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(claims_analytics_exports, "cleanup_export_artifacts", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: object(), raising=False)
    return normalized_calls


def test_sync_create_returns_ready_without_job(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    calls = _patch_common(monkeypatch, enabled=False)
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_ready_artifact",
        lambda actual_db, **_kwargs: _row("1", status="ready") if actual_db is db else pytest.fail("wrong db"),
    )
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_analytics_export",
        lambda **_kwargs: pytest.fail("synchronous export must not enqueue"),
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={
            "format": "json",
            "filters": {"event_type": "unsupported_ratio"},
            "pagination": {"limit": 10, "offset": 0},
        },
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 200
    assert body == {
        "export_id": _EXPORT_ID,
        "format": "json",
        "status": "ready",
        "download_url": f"/api/v1/claims/analytics/export/{_EXPORT_ID}",
        "created_at": "2026-08-08T12:00:01.000Z",
        "job_id": None,
        "job_status": None,
        "error_code": None,
        "snapshot_at": _SNAPSHOT,
    }
    assert calls and calls[0]["owner_user_id"] == "1"


@pytest.mark.parametrize(
    "export_request",
    [
        ClaimsAnalyticsExportRequest(format="json", filters=None, pagination=None),
        ClaimsAnalyticsExportRequest(
            format="json",
            filters=ClaimsAnalyticsExportFilters(workspace_id=None),
        ),
    ],
)
def test_explicit_null_optional_request_fields_are_treated_as_omitted(
    monkeypatch: pytest.MonkeyPatch,
    export_request: ClaimsAnalyticsExportRequest,
) -> None:
    db = _FakeDb()
    normalized_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(claims_jobs, "claims_analytics_export_jobs_enabled", lambda: False)
    monkeypatch.setattr(claims_analytics_exports, "reconcile_export_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(claims_analytics_exports, "cleanup_export_artifacts", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: object())

    def _create_ready(_db: Any, *, normalized: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        normalized_calls.append(normalized)
        return _row("1", status="ready")

    monkeypatch.setattr(claims_analytics_exports, "create_ready_artifact", _create_ready)

    body, response_status = claims_service.export_claims_analytics(
        payload=export_request.model_dump(exclude_unset=True),
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 200
    assert body["status"] == "ready"
    assert normalized_calls[0]["owner_user_id"] == "1"
    assert normalized_calls[0]["pagination"] == {"limit": 1000, "offset": 0}
    assert "workspace_id" not in normalized_calls[0]["filters"]


@pytest.mark.parametrize(
    ("claims_enabled", "exports_enabled"),
    [(False, False), (False, True), (True, False)],
)
def test_create_remains_synchronous_when_either_producer_flag_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    claims_enabled: bool,
    exports_enabled: bool,
) -> None:
    db = _FakeDb()
    real_enabled = claims_jobs.claims_analytics_export_jobs_enabled
    _patch_common(monkeypatch, enabled=False)
    monkeypatch.setattr(
        claims_jobs,
        "claims_analytics_export_jobs_enabled",
        lambda: real_enabled(
            {
                "CLAIMS_JOBS_ENABLED": claims_enabled,
                "CLAIMS_ANALYTICS_EXPORT_JOBS_ENABLED": exports_enabled,
            }
        ),
    )
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_ready_artifact",
        lambda actual_db, **_kwargs: _row("1", status="ready") if actual_db is db else pytest.fail("wrong db"),
    )
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_analytics_export",
        lambda **_kwargs: pytest.fail("disabled export producer must not enqueue"),
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json"},
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 200
    assert body["status"] == "ready"
    assert body["job_id"] is None


def test_jobs_client_failure_keeps_maintenance_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeDb()
    maintenance_calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(claims_jobs, "claims_analytics_export_jobs_enabled", lambda: False)
    monkeypatch.setattr(
        claims_analytics_exports,
        "normalize_export_request",
        lambda _payload, *, owner_user_id: _normalized(owner_user_id),
    )
    monkeypatch.setattr(
        claims_service,
        "jobs_manager_from_env",
        lambda: (_ for _ in ()).throw(RuntimeError("Jobs unavailable")),
    )
    monkeypatch.setattr(
        claims_analytics_exports,
        "reconcile_export_artifacts",
        lambda _db, *, job_manager, **_kwargs: maintenance_calls.append(("reconcile", job_manager)),
    )
    monkeypatch.setattr(
        claims_analytics_exports,
        "cleanup_export_artifacts",
        lambda _db, *, job_manager, **_kwargs: maintenance_calls.append(("cleanup", job_manager)),
    )
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_ready_artifact",
        lambda *_args, **_kwargs: _row("1", status="ready"),
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json"},
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 200
    assert body["status"] == "ready"
    assert maintenance_calls == [("reconcile", None), ("cleanup", None)]


def test_async_create_returns_accepted_job_without_inline_render(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    _patch_common(monkeypatch, enabled=True)
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_queued_artifact",
        lambda actual_db, **_kwargs: _row("1", status="queued") if actual_db is db else pytest.fail("wrong db"),
    )
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_ready_artifact",
        lambda *_args, **_kwargs: pytest.fail("asynchronous export must not render inline"),
    )
    enqueue_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_analytics_export",
        lambda **kwargs: enqueue_calls.append(kwargs) or {"id": 81, "status": "queued"},
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json"},
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 202
    assert body["status"] == "queued"
    assert body["job_id"] == 81
    assert body["job_status"] == "queued"
    assert enqueue_calls[0]["owner_user_id"] == "1"
    assert enqueue_calls[0]["export_id"] == _EXPORT_ID
    assert db.attach_calls == [{"export_id": _EXPORT_ID, "user_id": "1", "job_id": 81}]


def test_enqueue_failure_marks_artifact_failed_and_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    _patch_common(monkeypatch, enabled=True)
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_queued_artifact",
        lambda *_args, **_kwargs: _row("1", status="queued"),
    )
    secret = "postgresql://owner:password@private-db/claims"

    def _fail_enqueue(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(secret)

    monkeypatch.setattr(claims_jobs, "enqueue_claims_analytics_export", _fail_enqueue)

    with pytest.raises(HTTPException) as exc_info:
        claims_service.export_claims_analytics(
            payload={"format": "json"},
            principal=_principal(),
            current_user=_user(),
            db=db,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "code": "claims_export_enqueue_failed",
        "message": "Claims analytics export could not be queued.",
    }
    assert secret not in repr(exc_info.value.detail)
    assert db.transition_calls == [
        {
            "export_id": _EXPORT_ID,
            "user_id": "1",
            "from_statuses": ("queued",),
            "to_status": "failed",
            "error_code": "claims_export_enqueue_failed",
            "error_message": "Claims analytics export could not be queued.",
        }
    ]


@pytest.mark.parametrize("job_id", [None, 0, -1, True, "81"])
def test_malformed_jobs_acceptance_id_remains_accepted_for_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    job_id: object,
) -> None:
    db = _FakeDb()
    _patch_common(monkeypatch, enabled=True)
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_queued_artifact",
        lambda *_args, **_kwargs: _row("1", status="queued"),
    )
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_analytics_export",
        lambda **_kwargs: {"id": job_id, "status": "queued"},
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json"},
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 202
    assert body["job_id"] is None
    assert body["job_status"] == "queued"
    assert db.attach_calls == []
    assert db.transition_calls == []


@pytest.mark.parametrize(
    "accepted",
    [
        {"id": 81},
        {"id": 81, "status": None},
        {"id": 81, "status": ""},
        {"id": 81, "status": "   "},
        {"id": 81, "status": 1},
    ],
)
def test_malformed_jobs_acceptance_status_remains_accepted_with_null_projection(
    monkeypatch: pytest.MonkeyPatch,
    accepted: dict[str, Any],
) -> None:
    db = _FakeDb()
    _patch_common(monkeypatch, enabled=True)
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_queued_artifact",
        lambda *_args, **_kwargs: _row("1", status="queued"),
    )
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_analytics_export",
        lambda **_kwargs: accepted,
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json"},
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 202
    assert body["job_id"] == 81
    assert body["job_status"] is None
    assert db.attach_calls == [{"export_id": _EXPORT_ID, "user_id": "1", "job_id": 81}]
    assert db.transition_calls == []


@pytest.mark.parametrize("accepted", [None, [], "accepted", 81])
def test_non_mapping_jobs_acceptance_remains_accepted_for_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    accepted: object,
) -> None:
    db = _FakeDb()
    _patch_common(monkeypatch, enabled=True)
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_queued_artifact",
        lambda *_args, **_kwargs: _row("1", status="queued"),
    )
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_analytics_export",
        lambda **_kwargs: accepted,
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json"},
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 202
    assert body["job_id"] is None
    assert body["job_status"] is None
    assert db.attach_calls == []
    assert db.transition_calls == []


def test_attach_failure_after_jobs_acceptance_still_returns_202(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    db.attach_error = RuntimeError("private database path")
    _patch_common(monkeypatch, enabled=True)
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_queued_artifact",
        lambda *_args, **_kwargs: _row("1", status="queued"),
    )
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_analytics_export",
        lambda **_kwargs: {"id": 81, "status": "queued"},
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json"},
        principal=_principal(),
        current_user=_user(),
        db=db,
    )

    assert response_status == 202
    assert body["job_id"] == 81
    assert body["job_status"] == "queued"


def test_platform_admin_cross_owner_sqlite_routes_target_database(monkeypatch: pytest.MonkeyPatch) -> None:
    caller_db = _FakeDb(BackendType.SQLITE)
    target_db = _FakeDb(BackendType.SQLITE)
    _patch_common(monkeypatch, enabled=False, expected_owner="2")
    routed: list[int] = []

    @contextmanager
    def _override(user_id: int):
        routed.append(user_id)
        yield target_db, f"/users/{user_id}/Media_DB_v2.db"

    monkeypatch.setattr(claims_service, "_claims_user_override_db", _override)
    ready_calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_ready_artifact",
        lambda actual_db, *, owner_user_id, normalized: (
            ready_calls.append((actual_db, owner_user_id)) or _row("2", status="ready")
        ),
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json", "filters": {"workspace_id": "2"}},
        principal=_principal(platform_admin=True),
        current_user=_user(1),
        db=caller_db,
    )

    assert response_status == 200
    assert routed == [2]
    assert ready_calls == [(target_db, "2")]
    assert body["download_url"] == f"/api/v1/claims/analytics/export/{_EXPORT_ID}?workspace_id=2"


def test_non_platform_admin_cannot_route_cross_owner_export(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    monkeypatch.setattr(
        claims_analytics_exports,
        "normalize_export_request",
        lambda *_args, **_kwargs: pytest.fail("authorization must precede normalization"),
    )

    with pytest.raises(HTTPException) as exc_info:
        claims_service.export_claims_analytics(
            payload={"format": "json", "filters": {"workspace_id": "2"}},
            principal=_principal(platform_admin=False),
            current_user=_user(1),
            db=db,
        )

    assert exc_info.value.status_code == 403


def test_platform_admin_cross_owner_postgres_retains_shared_database(monkeypatch: pytest.MonkeyPatch) -> None:
    shared_db = _FakeDb(BackendType.POSTGRESQL)
    _patch_common(monkeypatch, enabled=False, expected_owner="2")
    ready_calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_ready_artifact",
        lambda actual_db, *, owner_user_id, normalized: (
            ready_calls.append((actual_db, owner_user_id)) or _row("2", status="ready")
        ),
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json", "filters": {"workspace_id": "2"}},
        principal=_principal(platform_admin=True),
        current_user=_user(1),
        db=shared_db,
    )

    assert response_status == 200
    assert ready_calls == [(shared_db, "2")]
    assert body["download_url"].endswith("?workspace_id=2")


def test_platform_admin_async_cross_owner_postgres_keeps_owner_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_db = _FakeDb(BackendType.POSTGRESQL)
    _patch_common(monkeypatch, enabled=True, expected_owner="2")
    queued_calls: list[tuple[object, str]] = []
    enqueue_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        claims_analytics_exports,
        "create_queued_artifact",
        lambda actual_db, *, owner_user_id, normalized: (
            queued_calls.append((actual_db, owner_user_id)) or _row("2", status="queued")
        ),
    )
    monkeypatch.setattr(
        claims_jobs,
        "enqueue_claims_analytics_export",
        lambda **kwargs: enqueue_calls.append(kwargs) or {"id": 81, "status": "queued"},
    )

    body, response_status = claims_service.export_claims_analytics(
        payload={"format": "json", "filters": {"workspace_id": "2"}},
        principal=_principal(platform_admin=True),
        current_user=_user(1),
        db=shared_db,
    )

    assert response_status == 202
    assert queued_calls == [(shared_db, "2")]
    assert enqueue_calls[0]["owner_user_id"] == "2"
    assert shared_db.attach_calls == [{"export_id": _EXPORT_ID, "user_id": "2", "job_id": 81}]
    assert body["download_url"].endswith("?workspace_id=2")


@pytest.mark.parametrize(
    ("enabled", "operation"),
    [(False, "create_ready_artifact"), (True, "create_queued_artifact")],
)
def test_artifact_storage_failures_return_sanitized_503(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    operation: str,
) -> None:
    db = _FakeDb()
    _patch_common(monkeypatch, enabled=enabled)
    secret = "sqlite:///private/owner/media.db"
    warning_calls: list[tuple[str, tuple[Any, ...]]] = []

    def _fail_storage(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(secret)

    monkeypatch.setattr(claims_analytics_exports, operation, _fail_storage)
    monkeypatch.setattr(
        claims_service.logger,
        "warning",
        lambda message, *args: warning_calls.append((message, args)),
    )

    with pytest.raises(HTTPException) as exc_info:
        claims_service.export_claims_analytics(
            payload={"format": "json"},
            principal=_principal(),
            current_user=_user(),
            db=db,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "code": "claims_export_storage_unavailable",
        "message": "Claims analytics export storage is temporarily unavailable.",
    }
    assert secret not in repr(exc_info.value.detail)
    assert warning_calls
    assert secret not in repr(warning_calls)


def test_sync_oversize_maps_safe_413(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDb()
    _patch_common(monkeypatch, enabled=False)
    failed_artifact = {"status": "processing", "payload_json": None, "payload_csv": None}

    def _too_large(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        failed_artifact["status"] = "failed"
        raise ClaimsAnalyticsExportError(
            "Claims analytics export exceeds the configured size limit.",
            code="claims_export_too_large",
            http_status=413,
        )

    monkeypatch.setattr(claims_analytics_exports, "create_ready_artifact", _too_large)

    with pytest.raises(HTTPException) as exc_info:
        claims_service.export_claims_analytics(
            payload={"format": "json"},
            principal=_principal(),
            current_user=_user(),
            db=db,
        )

    assert exc_info.value.status_code == 413
    assert exc_info.value.detail["code"] == "claims_export_too_large"
    assert failed_artifact == {"status": "failed", "payload_json": None, "payload_csv": None}


def test_export_endpoint_applies_service_response_status(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "export_id": _EXPORT_ID,
        "format": "json",
        "status": "queued",
        "download_url": f"/api/v1/claims/analytics/export/{_EXPORT_ID}",
        "created_at": None,
        "job_id": 81,
        "job_status": "queued",
        "error_code": None,
        "snapshot_at": _SNAPSHOT,
    }
    monkeypatch.setattr(claims_service, "export_claims_analytics", lambda **_kwargs: (body, 202))
    response = Response()

    result = claims_endpoint.export_claims_analytics(
        payload=ClaimsAnalyticsExportRequest(format="json"),
        response=response,
        principal=_principal(),
        current_user=_user(),
        db=_FakeDb(),
    )

    assert response.status_code == 202
    assert result == body
