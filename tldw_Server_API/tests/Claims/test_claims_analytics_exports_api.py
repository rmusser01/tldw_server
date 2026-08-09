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


@pytest.fixture(autouse=True)
def _clear_export_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CLAIMS_ANALYTICS_EXPORT_MAX_BYTES",
        "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC",
        "CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS",
    ):
        monkeypatch.delenv(key, raising=False)


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


class _ExportReadDb:
    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        backend_type: BackendType = BackendType.SQLITE,
    ) -> None:
        self.backend_type = backend_type
        self.rows = list(rows or [])
        self.list_calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []

    def list_claims_analytics_exports(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.list_calls.append(kwargs)
        matching = self._matching_rows(
            user_id=kwargs["user_id"],
            status=kwargs.get("status"),
            format=kwargs.get("format"),
        )
        offset = int(kwargs["offset"])
        return matching[offset : offset + int(kwargs["limit"])]

    def count_claims_analytics_exports(self, **kwargs: Any) -> int:
        self.count_calls.append(kwargs)
        return len(
            self._matching_rows(
                user_id=kwargs["user_id"],
                status=kwargs.get("status"),
                format=kwargs.get("format"),
            )
        )

    def get_claims_analytics_export(self, export_id: str, *, user_id: str) -> dict[str, Any] | None:
        self.get_calls.append({"export_id": export_id, "user_id": user_id})
        return next(
            (
                dict(row)
                for row in self.rows
                if row.get("export_id") == export_id and row.get("user_id") == user_id
            ),
            None,
        )

    def _matching_rows(
        self,
        *,
        user_id: str,
        status: str | None,
        format: str | None,
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.rows
            if row.get("user_id") == user_id
            and (status is None or row.get("status") == status)
            and (format is None or row.get("format") == format)
        ]


class _JobsReader:
    def __init__(
        self,
        jobs: dict[int, dict[str, Any]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.jobs = jobs or {}
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def get_jobs_by_ids(self, job_ids: list[int], **kwargs: Any) -> dict[int, dict[str, Any]]:
        self.calls.append({"job_ids": job_ids, **kwargs})
        if self.error is not None:
            raise self.error
        return {job_id: self.jobs[job_id] for job_id in job_ids if job_id in self.jobs}


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


def _stored_row(
    owner: str,
    *,
    export_id: str = _EXPORT_ID,
    status: str = "ready",
    format: str = "json",
    job_id: int | None = None,
    error_code: str | None = None,
    payload_json: str | None = '{"events":[{"id":1}]}',
    payload_csv: str | None = None,
) -> dict[str, Any]:
    return {
        "export_id": export_id,
        "user_id": owner,
        "format": format,
        "status": status,
        "job_id": job_id,
        "error_code": error_code,
        "error_message": "A safe stored message.",
        "snapshot_at": _SNAPSHOT,
        "filters_json": '{"event_type":"unsupported_ratio","end_time":"2026-08-08T12:00:00.000Z"}',
        "pagination_json": '{"limit":10,"offset":0}',
        "filters": {"workspace_id": "999"},
        "payload_json": payload_json,
        "payload_csv": payload_csv,
        "created_at": "2026-08-08T12:00:01.000Z",
        "updated_at": "2026-08-08T12:00:02.000Z",
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


def _patch_export_read_maintenance(
    monkeypatch: pytest.MonkeyPatch,
    manager: _JobsReader,
) -> list[tuple[str, object, str, int]]:
    calls: list[tuple[str, object, str, int]] = []
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: manager)
    monkeypatch.setattr(
        claims_analytics_exports,
        "reconcile_export_artifacts",
        lambda actual_db, *, owner_user_id, limit, **_kwargs: calls.append(
            ("reconcile", actual_db, owner_user_id, limit)
        ),
    )
    monkeypatch.setattr(
        claims_analytics_exports,
        "cleanup_export_artifacts",
        lambda actual_db, *, owner_user_id, limit, **_kwargs: calls.append(
            ("cleanup", actual_db, owner_user_id, limit)
        ),
    )
    return calls


@pytest.mark.parametrize(
    ("env_value", "expected_retention_hours"),
    [
        ("2", 2),
        ("", 24),
        ("true", 24),
        ("0", 24),
        ("-1", 24),
    ],
)
def test_export_maintenance_uses_validated_environment_retention_hours(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected_retention_hours: int,
) -> None:
    manager = object()
    retention_hours: list[Any] = []
    monkeypatch.setitem(claims_service.settings, "CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS", 999)
    monkeypatch.setenv("CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS", env_value)
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: manager)
    monkeypatch.setattr(
        claims_analytics_exports,
        "reconcile_export_artifacts",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        claims_analytics_exports,
        "cleanup_export_artifacts",
        lambda *_args, **kwargs: retention_hours.append(kwargs["retention_hours"]),
    )

    assert claims_service._claims_export_maintenance(db=object(), owner_user_id="1") is manager
    assert retention_hours == [expected_retention_hours]


def test_list_is_owner_scoped_and_batches_nullable_job_status_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _stored_row("1", status="ready", job_id=81)
    second = _stored_row("1", export_id="b" * 32, status="processing", job_id=82)
    unlinked = _stored_row("1", export_id="c" * 32, status="queued")
    other_owner = _stored_row("2", export_id="d" * 32, status="ready", job_id=83)
    db = _ExportReadDb([first, second, unlinked, other_owner])
    manager = _JobsReader(
        {
            81: {"id": 81, "status": "completed"},
            82: {"id": 82, "status": "retrying"},
            83: {"id": 83, "status": "completed"},
        }
    )
    maintenance_calls = _patch_export_read_maintenance(monkeypatch, manager)

    result = claims_service.list_claims_analytics_exports(
        limit=100,
        offset=0,
        status_filter=None,
        format_filter=None,
        workspace_id=None,
        principal=_principal(platform_admin=False),
        current_user=_user(1),
        db=db,
    )

    assert result["total"] == 3
    assert [(row["status"], row["job_status"]) for row in result["exports"]] == [
        ("ready", "completed"),
        ("processing", "retrying"),
        ("queued", None),
    ]
    assert result["exports"][0]["job_id"] == 81
    assert result["exports"][0]["error_code"] is None
    assert result["exports"][0]["snapshot_at"] == _SNAPSHOT
    assert result["exports"][0]["filters"] == {
        "event_type": "unsupported_ratio",
        "end_time": _SNAPSHOT,
    }
    assert result["exports"][0]["pagination"] == {"limit": 10, "offset": 0}
    assert manager.calls == [
        {
            "job_ids": [81, 82],
            "domain": "claims",
            "owner_user_id": "1",
            "include_archived": True,
        }
    ]
    assert db.list_calls[0]["user_id"] == "1"
    assert db.count_calls[0]["user_id"] == "1"
    assert maintenance_calls == [
        ("reconcile", db, "1", 100),
        ("cleanup", db, "1", 100),
    ]


def test_list_jobs_outage_returns_artifacts_with_null_job_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _stored_row("1", status="processing", job_id=81)
    db = _ExportReadDb([row])
    manager = _JobsReader(error=RuntimeError("jobs outage secret"))
    _patch_export_read_maintenance(monkeypatch, manager)

    result = claims_service.list_claims_analytics_exports(
        limit=10,
        offset=0,
        status_filter=None,
        format_filter=None,
        workspace_id=None,
        principal=_principal(platform_admin=False),
        current_user=_user(1),
        db=db,
    )

    assert result["exports"][0]["status"] == "processing"
    assert result["exports"][0]["job_status"] is None
    assert len(manager.calls) == 1


def test_list_status_filter_remains_artifact_only(monkeypatch: pytest.MonkeyPatch) -> None:
    ready = _stored_row("1", status="ready", job_id=81)
    processing = _stored_row("1", export_id="b" * 32, status="processing", job_id=82)
    db = _ExportReadDb([ready, processing])
    manager = _JobsReader(
        {
            81: {"id": 81, "status": "processing"},
            82: {"id": 82, "status": "completed"},
        }
    )
    _patch_export_read_maintenance(monkeypatch, manager)

    result = claims_service.list_claims_analytics_exports(
        limit=10,
        offset=0,
        status_filter="ready",
        format_filter=None,
        workspace_id=None,
        principal=_principal(platform_admin=False),
        current_user=_user(1),
        db=db,
    )

    assert [(row["status"], row["job_status"]) for row in result["exports"]] == [
        ("ready", "processing")
    ]
    assert db.list_calls[0]["status"] == "ready"
    assert db.count_calls[0]["status"] == "ready"


def test_platform_admin_cross_owner_list_routes_sqlite_and_canonicalizes_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_db = _ExportReadDb([_stored_row("1")])
    target_db = _ExportReadDb(
        [
            _stored_row("2", export_id="b" * 32),
            _stored_row("2", export_id="c" * 32),
        ]
    )
    manager = _JobsReader()
    _patch_export_read_maintenance(monkeypatch, manager)
    routed: list[int] = []

    @contextmanager
    def _override(user_id: int):
        routed.append(user_id)
        yield target_db, f"/users/{user_id}/Media_DB_v2.db"

    monkeypatch.setattr(claims_service, "_claims_user_override_db", _override)

    result = claims_service.list_claims_analytics_exports(
        limit=10,
        offset=0,
        status_filter=None,
        format_filter=None,
        workspace_id="2",
        principal=_principal(platform_admin=True),
        current_user=_user(1),
        db=caller_db,
    )

    assert routed == [2]
    assert caller_db.list_calls == []
    assert target_db.list_calls[0]["user_id"] == "2"
    assert [row["download_url"] for row in result["exports"]] == [
        f"/api/v1/claims/analytics/export/{'b' * 32}?workspace_id=2",
        f"/api/v1/claims/analytics/export/{'c' * 32}?workspace_id=2",
    ]


def test_platform_admin_cross_owner_list_scopes_shared_postgres_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_db = _ExportReadDb(
        [_stored_row("2", export_id="b" * 32, job_id=81)],
        backend_type=BackendType.POSTGRESQL,
    )
    manager = _JobsReader({81: {"id": 81, "status": "completed"}})
    _patch_export_read_maintenance(monkeypatch, manager)
    monkeypatch.setattr(
        claims_service,
        "_claims_user_override_db",
        lambda _user_id: pytest.fail("PostgreSQL must retain the shared database"),
    )

    result = claims_service.list_claims_analytics_exports(
        limit=10,
        offset=0,
        status_filter=None,
        format_filter=None,
        workspace_id="2",
        principal=_principal(platform_admin=True),
        current_user=_user(1),
        db=shared_db,
    )

    assert shared_db.list_calls == [
        {"user_id": "2", "status": None, "format": None, "limit": 10, "offset": 0}
    ]
    assert shared_db.count_calls == [{"user_id": "2", "status": None, "format": None}]
    assert manager.calls == [
        {
            "job_ids": [81],
            "domain": "claims",
            "owner_user_id": "2",
            "include_archived": True,
        }
    ]
    assert result["exports"][0]["download_url"].endswith("?workspace_id=2")


def test_non_platform_admin_cannot_list_cross_owner_exports() -> None:
    db = _ExportReadDb([_stored_row("2")])

    with pytest.raises(HTTPException) as exc_info:
        claims_service.list_claims_analytics_exports(
            limit=10,
            offset=0,
            status_filter=None,
            format_filter=None,
            workspace_id="2",
            principal=_principal(platform_admin=False),
            current_user=_user(1),
            db=db,
        )

    assert exc_info.value.status_code == 403
    assert db.list_calls == []


def _download(
    *,
    db: _ExportReadDb,
    workspace_id: str | None = None,
    principal: AuthPrincipal | None = None,
) -> Response:
    return claims_endpoint.download_claims_analytics_export(
        export_id=_EXPORT_ID,
        workspace_id=workspace_id,
        principal=principal or _principal(platform_admin=False),
        current_user=_user(1),
        db=db,
    )


def test_download_ready_json_returns_exact_payload_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = '{"events":[{"id":1}],"pagination":{"limit":10}}'
    db = _ExportReadDb([_stored_row("1", payload_json=payload)])
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: _JobsReader())

    response = _download(db=db)

    assert response.status_code == 200
    assert response.body == payload.encode("utf-8")
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_download_ready_csv_uses_exact_safe_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = "id,event_type\r\n1,unsupported_ratio\r\n"
    db = _ExportReadDb(
        [_stored_row("1", format="csv", payload_json=None, payload_csv=payload)]
    )
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: _JobsReader())

    response = _download(db=db)

    assert response.status_code == 200
    assert response.body == payload.encode("utf-8")
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"] == (
        f'attachment; filename="claims-analytics-{_EXPORT_ID}.csv"'
    )


def test_ready_download_survives_jobs_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _ExportReadDb([_stored_row("1")])
    monkeypatch.setattr(
        claims_service,
        "jobs_manager_from_env",
        lambda: (_ for _ in ()).throw(RuntimeError("jobs outage secret")),
    )

    response = _download(db=db)

    assert response.status_code == 200
    assert response.body == b'{"events":[{"id":1}]}'


@pytest.mark.parametrize(
    ("artifact_status", "job_status"),
    [
        ("queued", "queued"),
        ("processing", "processing"),
        ("processing", "retrying"),
    ],
)
def test_pending_download_returns_stable_not_ready_conflict(
    monkeypatch: pytest.MonkeyPatch,
    artifact_status: str,
    job_status: str,
) -> None:
    db = _ExportReadDb([_stored_row("1", status=artifact_status, job_id=81, payload_json=None)])
    monkeypatch.setattr(
        claims_service,
        "jobs_manager_from_env",
        lambda: _JobsReader({81: {"id": 81, "status": job_status}}),
    )

    with pytest.raises(HTTPException) as exc_info:
        _download(db=db)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "code": "claims_export_not_ready",
        "status": artifact_status,
        "job_status": job_status,
    }


@pytest.mark.parametrize(
    ("job_status", "expected_code"),
    [
        ("cancelled", "claims_export_job_cancelled"),
        ("quarantined", "claims_export_job_quarantined"),
    ],
)
def test_terminal_job_projection_returns_stable_conflict_code(
    monkeypatch: pytest.MonkeyPatch,
    job_status: str,
    expected_code: str,
) -> None:
    db = _ExportReadDb([_stored_row("1", status="queued", job_id=81, payload_json=None)])
    monkeypatch.setattr(
        claims_service,
        "jobs_manager_from_env",
        lambda: _JobsReader({81: {"id": 81, "status": job_status}}),
    )

    with pytest.raises(HTTPException) as exc_info:
        _download(db=db)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "code": expected_code,
        "status": "queued",
        "job_status": job_status,
    }


@pytest.mark.parametrize(
    ("stored_code", "expected_code"),
    [
        ("claims_export_too_large", "claims_export_too_large"),
        (None, "claims_export_failed"),
        ("postgresql://owner:secret@private-db", "claims_export_failed"),
    ],
)
def test_failed_download_uses_only_safe_stored_code(
    monkeypatch: pytest.MonkeyPatch,
    stored_code: str | None,
    expected_code: str,
) -> None:
    db = _ExportReadDb(
        [_stored_row("1", status="failed", error_code=stored_code, payload_json=None)]
    )
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: _JobsReader())

    with pytest.raises(HTTPException) as exc_info:
        _download(db=db)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {
        "code": expected_code,
        "status": "failed",
        "job_status": None,
    }
    assert "secret" not in repr(exc_info.value.detail)


def test_missing_wrong_owner_and_malformed_downloads_are_indistinguishable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: _JobsReader())
    missing_db = _ExportReadDb()
    wrong_owner_db = _ExportReadDb([_stored_row("2")])
    details: list[Any] = []

    for export_id, db in [
        (_EXPORT_ID, missing_db),
        (_EXPORT_ID, wrong_owner_db),
        ("../../private.csv", missing_db),
    ]:
        with pytest.raises(HTTPException) as exc_info:
            claims_service.get_claims_analytics_export(
                export_id=export_id,
                workspace_id=None,
                principal=_principal(platform_admin=False),
                current_user=_user(1),
                db=db,
            )
        assert exc_info.value.status_code == 404
        details.append(exc_info.value.detail)

    assert details[0] == details[1] == details[2]
    assert wrong_owner_db.get_calls == [{"export_id": _EXPORT_ID, "user_id": "1"}]
    assert missing_db.get_calls == [{"export_id": _EXPORT_ID, "user_id": "1"}]


def test_platform_admin_cross_owner_download_routes_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_db = _ExportReadDb([_stored_row("1")])
    target_db = _ExportReadDb([_stored_row("2")])
    routed: list[int] = []

    @contextmanager
    def _override(user_id: int):
        routed.append(user_id)
        yield target_db, f"/users/{user_id}/Media_DB_v2.db"

    monkeypatch.setattr(claims_service, "_claims_user_override_db", _override)
    monkeypatch.setattr(claims_service, "jobs_manager_from_env", lambda: _JobsReader())

    response = _download(
        db=caller_db,
        workspace_id="2",
        principal=_principal(platform_admin=True),
    )

    assert response.status_code == 200
    assert routed == [2]
    assert caller_db.get_calls == []
    assert target_db.get_calls == [{"export_id": _EXPORT_ID, "user_id": "2"}]


def test_platform_admin_cross_owner_download_scopes_shared_postgres_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_db = _ExportReadDb(
        [_stored_row("2")],
        backend_type=BackendType.POSTGRESQL,
    )
    monkeypatch.setattr(
        claims_service,
        "_claims_user_override_db",
        lambda _user_id: pytest.fail("PostgreSQL must retain the shared database"),
    )

    response = _download(
        db=shared_db,
        workspace_id="2",
        principal=_principal(platform_admin=True),
    )

    assert response.status_code == 200
    assert shared_db.get_calls == [{"export_id": _EXPORT_ID, "user_id": "2"}]


def test_non_platform_admin_cannot_download_cross_owner_export() -> None:
    db = _ExportReadDb([_stored_row("2")])

    with pytest.raises(HTTPException) as exc_info:
        claims_service.get_claims_analytics_export(
            export_id=_EXPORT_ID,
            workspace_id="2",
            principal=_principal(platform_admin=False),
            current_user=_user(1),
            db=db,
        )

    assert exc_info.value.status_code == 403
    assert db.get_calls == []
