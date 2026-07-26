from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileUpdateRequest,
)
from tldw_Server_API.app.api.v1.utils import deprecation as deprecation_utils
from tldw_Server_API.app.api.v2.endpoints import user_profiles as v2_profiles
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.Chatbooks.exceptions import ValidationError as ChatbookValidationError
from tldw_Server_API.app.core.UserProfiles.response_mappers import (
    LegacyProfileCommandResult,
)
from tldw_Server_API.app.main import app
from tldw_Server_API.app.services import admin_profiles_service

pytestmark = pytest.mark.unit

CallerName = Literal["v1_self", "v2_self", "admin", "chatbooks", "deprecated_email"]

PROFILE_VERSION = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
PROFILE_VERSION_JSON = "2026-01-02T03:04:05Z"
DEPRECATION_HEADERS = {
    "Deprecation": "true",
    "Sunset": "Fri, 01 Jan 2100 00:00:00 GMT",
    "Link": "</api/v1/users/me/profile>; rel=successor-version",
}
_MISSING_OVERRIDE = object()


@contextmanager
def _dependency_override_scope(
    overrides: dict[Any, Any],
) -> Iterator[None]:
    previous = {
        dependency: app.dependency_overrides.get(
            dependency,
            _MISSING_OVERRIDE,
        )
        for dependency in overrides
    }
    app.dependency_overrides.update(overrides)
    try:
        yield
    finally:
        for dependency, prior_override in previous.items():
            if prior_override is _MISSING_OVERRIDE:
                app.dependency_overrides.pop(dependency, None)
            else:
                app.dependency_overrides[dependency] = prior_override


class _FrozenDeprecationDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        frozen = cls(2099, 1, 1, tzinfo=timezone.utc)
        return frozen if tz is None else frozen.astimezone(tz)


@dataclass(frozen=True)
class CallerCase:
    name: str
    caller: CallerName
    dry_run: bool
    updates: tuple[tuple[str, Any], ...]
    status_code: int
    body: Any
    applied_keys: tuple[str, ...]
    headers: dict[str, str]
    audit_count: int
    command_result: LegacyProfileCommandResult | None = None
    expected_profile_version: datetime | None = None
    email_rowcount: int = 1
    email_failure: Literal["duplicate", "commit", "busy"] | None = None
    legacy_email_enabled: bool = True
    email_is_active: bool = True
    email_is_verified: bool = True
    email_execute_count: int | None = None
    email_transaction_exit_count: int | None = None


@dataclass(frozen=True)
class ObservedCallerContract:
    status_code: int
    body: Any
    applied_keys: tuple[str, ...]
    headers: dict[str, str]
    audit_count: int
    email_execute_count: int = 0
    email_transaction_exit_count: int = 0


def _legacy_success_body(
    applied: tuple[str, ...],
    skipped: tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    return {
        "profile_version": PROFILE_VERSION_JSON,
        "applied": list(applied),
        "skipped": list(skipped),
    }


def _v2_success_body(applied: tuple[str, ...]) -> dict[str, Any]:
    return {
        "profile_version": PROFILE_VERSION_JSON,
        "applied": list(applied),
    }


def _profile_error_body(
    *,
    error_code: str,
    detail: str,
    errors: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    return {
        "error_code": error_code,
        "detail": detail,
        "errors": list(errors),
    }


def _v2_error_body(
    *,
    error_code: str,
    detail: str,
    errors: tuple[dict[str, str], ...],
) -> dict[str, Any]:
    return {
        "detail": _profile_error_body(
            error_code=error_code,
            detail=detail,
            errors=errors,
        )
    }


def _deprecated_success_body(email: str) -> dict[str, Any]:
    return {
        "id": 7,
        "uuid": None,
        "username": "legacy-user",
        "email": email,
        "role": "user",
        "is_active": True,
        "is_verified": True,
        "created_at": "2026-01-01T00:00:00",
        "last_login": None,
        "storage_quota_mb": 5120,
        "storage_used_mb": 0.0,
        "warning": "deprecated_endpoint",
        "successor": "/api/v1/users/me/profile",
    }


_UNKNOWN_ERROR = ({"key": "preferences.ui.missing", "message": "unknown_key"},)
_STALE_ERROR = ({"key": "profile_version", "message": "mismatch"},)
_ROLLBACK_ERROR = ({"key": "preferences.ui.theme", "message": "execution_failed"},)

CALLER_CASES = (
    CallerCase(
        name="v1_duplicate_keys_preserve_applied_order",
        caller="v1_self",
        dry_run=False,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("preferences.ui.theme", "midnight"),
        ),
        status_code=200,
        body=_legacy_success_body(
            ("preferences.ui.theme", "preferences.ui.theme")
        ),
        applied_keys=("preferences.ui.theme", "preferences.ui.theme"),
        headers={},
        audit_count=1,
        command_result=LegacyProfileCommandResult(
            profile_version=PROFILE_VERSION,
            applied=("preferences.ui.theme", "preferences.ui.theme"),
        ),
    ),
    CallerCase(
        name="v1_dry_run_preserves_duplicate_accepted_order_without_audit",
        caller="v1_self",
        dry_run=True,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("preferences.ui.theme", "midnight"),
        ),
        status_code=200,
        body=_legacy_success_body(
            ("preferences.ui.theme", "preferences.ui.theme")
        ),
        applied_keys=("preferences.ui.theme", "preferences.ui.theme"),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            profile_version=PROFILE_VERSION,
            applied=("preferences.ui.theme", "preferences.ui.theme"),
        ),
    ),
    CallerCase(
        name="v1_mixed_accepted_and_rejected_uses_legacy_error_envelope",
        caller="v1_self",
        dry_run=False,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("preferences.ui.missing", "ignored"),
        ),
        status_code=400,
        body=_profile_error_body(
            error_code="profile_update_unknown_key",
            detail="One or more keys are not recognized",
            errors=_UNKNOWN_ERROR,
        ),
        applied_keys=(),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            status_code=400,
            applied=("preferences.ui.theme",),
            skipped=_UNKNOWN_ERROR,
            error_code="profile_update_unknown_key",
            detail="One or more keys are not recognized",
        ),
    ),
    CallerCase(
        name="v1_runtime_rollback_result_never_reports_applied_keys_or_audit",
        caller="v1_self",
        dry_run=False,
        updates=(("preferences.ui.theme", "paper"),),
        status_code=500,
        body=_profile_error_body(
            error_code="profile_update_failed",
            detail="Profile update rolled back",
            errors=_ROLLBACK_ERROR,
        ),
        applied_keys=(),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            status_code=500,
            skipped=_ROLLBACK_ERROR,
            error_code="profile_update_failed",
            detail="Profile update rolled back",
        ),
    ),
    CallerCase(
        name="v1_empty_updates_are_rejected_before_command_or_audit",
        caller="v1_self",
        dry_run=False,
        updates=(),
        status_code=400,
        body=_profile_error_body(
            error_code="profile_update_invalid",
            detail="No updates provided",
            errors=({"key": "updates", "message": "missing"},),
        ),
        applied_keys=(),
        headers={},
        audit_count=0,
    ),
    CallerCase(
        name="v1_stale_version_precedes_invalid_value_rejection",
        caller="v1_self",
        dry_run=False,
        updates=(("identity.email", "not-an-email"),),
        status_code=409,
        body=_profile_error_body(
            error_code="profile_version_mismatch",
            detail="profile_version_mismatch",
            errors=_STALE_ERROR,
        ),
        applied_keys=(),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            status_code=409,
            profile_version=PROFILE_VERSION,
            skipped=_STALE_ERROR,
            error_code="profile_version_mismatch",
            detail="profile_version_mismatch",
        ),
        expected_profile_version=datetime(2000, 1, 1, tzinfo=timezone.utc),
    ),
    CallerCase(
        name="v2_mixed_input_nests_the_exact_error_detail",
        caller="v2_self",
        dry_run=False,
        updates=(
            ("preferences.ui.theme", "paper"),
            ("preferences.ui.missing", "ignored"),
        ),
        status_code=400,
        body=_v2_error_body(
            error_code="profile_update_unknown_key",
            detail="One or more keys are not recognized",
            errors=_UNKNOWN_ERROR,
        ),
        applied_keys=(),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            status_code=400,
            applied=("preferences.ui.theme",),
            skipped=_UNKNOWN_ERROR,
            error_code="profile_update_unknown_key",
            detail="One or more keys are not recognized",
        ),
    ),
    CallerCase(
        name="v2_empty_updates_currently_succeed_and_emit_audit",
        caller="v2_self",
        dry_run=False,
        updates=(),
        status_code=200,
        body=_v2_success_body(()),
        applied_keys=(),
        headers={},
        audit_count=1,
        command_result=LegacyProfileCommandResult(profile_version=PROFILE_VERSION),
    ),
    CallerCase(
        name="v2_stale_version_precedes_invalid_value_rejection",
        caller="v2_self",
        dry_run=False,
        updates=(("identity.email", "not-an-email"),),
        status_code=409,
        body=_v2_error_body(
            error_code="profile_version_mismatch",
            detail="profile_version_mismatch",
            errors=_STALE_ERROR,
        ),
        applied_keys=(),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            status_code=409,
            profile_version=PROFILE_VERSION,
            skipped=_STALE_ERROR,
            error_code="profile_version_mismatch",
            detail="profile_version_mismatch",
        ),
        expected_profile_version=datetime(2000, 1, 1, tzinfo=timezone.utc),
    ),
    CallerCase(
        name="admin_stale_version_precedes_invalid_value_rejection",
        caller="admin",
        dry_run=False,
        updates=(("identity.email", "not-an-email"),),
        status_code=409,
        body=_profile_error_body(
            error_code="profile_version_mismatch",
            detail="profile_version_mismatch",
            errors=_STALE_ERROR,
        ),
        applied_keys=(),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            status_code=409,
            profile_version=PROFILE_VERSION,
            skipped=_STALE_ERROR,
            error_code="profile_version_mismatch",
            detail="profile_version_mismatch",
        ),
        expected_profile_version=datetime(2000, 1, 1, tzinfo=timezone.utc),
    ),
    CallerCase(
        name="admin_dry_run_preserves_duplicate_order_and_returns_audit_record",
        caller="admin",
        dry_run=True,
        updates=(
            ("limits.storage_quota_mb", 4096),
            ("limits.storage_quota_mb", 8192),
        ),
        status_code=200,
        body=_legacy_success_body(
            ("limits.storage_quota_mb", "limits.storage_quota_mb")
        ),
        applied_keys=("limits.storage_quota_mb", "limits.storage_quota_mb"),
        headers={},
        audit_count=1,
        command_result=LegacyProfileCommandResult(
            profile_version=PROFILE_VERSION,
            applied=("limits.storage_quota_mb", "limits.storage_quota_mb"),
        ),
    ),
    CallerCase(
        name="chatbooks_restore_preserves_adapter_update_order",
        caller="chatbooks",
        dry_run=False,
        updates=(
            ("identity.email", "restored@example.com"),
            ("preferences.ui.theme", "paper"),
        ),
        status_code=200,
        body={"account_profile": 1, "account_settings": 1},
        applied_keys=("identity.email", "preferences.ui.theme"),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            profile_version=PROFILE_VERSION,
            applied=("identity.email", "preferences.ui.theme"),
        ),
    ),
    CallerCase(
        name="chatbooks_restore_rejection_is_generic",
        caller="chatbooks",
        dry_run=False,
        updates=(("identity.email", "not-an-email"),),
        status_code=422,
        body={
            "detail": "Destination rejected one or more account profile/settings values"
        },
        applied_keys=(),
        headers={},
        audit_count=0,
        command_result=LegacyProfileCommandResult(
            status_code=422,
            skipped=({"key": "identity.email", "message": "invalid_email"},),
            error_code="profile_update_invalid",
            detail="One or more updates failed validation",
        ),
    ),
    CallerCase(
        name="deprecated_email_changed_is_lowercased_with_deprecation_headers",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "UPDATED@EXAMPLE.COM"),),
        status_code=200,
        body=_deprecated_success_body("updated@example.com"),
        applied_keys=("identity.email",),
        headers=DEPRECATION_HEADERS,
        audit_count=0,
    ),
    CallerCase(
        name="deprecated_email_unchanged_is_no_update",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "legacy@example.com"),),
        status_code=400,
        body={"detail": "No updates provided"},
        applied_keys=(),
        headers={},
        audit_count=0,
    ),
    CallerCase(
        name="deprecated_email_omitted_is_no_update",
        caller="deprecated_email",
        dry_run=False,
        updates=(),
        status_code=400,
        body={"detail": "No updates provided"},
        applied_keys=(),
        headers={},
        audit_count=0,
    ),
    CallerCase(
        name="deprecated_email_explicit_null_is_no_update",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", None),),
        status_code=400,
        body={"detail": "No updates provided"},
        applied_keys=(),
        headers={},
        audit_count=0,
    ),
    CallerCase(
        name="deprecated_email_invalid_value_uses_fastapi_validation_shape",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "not-an-email"),),
        status_code=422,
        body={
            "detail": [
                {
                    "type": "value_error",
                    "loc": ["body", "email"],
                    "msg": (
                        "value is not a valid email address: "
                        "An email address must have an @-sign."
                    ),
                    "input": "not-an-email",
                    "ctx": {"reason": "An email address must have an @-sign."},
                }
            ]
        },
        applied_keys=(),
        headers={},
        audit_count=0,
    ),
    CallerCase(
        name="deprecated_email_missing_user_is_404",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "updated@example.com"),),
        status_code=404,
        body={"detail": "User not found"},
        applied_keys=(),
        headers={},
        audit_count=0,
        email_rowcount=0,
    ),
    CallerCase(
        name="deprecated_email_inactive_user_is_403_without_deprecation_headers",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "updated@example.com"),),
        status_code=403,
        body={"detail": "User account is inactive"},
        applied_keys=(),
        headers={},
        audit_count=0,
        email_is_active=False,
    ),
    CallerCase(
        name="deprecated_email_unverified_user_is_403_without_deprecation_headers",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "updated@example.com"),),
        status_code=403,
        body={"detail": "Email verification required"},
        applied_keys=(),
        headers={},
        audit_count=0,
        email_is_verified=False,
    ),
    CallerCase(
        name="deprecated_email_duplicate_is_sanitized_500",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "updated@example.com"),),
        status_code=500,
        body={"detail": "Failed to update profile"},
        applied_keys=(),
        headers={},
        audit_count=0,
        email_failure="duplicate",
        email_execute_count=1,
        email_transaction_exit_count=0,
    ),
    CallerCase(
        name="deprecated_email_commit_failure_occurs_after_success_response",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "updated@example.com"),),
        status_code=200,
        body=_deprecated_success_body("updated@example.com"),
        applied_keys=("identity.email",),
        headers=DEPRECATION_HEADERS,
        audit_count=0,
        email_failure="commit",
        email_execute_count=1,
        email_transaction_exit_count=1,
    ),
    CallerCase(
        name="deprecated_email_database_busy_has_retry_after_only",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "updated@example.com"),),
        status_code=503,
        body={
            "detail": "Authentication database is busy. Please retry shortly."
        },
        applied_keys=(),
        headers={"Retry-After": "1"},
        audit_count=0,
        email_failure="busy",
    ),
    CallerCase(
        name="deprecated_email_disabled_is_410_warning",
        caller="deprecated_email",
        dry_run=False,
        updates=(("identity.email", "updated@example.com"),),
        status_code=410,
        body={
            "warning": "deprecated_endpoint",
            "successor": "/api/v1/users/me/profile",
        },
        applied_keys=(),
        headers={},
        audit_count=0,
        legacy_email_enabled=False,
    ),
)


def _principal(*, admin: bool = False) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=5 if admin else 7,
        username="admin" if admin else "legacy-user",
        roles=["admin"] if admin else ["user"],
        permissions=["*"] if admin else [],
        is_admin=admin,
        org_ids=[],
        team_ids=[],
        active_org_id=11 if admin else None,
        active_team_id=22 if admin else None,
    )


def _user_context() -> dict[str, Any]:
    return {
        "id": 7,
        "uuid": None,
        "username": "legacy-user",
        "email": "legacy@example.com",
        "role": "user",
        "is_active": True,
        "is_verified": True,
        "created_at": datetime(2026, 1, 1),
        "last_login": None,
        "storage_quota_mb": 5120,
        "storage_used_mb": 0.0,
    }


def _response_body(response: Any) -> Any:
    if isinstance(response, JSONResponse):
        return json.loads(response.body.decode("utf-8"))
    return response.model_dump(mode="json")


def _body_applied_keys(body: Any) -> tuple[str, ...]:
    if isinstance(body, dict) and isinstance(body.get("applied"), list):
        return tuple(str(key) for key in body["applied"])
    return ()


class CallerHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch

    async def invoke(self, case: CallerCase) -> ObservedCallerContract:
        if case.caller == "v1_self":
            return await self._invoke_v1(case)
        if case.caller == "v2_self":
            return await self._invoke_v2(case)
        if case.caller == "admin":
            return await self._invoke_admin(case)
        if case.caller == "chatbooks":
            return await self._invoke_chatbooks(case)
        return self._invoke_deprecated_email(case)

    def _command_service_type(self, case: CallerCase, calls: list[Any]):
        class _CommandService:
            def __init__(self, *, db_pool) -> None:
                del db_pool

            async def apply(self, command, *, db_conn, scope):
                calls.append((command, db_conn, scope))
                assert command.expected_profile_version == case.expected_profile_version
                if case.command_result is None:
                    raise AssertionError("caller reached command service unexpectedly")
                return case.command_result

        return _CommandService

    async def _invoke_v1(self, case: CallerCase) -> ObservedCallerContract:
        calls: list[Any] = []
        audits: list[dict[str, Any]] = []

        async def _resolve(_principal, *, allow_missing: bool = False):
            del _principal, allow_missing
            return _user_context()

        async def _pool():
            return object()

        async def _audit(*_args, **kwargs):
            audits.append(kwargs)

        self._monkeypatch.setattr(users_endpoints, "_resolve_user_context", _resolve)
        self._monkeypatch.setattr(users_endpoints, "get_db_pool", _pool)
        self._monkeypatch.setattr(
            users_endpoints,
            "ProfileCommandService",
            self._command_service_type(case, calls),
        )
        self._monkeypatch.setattr(users_endpoints, "_emit_user_profile_audit_event", _audit)
        response = await users_endpoints.update_current_user_profile(
            payload=UserProfileUpdateRequest(
                updates=[{"key": key, "value": value} for key, value in case.updates],
                dry_run=case.dry_run,
                profile_version=case.expected_profile_version,
            ),
            http_request=SimpleNamespace(),
            principal=_principal(),
            db=object(),
        )
        body = _response_body(response)
        return ObservedCallerContract(
            status_code=response.status_code if isinstance(response, JSONResponse) else 200,
            body=body,
            applied_keys=_body_applied_keys(body),
            headers={},
            audit_count=len(audits),
        )

    async def _invoke_v2(self, case: CallerCase) -> ObservedCallerContract:
        calls: list[Any] = []
        audits: list[dict[str, Any]] = []

        async def _active(_principal):
            return _user_context()

        async def _audit(*_args, **kwargs):
            audits.append(kwargs)

        service = self._command_service_type(case, calls)(db_pool=object())
        self._monkeypatch.setattr(
            v2_profiles,
            "_require_principal_active_verified",
            _active,
        )
        self._monkeypatch.setattr(v2_profiles, "_emit_user_profile_audit_event", _audit)
        try:
            response = await v2_profiles.update_current_user_profile_v2(
                payload=UserProfileUpdateRequest(
                    updates=[{"key": key, "value": value} for key, value in case.updates],
                    dry_run=case.dry_run,
                    profile_version=case.expected_profile_version,
                ),
                http_request=SimpleNamespace(),
                principal=_principal(),
                db=object(),
                command_service=service,
            )
        except HTTPException as exc:
            body = {"detail": exc.detail}
            return ObservedCallerContract(
                status_code=exc.status_code,
                body=body,
                applied_keys=(),
                headers=dict(exc.headers or {}),
                audit_count=len(audits),
            )
        body = _response_body(response)
        return ObservedCallerContract(
            status_code=200,
            body=body,
            applied_keys=_body_applied_keys(body),
            headers={},
            audit_count=len(audits),
        )

    async def _invoke_admin(self, case: CallerCase) -> ObservedCallerContract:
        calls: list[Any] = []

        async def _allow_scope(*_args, **_kwargs):
            return None

        async def _pool():
            return object()

        class _Repo:
            async def get_user_by_id(self, user_id: int):
                return {"id": user_id, "updated_at": PROFILE_VERSION}

        async def _repo_from_pool():
            return _Repo()

        self._monkeypatch.setattr(
            admin_profiles_service.admin_scope_service,
            "enforce_admin_user_scope",
            _allow_scope,
        )
        self._monkeypatch.setattr(admin_profiles_service, "get_db_pool", _pool)
        self._monkeypatch.setattr(
            admin_profiles_service.AuthnzUsersRepo,
            "from_pool",
            _repo_from_pool,
        )
        self._monkeypatch.setattr(
            admin_profiles_service,
            "ProfileCommandService",
            self._command_service_type(case, calls),
        )
        response, audit_info = await admin_profiles_service.update_user_profile(
            user_id=7,
            payload=UserProfileUpdateRequest(
                updates=[{"key": key, "value": value} for key, value in case.updates],
                dry_run=case.dry_run,
                profile_version=case.expected_profile_version,
            ),
            principal=_principal(admin=True),
            db=object(),
        )
        body = _response_body(response)
        return ObservedCallerContract(
            status_code=response.status_code if isinstance(response, JSONResponse) else 200,
            body=body,
            applied_keys=_body_applied_keys(body),
            headers={},
            audit_count=int(audit_info is not None),
        )

    async def _invoke_chatbooks(self, case: CallerCase) -> ObservedCallerContract:
        from tldw_Server_API.app.core.AuthNZ import database as database_module
        from tldw_Server_API.app.core.AuthNZ.repos import users_repo as users_repo_module
        from tldw_Server_API.app.core.UserProfiles import command_service as command_service_module

        calls: list[Any] = []

        class _Pool:
            @asynccontextmanager
            async def transaction(self):
                yield object()

        pool = _Pool()

        async def _pool():
            return pool

        class _Repo:
            def __init__(self, *, db_pool) -> None:
                assert db_pool is pool

            async def get_user_by_id(self, user_id: int):
                return {"id": user_id}

        self._monkeypatch.setattr(database_module, "get_db_pool", _pool)
        self._monkeypatch.setattr(users_repo_module, "AuthnzUsersRepo", _Repo)
        self._monkeypatch.setattr(
            command_service_module,
            "ProfileCommandService",
            self._command_service_type(case, calls),
        )

        profile = {
            key: value for key, value in case.updates if key == "identity.email"
        }
        overrides = {
            key: value for key, value in case.updates if key != "identity.email"
        }
        payloads: dict[str, dict[str, Any]] = {}
        if profile:
            payloads["account_profile"] = {"profile": profile}
        if overrides:
            payloads["account_settings"] = {"overrides": overrides}
        service = SimpleNamespace(user_id_int=7)
        try:
            body = await ChatbookService._restore_account_state_payloads(service, payloads)
        except ChatbookValidationError as exc:
            return ObservedCallerContract(
                status_code=422,
                body={"detail": str(exc)},
                applied_keys=(),
                headers={},
                audit_count=0,
            )
        command = calls[0][0] if calls else None
        applied_keys = tuple(key for key, _value in command.updates) if command else ()
        return ObservedCallerContract(
            status_code=200,
            body=body,
            applied_keys=applied_keys,
            headers={},
            audit_count=0,
        )

    def _invoke_deprecated_email(self, case: CallerCase) -> ObservedCallerContract:
        email_execute_count = 0
        email_transaction_exit_count = 0

        class _Cursor:
            rowcount = case.email_rowcount

        class _Db:
            async def execute(self, *_args, **_kwargs):
                return _Cursor()

        class _TransactionalConnection:
            async def execute(self, *_args, **_kwargs):
                nonlocal email_execute_count
                email_execute_count += 1
                if case.email_failure == "duplicate":
                    raise users_endpoints.DatabaseError("duplicate email")
                return _Cursor()

        transaction_connection = _TransactionalConnection()

        class _Transaction:
            async def __aenter__(self):
                return transaction_connection

            async def __aexit__(self, exc_type, exc, tb) -> bool:
                nonlocal email_transaction_exit_count
                if exc_type is None:
                    assert exc is None
                    assert tb is None
                    email_transaction_exit_count += 1
                    raise RuntimeError("commit failed at /private/authnz.db")
                return False

        class _TransactionPool:
            def transaction(self):
                return _Transaction()

        async def _transaction_pool():
            return _TransactionPool()

        async def _db_transaction():
            if case.email_failure == "busy":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Authentication database is busy. Please retry shortly."
                    ),
                    headers={"Retry-After": "1"},
                )
            yield _Db()

        async def _auth_principal():
            return _principal()

        async def _resolve(_principal_value, *, allow_missing: bool = False):
            del _principal_value, allow_missing
            user_context = _user_context()
            user_context["is_active"] = case.email_is_active
            user_context["is_verified"] = case.email_is_verified
            return user_context

        self._monkeypatch.setenv(
            "ENABLE_LEGACY_USER_ME_ENDPOINTS",
            "true" if case.legacy_email_enabled else "false",
        )
        self._monkeypatch.setenv("DEPRECATION_SUNSET_DAYS", "365")
        self._monkeypatch.setattr(
            deprecation_utils,
            "datetime",
            _FrozenDeprecationDateTime,
        )
        self._monkeypatch.setattr(users_endpoints, "_resolve_user_context", _resolve)
        use_normal_transaction_dependency = case.email_failure in {"duplicate", "commit"}
        if use_normal_transaction_dependency:
            self._monkeypatch.setenv("TEST_MODE", "0")
            self._monkeypatch.setenv("TLDW_TEST_MODE", "0")
            self._monkeypatch.setattr(
                auth_deps,
                "get_db_pool",
                _transaction_pool,
            )
        dependency_overrides = {get_auth_principal: _auth_principal}
        if not use_normal_transaction_dependency:
            dependency_overrides[get_db_transaction] = _db_transaction
        request_body = {
            "email": next(
                (value for key, value in case.updates if key == "identity.email"),
                None,
            )
        }
        if not case.updates:
            request_body = {}
        with _dependency_override_scope(dependency_overrides):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.put("/api/v1/users/me", json=request_body)
        headers = {
            key: response.headers[key]
            for key in (*DEPRECATION_HEADERS, "Retry-After")
            if key in response.headers
        }
        applied_keys = ("identity.email",) if response.status_code == 200 else ()
        if response.headers.get("content-type", "").startswith("application/json"):
            body = response.json()
        else:
            body = response.text
        return ObservedCallerContract(
            status_code=response.status_code,
            body=body,
            applied_keys=applied_keys,
            headers=headers,
            audit_count=0,
            email_execute_count=email_execute_count,
            email_transaction_exit_count=email_transaction_exit_count,
        )


@pytest.fixture
def caller_harness(monkeypatch: pytest.MonkeyPatch) -> CallerHarness:
    return CallerHarness(monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CALLER_CASES, ids=lambda case: case.name)
async def test_stage2_caller_contract_is_characterized(
    case: CallerCase,
    caller_harness: CallerHarness,
) -> None:
    observed = await caller_harness.invoke(case)

    assert observed.status_code == case.status_code
    assert observed.body == case.body
    assert observed.applied_keys == case.applied_keys
    assert observed.headers == case.headers
    assert observed.audit_count == case.audit_count
    if case.email_execute_count is not None:
        assert observed.email_execute_count == case.email_execute_count
    if case.email_transaction_exit_count is not None:
        assert (
            observed.email_transaction_exit_count
            == case.email_transaction_exit_count
        )


def test_dependency_override_scope_restores_preexisting_entries() -> None:
    async def _prior_db_override():
        yield object()

    async def _prior_principal_override():
        return _principal()

    async def _replacement_db_override():
        yield object()

    async def _replacement_principal_override():
        return _principal()

    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_db_transaction] = _prior_db_override
    app.dependency_overrides[get_auth_principal] = _prior_principal_override
    try:
        with _dependency_override_scope(
            {
                get_db_transaction: _replacement_db_override,
                get_auth_principal: _replacement_principal_override,
            }
        ):
            assert (
                app.dependency_overrides[get_db_transaction]
                is _replacement_db_override
            )
            assert (
                app.dependency_overrides[get_auth_principal]
                is _replacement_principal_override
            )
        assert app.dependency_overrides[get_db_transaction] is _prior_db_override
        assert (
            app.dependency_overrides[get_auth_principal]
            is _prior_principal_override
        )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)
