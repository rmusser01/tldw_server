from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import load_webhook_key_ring
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_memberships_for_user
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_single_user_instance
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicAlert, TopicMonitoringDB
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

_ADMIN_USERNAME = "admin"
_ADMIN_EMAIL = "admin@example.local"
_OWNER_USERNAME = "owner"
_OWNER_EMAIL = "owner@example.local"
_SUPER_ADMIN_USERNAME = "superadmin"
_SUPER_ADMIN_EMAIL = "superadmin@example.local"
_NON_ADMIN_USERNAME = "member"
_NON_ADMIN_EMAIL = "member@example.local"
_REQUESTER_USERNAME = "requester"
_REQUESTER_EMAIL = "requester@example.local"
_ORG_NAME = "Admin E2E"
_ORG_SLUG = "admin-e2e"
_SEEDED_ALERT_MESSAGE = "CPU high"
_SYSTEM_TEMP_ROOT = (Path(os.path.sep) / "tmp").resolve()
_ADMIN_E2E_ARTIFACT_ROOT = Path(tempfile.gettempdir()).resolve() / "tldw-admin-e2e"
_ADMIN_E2E_MIGRATION_OPERATION_ID = "whmig_" + ("c" * 32)
_ADMIN_E2E_DIGEST = "sha256:" + ("a" * 64)
_ADMIN_E2E_FINGERPRINT = "hmac-sha256:" + ("b" * 64)
_ADMIN_E2E_IDENTITY_FIELDS = (
    "import_operation_id",
    "fingerprint_key_id",
    "active_primary_key_id",
    "system_ops_webhook_fingerprint",
    "legacy_table_fingerprint",
    "redacted_report_digest",
    "protected_backup_ciphertext_digest",
    "report_file_identity",
    "backup_file_identity",
    "rollback_key_file_identity",
    "expected_ciphertext_digest",
)

_SEEDED_PRINCIPALS: dict[str, dict[str, Any]] = {}


def _seed_fixtures_payload(*, alerts: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return stable cross-feature fixtures shared by admin e2e scenarios."""
    return {
        "alerts": alerts or [],
        "organizations": [],
    }


async def _get_backup_schedules_repo():
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )

    pool = await get_db_pool()
    repo = AuthnzBackupSchedulesRepo(pool)
    await repo.ensure_schema()
    return repo


async def _get_admin_monitoring_repo():
    from tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo import (
        AuthnzAdminMonitoringRepo,
    )

    pool = await get_db_pool()
    repo = AuthnzAdminMonitoringRepo(pool)
    await repo.ensure_schema_ready_once()
    return repo


async def _soft_delete_all_backup_schedules() -> int:
    repo = await _get_backup_schedules_repo()
    deleted = 0
    deleted_at = datetime.now(timezone.utc).isoformat()
    while True:
        items, _ = await repo.list_schedules(limit=200, offset=0, include_deleted=False)
        if not items:
            break
        for item in items:
            if await repo.delete_schedule(str(item["id"]), deleted_at=deleted_at):
                deleted += 1
    return deleted


async def _clear_admin_monitoring_state() -> dict[str, int]:
    repo = await _get_admin_monitoring_repo()
    rules = await repo.list_rules()
    deleted_rules = 0
    for rule in rules:
        if await repo.delete_rule(int(rule["id"])):
            deleted_rules += 1

    overlay_reset = await repo.clear_state_and_events()

    return {
        "deleted_rules": deleted_rules,
        "deleted_states": overlay_reset["deleted_states"],
        "deleted_events": overlay_reset["deleted_events"],
    }


def _get_monitoring_alerts_db_path() -> Path:
    raw_path = str(os.getenv("MONITORING_ALERTS_DB") or "Databases/monitoring_alerts.db").strip()
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _clear_monitoring_alert_rows() -> int:
    db_path = _get_monitoring_alerts_db_path()
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as conn:
        deleted = conn.execute("DELETE FROM topic_alerts").rowcount or 0
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'topic_alerts'")
        conn.commit()
        return int(deleted)


def _seed_monitoring_alert_store() -> dict[str, Any]:
    monitoring_db = TopicMonitoringDB(db_path=str(_get_monitoring_alerts_db_path()))
    alert_id = monitoring_db.insert_alert(
        TopicAlert(
            user_id=None,
            scope_type="global",
            scope_id=None,
            source="system",
            watchlist_id="watch-cpu",
            rule_id="rule-cpu-high",
            rule_category="system",
            rule_severity="warning",
            pattern=_SEEDED_ALERT_MESSAGE,
            text_snippet=_SEEDED_ALERT_MESSAGE,
            metadata={"seeded_by": "admin_e2e_support"},
        )
    )
    return {
        "alert_id": str(alert_id),
        "alert_identity": f"alert:{alert_id}",
        "message": _SEEDED_ALERT_MESSAGE,
    }


async def _seed_monitoring_alert_store_async() -> dict[str, Any]:
    """Seed the monitoring alerts SQLite store off the event loop."""
    return await asyncio.to_thread(_seed_monitoring_alert_store)


async def _clear_monitoring_alert_rows_async() -> int:
    """Clear seeded monitoring-alert rows off the event loop."""
    return await asyncio.to_thread(_clear_monitoring_alert_rows)


def _resolve_safe_backup_dir() -> Path | None:
    raw_backup_root = str(os.getenv("TLDW_DB_BACKUP_PATH") or "").strip()
    if not raw_backup_root:
        return None

    backup_dir = Path(raw_backup_root).expanduser().resolve(strict=False)
    allowed_roots = {
        Path(tempfile.gettempdir()).resolve(),
        _SYSTEM_TEMP_ROOT,
    }
    for allowed_root in allowed_roots:
        if backup_dir == allowed_root:
            continue
        try:
            backup_dir.relative_to(allowed_root)
            return backup_dir
        except ValueError:
            continue

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="admin_e2e_backup_path_must_be_temp_scoped",
    )


def _clear_backup_artifacts() -> int:
    backup_dir = _resolve_safe_backup_dir()
    if backup_dir is None:
        return 0
    if not backup_dir.exists():
        return 0
    shutil.rmtree(backup_dir)
    return 1


async def _clear_backup_artifacts_async() -> int:
    """Delete seeded backup artifacts off the event loop."""
    return await asyncio.to_thread(_clear_backup_artifacts)


async def reset_admin_e2e_state() -> dict[str, Any]:
    """Clear transient seed state and delete admin-e2e backup schedule artifacts."""
    _SEEDED_PRINCIPALS.clear()
    deleted_schedules = await _soft_delete_all_backup_schedules()
    monitoring_reset = await _clear_admin_monitoring_state()
    deleted_monitoring_alerts = await _clear_monitoring_alert_rows_async()
    deleted_backup_dirs = await _clear_backup_artifacts_async()
    logger.debug(
        "Admin e2e reset completed: deleted_schedules={} deleted_rules={} deleted_states={} deleted_events={} deleted_monitoring_alerts={} deleted_backup_dirs={}",
        deleted_schedules,
        monitoring_reset["deleted_rules"],
        monitoring_reset["deleted_states"],
        monitoring_reset["deleted_events"],
        deleted_monitoring_alerts,
        deleted_backup_dirs,
    )
    return {"ok": True}


def _admin_e2e_migration_identity(primary_key_id: str) -> dict[str, object]:
    return {
        "import_operation_id": _ADMIN_E2E_MIGRATION_OPERATION_ID,
        "fingerprint_key_id": primary_key_id,
        "active_primary_key_id": primary_key_id,
        "system_ops_webhook_fingerprint": _ADMIN_E2E_FINGERPRINT,
        "legacy_table_fingerprint": _ADMIN_E2E_FINGERPRINT,
        "redacted_report_digest": _ADMIN_E2E_DIGEST,
        "protected_backup_ciphertext_digest": _ADMIN_E2E_DIGEST,
        "active_report_path": str(_ADMIN_E2E_ARTIFACT_ROOT / "webhook-report.json"),
        "active_backup_path": str(_ADMIN_E2E_ARTIFACT_ROOT / "webhook-backup.enc"),
        "active_key_path": str(_ADMIN_E2E_ARTIFACT_ROOT / "webhook-backup.key"),
        "staging_report_path": str(_ADMIN_E2E_ARTIFACT_ROOT / "webhook-report.json.staging"),
        "staging_backup_path": str(_ADMIN_E2E_ARTIFACT_ROOT / "webhook-backup.enc.staging"),
        "staging_key_path": str(_ADMIN_E2E_ARTIFACT_ROOT / "webhook-backup.key.staging"),
        "report_owner_id": 1,
        "report_group_id": 1,
        "report_mode": 384,
        "report_file_identity": "admin-e2e:report",
        "backup_owner_id": 1,
        "backup_group_id": 1,
        "backup_mode": 384,
        "backup_file_identity": "admin-e2e:backup",
        "rollback_key_owner_id": 1,
        "rollback_key_group_id": 1,
        "rollback_key_mode": 384,
        "rollback_key_file_identity": "admin-e2e:key",
        "rollback_retirement_phase": "retained",
        "expected_ciphertext_digest": _ADMIN_E2E_DIGEST,
    }


async def prepare_admin_webhooks_for_admin_e2e() -> dict[str, Any]:
    """Complete only a fresh canonical-webhook migration for real-backend e2e."""
    settings = AdminWebhookSettings.from_environment(os.environ)
    if settings.mode is not AdminWebhookMode.ON:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="admin_e2e_webhook_mode_not_on",
        )

    key_ring_result = load_webhook_key_ring()
    ring = key_ring_result.ring
    if ring is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="admin_e2e_webhook_key_unavailable",
        )

    pool = await get_db_pool()
    repository = AdminWebhookRepository(pool)
    now = datetime.now(timezone.utc)
    migration_identity = _admin_e2e_migration_identity(ring.primary_id)
    async with repository.transaction() as tx:
        migration = await tx.lock_migration_state()
        if migration.phase == "complete":
            if migration.active_primary_key_id != ring.primary_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="admin_e2e_webhook_key_mismatch",
                )
            if any(
                getattr(migration, field) != expected
                for field, expected in migration_identity.items()
                if field in _ADMIN_E2E_IDENTITY_FIELDS
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="admin_e2e_webhook_state_mismatch",
                )
            completed = migration
        elif migration.phase != "migration_pending":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="admin_e2e_webhook_migration_in_progress",
            )
        else:
            completed = await tx.compare_and_set_migration_state(
                expected_revision=migration.state_revision,
                updates={
                    "phase": "complete",
                    "import_operator_id": 1,
                    "import_started_at": now,
                    "import_approved_at": now,
                    "artifacts_ready_at": now,
                    "database_committed_at": now,
                    "completed_at": now,
                    "rollback_expires_at": now + timedelta(days=7),
                    **migration_identity,
                },
                at=now,
            )

    return {
        "ok": True,
        "phase": "complete",
        "active_primary_key_id": ring.primary_id,
        "state_revision": completed.state_revision,
    }


def _fixture_secret(name: str) -> str:
    """Return a required fixture secret for admin e2e multi-user scenarios."""
    configured = str(os.getenv(name) or "").strip()
    if configured:
        return configured
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="admin_e2e_fixture_secret_missing",
    )


def _hash_fixture_password(password: str) -> str:
    """Hash a fixture password without applying password-strength policy."""
    return PasswordService().hasher.hash(password)


async def _ensure_user(
    *,
    users_repo: AuthnzUsersRepo,
    username: str,
    email: str,
    password: str,
    role: str,
) -> dict[str, Any]:
    users_db = UsersDB(users_repo.db_pool)
    await users_db.initialize()
    password_hash = _hash_fixture_password(password)

    existing = await users_db.get_user_by_username(username)
    if existing is None:
        existing = await users_db.get_user_by_email(email)

    if existing is None:
        existing = await users_db.create_user(
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            is_active=True,
            is_verified=True,
        )
    else:
        existing = await users_db.update_user(
            int(existing["id"]),
            email=email,
            password_hash=password_hash,
            is_active=True,
            is_verified=True,
            role=role,
        )

    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="admin_e2e_user_seed_failed",
        )

    await users_repo.assign_role_if_missing(user_id=int(existing["id"]), role_name=role)
    return existing


async def _ensure_org(
    *,
    orgs_repo: AuthnzOrgsTeamsRepo,
    owner_user_id: int,
) -> dict[str, Any]:
    rows, _ = await orgs_repo.list_organizations(limit=50, offset=0, q=_ORG_SLUG, with_total=False)
    for row in rows:
        if str(row.get("slug") or "").strip().lower() == _ORG_SLUG:
            return row

    rows, _ = await orgs_repo.list_organizations(limit=50, offset=0, q=_ORG_NAME, with_total=False)
    for row in rows:
        if str(row.get("name") or "").strip() == _ORG_NAME:
            return row

    return await orgs_repo.create_organization(
        name=_ORG_NAME,
        slug=_ORG_SLUG,
        owner_user_id=owner_user_id,
        metadata={"created_by": "admin_e2e_support"},
    )


async def _build_scope_claims_for_user(user_id: int) -> dict[str, Any]:
    memberships = await list_memberships_for_user(int(user_id))
    team_ids = sorted({membership.get("team_id") for membership in memberships if membership.get("team_id") is not None})
    org_ids = sorted({membership.get("org_id") for membership in memberships if membership.get("org_id") is not None})

    claims: dict[str, Any] = {}
    if team_ids:
        claims["team_ids"] = team_ids
    if org_ids:
        claims["org_ids"] = org_ids
    if len(team_ids) == 1:
        claims["active_team_id"] = team_ids[0]
    if len(org_ids) == 1:
        claims["active_org_id"] = org_ids[0]
    return claims


def _recreate_sqlite_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return sqlite3.connect(path)


def _seed_media_store(*, user_id: int, media_count: int) -> None:
    media_db_path = DatabasePaths.get_media_db_path(user_id)
    with _recreate_sqlite_db(media_db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY,
                deleted INTEGER DEFAULT 0,
                is_trash INTEGER DEFAULT 0
            )
            """
        )
        for index in range(media_count):
            conn.execute(
                "INSERT INTO Media (id, deleted, is_trash) VALUES (?, 0, 0)",
                (index + 1,),
            )
        conn.execute("INSERT INTO Media (id, deleted, is_trash) VALUES (?, 1, 0)", (9001,))
        conn.execute("INSERT INTO Media (id, deleted, is_trash) VALUES (?, 0, 1)", (9002,))
        conn.commit()


def _seed_chacha_store(*, user_id: int, note_count: int, message_count: int) -> None:
    chacha_db_path = DatabasePaths.get_chacha_db_path(user_id)
    with _recreate_sqlite_db(chacha_db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, deleted INTEGER DEFAULT 0)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                deleted INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                deleted INTEGER DEFAULT 0
            )
            """
        )
        for index in range(note_count):
            conn.execute("INSERT INTO notes (id, deleted) VALUES (?, 0)", (f"note-{index}",))
        conn.execute("INSERT INTO notes (id, deleted) VALUES (?, 1)", ("note-deleted",))
        conn.execute(
            "INSERT INTO conversations (id, client_id, deleted) VALUES (?, ?, 0)",
            ("conv-1", str(user_id)),
        )
        conn.execute(
            "INSERT INTO conversations (id, client_id, deleted) VALUES (?, ?, 1)",
            ("conv-deleted", str(user_id)),
        )
        for index in range(message_count):
            conn.execute(
                "INSERT INTO messages (id, conversation_id, deleted) VALUES (?, ?, 0)",
                (f"msg-{index}", "conv-1"),
            )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, deleted) VALUES (?, ?, 1)",
            ("msg-deleted", "conv-1"),
        )
        conn.commit()


def _seed_audit_store(*, user_id: int, audit_count: int) -> None:
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import _resolve_audit_storage_mode

    audit_mode = _resolve_audit_storage_mode()
    if audit_mode == "shared":
        audit_db_path = DatabasePaths.get_shared_audit_db_path()
        audit_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(audit_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    context_user_id TEXT,
                    tenant_user_id TEXT
                )
                """
            )
            conn.execute("DELETE FROM audit_events WHERE tenant_user_id = ?", (str(user_id),))
            for index in range(audit_count):
                conn.execute(
                    """
                    INSERT INTO audit_events (
                        event_id,
                        timestamp,
                        category,
                        event_type,
                        severity,
                        context_user_id,
                        tenant_user_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"evt-{user_id}-{index}",
                        f"2026-03-11T20:00:0{index}Z",
                        "system",
                        "admin.e2e.seeded",
                        "low",
                        str(user_id),
                        str(user_id),
                    ),
                )
            conn.commit()
        return

    audit_db_path = DatabasePaths.get_audit_db_path(user_id)
    with _recreate_sqlite_db(audit_db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                context_user_id TEXT,
                tenant_user_id TEXT
            )
            """
        )
        for index in range(audit_count):
            conn.execute(
                """
                INSERT INTO audit_events (
                    event_id,
                    timestamp,
                    category,
                    event_type,
                    severity,
                    context_user_id,
                    tenant_user_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"evt-{user_id}-{index}",
                    f"2026-03-11T20:00:0{index}Z",
                    "system",
                    "admin.e2e.seeded",
                    "low",
                    str(user_id),
                    str(user_id),
                ),
            )
        conn.commit()


def _seed_dsr_subject_store_data(*, user_id: int) -> None:
    _seed_media_store(user_id=user_id, media_count=3)
    _seed_chacha_store(user_id=user_id, note_count=2, message_count=2)
    _seed_audit_store(user_id=user_id, audit_count=2)


async def _seed_dsr_subject_store_data_async(*, user_id: int) -> None:
    """Seed per-user DSR fixture stores off the event loop."""
    await asyncio.to_thread(_seed_dsr_subject_store_data, user_id=user_id)


async def seed_admin_e2e_scenario(scenario: str) -> dict[str, Any]:
    """Create deterministic users and fixtures for admin-ui real-backend browser tests."""
    settings = get_settings()
    await _clear_monitoring_alert_rows_async()
    seeded_alert = await _seed_monitoring_alert_store_async()
    if settings.AUTH_MODE != "multi_user":
        if scenario == "single_user_admin":
            single_user = get_single_user_instance()
            api_key = str(settings.SINGLE_USER_API_KEY or "").strip()
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="admin_e2e_single_user_key_missing",
                )
            return {
                "scenario": scenario,
                "users": {
                    "admin": {
                        "id": int(single_user.id_int or settings.SINGLE_USER_FIXED_ID),
                        "username": str(single_user.username),
                        "email": str(single_user.email or ""),
                        "key": api_key,
                    },
                },
                "fixtures": _seed_fixtures_payload(alerts=[seeded_alert]),
            }
        return {
            "scenario": scenario,
            "users": {},
            "fixtures": _seed_fixtures_payload(alerts=[seeded_alert]),
        }

    pool = await get_db_pool()
    users_repo = AuthnzUsersRepo(db_pool=pool)
    orgs_repo = AuthnzOrgsTeamsRepo(db_pool=pool)

    admin_password = _fixture_secret("TLDW_ADMIN_E2E_ADMIN_PASSWORD")
    owner_password = _fixture_secret("TLDW_ADMIN_E2E_OWNER_PASSWORD")
    super_admin_password = _fixture_secret("TLDW_ADMIN_E2E_SUPER_ADMIN_PASSWORD")
    member_password = _fixture_secret("TLDW_ADMIN_E2E_MEMBER_PASSWORD")
    requester_password = _fixture_secret("TLDW_ADMIN_E2E_REQUESTER_PASSWORD")

    admin_user = await _ensure_user(
        users_repo=users_repo,
        username=_ADMIN_USERNAME,
        email=_ADMIN_EMAIL,
        password=admin_password,
        role="admin",
    )
    owner_user = await _ensure_user(
        users_repo=users_repo,
        username=_OWNER_USERNAME,
        email=_OWNER_EMAIL,
        password=owner_password,
        role="owner",
    )
    super_admin_user = await _ensure_user(
        users_repo=users_repo,
        username=_SUPER_ADMIN_USERNAME,
        email=_SUPER_ADMIN_EMAIL,
        password=super_admin_password,
        role="super_admin",
    )
    member_user = await _ensure_user(
        users_repo=users_repo,
        username=_NON_ADMIN_USERNAME,
        email=_NON_ADMIN_EMAIL,
        password=member_password,
        role="user",
    )
    requester_user = await _ensure_user(
        users_repo=users_repo,
        username=_REQUESTER_USERNAME,
        email=_REQUESTER_EMAIL,
        password=requester_password,
        role="user",
    )

    org = await _ensure_org(orgs_repo=orgs_repo, owner_user_id=int(admin_user["id"]))
    org_id = int(org["id"])
    await orgs_repo.add_org_member(org_id=org_id, user_id=int(admin_user["id"]), role="admin")
    await orgs_repo.add_org_member(org_id=org_id, user_id=int(owner_user["id"]), role="owner")
    await orgs_repo.add_org_member(org_id=org_id, user_id=int(super_admin_user["id"]), role="admin")
    await orgs_repo.add_org_member(org_id=org_id, user_id=int(member_user["id"]), role="member")
    await orgs_repo.add_org_member(org_id=org_id, user_id=int(requester_user["id"]), role="member")

    if scenario == "dsr_jwt_admin":
        await _seed_dsr_subject_store_data_async(user_id=int(requester_user["id"]))

    _SEEDED_PRINCIPALS["jwt_admin"] = {
        "user_id": int(admin_user["id"]),
        "username": str(admin_user["username"]),
        "role": str(admin_user.get("role") or "admin"),
    }
    _SEEDED_PRINCIPALS["jwt_owner"] = {
        "user_id": int(owner_user["id"]),
        "username": str(owner_user["username"]),
        "role": str(owner_user.get("role") or "owner"),
    }
    _SEEDED_PRINCIPALS["jwt_super_admin"] = {
        "user_id": int(super_admin_user["id"]),
        "username": str(super_admin_user["username"]),
        "role": str(super_admin_user.get("role") or "super_admin"),
    }
    _SEEDED_PRINCIPALS["jwt_non_admin"] = {
        "user_id": int(member_user["id"]),
        "username": str(member_user["username"]),
        "role": str(member_user.get("role") or "user"),
    }

    return {
        "scenario": scenario,
        "users": {
            "admin": {
                "id": int(admin_user["id"]),
                "username": str(admin_user["username"]),
                "email": str(admin_user["email"]),
                "key": "jwt_admin",
            },
            "owner": {
                "id": int(owner_user["id"]),
                "username": str(owner_user["username"]),
                "email": str(owner_user["email"]),
                "key": "jwt_owner",
            },
            "super_admin": {
                "id": int(super_admin_user["id"]),
                "username": str(super_admin_user["username"]),
                "email": str(super_admin_user["email"]),
                "key": "jwt_super_admin",
            },
            "non_admin": {
                "id": int(member_user["id"]),
                "username": str(member_user["username"]),
                "email": str(member_user["email"]),
                "key": "jwt_non_admin",
            },
            "requester": {
                "id": int(requester_user["id"]),
                "username": str(requester_user["username"]),
                "email": str(requester_user["email"]),
                "key": "requester_user",
            },
        },
        "fixtures": {
            **_seed_fixtures_payload(alerts=[seeded_alert]),
            "organizations": [
                {
                    "id": org_id,
                    "name": str(org["name"]),
                    "slug": str(org.get("slug") or _ORG_SLUG),
                }
            ],
        },
    }


async def bootstrap_admin_e2e_jwt_session(principal_key: str) -> dict[str, Any]:
    """Create a real JWT session payload for Playwright cookie injection."""
    settings = get_settings()
    if settings.AUTH_MODE != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="admin_e2e_jwt_requires_multi_user_mode",
        )

    seeded = _SEEDED_PRINCIPALS.get(principal_key)
    if seeded is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="admin_e2e_principal_not_seeded",
        )

    session_manager = await get_session_manager()
    jwt_service = JWTService(settings)

    temp_session = await session_manager.create_session(
        user_id=int(seeded["user_id"]),
        access_token=secrets.token_urlsafe(24),
        refresh_token=secrets.token_urlsafe(24),
        ip_address="127.0.0.1",
        user_agent="admin-ui-real-backend-e2e",
    )
    session_id = int(temp_session["session_id"])

    additional_claims = await _build_scope_claims_for_user(int(seeded["user_id"]))
    additional_claims["session_id"] = session_id
    access_token = jwt_service.create_access_token(
        user_id=int(seeded["user_id"]),
        username=str(seeded["username"]),
        role=str(seeded["role"]),
        additional_claims=additional_claims,
    )
    refresh_token = jwt_service.create_refresh_token(
        user_id=int(seeded["user_id"]),
        username=str(seeded["username"]),
        additional_claims=additional_claims,
    )
    await session_manager.update_session_tokens(
        session_id=session_id,
        access_token=access_token,
        refresh_token=refresh_token,
    )

    return {
        "principal_key": principal_key,
        "cookies": [
            {
                "name": "access_token",
                "value": access_token,
                "path": "/",
                "http_only": True,
                "same_site": "Lax",
            },
            {
                "name": "refresh_token",
                "value": refresh_token,
                "path": "/",
                "http_only": True,
                "same_site": "Lax",
            },
            {
                "name": "admin_session",
                "value": "1",
                "path": "/",
                "http_only": False,
                "same_site": "Lax",
            },
            {
                "name": "admin_auth_mode",
                "value": "jwt",
                "path": "/",
                "http_only": False,
                "same_site": "Lax",
            },
        ],
    }


async def run_due_backup_schedules_for_admin_e2e() -> dict[str, Any]:
    """Force active backup schedules due now and process the resulting queued jobs once."""
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Storage.backup_schedule_jobs import (
        BACKUP_SCHEDULE_DOMAIN,
        BACKUP_SCHEDULE_JOB_TYPE,
    )
    from tldw_Server_API.app.services.admin_backup_jobs_worker import handle_backup_schedule_job
    from tldw_Server_API.app.services.admin_backup_scheduler import _AdminBackupScheduler

    repo = await _get_backup_schedules_repo()
    scheduler = _AdminBackupScheduler(repo=repo)
    jobs = JobManager()

    active_schedules: list[dict[str, Any]] = []
    offset = 0
    page_size = 200
    while True:
        page, total = await repo.list_schedules(limit=page_size, offset=offset, include_deleted=False)
        active_schedules.extend([item for item in page if not bool(item.get("is_paused"))])
        offset += len(page)
        if not page or offset >= int(total):
            break

    if not active_schedules:
        return {"ok": True, "triggered_runs": 0}

    existing_job_ids = {
        str(job["id"])
        for job in jobs.list_jobs(
            domain=BACKUP_SCHEDULE_DOMAIN,
            job_type=BACKUP_SCHEDULE_JOB_TYPE,
            limit=500,
        )
    }

    forced_due = datetime.now(timezone.utc).replace(second=0, microsecond=0).isoformat()
    for schedule in active_schedules:
        await repo.update_schedule(
            str(schedule["id"]),
            next_run_at=forced_due,
            updated_by_user_id=(
                int(schedule["updated_by_user_id"])
                if schedule.get("updated_by_user_id") is not None
                else None
            ),
        )
        await scheduler._run_schedule(str(schedule["id"]))

    queued_new_jobs = [
        job
        for job in jobs.list_jobs(
            domain=BACKUP_SCHEDULE_DOMAIN,
            job_type=BACKUP_SCHEDULE_JOB_TYPE,
            status="queued",
            limit=500,
        )
        if str(job["id"]) not in existing_job_ids
    ]

    triggered_runs = 0
    for job in sorted(queued_new_jobs, key=lambda item: int(item["id"])):
        result = await handle_backup_schedule_job(job, repo=repo)
        jobs.complete_job(int(job["id"]), result=result, enforce=False)
        triggered_runs += 1

    return {"ok": True, "triggered_runs": triggered_runs}
