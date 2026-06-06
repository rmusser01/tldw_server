from __future__ import annotations

import base64
import json

import pytest

from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.secret_store import CalendarSecretStore
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase

pytestmark = pytest.mark.unit


@pytest.fixture
def calendar_db(tmp_path):
    db = CalendarDatabase(db_path=tmp_path / "calendar_secrets.db")
    db.ensure_schema()
    return db


@pytest.fixture
def calendar_secret_key(monkeypatch):
    key = base64.b64encode(b"calendar-secret-store-key-32-bytes").decode("ascii")
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", key)
    return key


def test_secret_store_encrypts_payload_and_resolves_only_for_owner(
    calendar_db,
    calendar_secret_key,
) -> None:
    store = CalendarSecretStore(db=calendar_db, tenant_id="default")

    secret_ref = store.create_secret(
        owner_user_id=1,
        provider="caldav",
        payload={"server_url": "https://caldav.example.test", "username": "reader@example.test", "password": "app-secret"},
    )

    assert secret_ref.startswith("calendar_secret_")
    encrypted_payload = calendar_db.resolve_secret_ref(secret_ref)
    assert "reader@example.test" not in encrypted_payload
    assert "app-secret" not in encrypted_payload
    assert json.loads(encrypted_payload)["_enc"] == "aesgcm:v1"
    assert store.resolve_secret(owner_user_id=1, secret_ref=secret_ref) == {
        "server_url": "https://caldav.example.test",
        "username": "reader@example.test",
        "password": "app-secret",
    }
    with pytest.raises(CalendarValidationError):
        store.resolve_secret(owner_user_id=2, secret_ref=secret_ref)


@pytest.mark.parametrize("cleanup_method", ["delete_secret", "delete_account", "revoke_account"])
def test_secret_store_deletes_secret_material(
    calendar_db,
    calendar_secret_key,
    cleanup_method,
) -> None:
    store = CalendarSecretStore(db=calendar_db, tenant_id="default")
    secret_ref = store.create_secret(
        owner_user_id=1,
        provider="caldav",
        payload={"username": "reader@example.test", "password": "app-secret"},
    )
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        display_name="Fastmail",
        secret_ref=secret_ref,
    )

    if cleanup_method == "delete_secret":
        assert store.delete_secret(owner_user_id=1, secret_ref=secret_ref) is True
    elif cleanup_method == "delete_account":
        calendar_db.delete_external_account(account.id)
    else:
        calendar_db.revoke_external_account(account.id)

    with pytest.raises(CalendarValidationError):
        store.resolve_secret(owner_user_id=1, secret_ref=secret_ref)
    with calendar_db.connection() as conn:
        row = conn.execute(
            """
            SELECT encrypted_payload, deleted_at
            FROM calendar_external_account_secrets
            WHERE secret_ref = ?
            """,
            (secret_ref,),
        ).fetchone()
    assert row is not None
    assert row["encrypted_payload"] == ""
    assert row["deleted_at"] is not None


def test_secret_store_requires_calendar_encryption_key(calendar_db, monkeypatch) -> None:
    monkeypatch.delenv("CALENDAR_SECRET_ENCRYPTION_KEY", raising=False)
    store = CalendarSecretStore(db=calendar_db, tenant_id="default")

    with pytest.raises(CalendarValidationError, match="CALENDAR_SECRET_ENCRYPTION_KEY"):
        store.create_secret(
            owner_user_id=1,
            provider="caldav",
            payload={"username": "reader@example.test", "password": "app-secret"},
        )
