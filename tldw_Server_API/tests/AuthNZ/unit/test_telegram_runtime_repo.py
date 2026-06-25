from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)


@pytest.mark.asyncio
async def test_telegram_runtime_repo_persists_receipts_pairing_codes_and_links(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.telegram_runtime_repo import TelegramRuntimeRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = TelegramRuntimeRepo(pool)
    await repo.ensure_tables()

    now = datetime.now(timezone.utc)
    first_store = await repo.store_webhook_receipt(
        dedupe_key="team:22:9001",
        scope_type="team",
        scope_id=22,
        update_id=9001,
        expires_at=now + timedelta(minutes=5),
        now=now,
    )
    second_store = await repo.store_webhook_receipt(
        dedupe_key="team:22:9001",
        scope_type="team",
        scope_id=22,
        update_id=9001,
        expires_at=now + timedelta(minutes=5),
        now=now,
    )
    third_store = await repo.store_webhook_receipt(
        dedupe_key="team:22:9001",
        scope_type="team",
        scope_id=22,
        update_id=9001,
        expires_at=now + timedelta(minutes=10),
        now=now + timedelta(minutes=6),
    )
    assert first_store is True
    assert second_store is False
    assert third_store is True

    pairing = await repo.create_pairing_code(
        pairing_code="ABCD1234",
        scope_type="team",
        scope_id=22,
        auth_user_id=901,
        expires_at=now + timedelta(minutes=5),
        now=now,
    )
    assert pairing["pairing_code"] == "ABCD1234"

    stored_pairing = await pool.fetchone(
        """
        SELECT pairing_code
        FROM telegram_pairing_codes
        WHERE id = ?
        """,
        (pairing["id"],),
    )
    assert stored_pairing is not None
    assert stored_pairing["pairing_code"] != "ABCD1234"
    assert str(stored_pairing["pairing_code"]).startswith("hmac_sha256:")

    wrong_scope_pairing = await repo.consume_pairing_code(
        "abcd1234",
        scope_type="team",
        scope_id=23,
        now=now + timedelta(minutes=1),
    )
    consumed_pairing = await repo.consume_pairing_code(
        "abcd1234",
        scope_type="team",
        scope_id=22,
        now=now + timedelta(minutes=1),
    )
    missing_pairing = await repo.consume_pairing_code(
        "ABCD1234",
        scope_type="team",
        scope_id=22,
        now=now + timedelta(minutes=1),
    )
    assert wrong_scope_pairing is None
    assert consumed_pairing is not None
    assert consumed_pairing["auth_user_id"] == 901
    assert missing_pairing is None

    linked = await repo.upsert_actor_link(
        scope_type="team",
        scope_id=22,
        telegram_user_id=77,
        auth_user_id=901,
        telegram_username="linked-user",
        now=now,
    )
    fetched = await repo.get_actor_link(
        scope_type="team",
        scope_id=22,
        telegram_user_id=77,
    )
    assert linked["auth_user_id"] == 901
    assert fetched is not None
    assert fetched["telegram_username"] == "linked-user"
