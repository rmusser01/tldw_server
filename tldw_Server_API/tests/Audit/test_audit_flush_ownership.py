"""A strict flush must not acknowledge another flush's uncommitted events."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from tldw_Server_API.app.core.Audit.unified_audit_service import AuditContext, AuditEventType, UnifiedAuditService


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["commit", "fail", "cancel"])
async def test_strict_flush_waits_for_inflight_event_ownership(tmp_path, monkeypatch, outcome):
    audit = UnifiedAuditService(
        db_path=str(tmp_path / "audit.db"),
        storage_mode="per_user",
        enable_pii_detection=False,
        enable_risk_scoring=False,
    )
    await audit.initialize(start_background_tasks=False)
    event_id = await audit.log_event(
        AuditEventType.CONFIG_CHANGED, context=AuditContext(user_id="41"), action="controlled.rule.change"
    )
    entered, release, required_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = audit._immediate_write_transaction
    attempts = 0
    tasks = []

    @asynccontextmanager
    async def held_transaction(db):
        nonlocal attempts
        attempts += 1
        async with original(db):
            if attempts == 1:
                entered.set()
                await release.wait()
            if outcome == "fail":
                raise OSError("controlled audit write failure")
            yield

    async def strict():
        required_started.set()
        return await audit.flush(raise_on_failure=True)

    async def committed_ids():
        async with audit._read_db() as db:
            return await audit._fetch_existing_event_ids(db, [event_id])

    try:
        with monkeypatch.context() as injection:
            injection.setattr(audit, "_immediate_write_transaction", held_transaction)
            first = asyncio.create_task(audit.flush())
            tasks.append(first)
            await asyncio.wait_for(entered.wait(), 5)
            assert await committed_ids() == set()  # nosec B101 -- Observe storage without query_events auto-flush.
            required = asyncio.create_task(strict())
            tasks.append(required)
            await asyncio.wait_for(required_started.wait(), 5)
            assert not required.done()  # nosec B101 -- A buffer drain is not a durable acknowledgement.
            if outcome == "cancel":
                first.cancel()
            release.set()
            if outcome == "fail":
                assert await asyncio.wait_for(first, 5) is False  # nosec B101 -- Original background failure is retained.
                with pytest.raises(OSError, match="controlled audit write failure"):
                    await asyncio.wait_for(required, 5)
                assert await committed_ids() == set()  # nosec B101 -- A failed transaction never became durable.
            else:
                if outcome == "cancel":
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(first, 5)
                else:
                    assert await asyncio.wait_for(first, 5) is True  # nosec B101 -- Real first transaction committed.
                assert await asyncio.wait_for(required, 5) is True  # nosec B101 -- Strict completion follows ownership release.
                assert await committed_ids() == {event_id}  # nosec B101 -- Exact event is present without read-triggered flush.
    finally:
        release.set()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
        await audit.stop()


@pytest.mark.asyncio
async def test_exact_event_flush_rejects_missing_identity_before_storage(tmp_path):
    from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError

    path = tmp_path / "not-created.db"
    audit = UnifiedAuditService(db_path=str(path))
    for event_id in ("", None, 42):
        with pytest.raises(MandatoryAuditWriteError, match="identity missing"):
            await audit.flush_event(event_id)
    assert not path.exists()  # nosec B101 -- Invalid event identity cannot start storage work.
