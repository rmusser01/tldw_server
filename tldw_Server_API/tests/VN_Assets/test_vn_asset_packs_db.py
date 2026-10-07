import asyncio
import json
import sqlite3
import threading
from collections.abc import Generator, Mapping
from contextlib import closing
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import (
    VNAssetPacksRepository,
    ensure_vn_asset_tables,
)
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.VN_Assets.jobs import (
    build_legacy_activity_reader,
    create_generate_variant_job,
    vn_asset_generation_jobs_queue,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-assets-test-client")
    yield database
    database.close_connection()


@pytest.fixture
def sqlite_timestamp(chacha_db: CharactersRAGDB) -> dict[str, str]:
    """Control native CURRENT_TIMESTAMP on this test's owning SQLite connection."""
    clock = {"now": "2000-01-01 00:00:00"}
    chacha_db.get_connection().create_function("current_timestamp", 0, lambda: clock["now"])
    return clock


@pytest.fixture
def integrity_state(chacha_db: CharactersRAGDB, tmp_path: Path) -> dict[str, Any]:
    """Create damaged history, approvals, claimed work and a live sibling batch."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slots = [repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key=name) for name in ("first", "second")]
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=6,
        recipes=[{"slot_id": slots[index == 4]["id"], "variant_index": index, "recipe": {}} for index in range(6)],
    )
    approvals = []
    for index in (0, 5):
        path = tmp_path / f"approved-{index}.png"
        path.write_bytes(b"approved")
        item = repo.reserve_variant_item(
            batch_id=batch["id"], slot_id=slots[0]["id"], variant_index=index,
            item_fields={"pack_id": pack["id"], "generated_file_id": 17 + index,
                         "storage_ref": str(path), "bytes": 8},
        )
        repo.complete_variant(batch_id=batch["id"], slot_id=slots[0]["id"], variant_index=index, item_id=item["id"])
        repo.update_item_review(item["id"], review_status="approved", preferred=False)
        approvals.append(repo.get_item(item["id"]))
    repo.fail_variant(batch_id=batch["id"], slot_id=slots[0]["id"], variant_index=1, error="historical")
    with chacha_db.transaction() as conn:
        conn.execute("UPDATE vn_asset_generation_recipes SET outcome_status='cancelled' WHERE batch_id=? AND variant_index=2", (batch["id"],))
        conn.execute("DELETE FROM vn_asset_generation_recipes WHERE batch_id=? AND variant_index=5", (batch["id"],))
    repo.update_batch(batch["id"], {"cancelled_count": 1, "enqueued_count": 3})
    old_attempt, sibling_attempt = "old", "sibling"
    reserved = repo.claim_variant(
        batch_id=batch["id"], slot_id=slots[0]["id"], variant_index=3,
        lease_id="old", attempt_token=old_attempt, item_fields={"pack_id": pack["id"]},
    )
    repo.reserve_variant_item(
        batch_id=batch["id"], slot_id=slots[1]["id"], variant_index=4,
        item_fields={"pack_id": pack["id"]},
    )
    sibling = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slots[0]["id"], "variant_index": 0, "recipe": {}}],
    )
    repo.claim_variant(
        batch_id=sibling["id"], slot_id=slots[0]["id"], variant_index=0,
        lease_id="inline", attempt_token=sibling_attempt, item_fields={"pack_id": pack["id"]},
    )
    repo.start_variant_generation(batch_id=sibling["id"], slot_id=slots[0]["id"], variant_index=0, attempt_token=sibling_attempt)
    return {"repo": repo, "pack": pack, "batch": batch, "slots": slots,
            "approvals": approvals, "reserved": reserved, "sibling": sibling, "old_attempt": old_attempt}


@pytest.mark.integration
@pytest.mark.parametrize("cancelled", [False, True])
def test_integrity_failure_preserves_history_approvals_cancellation_and_sibling(
    integrity_state: dict[str, Any], cancelled: bool,
) -> None:
    """Fail only surviving unfinished recipes once, without recounting lost history."""
    state = integrity_state
    repo, batch_id = state["repo"], state["batch"]["id"]
    if cancelled:
        repo.cancel_batch(batch_id)
    sibling_before = repo.get_batch(state["sibling"]["id"])
    first = repo.fail_batch_integrity(batch_id, error="vn_asset_recipe_count_mismatch")
    second = repo.fail_batch_integrity(batch_id, error="vn_asset_recipe_count_mismatch")
    assert second == first
    assert (first["completed_count"], first["failed_count"], first["cancelled_count"]) == ((2, 1, 3) if cancelled else (2, 3, 1))
    assert first["status"] == ("cancelled" if cancelled else "failed")
    assert first["planned_count"] == 6 and first["enqueued_count"] == 3
    assert repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 5) is None
    assert repo.get_batch(state["sibling"]["id"]) == sibling_before
    assert repo.count_items_for_generation(state["pack"]["id"]) == 3
    assert repo.get_slot(state["slots"][0]["id"])["status"] == "generating"
    assert repo.get_slot(state["slots"][1]["id"])["status"] == ("cancelled" if cancelled else "failed")
    for item in state["approvals"]:
        assert repo.get_item(item["id"]) == item
        assert Path(item["storage_ref"]).read_bytes() == b"approved"
    identity = {"batch_id": batch_id, "slot_id": state["slots"][0]["id"], "variant_index": 3}
    late_attempt = "late"
    with pytest.raises(VNAssetGenerationError):
        repo.claim_variant(**identity, lease_id="late", attempt_token=late_attempt, item_fields={"pack_id": state["pack"]["id"]})
    with pytest.raises(VNAssetGenerationError):
        repo.update_item_storage(state["reserved"]["id"], **identity, attempt_token=state["old_attempt"],
                                 generated_file_id=88, storage_ref="late.png", mime_type="image/png",
                                 width=10, height=10, bytes=3)
    with pytest.raises(VNAssetGenerationError):
        repo.complete_variant(**identity, item_id=state["reserved"]["id"], attempt_token=state["old_attempt"])


@pytest.mark.integration
def test_integrity_failure_rolls_back_recipe_counters_and_every_slot(
    integrity_state: dict[str, Any],
) -> None:
    """A slot reconciliation failure rolls back the complete terminal transition."""
    state = integrity_state
    repo, batch_id = state["repo"], state["batch"]["id"]
    _legacy_integrity_batch(repo, state)
    repo.update_slot(state["slots"][0]["id"], {"status": "queued"})
    batch_before = repo.get_batch(batch_id)
    slots_before = [repo.get_slot(slot["id"]) for slot in state["slots"]]
    outcome_before = repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 3)
    def fail_second(
        _pack_id: int, slot_id: int, _user_id: int, _batches: Mapping[int, str],
        _settled: set[tuple[int, int, str]], _finishing: tuple[int, str] | None,
    ) -> tuple[bool, bool]:
        """Reject the public activity read after real first-slot reconciliation."""
        if slot_id == state["slots"][1]["id"]:
            assert repo.db.get_connection().in_transaction
            assert repo.get_batch(batch_id)["failed_count"] == 3
            assert repo.get_slot(state["slots"][0]["id"])["status"] == "generating"
            raise RuntimeError("reconciliation failed")
        return False, False

    repo.legacy_activity_reader = fail_second
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        repo.fail_batch_integrity(batch_id, error="vn_asset_recipe_count_mismatch")
    assert repo.get_batch(batch_id) == batch_before
    assert [repo.get_slot(slot["id"]) for slot in state["slots"]] == slots_before
    assert repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 3) == outcome_before
    assert repo.count_items_for_generation(state["pack"]["id"]) == 5


@pytest.mark.integration
def test_integrity_failure_reconciles_interrupted_cancellation_without_recounting_history(
    integrity_state: dict[str, Any],
) -> None:
    """Already-cancelled batches cancel leftover reservations rather than fail them."""
    state = integrity_state
    repo, batch_id = state["repo"], state["batch"]["id"]
    repo.update_batch(batch_id, {"status": "cancelled", "cancelled_count": 7})
    cancelled = repo.fail_batch_integrity(batch_id, error="vn_asset_recipe_count_mismatch")
    assert (cancelled["status"], cancelled["completed_count"], cancelled["failed_count"], cancelled["cancelled_count"]) == ("cancelled", 2, 1, 9)
    assert repo.count_items_for_generation(state["pack"]["id"]) == 3
    assert repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 3)["outcome_status"] == "cancelled"
    assert repo.get_slot(state["slots"][0]["id"])["status"] == "generating"
    assert repo.fail_batch_integrity(batch_id, error="vn_asset_recipe_count_mismatch") == cancelled


def _integrity_provenance_state(
    chacha_db: CharactersRAGDB, *, authored: bool = True,
) -> dict[str, Any]:
    """Start B after A failed, retaining A's ID but clearing its older error."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Integrity Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Integrity Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    recipe = {"slots": [{"slot_id": slot["id"], "variant_count": 1}]}
    previous = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe=recipe if authored else None,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "previous"}}],
    )
    repo.fail_variant(batch_id=previous["id"], slot_id=slot["id"], variant_index=0, error="previous failure")
    current = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe=recipe if authored else None, execution_recipe={"backend": "pinned"},
        source_batch_id=previous["id"],
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "current"}}],
    )
    identity = {"batch_id": current["id"], "slot_id": slot["id"], "variant_index": 0}
    attempt_token = f"integrity-inline-{current['id']}"
    item = repo.claim_variant(
        **identity, lease_id="inline", attempt_token=attempt_token, item_fields={"pack_id": pack["id"]},
    )
    repo.start_variant_generation(**identity, attempt_token=attempt_token)
    return {"repo": repo, "pack": pack, "slot": slot, "previous": previous,
            "current": current, "identity": identity, "attempt_token": attempt_token, "item": item}


@pytest.mark.parametrize("authored", [True, False], ids=["latest-authored-owner", "unowned-legacy"])
def test_integrity_provenance_replaces_previous_failed_source(
    chacha_db: CharactersRAGDB, sqlite_timestamp: dict[str, str], authored: bool,
) -> None:
    """Catch integrity failure retaining A as Retry source after B starts."""
    state = _integrity_provenance_state(chacha_db, authored=authored)
    repo, current, slot_id = state["repo"], state["current"], state["slot"]["id"]
    before = repo.get_slot(slot_id)
    assert (before["status"], before["last_failed_batch_id"], before["last_error"]) == (
        "generating", state["previous"]["id"], None,
    )

    failed = repo.fail_batch_integrity(current["id"], error="vn_asset_recipe_count_mismatch")

    stored = repo.get_slot(slot_id)
    assert (stored["status"], stored["last_failed_batch_id"], stored["last_error"],
            stored["latest_generation_batch_id"]) == (
        "failed", current["id"], "vn_asset_recipe_count_mismatch", current["id"] if authored else None,
    )
    assert (failed["status"], failed["completed_count"], failed["failed_count"], failed["cancelled_count"]) == (
        "failed", 0, 1, 0,
    )
    for field in ("recipe_json", "execution_recipe_json", "source_batch_id"):
        assert failed[field] == current[field]
    assert repo.get_variant_outcome(**state["identity"])["outcome_status"] == "failed"
    sqlite_timestamp["now"] = "2000-01-01 00:00:01"
    assert repo.fail_batch_integrity(current["id"], error="different replay error") == failed
    assert repo.get_slot(slot_id) == stored
    with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
        repo.start_variant_generation(**state["identity"], attempt_token=state["attempt_token"])


def test_integrity_provenance_does_not_overwrite_newer_failed_owner(chacha_db: CharactersRAGDB) -> None:
    """Catch older B publishing provenance over the latest authored owner C."""
    state = _integrity_provenance_state(chacha_db)
    repo, slot_id = state["repo"], state["slot"]["id"]
    newer = repo.create_batch(
        pack_id=state["pack"]["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe={"slots": [{"slot_id": slot_id, "variant_count": 1}]},
        recipes=[{"slot_id": slot_id, "variant_index": 0, "recipe": {"prompt": "newer"}}],
    )
    repo.fail_batch_enqueue(newer["id"], "newer failure")
    slot_before, newer_before = repo.get_slot(slot_id), repo.get_batch(newer["id"])

    failed = repo.fail_batch_integrity(state["current"]["id"], error="older integrity failure")

    assert (failed["status"], failed["failed_count"]) == ("failed", 1)
    assert repo.get_slot(slot_id) == slot_before
    assert repo.get_batch(newer["id"]) == newer_before


def test_integrity_provenance_preserves_interrupted_cancellation(chacha_db: CharactersRAGDB) -> None:
    """Catch cancelled B replacing cancellation or publishing failure provenance."""
    state = _integrity_provenance_state(chacha_db)
    repo, batch_id, slot_id = state["repo"], state["current"]["id"], state["slot"]["id"]
    repo.update_batch(batch_id, {"status": "cancelled", "enqueue_error": "cancelled before reconciliation"})

    cancelled = repo.fail_batch_integrity(batch_id, error="must not replace cancellation")

    assert (cancelled["status"], cancelled["failed_count"], cancelled["cancelled_count"], cancelled["enqueue_error"]) == (
        "cancelled", 0, 1, "cancelled before reconciliation",
    )
    stored = repo.get_slot(slot_id)
    assert (stored["last_failed_batch_id"], stored["last_error"], stored["latest_generation_batch_id"]) == (
        state["previous"]["id"], None, batch_id,
    )
    assert repo.get_variant_outcome(**state["identity"])["outcome_status"] == "cancelled"
    assert repo.fail_batch_integrity(batch_id, error="another failure") == cancelled


def test_integrity_provenance_only_marks_newly_failed_slots(chacha_db: CharactersRAGDB) -> None:
    """Catch stamping terminal-only slots or hiding published/deleted outcomes."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Integrity Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Integrity Pack")
    slots = [repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key=name)
             for name in ("partial", "approved", "deleted")]
    entries = [{"slot_id": slot["id"], "variant_index": index, "recipe": {}}
               for slot, count in zip(slots, (2, 1, 1), strict=True) for index in range(count)]
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=4,
        recipe={"slots": [{"slot_id": slot["id"], "variant_count": count}
                          for slot, count in zip(slots, (2, 1, 1), strict=True)]},
        recipes=entries,
    )
    terminal_outcomes = []
    for slot, index in zip(slots, (1, 0, 0), strict=True):
        identity = {"batch_id": batch["id"], "slot_id": slot["id"], "variant_index": index}
        item = repo.reserve_variant_item(
            **identity, item_fields={"pack_id": pack["id"], "generated_file_id": 17 + slot["id"]},
        )
        repo.complete_variant(**identity, item_id=item["id"])
        if slot == slots[1]:
            repo.update_item_review(item["id"], review_status="approved", preferred=False)
        elif slot == slots[2]:
            repo.delete_item(item["id"])
        terminal_outcomes.append(repo.get_variant_outcome(**identity))
    items_before = repo.list_items(pack["id"])
    terminal_slots_before = [repo.get_slot(slot["id"]) for slot in slots[1:]]

    failed = repo.fail_batch_integrity(batch["id"], error="unfinished recipe corrupt")

    partial = repo.get_slot(slots[0]["id"])
    assert (partial["status"], partial["last_failed_batch_id"], partial["last_error"]) == (
        "reviewing", batch["id"], "unfinished recipe corrupt",
    )
    assert (failed["completed_count"], failed["failed_count"], failed["cancelled_count"]) == (3, 1, 0)
    assert repo.list_items(pack["id"]) == items_before
    for slot, before, status in zip(slots[1:], terminal_slots_before, ("approved", "planned"), strict=True):
        stored = repo.get_slot(slot["id"])
        assert (stored["status"], stored["last_failed_batch_id"], stored["last_error"]) == (
            status, before["last_failed_batch_id"], before["last_error"],
        )
    for slot, index, outcome in zip(slots, (1, 0, 0), terminal_outcomes, strict=True):
        assert repo.get_variant_outcome(batch["id"], slot["id"], index) == outcome


def _file_integrity_repository(state: dict[str, Any], tmp_path: Path) -> VNAssetPacksRepository:
    """Copy the existing damaged-history fixture to a real file-backed repository."""
    path = tmp_path / "integrity.db"
    with closing(sqlite3.connect(path)) as target:
        state["repo"].db.get_connection().backup(target)
    return VNAssetPacksRepository.initialized(CharactersRAGDB(path, client_id="integrity-owner"))


def _legacy_integrity_batch(repo: VNAssetPacksRepository, state: dict[str, Any]) -> dict[str, Any]:
    """Add real V0 work so reconciliation consults its supported activity reader."""
    return repo.create_batch(
        pack_id=state["pack"]["id"], requested_by_user_id=1, total_variants=2,
        options={"slot_ids": [slot["id"] for slot in state["slots"]], "variant_count": 1},
    )


def _assert_integrity_handles_closed(
    owned: list[tuple[threading.Thread, sqlite3.Connection]], owner: sqlite3.Connection,
) -> None:
    """Prove callback-acquired handles close while the caller handle stays usable."""
    assert owned, "the activity callback did not expose the owned reconciliation handle"
    for thread, connection in owned:
        assert thread.ident != threading.get_ident() and not thread.is_alive()
        assert connection is not owner
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
    assert not owner.in_transaction
    assert owner.execute("SELECT 1").fetchone()[0] == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_async_integrity_preserves_history_and_terminal_fences(
    integrity_state: dict[str, Any], tmp_path: Path, cancelled: bool,
) -> None:
    """Owned reconciliation preserves approvals, sibling activity and terminal counts."""
    state = integrity_state
    repo = _file_integrity_repository(state, tmp_path)
    batch_id, slot_id = state["batch"]["id"], state["slots"][0]["id"]
    try:
        if cancelled:
            repo.update_batch(batch_id, {"status": "cancelled", "cancelled_count": 7})
        sibling_before = repo.get_batch(state["sibling"]["id"])
        first = await repo.fail_batch_integrity_async(batch_id, error="vn_asset_recipe_not_found")
        assert await repo.fail_batch_integrity_async(batch_id, error="vn_asset_recipe_not_found") == first
        assert (first["completed_count"], first["failed_count"], first["cancelled_count"]) == ((2, 1, 9) if cancelled else (2, 3, 1))
        assert first["status"] == ("cancelled" if cancelled else "failed")
        assert (first["planned_count"], first["enqueued_count"]) == (6, 3)
        assert repo.get_batch(state["sibling"]["id"]) == sibling_before
        assert repo.get_variant_outcome(batch_id, slot_id, 5) is None
        assert repo.count_items_for_generation(state["pack"]["id"]) == 3
        assert repo.get_slot(slot_id)["status"] == "generating"
        assert repo.get_slot(state["slots"][1]["id"])["status"] == ("cancelled" if cancelled else "failed")
        for item in state["approvals"]:
            assert repo.get_item(item["id"]) == item
            assert Path(item["storage_ref"]).read_bytes() == b"approved"
        identity = {"batch_id": batch_id, "slot_id": slot_id, "variant_index": 3}
        late_attempt = "late"
        with pytest.raises(VNAssetGenerationError):
            repo.claim_variant(**identity, lease_id="late", attempt_token=late_attempt, item_fields={"pack_id": state["pack"]["id"]})
        with pytest.raises(VNAssetGenerationError):
            repo.complete_variant(**identity, item_id=state["reserved"]["id"], attempt_token=state["old_attempt"])
        with pytest.raises(VNAssetGenerationError):
            repo.update_item_storage(state["reserved"]["id"], **identity, attempt_token=state["old_attempt"],
                                     generated_file_id=88, storage_ref="late.png", mime_type="image/png",
                                     width=10, height=10, bytes=3)
    finally:
        repo.db.close_connection()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_delivery", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_async_integrity_owns_atomic_reconciliation_and_cleanup(
    integrity_state: dict[str, Any], tmp_path: Path,
    fail: bool, cancel_delivery: bool,
) -> None:
    """Readers see no partial writes; cancellation drains commit/rollback and closure."""
    state = integrity_state
    repo = _file_integrity_repository(state, tmp_path)
    database, batch_id = repo.db, state["batch"]["id"]
    owner = database.get_connection()
    _legacy_integrity_batch(repo, state)
    repo.update_slot(state["slots"][0]["id"], {"status": "queued"})
    batch_before = repo.get_batch(batch_id)
    slots_before = [repo.get_slot(slot["id"]) for slot in state["slots"]]
    outcome_before = repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 3)
    blocked, release = threading.Event(), threading.Event()
    owned: list[tuple[threading.Thread, sqlite3.Connection]] = []
    failure = RuntimeError("second slot reconciliation failed")

    def blocked_reader(
        _pack_id: int, slot_id: int, _user_id: int, _batches: Mapping[int, str],
        _settled: set[tuple[int, int, str]], _finishing: tuple[int, str] | None,
    ) -> tuple[bool, bool]:
        """Hold the real write after terminal writes and first-slot reconciliation."""
        connection = database.get_connection()
        owned.append((threading.current_thread(), connection))
        assert connection.in_transaction
        if slot_id == state["slots"][1]["id"]:
            assert repo.get_batch(batch_id)["failed_count"] == 3
            assert repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 3)["outcome_status"] == "failed"
            assert repo.get_slot(state["slots"][0]["id"])["status"] == "generating"
            assert repo.count_items_for_generation(state["pack"]["id"]) == 3
            blocked.set()
            assert release.wait(5), "test reconciliation was not released"
            if fail:
                raise failure
        return False, False

    repo.legacy_activity_reader = blocked_reader
    pending = asyncio.create_task(repo.fail_batch_integrity_async(batch_id, error="vn_asset_recipe_not_found"))
    try:
        assert await asyncio.to_thread(blocked.wait, 3)
        assert repo.get_batch(batch_id) == batch_before
        assert [repo.get_slot(slot["id"]) for slot in state["slots"]] == slots_before
        assert repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 3) == outcome_before
        assert repo.count_items_for_generation(state["pack"]["id"]) == 5
        if cancel_delivery:
            for _ in range(2):
                pending.cancel()
                await asyncio.sleep(0)
                assert not pending.done(), "cancellation abandoned an owned transaction"
        release.set()
        if fail:
            with pytest.raises(RuntimeError) as raised:
                await pending
            assert raised.value is failure
            assert repo.get_batch(batch_id) == batch_before
            assert [repo.get_slot(slot["id"]) for slot in state["slots"]] == slots_before
            assert repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 3) == outcome_before
            assert repo.count_items_for_generation(state["pack"]["id"]) == 5
        else:
            if cancel_delivery:
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                assert (await pending)["failed_count"] == 3
            assert repo.get_batch(batch_id)["failed_count"] == 3
            assert repo.get_slot(state["slots"][1]["id"])["status"] == "failed"
            assert repo.count_items_for_generation(state["pack"]["id"]) == 3
        _assert_integrity_handles_closed(owned, owner)
        assert database.get_connection() is owner and not owner.in_transaction
        assert owner.execute("SELECT 1").fetchone()[0] == 1
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        database.close_connection()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private-memory", "caller-transaction"])
async def test_async_integrity_preserves_owner_thread_fallback(
    integrity_state: dict[str, Any], tmp_path: Path, mode: str,
) -> None:
    """Connection-local writes stay inline and caller rollback retains ownership."""
    repo = integrity_state["repo"] if mode == "private-memory" else _file_integrity_repository(integrity_state, tmp_path)
    owner = repo.db.get_connection()
    main_thread = threading.get_ident()
    _legacy_integrity_batch(repo, integrity_state)
    batch_id = integrity_state["batch"]["id"]
    before = repo.get_batch(batch_id)
    threads: list[int] = []

    def observed_reader(
        _pack_id: int, _slot_id: int, _user_id: int, _batches: Mapping[int, str],
        _settled: set[tuple[int, int, str]], _finishing: tuple[int, str] | None,
    ) -> tuple[bool, bool]:
        """Observe the public callback's caller-owned transaction and thread."""
        assert repo.db.get_connection() is owner and owner.in_transaction
        assert repo.get_batch(batch_id)["failed_count"] == 3
        threads.append(threading.get_ident())
        return False, False

    repo.legacy_activity_reader = observed_reader
    try:
        if mode == "caller-transaction":
            owner.execute("BEGIN IMMEDIATE")
            owner.execute("UPDATE vn_asset_batches SET enqueue_error='caller-uncommitted' WHERE id=?", (batch_id,))
        result = await repo.fail_batch_integrity_async(batch_id, error="vn_asset_recipe_not_found")
        assert (result["status"], result["failed_count"]) == ("failed", 3)
        assert threads and set(threads) == {main_thread}
        assert repo.db.get_connection() is owner
        if mode == "caller-transaction":
            assert owner.in_transaction
            owner.rollback()
            assert repo.get_batch(batch_id) == before
            assert repo.count_items_for_generation(integrity_state["pack"]["id"]) == 5
        assert owner.execute("SELECT 1").fetchone()[0] == 1
    finally:
        if owner.in_transaction:
            owner.rollback()
        if mode != "private-memory":
            repo.db.close_connection()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("activity", ["inline", "queued-jobs", "active-jobs", "reader-error"])
async def test_async_integrity_preserves_inline_and_actual_jobs_activity(
    integrity_state: dict[str, Any], tmp_path: Path, activity: str,
) -> None:
    """Reconciliation keeps instance-local activity and the real Jobs reader contract."""
    state = integrity_state
    repo = _file_integrity_repository(state, tmp_path)
    batch_id, slot_id, pack_id = state["batch"]["id"], state["slots"][1]["id"], state["pack"]["id"]
    legacy = repo.create_batch(pack_id=pack_id, requested_by_user_id=1, total_variants=1,
                               options={"slot_ids": [slot_id], "variant_count": 1})
    jobs = JobManager(db_path=tmp_path / "integrity-jobs.db")
    job = None
    if activity != "inline":
        job = create_generate_variant_job(jobs, pack_id=pack_id, slot_id=slot_id, batch_id=legacy["id"], variant_index=0, user_id=1)
    if activity == "active-jobs":
        assert jobs.acquire_next_job(domain="vn_assets", queue=vn_asset_generation_jobs_queue(),
                                     worker_id="legacy", lease_seconds=120) is not None
    if activity == "inline":
        repo.begin_inline_legacy_display(legacy["id"], slot_id)
    context = ContextVar("integrity-reader-context", default="missing")
    token = context.set("owner-context")
    observed_context: list[str] = []
    owned: list[tuple[threading.Thread, sqlite3.Connection]] = []
    owner = repo.db.get_connection()
    reader = build_legacy_activity_reader(jobs)
    failure = RuntimeError("Jobs activity read failed")

    def observed_reader(
        pack_id: int, slot_id: int, user_id: int, batches: Mapping[int, str],
        settled: set[tuple[int, int, str]], finishing: tuple[int, str] | None,
    ) -> tuple[bool, bool]:
        """Exercise the real activity callback with the originating context."""
        connection = repo.db.get_connection()
        assert connection.in_transaction
        owned.append((threading.current_thread(), connection))
        observed_context.append(context.get())
        result = reader(pack_id, slot_id, user_id, batches, settled, finishing)
        if activity == "reader-error":
            raise failure
        return result

    repo.legacy_activity_reader = observed_reader
    before = repo.get_batch(batch_id)
    slots_before = [repo.get_slot(slot["id"]) for slot in state["slots"]]
    job_before = jobs.get_job(job["id"], owner_user_id="1") if job is not None else None
    try:
        if activity == "reader-error":
            with pytest.raises(RuntimeError) as raised:
                await repo.fail_batch_integrity_async(batch_id, error="vn_asset_recipe_not_found")
            assert raised.value is failure
            assert repo.get_batch(batch_id) == before
            assert [repo.get_slot(slot["id"]) for slot in state["slots"]] == slots_before
        else:
            await repo.fail_batch_integrity_async(batch_id, error="vn_asset_recipe_not_found")
            assert repo.get_slot(slot_id)["status"] == ("queued" if activity == "queued-jobs" else "generating")
            assert repo.get_slot(state["slots"][0]["id"])["status"] == "generating"
        assert observed_context and set(observed_context) == {"owner-context"}
        _assert_integrity_handles_closed(owned, owner)
        assert repo.db.get_connection() is owner
        if job is not None:
            assert jobs.get_job(job["id"], owner_user_id="1") == job_before
        if activity == "inline":
            repo.finish_legacy_display(legacy["id"], slot_id, inline=True, fallback_status=None)
            assert repo.get_slot(slot_id)["status"] == "failed"
            assert repo.get_slot(state["slots"][0]["id"])["status"] == "generating"
    finally:
        context.reset(token)
        repo.db.close_connection()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private-memory", "caller-transaction"])
async def test_async_outcome_read_preserves_owner_connection_fallback(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain original connection ownership when reading complete outcome metadata.

    Args:
        mode: Private memory or an uncommitted caller transaction.
        tmp_path: File-backed caller database location.
        monkeypatch: Reject any attempted transfer of owner-only work.

    Returns:
        None; asserts exact outcome metadata and original connection ownership.
    """
    database = CharactersRAGDB(":memory:" if mode == "private-memory" else str(tmp_path / "owner.db"), client_id="owner-read")
    repo = VNAssetPacksRepository.initialized(database)
    character = database.add_character_card({"name": "Owner"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character, title="Owner")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="owner")
    batch = repo.create_batch(pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
                             recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {}}])

    async def forbidden_offload(*_args: Any, **_kwargs: Any) -> None:
        """Reject transfer of private or transactional work to another thread."""
        pytest.fail("owner-only read was offloaded")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_offload)
    owner = database.get_connection()
    try:
        if mode == "caller-transaction":
            owner.execute("BEGIN")
            owner.execute("UPDATE vn_asset_generation_recipes SET claim_token='uncommitted'")
        outcome = await repo.get_variant_outcome_async(batch["id"], slot["id"], 0)
        assert outcome == {"outcome_status": "planned", "item_id": None, "claim_token": "uncommitted" if mode == "caller-transaction" else None, "claim_lease_id": None, "deleted_item_json": None}
        assert await repo.get_variant_outcome_async(batch["id"], slot["id"], 99) is None
        assert database.get_connection() is owner
        if mode == "caller-transaction":
            assert owner.in_transaction
            owner.rollback()
            assert repo.get_variant_outcome(batch["id"], slot["id"], 0)["claim_token"] is None
        assert owner.execute("SELECT 1").fetchone()[0] == 1
    finally:
        if owner.in_transaction:
            owner.rollback()
        database.close_connection()


def test_vn_asset_tables_are_created(chacha_db: CharactersRAGDB) -> None:
    ensure_vn_asset_tables(chacha_db)

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vn_asset_%'"
    )
    table_names = {row[0] for row in cursor.fetchall()}
    assert {
        "vn_asset_packs",
        "vn_asset_slots",
        "vn_asset_items",
        "vn_asset_batches",
        "vn_asset_generation_recipes",
    }.issubset(table_names)


@pytest.mark.integration
def test_existing_batch_table_gains_recipe_version(chacha_db: CharactersRAGDB) -> None:
    chacha_db.execute_query(
        "CREATE TABLE vn_asset_batches ("
        "id INTEGER PRIMARY KEY, pack_id INTEGER NOT NULL, job_batch_id TEXT)"
    )

    ensure_vn_asset_tables(chacha_db)

    columns = {
        row[1]
        for row in chacha_db.execute_query("PRAGMA table_info(vn_asset_batches)").fetchall()
    }
    assert "recipe_version" in columns


@pytest.mark.integration
def test_existing_recipe_table_gains_outcome_columns(chacha_db: CharactersRAGDB) -> None:
    """Upgrade a pre-outcome ledger without inventing deletion receipts.

    Args:
        chacha_db: Isolated native SQLite database with the old recipe schema.

    Returns:
        None; asserts additive outcome, fence and receipt columns.
    """
    chacha_db.execute_query(
        "CREATE TABLE vn_asset_generation_recipes ("
        "batch_id INTEGER, slot_id INTEGER, variant_index INTEGER, recipe_json TEXT)"
    )

    ensure_vn_asset_tables(chacha_db)

    columns = {
        row[1]
        for row in chacha_db.execute_query(
            "PRAGMA table_info(vn_asset_generation_recipes)"
        ).fetchall()
    }
    assert {"outcome_status", "item_id", "claim_token", "claim_lease_id", "deleted_item_json"}.issubset(columns)


@pytest.mark.integration
def test_existing_idempotency_table_gains_batch_link(chacha_db: CharactersRAGDB) -> None:
    chacha_db.execute_query(
        "CREATE TABLE vn_asset_idempotency_records ("
        "id INTEGER PRIMARY KEY, owner_user_id INTEGER NOT NULL, scope TEXT NOT NULL, "
        "resource_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, "
        "payload_hash TEXT NOT NULL, status TEXT NOT NULL, response_json TEXT NOT NULL)"
    )

    ensure_vn_asset_tables(chacha_db)

    columns = {
        row[1]
        for row in chacha_db.execute_query(
            "PRAGMA table_info(vn_asset_idempotency_records)"
        ).fetchall()
    }
    assert "batch_id" in columns


@pytest.mark.integration
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled", "planned"])
def test_delete_item_preserves_terminal_recipe_ledger_on_existing_schema(
    chacha_db: CharactersRAGDB, outcome: str,
) -> None:
    # Explicitly retain the original NO ACTION foreign key, including after initialization.
    chacha_db.execute_query(
        "CREATE TABLE vn_asset_generation_recipes ("
        "batch_id INTEGER, slot_id INTEGER, variant_index INTEGER, recipe_json TEXT, "
        "item_id INTEGER REFERENCES vn_asset_items(id), "
        "PRIMARY KEY (batch_id, slot_id, variant_index))"
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "historical"}}],
    )
    item = repo.reserve_variant_item(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0, item_fields={"pack_id": pack["id"]},
    )
    with chacha_db.transaction() as conn:
        conn.execute("UPDATE vn_asset_generation_recipes SET outcome_status = ?", (outcome,))
    repo.update_batch(batch["id"], {"status": "cancelled", "completed_count": 1})
    before = repo.get_variant_outcome(batch["id"], slot["id"], 0)
    before_batch = repo.get_batch(batch["id"])
    child = repo.create_item(
        pack_id=pack["id"], slot_id=slot["id"], variant_index=1,
        parent_item_id=item["id"], depth_kind="estimated",
    )

    assert repo.delete_item(item["id"]) is True

    assert repo.get_item(item["id"]) is None
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0) == {**before, "item_id": None}
    assert repo.get_batch(batch["id"]) == before_batch
    assert repo.get_item(child["id"])["parent_item_id"] is None
    assert chacha_db.execute_query("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.integration
def test_delete_item_rejects_active_variant_reservation(chacha_db: CharactersRAGDB) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "active"}}],
    )
    item = repo.reserve_variant_item(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0, item_fields={"pack_id": pack["id"]},
    )
    before = repo.get_variant_outcome(batch["id"], slot["id"], 0)
    with pytest.raises(VNAssetGenerationError, match="vn_asset_variant_in_progress"):
        repo.delete_item(item["id"])
    assert repo.get_item(item["id"]) is not None
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0) == before


@pytest.mark.integration
def test_batch_and_recipes_roll_back_without_matching_receipt(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")

    with pytest.raises(ValueError, match="vn_asset_generation_receipt_not_claimed"):
        repo.create_batch(
            pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
            recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
            idempotency_receipt={
                "scope": "vn_asset_generate", "resource_id": f"pack:{pack['id']}",
                "idempotency_key": "missing", "payload_hash": "missing",
            },
        )

    assert repo.list_batches(pack["id"]) == []


@pytest.mark.integration
def test_stale_unlinked_generation_claim_can_be_reclaimed(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    receipt = {
        "owner_user_id": 1,
        "scope": "vn_asset_generate",
        "resource_id": "pack:7",
        "idempotency_key": "stale-generation",
        "payload_hash": "same-payload",
    }
    first, first_claimed = repo.claim_idempotency_record(**receipt)
    _, immediate_claimed = repo.claim_idempotency_record(**receipt)
    with chacha_db.transaction() as conn:
        conn.execute(
            "UPDATE vn_asset_idempotency_records SET updated_at = '2000-01-01 00:00:00' WHERE id = ?",
            (first["id"],),
        )

    reclaimed, stale_claimed = repo.claim_idempotency_record(**receipt)

    assert first_claimed is True
    assert immediate_claimed is False
    assert stale_claimed is True
    assert reclaimed["id"] == first["id"]
    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.claim_idempotency_record(**{**receipt, "payload_hash": "different"})


@pytest.mark.integration
def test_duplicate_recipe_rolls_back_batch(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    recipe = {"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}

    with pytest.raises(sqlite3.IntegrityError):
        repo.create_batch(
            pack_id=pack["id"],
            requested_by_user_id=1,
            total_variants=2,
            recipes=[recipe, recipe],
        )

    assert repo.list_batches(pack["id"]) == []


@pytest.mark.integration
def test_failed_variant_cannot_publish_reserved_item(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    item = repo.reserve_variant_item(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0,
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    assert repo.list_items(pack["id"]) == []
    assert repo.count_items_for_generation(pack["id"]) == 1
    repo.fail_variant(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0, error="failed"
    )

    with pytest.raises(ValueError, match="vn_asset_variant_failed"):
        repo.complete_variant(
            batch_id=batch["id"], slot_id=slot["id"],
            variant_index=0, item_id=item["id"],
        )

    assert repo.get_item(item["id"])["review_status"] == "hidden"
    assert repo.list_items(pack["id"]) == []
    assert repo.count_items_for_generation(pack["id"]) == 0
    assert repo.get_batch(batch["id"])["failed_count"] == 1


@pytest.mark.integration
def test_cancel_batch_terminalizes_reserved_variants_and_frees_capacity(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    item = repo.reserve_variant_item(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0,
        item_fields={"pack_id": pack["id"]},
    )
    assert repo.count_items_for_generation(pack["id"]) == 1

    cancelled = repo.cancel_batch(batch["id"])
    repo.cancel_batch(batch["id"])
    repo.fail_variant(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0, error="late failure",
    )

    assert cancelled["status"] == "cancelled"
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0)["outcome_status"] == "cancelled"
    assert repo.count_items_for_generation(pack["id"]) == 0
    assert repo.get_item(item["id"])["review_status"] == "hidden"
    assert repo.get_batch(batch["id"])["cancelled_count"] == 1


@pytest.mark.integration
def test_variant_claim_requires_a_new_lease_to_replace_active_claim(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    args = {
        "batch_id": batch["id"], "slot_id": slot["id"], "variant_index": 0,
        "item_fields": {"pack_id": pack["id"]},
    }
    first = repo.claim_variant(**args, lease_id="lease-1", attempt_token="attempt-1")
    with pytest.raises(ValueError, match="vn_asset_variant_in_progress"):
        repo.claim_variant(**args, lease_id="lease-1", attempt_token="attempt-2")
    with pytest.raises(ValueError, match="vn_asset_variant_in_progress"):
        repo.claim_variant(**args, lease_id="lease-2", attempt_token="attempt-2")
    second = repo.claim_variant(
        **args, lease_id="lease-2", attempt_token="attempt-2", allow_takeover=True,
        expected_claim_token="attempt-1", validate_authority=lambda: None,
    )

    assert second["id"] == first["id"]
    assert repo.count_items_for_generation(pack["id"]) == 1
    with pytest.raises(ValueError, match="vn_asset_variant_claim_lost"):
        repo.update_item_storage(
            first["id"], generated_file_id=77, storage_ref="asset.png", mime_type="image/png",
            width=10, height=10, bytes=3, batch_id=batch["id"], slot_id=slot["id"],
            variant_index=0, attempt_token="attempt-1",
        )


@pytest.mark.integration
@pytest.mark.parametrize("already_cancelled", [False, True])
def test_cancel_batch_preserves_completed_and_failed_outcomes(
    chacha_db: CharactersRAGDB, already_cancelled: bool,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=3,
        recipes=[{"slot_id": slot["id"], "variant_index": index, "recipe": {"prompt": "frozen"}} for index in range(3)],
    )
    item = repo.reserve_variant_item(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0,
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    repo.complete_variant(batch_id=batch["id"], slot_id=slot["id"], variant_index=0, item_id=item["id"])
    repo.fail_variant(batch_id=batch["id"], slot_id=slot["id"], variant_index=1, error="failed")
    if already_cancelled:
        repo.update_batch(batch["id"], {"status": "cancelled"})

    cancelled = repo.cancel_batch(batch["id"])

    assert (cancelled["completed_count"], cancelled["failed_count"], cancelled["cancelled_count"]) == (1, 1, 1)
    assert [repo.get_variant_outcome(batch["id"], slot["id"], index)["outcome_status"] for index in range(3)] == ["completed", "failed", "cancelled"]


def test_cancelled_v1_replay_preserves_exact_batch_row_across_timestamp_change(
    chacha_db: CharactersRAGDB, sqlite_timestamp: dict[str, str],
) -> None:
    """Catch replay stamping an already reconciled cancelled V1 batch."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Cancellation Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Cancellation Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    original = repo.cancel_batch(batch["id"])
    outcome = repo.get_variant_outcome(batch["id"], slot["id"], 0)
    assert original["updated_at"] == "2000-01-01 00:00:00"
    sqlite_timestamp["now"] = "2000-01-01 00:00:01"

    assert repo.cancel_batch(batch["id"]) == original
    assert repo.cancel_batch(batch["id"]) == original
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0) == outcome


def test_cancelled_v1_replay_reconciles_unfinished_recipes_once(
    chacha_db: CharactersRAGDB, sqlite_timestamp: dict[str, str],
) -> None:
    """Catch an early replay return skipping an interrupted cancellation's work."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Cancellation Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Cancellation Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    repo.update_batch(batch["id"], {"status": "cancelled", "cancelled_count": 1})
    sqlite_timestamp["now"] = "2000-01-01 00:00:01"

    reconciled = repo.cancel_batch(batch["id"])

    assert reconciled["updated_at"] == "2000-01-01 00:00:01"
    assert reconciled["cancelled_count"] == 1
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0)["outcome_status"] == "cancelled"
    assert repo.get_slot(slot["id"])["status"] == "cancelled"
    sqlite_timestamp["now"] = "2000-01-01 00:00:02"
    assert repo.cancel_batch(batch["id"]) == reconciled


def test_cancelled_v1_replay_repairs_unreconciled_cancelled_count(
    chacha_db: CharactersRAGDB, sqlite_timestamp: dict[str, str],
) -> None:
    """Catch skipping a stale counter when all surviving recipes are cancelled."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Cancellation Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Cancellation Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    repo.cancel_batch(batch["id"])
    original = repo.update_batch(batch["id"], {"cancelled_count": 0})
    outcome = repo.get_variant_outcome(batch["id"], slot["id"], 0)
    sqlite_timestamp["now"] = "2000-01-01 00:00:01"

    reconciled = repo.cancel_batch(batch["id"])

    assert reconciled == {**original, "cancelled_count": 1, "updated_at": "2000-01-01 00:00:01"}
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0) == outcome


def test_legacy_cancel_replay_retains_unconditional_timestamp_update(
    chacha_db: CharactersRAGDB, sqlite_timestamp: dict[str, str],
) -> None:
    """Catch applying immutable V1 replay semantics to unconditional V0 cancellation."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Cancellation Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Cancellation Pack")
    batch = repo.create_batch(pack_id=pack["id"], requested_by_user_id=1, status="completed")
    original = repo.cancel_batch(batch["id"])
    assert original["updated_at"] == "2000-01-01 00:00:00"
    sqlite_timestamp["now"] = "2000-01-01 00:00:01"

    assert repo.cancel_batch(batch["id"]) == {**original, "updated_at": "2000-01-01 00:00:01"}


@pytest.mark.integration
def test_stale_observation_cannot_replace_newer_variant_claim(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    args = {
        "batch_id": batch["id"], "slot_id": slot["id"], "variant_index": 0,
        "item_fields": {"pack_id": pack["id"]}, "allow_takeover": True,
        "validate_authority": lambda: None,
    }
    repo.claim_variant(**args, lease_id="lease-1", attempt_token="attempt-1", expected_claim_token=None)
    repo.claim_variant(**args, lease_id="lease-2", attempt_token="attempt-2", expected_claim_token="attempt-1")
    with pytest.raises(ValueError, match="vn_asset_variant_claim_changed"):
        repo.claim_variant(**args, lease_id="stale", attempt_token="stale", expected_claim_token="attempt-1")
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0)["claim_token"] == "attempt-2"


@pytest.mark.integration
@pytest.mark.parametrize("transition", ["attach", "complete", "fail"])
def test_missing_token_cannot_mutate_claimed_variant(
    chacha_db: CharactersRAGDB, transition: str,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    identity = {"batch_id": batch["id"], "slot_id": slot["id"], "variant_index": 0}
    item = repo.claim_variant(
        **identity, lease_id="lease-1", attempt_token="attempt-1",
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    if transition == "fail":
        repo.fail_variant(**identity, error="stale unclaimed failure")
    else:
        with pytest.raises(VNAssetGenerationError, match="vn_asset_variant_claim_lost"):
            if transition == "complete":
                repo.complete_variant(**identity, item_id=item["id"])
            else:
                repo.update_item_storage(
                    item["id"], generated_file_id=88, storage_ref="stale.png",
                    mime_type="image/png", width=10, height=10, bytes=3,
                )
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0)["outcome_status"] == "planned"
    assert repo.get_item(item["id"])["generated_file_id"] == 17


@pytest.mark.integration
def test_released_inline_claim_cannot_complete_with_stale_token(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    identity = {"batch_id": batch["id"], "slot_id": slot["id"], "variant_index": 0}
    item = repo.claim_variant(
        **identity, lease_id="inline", attempt_token="attempt-1",
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    repo.release_variant_claim(**identity, attempt_token="attempt-1")
    with pytest.raises(VNAssetGenerationError, match="vn_asset_variant_claim_lost"):
        repo.complete_variant(**identity, item_id=item["id"], attempt_token="attempt-1")


@pytest.mark.integration
def test_cancel_legacy_batch_preserves_unconditional_cancellation(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    batch = repo.create_batch(pack_id=pack["id"], requested_by_user_id=1, status="completed")
    assert repo.cancel_batch(batch["id"])["status"] == "cancelled"


@pytest.mark.unit
def test_generation_error_preserves_public_code_and_internal_context() -> None:
    error = VNAssetGenerationError("vn_asset_variant_claim_lost", retryable=True, batch_id=17)
    assert isinstance(error, ValueError)
    assert str(error) == error.code == "vn_asset_variant_claim_lost"
    assert error.retryable is True
    assert error.context == {"batch_id": 17}
    with pytest.raises(TypeError):
        error.context["batch_id"] = 18


@pytest.mark.integration
@pytest.mark.parametrize("code", [
    "vn_asset_recipe_item_mismatch", "vn_asset_variant_failed",
    "vn_asset_batch_terminal", "vn_asset_item_storage_missing",
])
def test_completion_guards_use_typed_codes_and_variant_context(
    chacha_db: CharactersRAGDB, code: str,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}],
    )
    identity = {"batch_id": batch["id"], "slot_id": slot["id"], "variant_index": 0}
    item = repo.reserve_variant_item(**identity, item_fields={"pack_id": pack["id"]})
    item_id = item["id"]
    if code == "vn_asset_recipe_item_mismatch":
        item_id += 1
    elif code == "vn_asset_variant_failed":
        repo.fail_variant(**identity, error="failed")
    elif code == "vn_asset_batch_terminal":
        repo.update_batch(batch["id"], {"status": "cancelled"})
    with pytest.raises(VNAssetGenerationError, match=code) as caught:
        repo.complete_variant(**identity, item_id=item_id)
    assert str(caught.value) == caught.value.code == code
    assert caught.value.context == {**identity, "item_id": item_id, "operation": "complete_variant"}


@pytest.mark.integration
def test_partial_variant_failure_keeps_completed_slot_reviewable(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=2,
        recipes=[
            {"slot_id": slot["id"], "variant_index": index, "recipe": {"prompt": "frozen"}}
            for index in range(2)
        ],
    )
    item = repo.reserve_variant_item(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0,
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    repo.complete_variant(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0, item_id=item["id"]
    )

    repo.fail_variant(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=1, error="provider failed"
    )

    assert repo.get_slot(slot["id"])["status"] == "reviewing"
    assert repo.get_batch(batch["id"])["completed_count"] == 1
    assert repo.get_batch(batch["id"])["failed_count"] == 1


def test_batch_recipe_columns_migrate_and_execution_recipe_is_first_writer_wins(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Recipe Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Recipe Pack")
    recipe = {"version": 1, "slots": [{"slot_id": 7, "prompt": "original"}]}

    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, recipe=recipe, source_batch_id=13,
    )
    first = repo.set_execution_recipe_if_absent(batch["id"], {"version": 1, "slots": [{"slot_id": 7, "backend": "a"}]})
    second = repo.set_execution_recipe_if_absent(batch["id"], {"version": 1, "slots": [{"slot_id": 7, "backend": "b"}]})

    assert json.loads(batch["recipe_json"]) == recipe
    assert batch["source_batch_id"] == 13
    assert json.loads(first["execution_recipe_json"])["slots"][0]["backend"] == "a"
    assert second["execution_recipe_json"] == first["execution_recipe_json"]


def test_existing_batch_table_receives_nullable_recipe_columns(chacha_db: CharactersRAGDB) -> None:
    with chacha_db.transaction() as conn:
        conn.execute(
            "CREATE TABLE vn_asset_batches (id INTEGER PRIMARY KEY, pack_id INTEGER, "
            "job_batch_id TEXT, options_json TEXT)"
        )
        conn.execute("INSERT INTO vn_asset_batches (id, options_json) VALUES (1, '{}')")

    ensure_vn_asset_tables(chacha_db)

    row = chacha_db.execute_query(
        "SELECT recipe_json, execution_recipe_json, source_batch_id FROM vn_asset_batches WHERE id = 1"
    ).fetchone()
    assert tuple(row) == (None, None, None)


def test_existing_slot_table_receives_failure_batch_column(chacha_db: CharactersRAGDB) -> None:
    with chacha_db.transaction() as conn:
        conn.execute(
            "CREATE TABLE vn_asset_slots (id INTEGER PRIMARY KEY, pack_id INTEGER, "
            "depends_on_slot_id INTEGER)"
        )
        conn.execute("INSERT INTO vn_asset_slots (id, pack_id) VALUES (1, 2)")

    ensure_vn_asset_tables(chacha_db)

    row = chacha_db.execute_query(
        "SELECT last_failed_batch_id, latest_generation_batch_id FROM vn_asset_slots WHERE id = 1"
    ).fetchone()
    assert tuple(row) == (None, None)


def test_union_batch_preserves_independent_snapshots_receipt_and_slot_ownership(
    chacha_db: CharactersRAGDB,
) -> None:
    """Catch dropping either recipe form, receipt linkage, or queued-slot ownership."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Union Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Union Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    skipped = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="skipped")
    authored = {"version": 1, "slots": [
        {"slot_id": slot["id"], "variant_count": 1, "prompt": "authored"},
        {"slot_id": skipped["id"], "variant_count": 0, "prompt": "not requested"},
    ]}
    execution = {"version": 1, "slots": [{"slot_id": slot["id"], "backend": "pinned"}]}
    frozen = {"prompt": "variant-specific", "seed": 42, "backend": "pinned"}
    receipt = {"scope": "vn_asset_generate", "resource_id": f"pack:{pack['id']}",
               "idempotency_key": "union", "payload_hash": "original"}
    repo.claim_idempotency_record(owner_user_id=1, **receipt)

    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_slots=1,
        total_variants=1, recipe=authored, execution_recipe=execution, source_batch_id=13,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": frozen}],
        idempotency_receipt=receipt,
    )
    authored["slots"][0]["prompt"] = "edited later"
    frozen["prompt"] = "edited later"
    stored = repo.set_execution_recipe_if_absent(batch["id"], {"slots": []})
    response = {"batch_id": batch["id"], "status": "queued"}
    original = repo.complete_idempotency_record(owner_user_id=1, **receipt, response=response)
    replayed = repo.complete_idempotency_record(
        owner_user_id=1, **receipt, response={"batch_id": batch["id"], "status": "completed"},
    )

    assert json.loads(stored["recipe_json"])["slots"][0]["prompt"] == "authored"
    assert json.loads(stored["execution_recipe_json"]) == execution
    assert stored["source_batch_id"] == 13 and stored["recipe_version"] == 1
    assert repo.get_batch_recipe(batch["id"], slot["id"], 0) == {
        "prompt": "variant-specific", "seed": 42, "backend": "pinned",
    }
    assert repo.get_slot(slot["id"])["latest_generation_batch_id"] == batch["id"]
    assert repo.get_slot(skipped["id"])["latest_generation_batch_id"] is None
    assert replayed == original
    assert original["batch_id"] == batch["id"]
    assert json.loads(original["response_json"]) == response


def test_union_batch_recipe_failure_rolls_back_receipt_and_preserves_slot_owner(
    chacha_db: CharactersRAGDB,
) -> None:
    """Catch committing union metadata or a receipt when ledger insertion fails."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Union Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Union Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    authored = {"slots": [{"slot_id": slot["id"], "variant_count": 1}]}
    older = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", recipe=authored,
    )
    repo.mark_slot_generation_failed(slot["id"], older["id"], "original failure")
    before = repo.get_slot(slot["id"])
    receipt = {"scope": "vn_asset_generate", "resource_id": f"pack:{pack['id']}",
               "idempotency_key": "rollback", "payload_hash": "same-payload"}
    claimed, _ = repo.claim_idempotency_record(owner_user_id=1, **receipt)
    entry = {"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen"}}

    with pytest.raises(sqlite3.IntegrityError):
        repo.create_batch(
            pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=2,
            recipes=[entry, entry], idempotency_receipt=receipt, recipe=authored,
            execution_recipe={"slots": []}, source_batch_id=older["id"],
        )

    assert repo.list_batches(pack["id"]) == [older]
    assert repo.get_slot(slot["id"]) == before
    assert repo.get_idempotency_record(
        owner_user_id=1, **{key: value for key, value in receipt.items() if key != "payload_hash"},
    ) == claimed


@pytest.mark.parametrize("history", ["parent", "dev"])
def test_union_migrations_preserve_existing_history_values(
    chacha_db: CharactersRAGDB, history: str,
) -> None:
    """Catch additive migration rewriting snapshots, counters, fences or receipts."""
    with chacha_db.transaction() as conn:
        if history == "parent":
            conn.execute(
                "CREATE TABLE vn_asset_batches (id INTEGER PRIMARY KEY, pack_id INTEGER, "
                "job_batch_id TEXT, recipe_version INTEGER, completed_count INTEGER, "
                "failed_count INTEGER, cancelled_count INTEGER)"
            )
            conn.execute("INSERT INTO vn_asset_batches VALUES (1, 2, 'job', 1, 3, 4, 5)")
            conn.execute(
                "CREATE TABLE vn_asset_generation_recipes (batch_id INTEGER, slot_id INTEGER, "
                "variant_index INTEGER, recipe_json TEXT, outcome_status TEXT, item_id INTEGER, "
                "deleted_item_json TEXT, claim_lease_id TEXT, claim_token TEXT)"
            )
            conn.execute(
                "INSERT INTO vn_asset_generation_recipes VALUES "
                "(1, 7, 0, '{\"prompt\":\"frozen\"}', 'completed', NULL, "
                "'{\"id\":9}', 'original-lease', 'original-token')"
            )
            conn.execute(
                "CREATE TABLE vn_asset_slots (id INTEGER PRIMARY KEY, pack_id INTEGER, "
                "depends_on_slot_id INTEGER, status TEXT, last_error TEXT)"
            )
            conn.execute("INSERT INTO vn_asset_slots VALUES (7, 2, NULL, 'failed', 'original error')")
        else:
            conn.execute(
                "CREATE TABLE vn_asset_batches (id INTEGER PRIMARY KEY, pack_id INTEGER, "
                "job_batch_id TEXT, recipe_json TEXT, execution_recipe_json TEXT, "
                "source_batch_id INTEGER, completed_count INTEGER, failed_count INTEGER, "
                "cancelled_count INTEGER)"
            )
            conn.execute(
                "INSERT INTO vn_asset_batches VALUES (1, 2, 'job', '{\"slots\":[]}', "
                "'{\"backend\":\"pinned\"}', 13, 3, 4, 5)"
            )
            conn.execute(
                "CREATE TABLE vn_asset_slots (id INTEGER PRIMARY KEY, pack_id INTEGER, "
                "depends_on_slot_id INTEGER, status TEXT, last_error TEXT, "
                "last_failed_batch_id INTEGER, latest_generation_batch_id INTEGER)"
            )
            conn.execute(
                "INSERT INTO vn_asset_slots VALUES (7, 2, NULL, 'failed', 'original error', 1, 1)"
            )
        conn.execute(
            "CREATE TABLE vn_asset_idempotency_records (id INTEGER PRIMARY KEY, "
            "owner_user_id INTEGER, scope TEXT, resource_id TEXT, idempotency_key TEXT, "
            "payload_hash TEXT, status TEXT, response_json TEXT)"
        )
        conn.execute(
            "INSERT INTO vn_asset_idempotency_records VALUES "
            "(1, 1, 'vn_asset_generate', 'pack:2', 'original', 'hash', 'completed', "
            "'{\"batch_id\":1,\"status\":\"queued\"}')"
        )
        if history == "parent":
            conn.execute("ALTER TABLE vn_asset_idempotency_records ADD COLUMN batch_id INTEGER")
            conn.execute("UPDATE vn_asset_idempotency_records SET batch_id = 1")
        batches_before = dict(conn.execute("SELECT * FROM vn_asset_batches").fetchone())
        slots_before = dict(conn.execute("SELECT * FROM vn_asset_slots").fetchone())
        receipt_before = dict(conn.execute("SELECT * FROM vn_asset_idempotency_records").fetchone())
        outcome_before = (
            dict(conn.execute("SELECT * FROM vn_asset_generation_recipes").fetchone())
            if history == "parent" else None
        )

    ensure_vn_asset_tables(chacha_db)
    ensure_vn_asset_tables(chacha_db)

    batch = dict(chacha_db.execute_query("SELECT * FROM vn_asset_batches").fetchone())
    slot = dict(chacha_db.execute_query("SELECT * FROM vn_asset_slots").fetchone())
    receipt = dict(chacha_db.execute_query("SELECT * FROM vn_asset_idempotency_records").fetchone())
    assert {key: batch[key] for key in batches_before} == batches_before
    assert {key: slot[key] for key in slots_before} == slots_before
    assert {key: receipt[key] for key in receipt_before} == receipt_before
    if history == "parent":
        assert (batch["recipe_json"], batch["execution_recipe_json"], batch["source_batch_id"]) == (None, None, None)
        assert (slot["last_failed_batch_id"], slot["latest_generation_batch_id"]) == (None, None)
        assert dict(chacha_db.execute_query("SELECT * FROM vn_asset_generation_recipes").fetchone()) == outcome_before
    else:
        assert batch["recipe_version"] == 0
        assert receipt["batch_id"] is None
        assert chacha_db.execute_query("SELECT * FROM vn_asset_generation_recipes").fetchall() == []


@pytest.mark.parametrize("transition", ["start", "complete", "fail"])
def test_union_stale_variant_transition_preserves_newer_failed_slot_owner(
    chacha_db: CharactersRAGDB, transition: str,
) -> None:
    """Catch an older V1 outcome replacing the latest authored batch's failure."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Union Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Union Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    authored = {"slots": [{"slot_id": slot["id"], "variant_count": 1}]}
    older = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe=authored,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "older"}}],
    )
    identity = {"batch_id": older["id"], "slot_id": slot["id"], "variant_index": 0}
    attempt_token = f"older-inline-{older['id']}"
    item = repo.claim_variant(
        **identity, lease_id="inline", attempt_token=attempt_token,
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    newer = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe=authored,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "newer"}}],
    )
    repo.fail_batch_enqueue(newer["id"], "latest enqueue rejected")
    before = repo.get_slot(slot["id"])

    if transition == "start":
        repo.start_variant_generation(**identity, attempt_token=attempt_token)
    elif transition == "complete":
        repo.complete_variant(**identity, item_id=item["id"], attempt_token=attempt_token)
    else:
        repo.fail_variant(**identity, error="older provider failure", attempt_token=attempt_token)

    stored = repo.get_slot(slot["id"])
    assert {key: stored[key] for key in ("status", "last_error", "last_failed_batch_id", "latest_generation_batch_id")} == {
        key: before[key] for key in ("status", "last_error", "last_failed_batch_id", "latest_generation_batch_id")
    }
    assert repo.get_variant_outcome(older["id"], slot["id"], 0)["outcome_status"] == (
        {"start": "planned", "complete": "completed", "fail": "failed"}[transition]
    )


def test_union_partial_failure_retains_provenance_and_completed_candidate(
    chacha_db: CharactersRAGDB,
) -> None:
    """Catch clearing a sibling failure or hiding a durable completed variant."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Union Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Union Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=2,
        recipe={"slots": [{"slot_id": slot["id"], "variant_count": 2}]},
        recipes=[{"slot_id": slot["id"], "variant_index": index, "recipe": {}} for index in range(2)],
    )
    success_token, failure_token = "successful-inline", "failed-inline"
    item = repo.claim_variant(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0,
        lease_id="inline", attempt_token=success_token,
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    repo.claim_variant(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=1,
        lease_id="inline", attempt_token=failure_token, item_fields={"pack_id": pack["id"]},
    )
    repo.fail_variant(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=1,
        error="sibling failed", attempt_token=failure_token,
    )
    assert repo.get_batch(batch["id"])["status"] == "processing"
    repo.start_variant_generation(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0, attempt_token=success_token,
    )
    assert repo.get_slot(slot["id"])["last_error"] == "sibling failed"
    repo.complete_variant(
        batch_id=batch["id"], slot_id=slot["id"], variant_index=0,
        item_id=item["id"], attempt_token=success_token,
    )

    stored = repo.get_slot(slot["id"])
    assert (stored["status"], stored["last_error"], stored["last_failed_batch_id"]) == (
        "reviewing", "sibling failed", batch["id"],
    )
    assert [candidate["id"] for candidate in repo.list_items(pack["id"])] == [item["id"]]
    assert (repo.get_batch(batch["id"])["completed_count"], repo.get_batch(batch["id"])["failed_count"]) == (1, 1)


def _legacy_provenance_state(chacha_db: CharactersRAGDB, *, authored: bool) -> dict[str, Any]:
    """Create real published V1 history and a later legacy batch on the same slot."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Legacy Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Legacy Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary", variant_count=2)
    history = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "published history"}}],
    )
    item = repo.reserve_variant_item(
        batch_id=history["id"], slot_id=slot["id"], variant_index=0,
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    repo.complete_variant(batch_id=history["id"], slot_id=slot["id"], variant_index=0, item_id=item["id"])
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=2,
        options={"slot_ids": [slot["id"]], "variant_count": 2},
        recipe={"slots": [{"slot_id": slot["id"], "variant_count": 2}]} if authored else None,
    )
    assert batch["recipe_version"] == 0
    assert repo.list_batch_recipes(batch["id"]) == []
    return {"repo": repo, "pack": pack, "slot": slot, "batch": batch, "history": history}


@pytest.mark.parametrize("authored", [True, False], ids=["authored-owner", "null-legacy-owner"])
@pytest.mark.parametrize("failure", ["fanout", "variant"])
def test_legacy_provenance_remaining_success_preserves_failure(
    chacha_db: CharactersRAGDB, authored: bool, failure: str,
) -> None:
    """Catch NULL fanout omission and mixed display erasing a real V0 failure."""
    state = _legacy_provenance_state(chacha_db, authored=authored)
    repo, batch_id, slot_id = state["repo"], state["batch"]["id"], state["slot"]["id"]
    repo.begin_inline_legacy_display(batch_id, slot_id)
    repo.mark_slot_generation_started(slot_id, batch_id)
    error = "legacy exhausted failure"
    if failure == "fanout":
        repo.fail_batch_fanout_if_active(
            batch_id, error=error, planned_count=2, enqueued_count=1, failed_slot_ids=[slot_id],
        )
    else:
        repo.record_batch_variant_failure(batch_id, slot_id=slot_id, error=error)
    failure_before = repo.get_slot(slot_id)
    assert (failure_before["status"], failure_before["last_error"], failure_before["last_failed_batch_id"]) == (
        "failed", error, batch_id,
    )

    repo.mark_slot_generation_started(slot_id, batch_id)
    item = repo.create_item(
        pack_id=state["pack"]["id"], slot_id=slot_id, variant_index=0, generated_file_id=18,
        source_context_snapshot={"batch_id": batch_id, "variant_index": 0},
    )
    assert repo.get_slot(slot_id)["status"] == "failed"
    repo.mark_slot_generation_succeeded(slot_id, batch_id)
    assert repo.get_slot(slot_id)["status"] == "failed"
    repo.record_batch_variant_success(batch_id)
    repo.finish_legacy_display(batch_id, slot_id, inline=True, fallback_status="reviewing")

    stored = repo.get_slot(slot_id)
    assert (stored["status"], stored["last_error"], stored["last_failed_batch_id"],
            stored["latest_generation_batch_id"]) == (
        "failed", error, batch_id, batch_id if authored else None,
    )
    batch = repo.get_batch(batch_id)
    assert (batch["status"], batch["completed_count"], batch["failed_count"]) == (
        "failed", 1, 0 if failure == "fanout" else 1,
    )
    assert item["id"] in [candidate["id"] for candidate in repo.list_items(state["pack"]["id"])]
    assert repo.get_variant_outcome(state["history"]["id"], slot_id, 0)["outcome_status"] == "completed"


def test_legacy_provenance_older_success_preserves_newer_failed_owner(chacha_db: CharactersRAGDB) -> None:
    """Catch NULL legacy completion publishing over a newer real authored failure."""
    state = _legacy_provenance_state(chacha_db, authored=False)
    repo, batch_id, slot_id = state["repo"], state["batch"]["id"], state["slot"]["id"]
    newer = repo.create_batch(
        pack_id=state["pack"]["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe={"slots": [{"slot_id": slot_id, "variant_count": 1}]},
    )
    repo.record_batch_variant_failure(newer["id"], slot_id=slot_id, error="newer actual failure")
    slot_before, newer_before = repo.get_slot(slot_id), repo.get_batch(newer["id"])

    repo.fail_batch_fanout_if_active(
        batch_id, error="older fanout failure", planned_count=2, enqueued_count=1, failed_slot_ids=[slot_id],
    )
    repo.mark_slot_generation_started(slot_id, batch_id)
    repo.create_item(
        pack_id=state["pack"]["id"], slot_id=slot_id, generated_file_id=18,
        source_context_snapshot={"batch_id": batch_id, "variant_index": 0},
    )
    repo.mark_slot_generation_succeeded(slot_id, batch_id)
    repo.record_batch_variant_success(batch_id)
    repo.finish_legacy_display(batch_id, slot_id, inline=False, fallback_status="reviewing")

    assert repo.get_slot(slot_id) == slot_before
    assert repo.get_batch(newer["id"]) == newer_before


def test_legacy_provenance_null_guard_retains_v1_review_precedence(chacha_db: CharactersRAGDB) -> None:
    """Catch applying the NULL V0 display guard to a real V1 failure/candidate."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "V1 Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="V1 Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    failed = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "real V1 failure"}}],
    )
    repo.fail_variant(batch_id=failed["id"], slot_id=slot["id"], variant_index=0, error="V1 failure")
    assert repo.get_slot(slot["id"])["status"] == "failed"
    legacy = repo.create_batch(pack_id=pack["id"], requested_by_user_id=1, total_variants=1)
    item = repo.create_item(
        pack_id=pack["id"], slot_id=slot["id"], generated_file_id=17,
        source_context_snapshot={"batch_id": legacy["id"], "variant_index": 0},
    )

    repo.finish_legacy_display(legacy["id"], slot["id"], inline=False, fallback_status="reviewing")

    stored = repo.get_slot(slot["id"])
    assert (stored["status"], stored["last_failed_batch_id"], stored["last_error"],
            stored["latest_generation_batch_id"]) == ("reviewing", failed["id"], "V1 failure", None)
    assert repo.list_items(pack["id"])[0]["id"] == item["id"]
    assert repo.get_variant_outcome(failed["id"], slot["id"], 0)["outcome_status"] == "failed"


@pytest.fixture
def display_outage_state(chacha_db: CharactersRAGDB) -> dict[str, Any]:
    """Create a recipe-free legacy delivery rejected by a real authored owner."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Display Outage Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Display Outage Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    legacy = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        options={"slot_ids": [slot["id"]], "variant_count": 1},
    )
    owner = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe={"slots": [{"slot_id": slot["id"], "variant_count": 1}]},
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen owner"}}],
    )
    repo.cancel_batch(owner["id"])
    repo.update_slot(slot["id"], {"last_error": "retain authored diagnostics"})
    reads = {"count": 0}

    def unavailable_reader(
        _pack_id: int, _slot_id: int, _user_id: int, _batches: Mapping[int, str],
        _settled: set[tuple[int, int, str]], _finishing_delivery: tuple[int, str] | None,
    ) -> tuple[bool, bool]:
        """Fail only the observational reader, without changing native Jobs state."""
        reads["count"] += 1
        raise OSError("legacy display unavailable")

    repo.legacy_activity_reader = unavailable_reader
    return {"repo": repo, "slot_id": slot["id"], "legacy_id": legacy["id"], "owner_id": owner["id"], "reads": reads}


@pytest.mark.parametrize("verified_inline", [False, True], ids=["unverified-jobs", "verified-inline"])
def test_db_display_outage_start_retains_only_verified_activity(
    display_outage_state: dict[str, Any], verified_inline: bool,
) -> None:
    """An advisory Jobs outage cannot block a legacy start or invent liveness."""
    state = display_outage_state
    repo, slot_id, legacy_id = state["repo"], state["slot_id"], state["legacy_id"]
    before = repo.get_slot(slot_id)
    batches_before = [repo.get_batch(batch_id) for batch_id in (legacy_id, state["owner_id"])]
    outcome_before = repo.get_variant_outcome(state["owner_id"], slot_id, 0)
    if verified_inline:
        repo.begin_inline_legacy_display(legacy_id, slot_id)
    try:
        assert repo.mark_slot_generation_started(slot_id, legacy_id) is None
        stored = repo.get_slot(slot_id)
        if verified_inline:
            assert stored["status"] == "generating"
            assert {key: stored[key] for key in ("last_error", "last_failed_batch_id", "latest_generation_batch_id")} == {
                key: before[key] for key in ("last_error", "last_failed_batch_id", "latest_generation_batch_id")
            }
        else:
            assert stored == before
        assert state["reads"]["count"] == 1
        assert [repo.get_batch(batch_id) for batch_id in (legacy_id, state["owner_id"])] == batches_before
        assert repo.get_variant_outcome(state["owner_id"], slot_id, 0) == outcome_before
    finally:
        repo.legacy_activity_reader = None
        if verified_inline:
            repo.finish_legacy_display(legacy_id, slot_id, inline=True, fallback_status=None)


def test_db_display_outage_start_does_not_mask_native_display_write_failure(
    display_outage_state: dict[str, Any], chacha_db: CharactersRAGDB,
) -> None:
    """A native SQLite mutation error still propagates after an advisory outage."""
    state = display_outage_state
    repo, slot_id, legacy_id = state["repo"], state["slot_id"], state["legacy_id"]
    before = repo.get_slot(slot_id)
    repo.begin_inline_legacy_display(legacy_id, slot_id)
    denied: list[str | None] = []

    def deny_display_write(
        action: int, table: str | None, column: str | None, _database: str | None, _source: str | None,
    ) -> int:
        """Reject the real derived UPDATE only after the reader has failed."""
        if action == sqlite3.SQLITE_UPDATE and table == "vn_asset_slots" and state["reads"]["count"]:
            denied.append(column)
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    connection = chacha_db.get_connection()
    connection.set_authorizer(deny_display_write)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            repo.mark_slot_generation_started(slot_id, legacy_id)
        assert denied == ["status"]
        assert repo.get_slot(slot_id) == before
    finally:
        connection.set_authorizer(None)
        repo.legacy_activity_reader = None
        repo.finish_legacy_display(legacy_id, slot_id, inline=True, fallback_status=None)


@pytest.mark.parametrize("checkpoint", ["queued", "generating", "reviewing"])
@pytest.mark.parametrize("newer_failed_owner", [False, True], ids=["null-owner", "newer-failed-owner"])
def test_db_d1_flat_v1_progress_after_null_owned_v0_failure(
    chacha_db: CharactersRAGDB, checkpoint: str, newer_failed_owner: bool,
) -> None:
    """Reconcile actual later flat V1 work without fabricating authored ownership."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "D1 Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="D1 Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary", variant_count=2)
    legacy = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=2,
        options={"slot_ids": [slot["id"]], "variant_count": 2},
    )
    repo.fail_batch_fanout_if_active(
        legacy["id"], error="old exhausted fanout", planned_count=2, enqueued_count=1,
        failed_slot_ids=[slot["id"]],
    )
    legacy_before = repo.get_batch(legacy["id"])
    failed_slot = repo.get_slot(slot["id"])
    assert (failed_slot["status"], failed_slot["last_failed_batch_id"],
            failed_slot["latest_generation_batch_id"]) == ("failed", legacy["id"], None)
    assert legacy_before["recipe_json"] is None and repo.list_batch_recipes(legacy["id"]) == []
    entries = [{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen flat V1"}}]
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe=None, recipes=entries,
    )
    identity = {"batch_id": batch["id"], "slot_id": slot["id"], "variant_index": 0}
    attempt_token = f"d1-inline-{batch['id']}"
    item = repo.claim_variant(
        **identity, lease_id="inline", attempt_token=attempt_token,
        item_fields={"pack_id": pack["id"], "generated_file_id": 17},
    )
    newer_before = slot_before = None
    if newer_failed_owner:
        newer = repo.create_batch(
            pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
            recipe={"slots": [{"slot_id": slot["id"], "variant_count": 1}]}, recipes=entries,
        )
        repo.fail_batch_enqueue(newer["id"], "newer enqueue rejected")
        newer_before, slot_before = repo.get_batch(newer["id"]), repo.get_slot(slot["id"])
        assert (slot_before["status"], slot_before["last_failed_batch_id"],
                slot_before["latest_generation_batch_id"]) == ("failed", newer["id"], newer["id"])

    if checkpoint == "queued":
        repo.release_variant_claim(**identity, attempt_token=attempt_token)
    else:
        repo.start_variant_generation(**identity, attempt_token=attempt_token)
    if checkpoint == "reviewing":
        repo.complete_variant(**identity, item_id=item["id"], attempt_token=attempt_token)

    stored = repo.get_slot(slot["id"])
    if newer_failed_owner:
        assert stored == slot_before
        assert repo.get_batch(newer_before["id"]) == newer_before
    else:
        assert (stored["status"], stored["last_error"], stored["last_failed_batch_id"],
                stored["latest_generation_batch_id"]) == (
            checkpoint, "old exhausted fanout" if checkpoint == "queued" else None,
            None if checkpoint == "reviewing" else legacy["id"], None,
        )
    outcome = repo.get_variant_outcome(**identity)
    completed = checkpoint == "reviewing"
    assert (outcome["outcome_status"], outcome["claim_token"], outcome["claim_lease_id"], outcome["item_id"]) == (
        "completed" if completed else "planned", None if checkpoint == "queued" else attempt_token,
        None if checkpoint == "queued" else "inline", item["id"],
    )
    stored_batch = repo.get_batch(batch["id"])
    assert (stored_batch["recipe_version"], stored_batch["recipe_json"],
            stored_batch["execution_recipe_json"], stored_batch["source_batch_id"]) == (1, None, None, None)
    assert (stored_batch["status"], stored_batch["completed_count"], stored_batch["failed_count"],
            stored_batch["cancelled_count"]) == ("completed" if completed else "queued", int(completed), 0, 0)
    assert repo.get_batch_recipe(**identity) == entries[0]["recipe"]
    assert [candidate["id"] for candidate in repo.list_items(pack["id"])] == ([item["id"]] if completed else [])
    assert repo.get_batch(legacy["id"]) == legacy_before


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("jobs_delivery", [False, True], ids=["inline", "real-jobs"])
def test_active_legacy_cancel_preserves_verified_progress_and_history(
    chacha_db: CharactersRAGDB, tmp_path: Path, approved: bool, jobs_delivery: bool,
) -> None:
    """Catch latest-owner provenance fencing hiding real recipe-free activity."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Active Legacy Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Active Legacy Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    approved_item = None
    if approved:
        path = tmp_path / "approved-history.png"
        path.write_bytes(b"approved history")
        approved_item = repo.create_item(
            pack_id=pack["id"], slot_id=slot["id"], review_status="approved",
            generated_file_id=17, storage_ref=str(path), bytes=16,
        )
    v1 = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe={"slots": [{"slot_id": slot["id"], "variant_count": 1}]},
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {"prompt": "frozen V1"}}],
    )
    legacy = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        options={"slot_ids": [slot["id"]], "variant_count": 1},
    )
    repo.update_slot(slot["id"], {"last_error": "retain authored diagnostics"})
    assert legacy["recipe_json"] is None and repo.list_batch_recipes(legacy["id"]) == []
    jobs = JobManager(db_path=tmp_path / "active-legacy-jobs.db")
    repo.legacy_activity_reader = build_legacy_activity_reader(jobs)
    job = None
    if jobs_delivery:
        create_generate_variant_job(
            jobs, pack_id=pack["id"], slot_id=slot["id"], batch_id=legacy["id"], variant_index=0, user_id=1,
        )
        job = jobs.acquire_next_job(
            domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="active-legacy", lease_seconds=120,
        )
        assert job is not None
    else:
        repo.begin_inline_legacy_display(legacy["id"], slot["id"])
    try:
        repo.mark_slot_generation_started(slot["id"], legacy["id"])
        stored = repo.get_slot(slot["id"])
        assert (stored["status"], stored["last_error"], stored["last_failed_batch_id"],
                stored["latest_generation_batch_id"]) == (
            "generating", "retain authored diagnostics", None, v1["id"],
        )
        cancelled = repo.cancel_batch(v1["id"])
        assert (cancelled["status"], cancelled["cancelled_count"]) == ("cancelled", 1)
        assert repo.get_slot(slot["id"])["status"] == "generating"
        assert repo.get_batch(legacy["id"]) == legacy
        if approved_item is not None:
            assert repo.get_item(approved_item["id"]) == approved_item
            assert Path(approved_item["storage_ref"]).read_bytes() == b"approved history"
        repo.create_item(
            pack_id=pack["id"], slot_id=slot["id"], generated_file_id=18,
            source_context_snapshot={"batch_id": legacy["id"], "variant_index": 0},
        )
        repo.mark_slot_generation_succeeded(slot["id"], legacy["id"])
        repo.record_batch_variant_success(legacy["id"])
    finally:
        if job is not None:
            assert jobs.complete_job(job["id"], result={}, worker_id="active-legacy", lease_id=job["lease_id"])
        repo.finish_legacy_display(legacy["id"], slot["id"], inline=not jobs_delivery, fallback_status="reviewing")
    stored = repo.get_slot(slot["id"])
    assert (stored["status"], stored["last_error"], stored["latest_generation_batch_id"]) == (
        "reviewing", "retain authored diagnostics", v1["id"],
    )
    assert repo.get_batch(legacy["id"])["completed_count"] == 1
    if approved_item is not None:
        assert repo.get_item(approved_item["id"]) == approved_item


@pytest.mark.parametrize("failed_owner", [False, True], ids=["unverified-marker", "newer-failed-owner"])
def test_active_legacy_cancel_start_preserves_unverified_or_failed_owner(
    chacha_db: CharactersRAGDB, failed_owner: bool,
) -> None:
    """Catch stale markers clearing provenance or inventing liveness from counts."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Guarded Legacy Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Guarded Legacy Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    v1 = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe={"slots": [{"slot_id": slot["id"], "variant_count": 1}]},
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {}}],
    )
    legacy = repo.create_batch(pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1)
    if failed_owner:
        repo.fail_variant(batch_id=v1["id"], slot_id=slot["id"], variant_index=0, error="newer actual failure")
        repo.begin_inline_legacy_display(legacy["id"], slot["id"])
    before = repo.get_slot(slot["id"])
    try:
        repo.mark_slot_generation_started(slot["id"], legacy["id"])
        assert repo.get_slot(slot["id"]) == before
    finally:
        if failed_owner:
            repo.finish_legacy_display(legacy["id"], slot["id"], inline=True, fallback_status=None)


@pytest.mark.parametrize("ledger_version", [0, 1])
def test_union_partial_fanout_error_preserves_v1_outcomes_and_v0_failure_ownership(
    chacha_db: CharactersRAGDB, ledger_version: int,
) -> None:
    """Catch using legacy terminal fanout failure for an unfinished V1 ledger."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Union Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Union Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=2,
        recipe={"slots": [{"slot_id": slot["id"], "variant_count": 2}]},
        recipes=[{"slot_id": slot["id"], "variant_index": index, "recipe": {}} for index in range(2)]
        if ledger_version else None,
    )
    if ledger_version:
        repo.fail_variant(batch_id=batch["id"], slot_id=slot["id"], variant_index=0, error="first variant failed")
    before_batch = repo.get_batch(batch["id"])
    before_slot = repo.get_slot(slot["id"])
    before_outcome = repo.get_variant_outcome(batch["id"], slot["id"], 1)

    failed = repo.fail_batch_fanout_if_active(
        batch["id"], error="partial enqueue", planned_count=2, enqueued_count=1,
        failed_slot_ids=[slot["id"]],
    )

    assert failed["status"] == ("processing" if ledger_version else "failed")
    assert failed["enqueue_error"] == "partial enqueue"
    assert (failed["planned_count"], failed["enqueued_count"]) == (2, 1)
    assert (failed["completed_count"], failed["failed_count"]) == (
        before_batch["completed_count"], before_batch["failed_count"],
    )
    assert repo.get_variant_outcome(batch["id"], slot["id"], 1) == before_outcome
    if ledger_version:
        assert repo.get_slot(slot["id"]) == before_slot
        resumed = repo.complete_batch_fanout(batch["id"], planned_count=2, enqueued_count=2, total_slots=1)
        assert (resumed["status"], resumed["enqueue_error"], resumed["failed_count"]) == ("processing", None, 1)
    else:
        assert (repo.get_slot(slot["id"])["last_failed_batch_id"], repo.get_slot(slot["id"])["last_error"]) == (
            batch["id"], "partial enqueue",
        )


@pytest.mark.parametrize("ledger_version", [0, 1])
def test_union_fanout_completion_recovers_legacy_error_without_reopening_terminal_v1(
    chacha_db: CharactersRAGDB, ledger_version: int,
) -> None:
    """Catch reopening a terminal V1 batch under V0 fanout-recovery rules."""
    repo = VNAssetPacksRepository.initialized(chacha_db)
    character_id = chacha_db.add_character_card({"name": "Union Primary"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Union Pack")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="primary")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, status="queued", total_variants=1,
        recipe={"slots": [{"slot_id": slot["id"], "variant_count": 1}]},
        recipes=[{"slot_id": slot["id"], "variant_index": 0, "recipe": {}}] if ledger_version else None,
    )
    repo.update_batch(batch["id"], {"status": "failed", "enqueue_error": "original failure"})
    repo.mark_slot_generation_failed(slot["id"], batch["id"], "original failure")
    before_slot = repo.get_slot(slot["id"])
    before_outcome = repo.get_variant_outcome(batch["id"], slot["id"], 0)

    resumed = repo.complete_batch_fanout(batch["id"], planned_count=1, enqueued_count=1, total_slots=1)

    assert resumed["status"] == ("failed" if ledger_version else "enqueued")
    assert repo.get_variant_outcome(batch["id"], slot["id"], 0) == before_outcome
    if ledger_version:
        assert repo.get_slot(slot["id"]) == before_slot
    else:
        assert (repo.get_slot(slot["id"])["status"], repo.get_slot(slot["id"])["last_failed_batch_id"]) == (
            "generating", None,
        )


def test_ensure_vn_asset_tables_rejects_non_sqlite_before_transaction() -> None:
    class NonSqliteDB:
        backend_type = BackendType.POSTGRESQL

        def transaction(self):
            raise AssertionError("transaction should not be opened for unsupported backends")

    with pytest.raises(NotImplementedError, match="SQLite ChaChaNotes"):
        ensure_vn_asset_tables(NonSqliteDB())  # type: ignore[arg-type]


def test_repository_rejects_non_sqlite_backend() -> None:
    class NonSqliteDB:
        backend_type = BackendType.POSTGRESQL

    with pytest.raises(NotImplementedError, match="SQLite ChaChaNotes"):
        VNAssetPacksRepository(NonSqliteDB())  # type: ignore[arg-type]


def test_repository_constructor_does_not_create_schema(chacha_db: CharactersRAGDB) -> None:
    VNAssetPacksRepository(chacha_db)

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vn_asset_%'"
    )
    assert cursor.fetchall() == []


def test_idempotency_claim_replays_completed_response_and_rejects_payload_conflict(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)

    first_record, claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="export-1",
        payload_hash="hash-a",
    )
    assert claimed is True
    assert first_record["status"] == "in_progress"

    second_record, second_claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="export-1",
        payload_hash="hash-a",
    )
    assert second_claimed is False
    assert second_record["status"] == "in_progress"

    repo.complete_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="export-1",
        payload_hash="hash-a",
        response={"job_id": "job-1"},
    )
    completed_record, replay_claimed = repo.claim_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="export-1",
        payload_hash="hash-a",
    )
    assert replay_claimed is False
    assert completed_record["status"] == "completed"
    assert json.loads(completed_record["response_json"]) == {"job_id": "job-1"}

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.claim_idempotency_record(
            owner_user_id=42,
            scope="vn_asset_export",
            resource_id="pack:1",
            idempotency_key="export-1",
            payload_hash="hash-b",
        )


def test_create_idempotency_record_is_conflict_tolerant_for_same_payload(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNAssetPacksRepository.initialized(chacha_db)

    first = repo.create_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="legacy-export-1",
        payload_hash="hash-a",
        response={"job_id": "job-1"},
    )
    second = repo.create_idempotency_record(
        owner_user_id=42,
        scope="vn_asset_export",
        resource_id="pack:1",
        idempotency_key="legacy-export-1",
        payload_hash="hash-a",
        response={"job_id": "job-1"},
    )

    assert second["id"] == first["id"]
    assert second["status"] == "completed"
    assert json.loads(second["response_json"]) == {"job_id": "job-1"}
    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_idempotency_record(
            owner_user_id=42,
            scope="vn_asset_export",
            resource_id="pack:1",
            idempotency_key="legacy-export-1",
            payload_hash="hash-b",
            response={"job_id": "job-2"},
        )


def test_ensure_vn_asset_tables_preserves_outer_transaction_rollback(chacha_db: CharactersRAGDB) -> None:
    character_name = "Rolled Back Before VN Schema"

    with pytest.raises(RuntimeError, match="force rollback"):
        with chacha_db.transaction() as conn:
            conn.execute(
                "INSERT INTO character_cards (name, client_id, version) VALUES (?, ?, 1)",
                (character_name, chacha_db.client_id),
            )
            ensure_vn_asset_tables(chacha_db)
            raise RuntimeError("force rollback")

    cursor = chacha_db.execute_query(
        "SELECT id FROM character_cards WHERE name = ?",
        (character_name,),
    )
    assert cursor.fetchone() is None


def test_create_pack_requires_existing_primary_character(chacha_db: CharactersRAGDB) -> None:
    repo = VNAssetPacksRepository(chacha_db)

    with pytest.raises(ValueError, match="primary_character_not_found"):
        repo.create_pack(owner_user_id=1, primary_character_id=9999, title="Pack")


def test_create_pack_writes_minimum_row_and_json_defaults(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository(chacha_db)

    pack = repo.create_pack(owner_user_id=42, primary_character_id=character_id, title="Starter Pack")

    cursor = chacha_db.execute_query("SELECT * FROM vn_asset_packs WHERE id = ?", (pack["id"],))
    row = cursor.fetchone()
    assert row is not None
    assert row["owner_user_id"] == 42
    assert row["title"] == "Starter Pack"
    assert row["primary_character_id"] == character_id
    assert row["status"] == "draft"
    assert row["content_rating"] == "general"
    assert json.loads(row["source_world_book_ids_json"]) == []
    assert row["deleted"] == 0
    assert row["version"] == 1


def test_matrix_slot_creation_supports_multi_hop_dependencies(chacha_db: CharactersRAGDB) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(owner_user_id=42, primary_character_id=character_id, title="Starter Pack")

    slots = repo.create_slots_for_matrix(
        pack_id=pack["id"],
        slot_specs=[
            {
                "asset_type": "background",
                "slot_key": "background.interior",
                "variant_count": 1,
            },
            {
                "asset_type": "depth_companion",
                "slot_key": "depth.interior",
                "variant_count": 0,
                "depends_on_slot_key": "background.interior",
            },
            {
                "asset_type": "trim_mask",
                "slot_key": "trim.depth.interior",
                "variant_count": 0,
                "depends_on_slot_key": "depth.interior",
            },
        ],
    )

    slots_by_key = {slot["slot_key"]: slot for slot in slots}
    assert slots_by_key["depth.interior"]["depends_on_slot_id"] == slots_by_key["background.interior"]["id"]
    assert slots_by_key["trim.depth.interior"]["depends_on_slot_id"] == slots_by_key["depth.interior"]["id"]


def test_list_packs_for_setup_applies_owner_query_and_bounded_pagination(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    repo.create_pack(
        owner_user_id=42,
        primary_character_id=character_id,
        title="Archive Alpha",
        description="Station archive pack.",
    )
    repo.create_pack(
        owner_user_id=42,
        primary_character_id=character_id,
        title="Archive Beta",
        description="Secondary archive pack.",
    )
    repo.create_pack(
        owner_user_id=7,
        primary_character_id=character_id,
        title="Archive Other Owner",
    )

    rows, has_more = repo.list_packs_for_setup(
        owner_user_id=42,
        query="archive",
        limit=1,
        offset=0,
    )

    assert [row["title"] for row in rows] == ["Archive Alpha"]
    assert has_more is True

    next_rows, next_has_more = repo.list_packs_for_setup(
        owner_user_id=42,
        query="archive",
        limit=1,
        offset=1,
    )

    assert [row["title"] for row in next_rows] == ["Archive Beta"]
    assert next_has_more is False


def test_latest_completed_import_provenance_by_pack_ids_uses_latest_completed_row(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id = chacha_db.add_character_card({"name": "VN Primary"})
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(
        owner_user_id=42,
        primary_character_id=character_id,
        title="Imported Pack",
    )
    other_pack = repo.create_pack(
        owner_user_id=7,
        primary_character_id=character_id,
        title="Other Owner Pack",
    )
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job",
        status="completed",
        archive_path="test-artifacts/preview.vnpack",
    )
    repo.create_import_journal(
        owner_user_id=42,
        preview_id=int(preview["id"]),
        job_id="older-completed",
        status="completed",
        stage="completed",
        trust_mode="trusted_restore",
        target_mode="create_new",
        target_pack_id=int(pack["id"]),
        completed_at="2026-05-08T00:00:00Z",
    )
    repo.create_import_journal(
        owner_user_id=42,
        preview_id=int(preview["id"]),
        job_id="failed-newer",
        status="failed",
        stage="failed",
        trust_mode="trusted_restore",
        target_mode="create_new",
        target_pack_id=int(pack["id"]),
        completed_at="2026-05-10T00:00:00Z",
    )
    repo.create_import_journal(
        owner_user_id=42,
        preview_id=int(preview["id"]),
        job_id="newer-completed",
        status="completed",
        stage="completed",
        trust_mode="untrusted_import",
        target_mode="create_new",
        target_pack_id=int(pack["id"]),
        completed_at="2026-05-09T00:00:00Z",
    )
    other_preview = repo.create_import_preview(
        owner_user_id=7,
        job_id="other-preview",
        status="completed",
        archive_path="test-artifacts/other.vnpack",
    )
    repo.create_import_journal(
        owner_user_id=7,
        preview_id=int(other_preview["id"]),
        job_id="other-owner",
        status="completed",
        stage="completed",
        trust_mode="trusted_restore",
        target_mode="create_new",
        target_pack_id=int(other_pack["id"]),
        completed_at="2026-05-10T00:00:00Z",
    )

    provenance = repo.latest_completed_import_provenance_by_pack_ids(
        owner_user_id=42,
        pack_ids=[int(pack["id"]), int(other_pack["id"])],
    )

    assert set(provenance) == {int(pack["id"])}
    assert provenance[int(pack["id"])]["job_id"] == "newer-completed"
    assert provenance[int(pack["id"])]["trust_mode"] == "untrusted_import"
