import asyncio
import json
import sqlite3
from collections.abc import Generator
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


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-assets-test-client")
    yield database
    database.close_connection()


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
    integrity_state: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slot reconciliation failure rolls back the complete terminal transition."""
    state = integrity_state
    repo, batch_id = state["repo"], state["batch"]["id"]
    batch_before = repo.get_batch(batch_id)
    slots_before = [repo.get_slot(slot["id"]) for slot in state["slots"]]
    outcome_before = repo.get_variant_outcome(batch_id, state["slots"][0]["id"], 3)
    original = repo._refresh_slot_generation_status

    def fail_second(conn: Any, slot_id: int, **kwargs: Any) -> None:
        """Fail after the first slot has already been reconciled in the transaction."""
        if slot_id == state["slots"][1]["id"]:
            raise RuntimeError("reconciliation failed")
        original(conn, slot_id, **kwargs)

    monkeypatch.setattr(repo, "_refresh_slot_generation_status", fail_second)
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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["private-memory", "caller-transaction"])
async def test_async_outcome_read_preserves_owner_connection_fallback(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private memory and uncommitted caller work retain their original connection."""
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
        assert outcome == {"outcome_status": "planned", "item_id": None, "claim_token": "uncommitted" if mode == "caller-transaction" else None, "claim_lease_id": None}
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
    assert {"outcome_status", "item_id", "claim_token", "claim_lease_id"}.issubset(columns)


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
