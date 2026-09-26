"""First-completed VN receipt snapshots and exact scoped payload conflicts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService

pytestmark = pytest.mark.integration
SCOPES = (
    "vn_asset_generate", "vn_asset_slot_retry", "vn_asset_item_regenerate", "vn_asset_item_upload",
    "vn_asset_cleanup", "vn_asset_export", "vn_asset_import_preview", "vn_asset_import_commit",
)
RECEIPT = {"scope": "vn_asset_generate", "resource_id": "pack:1", "idempotency_key": "same-key", "payload_hash": "same-payload"}


@pytest.fixture
def service(tmp_path: Path) -> Iterator[VNAssetPackService]:
    """Use real isolated receipt transactions without Jobs or external generation."""
    db = CharactersRAGDB(str(tmp_path / "vn.db"), client_id="receipt-first-completion")
    try:
        yield VNAssetPackService(db, owner_user_id=42)
    finally:
        db.close_connection()


@pytest.mark.parametrize("scope", SCOPES)
def test_each_receipt_scope_preserves_first_completed_json(service: VNAssetPackService, scope: str) -> None:
    """All VN operations preserve the first snapshot, not just batch recovery."""
    receipt = {**RECEIPT, "scope": scope}
    assert service.claim_or_replay_idempotency(owner_user_id=42, **receipt) is None
    first = {"first": True, "optional": None, "nested": {"ids": [1, 2]}}
    assert service.complete_idempotency_response(owner_user_id=42, **receipt, response=first) is None
    service.complete_idempotency_response(owner_user_id=42, **receipt, response={"later": True})
    assert service.claim_or_replay_idempotency(owner_user_id=42, **receipt) == first


@pytest.mark.parametrize("completed", [False, True])
def test_completion_rejects_payload_conflict_without_changing_record(
    service: VNAssetPackService, completed: bool,
) -> None:
    """Completion must not replace the claimed payload hash, including after completion."""
    service.repo.claim_idempotency_record(owner_user_id=42, **RECEIPT)
    if completed:
        service.complete_idempotency_response(owner_user_id=42, **RECEIPT, response={"first": True})
    identity = {key: value for key, value in RECEIPT.items() if key != "payload_hash"}
    before = service.repo.get_idempotency_record(owner_user_id=42, **identity)
    with pytest.raises(ValueError, match="^idempotency_key_conflict$"):
        service.complete_idempotency_response(
            owner_user_id=42, **{**RECEIPT, "payload_hash": "different"}, response={"overwrite": True},
        )
    assert service.repo.get_idempotency_record(owner_user_id=42, **identity) == before


def test_first_completion_without_claim_and_scope_isolation(service: VNAssetPackService) -> None:
    """Keep existing insert-on-completion behavior without mixing owners or resources."""
    service.complete_idempotency_response(owner_user_id=42, **RECEIPT, response={"first": True})
    for overrides in ({"owner_user_id": 43}, {"scope": "vn_asset_export"}, {"resource_id": "pack:2"}):
        facts = {"owner_user_id": 42, **RECEIPT, **overrides}
        service.complete_idempotency_response(**facts, response={"isolated": overrides})
        assert service.claim_or_replay_idempotency(**facts) == {"isolated": overrides}
    assert service.claim_or_replay_idempotency(owner_user_id=42, **RECEIPT) == {"first": True}


def test_concurrent_completions_return_one_authoritative_snapshot(service: VNAssetPackService) -> None:
    """Independent real transactions converge on whichever completion committed first."""
    service.repo.claim_idempotency_record(owner_user_id=42, **RECEIPT)
    barrier = Barrier(2)

    def complete(index: int) -> dict[str, Any]:
        """Complete through a caller-owned DB connection, preserving the returned record."""
        db = CharactersRAGDB(str(service.repo.db.db_path), client_id=f"complete-{index}")
        caller = VNAssetPackService(db, owner_user_id=42)
        try:
            barrier.wait(timeout=15)
            record = caller.repo.complete_idempotency_record(owner_user_id=42, **RECEIPT, response={"caller": index})
            return json.loads(record["response_json"])
        finally:
            db.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        snapshots = list(executor.map(complete, (1, 2)))
    assert snapshots[0] == snapshots[1]
    assert snapshots[0] in ({"caller": 1}, {"caller": 2})
    assert service.claim_or_replay_idempotency(owner_user_id=42, **RECEIPT) == snapshots[0]
