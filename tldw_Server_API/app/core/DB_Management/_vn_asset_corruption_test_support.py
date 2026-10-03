"""Private damaged VN persistence fixtures; imported only by isolated tests."""

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository


def delete_recipe_rows(
    repo: VNAssetPacksRepository, batch_id: int, *,
    slot_id: int | None = None, variant_index: int | None = None,
) -> None:
    """Remove exactly the selected recipe rows to exercise fail-closed readers.

    Args:
        repo (VNAssetPacksRepository): Real isolated VN database repository.
        batch_id (int): Batch whose recipe ledger is deliberately damaged.
        slot_id (int | None): Optional restriction; None selects every slot.
        variant_index (int | None): Optional restriction; None selects every variant.

    Returns:
        None: Completes the native database write; database errors propagate.
    """
    repo.db.execute_query(
        """DELETE FROM vn_asset_generation_recipes WHERE batch_id = ?
           AND (? IS NULL OR slot_id = ?) AND (? IS NULL OR variant_index = ?)""",
        (batch_id, slot_id, slot_id, variant_index, variant_index),
    )


def corrupt_recipe_version(repo: VNAssetPacksRepository, batch_id: int) -> None:
    """Set the explicit unsupported version used by damaged-ledger controls.

    Args:
        repo (VNAssetPacksRepository): Real isolated VN database repository.
        batch_id (int): Batch whose persisted version is deliberately damaged.

    Returns:
        None: Writes unsupported version99; native errors propagate.
    """
    repo.db.execute_query(
        "UPDATE vn_asset_batches SET recipe_version = 99 WHERE id = ?", (batch_id,),
    )


def set_recipe_item(repo: VNAssetPacksRepository, batch_id: int, slot_id: int, item_id: int | None) -> None:
    """Corrupt or restore the first variant's item link in a test database.

    Args:
        repo (VNAssetPacksRepository): Test-owned repository.
        batch_id (int): Selected batch.
        slot_id (int): Selected slot.
        item_id (int | None): Replacement item link, including an unmarked null link.

    Returns:
        None: Updates only the selected test recipe; native errors propagate.
    """
    repo.db.execute_query(
        "UPDATE vn_asset_generation_recipes SET item_id = ? WHERE batch_id = ? AND slot_id = ? AND variant_index = 0",
        (item_id, batch_id, slot_id),
    )


def set_deletion_receipt(repo: VNAssetPacksRepository, batch_id: int, slot_id: int, receipt: str | None) -> None:
    """Corrupt or restore the first variant's deletion receipt for fail-closed tests.

    Args:
        repo (VNAssetPacksRepository): Test-owned repository.
        batch_id (int): Selected batch.
        slot_id (int): Selected slot.
        receipt (str | None): Serialized replacement, deliberately not validated.

    Returns:
        None: Updates only the selected test recipe; native errors propagate.
    """
    repo.db.execute_query(
        "UPDATE vn_asset_generation_recipes SET deleted_item_json = ? "
        "WHERE batch_id = ? AND slot_id = ? AND variant_index = 0",
        (receipt, batch_id, slot_id),
    )


def drop_deletion_receipt_column(repo: VNAssetPacksRepository) -> None:
    """Restore the pre-receipt schema to exercise additive upgrade behavior.

    Args:
        repo (VNAssetPacksRepository): Isolated repository with no deletion receipts.

    Returns:
        None: Removes the column without altering recipe data; native errors propagate.
    """
    repo.db.execute_query("ALTER TABLE vn_asset_generation_recipes DROP COLUMN deleted_item_json")
