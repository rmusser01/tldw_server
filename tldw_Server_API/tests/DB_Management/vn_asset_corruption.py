"""Test-only damaged VN persistence fixtures; never imported by runtime code."""

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository


def delete_recipe_rows(
    repo: VNAssetPacksRepository, batch_id: int, *,
    slot_id: int | None = None, variant_index: int | None = None,
) -> None:
    """Remove exactly the selected recipe rows to exercise fail-closed readers.

    Args:
        repo: Real isolated VN database repository.
        batch_id: Batch whose recipe ledger is deliberately damaged.
        slot_id: Optional slot restriction; None selects every slot.
        variant_index: Optional variant restriction; None selects every variant.

    Returns:
        None after the native database write; database errors propagate.
    """
    repo.db.execute_query(
        """DELETE FROM vn_asset_generation_recipes WHERE batch_id = ?
           AND (? IS NULL OR slot_id = ?) AND (? IS NULL OR variant_index = ?)""",
        (batch_id, slot_id, slot_id, variant_index, variant_index),
    )


def corrupt_recipe_version(repo: VNAssetPacksRepository, batch_id: int) -> None:
    """Set the explicit unsupported version used by damaged-ledger controls.

    Args:
        repo: Real isolated VN database repository.
        batch_id: Batch whose persisted version is deliberately damaged.

    Returns:
        None after writing unsupported version99; native errors propagate.
    """
    repo.db.execute_query(
        "UPDATE vn_asset_batches SET recipe_version = 99 WHERE id = ?", (batch_id,),
    )


def set_recipe_item(repo: VNAssetPacksRepository, batch_id: int, slot_id: int, item_id: int | None) -> None:
    """Corrupt or restore the first variant's item link in a test database.

    Args:
        repo: Test-owned repository.
        batch_id: Selected batch.
        slot_id: Selected slot.
        item_id: Replacement item link, including an unmarked null link.

    Returns:
        None; updates only the selected test recipe.
    """
    repo.db.execute_query(
        "UPDATE vn_asset_generation_recipes SET item_id = ? WHERE batch_id = ? AND slot_id = ? AND variant_index = 0",
        (item_id, batch_id, slot_id),
    )


def set_deletion_receipt(repo: VNAssetPacksRepository, batch_id: int, slot_id: int, receipt: str | None) -> None:
    """Corrupt or restore the first variant's deletion receipt for fail-closed tests.

    Args:
        repo: Test-owned repository.
        batch_id: Selected batch.
        slot_id: Selected slot.
        receipt: Serialized replacement, deliberately not validated by the fixture.

    Returns:
        None; updates only the selected test recipe.
    """
    repo.db.execute_query(
        "UPDATE vn_asset_generation_recipes SET deleted_item_json = ? "
        "WHERE batch_id = ? AND slot_id = ? AND variant_index = 0",
        (receipt, batch_id, slot_id),
    )


def drop_deletion_receipt_column(repo: VNAssetPacksRepository) -> None:
    """Restore the pre-receipt schema to exercise additive upgrade behavior.

    Args:
        repo: Isolated test repository with no deliberate deletion receipts.

    Returns:
        None; removes the new test column without altering old recipe data.
    """
    repo.db.execute_query("ALTER TABLE vn_asset_generation_recipes DROP COLUMN deleted_item_json")
