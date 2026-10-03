"""Native SQLite recipe lookup plans and unchanged public metadata outcomes."""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError

INDEX_NAME = "idx_vn_asset_generation_recipes_item_outcome"


def _seed(repo: VNAssetPacksRepository) -> dict[str, int]:
    """Create published, active, failed, cancelled and unlinked items publicly."""
    character_id = repo.db.add_character_card({"name": "Recipe index control"})
    pack = repo.create_pack(owner_user_id=1, primary_character_id=character_id, title="Index control")
    slot = repo.create_slot(pack_id=pack["id"], asset_type="sprite", slot_key="sprite")
    batch = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=3,
        recipes=[{"slot_id": slot["id"], "variant_index": index, "recipe": {"index": index}}
                 for index in range(3)],
    )
    state = {"pack": pack["id"], "slot": slot["id"], "batch": batch["id"]}
    for index, name in enumerate(("planned", "completed", "failed")):
        item = repo.reserve_variant_item(
            batch_id=batch["id"], slot_id=slot["id"], variant_index=index,
            item_fields={"pack_id": pack["id"], "generated_file_id": 70 + index},
        )
        state[name] = item["id"]
    repo.complete_variant(batch_id=batch["id"], slot_id=slot["id"], variant_index=1, item_id=state["completed"])
    repo.update_item_review(state["completed"], review_status="approved", preferred=True)
    repo.fail_variant(batch_id=batch["id"], slot_id=slot["id"], variant_index=2, error="control failure")
    cancelled = repo.create_batch(
        pack_id=pack["id"], requested_by_user_id=1, total_variants=1,
        recipes=[{"slot_id": slot["id"], "variant_index": 3, "recipe": {}}],
    )
    state["cancelled_batch"] = cancelled["id"]
    item = repo.reserve_variant_item(
        batch_id=cancelled["id"], slot_id=slot["id"], variant_index=3,
        item_fields={"pack_id": pack["id"], "generated_file_id": 73},
    )
    state["cancelled"] = item["id"]
    repo.cancel_batch(cancelled["id"])
    state["legacy"] = repo.create_item(pack_id=pack["id"], slot_id=slot["id"])["id"]
    state["child"] = repo.create_item(
        pack_id=pack["id"], slot_id=slot["id"], parent_item_id=state["completed"], generated_file_id=71,
    )["id"]
    return state


def _rows(connection: sqlite3.Connection) -> list[str]:
    """Snapshot all VN table rows without touching production query algorithms."""
    return [statement for statement in connection.iterdump() if statement.startswith('INSERT INTO "vn_')]


def _public_plan(repo: VNAssetPacksRepository, state: dict[str, int], operation: str) -> list[str]:
    """EXPLAIN the actual SQL traced from a public visibility or capacity call."""
    connection = repo.db.get_connection()
    statements: list[str] = []
    connection.set_trace_callback(statements.append)
    try:
        argument = state["planned"] if operation == "item_is_unpublished" else state["pack"]
        getattr(repo, operation)(argument)
    finally:
        connection.set_trace_callback(None)
    queries = [statement for statement in statements
               if statement.lstrip().upper().startswith("SELECT") and "vn_asset_generation_recipes" in statement]
    assert len(queries) == 1, statements
    plan = [row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + queries[0])]
    return plan


@pytest.fixture(params=["new", "reopened_existing"])
def recipe_repo(request: pytest.FixtureRequest, tmp_path: Path) -> Generator[
    tuple[VNAssetPacksRepository, dict[str, int]], None, None
]:
    """Initialize a fresh DB or reopen real pre-index rows and preserve them."""
    path = tmp_path / "recipe-index.db"
    database = CharactersRAGDB(path, client_id="recipe-index-control")
    try:
        repo = VNAssetPacksRepository.initialized(database)
        state = _seed(repo)
        if request.param == "reopened_existing":
            with database.transaction() as connection:
                connection.execute("DROP INDEX IF EXISTS idx_vn_asset_generation_recipes_item_outcome")
            for operation in ("list_items", "count_items_for_generation", "item_is_unpublished"):
                assert any("SCAN" in detail and "recipe" in detail for detail in _public_plan(repo, state, operation))
            before = _rows(database.get_connection())
            database.close_connection()
            database = CharactersRAGDB(path, client_id="recipe-index-reopen")
            repo = VNAssetPacksRepository.initialized(database)
            assert _rows(database.get_connection()) == before
        yield repo, state
    finally:
        database.close_connection()


@pytest.mark.integration
@pytest.mark.parametrize("operation", ["list_items", "count_items_for_generation", "item_is_unpublished"])
def test_public_item_predicates_use_covering_recipe_index(
    recipe_repo: tuple[VNAssetPacksRepository, dict[str, int]], operation: str,
) -> None:
    """New/reopened DB public predicates search a covering item-leading index."""
    repo, state = recipe_repo
    plan = _public_plan(repo, state, operation)
    assert any("SEARCH" in detail and "COVERING INDEX" in detail and "item_id=?" in detail
               and "recipe" in detail for detail in plan), plan


@pytest.mark.integration
def test_recipe_index_schema_is_idempotent_without_row_changes(
    recipe_repo: tuple[VNAssetPacksRepository, dict[str, int]],
) -> None:
    """Repeated public initialization retains every row and the ordered index."""
    repo, _state = recipe_repo
    connection = repo.db.get_connection()
    before = _rows(connection)
    schema_before = [tuple(row) for row in connection.execute("SELECT * FROM sqlite_master ORDER BY name")]
    repo.initialize_schema()
    repo.initialize_schema()
    assert _rows(connection) == before
    assert [tuple(row) for row in connection.execute("SELECT * FROM sqlite_master ORDER BY name")] == schema_before
    columns = [row[2] for row in connection.execute("PRAGMA index_info(idx_vn_asset_generation_recipes_item_outcome)")]
    assert columns == ["item_id", "outcome_status"]


@pytest.mark.integration
def test_recipe_index_preserves_visibility_approval_capacity_and_reference_cleanup(
    recipe_repo: tuple[VNAssetPacksRepository, dict[str, int]],
) -> None:
    """Public outcomes retain hidden reservations, approval and cleanup fences."""
    repo, state = recipe_repo
    assert [item["id"] for item in repo.list_items(state["pack"])] == [
        state["completed"], state["legacy"], state["child"],
    ]
    assert repo.count_items_for_generation(state["pack"]) == 4
    for name in ("planned", "failed", "cancelled"):
        assert repo.item_is_unpublished(state[name])
        assert repo.get_item(state[name])["review_status"] == "hidden"
    approved = repo.get_item(state["completed"])
    assert (approved["review_status"], approved["preferred"]) == ("approved", 1)
    assert not repo.item_is_unpublished(state["completed"])
    assert repo.count_items_referencing_generated_file(71) == 2
    with pytest.raises(VNAssetGenerationError, match="vn_asset_variant_in_progress"):
        repo.delete_item(state["planned"])
    assert repo.get_item(state["completed"]) == approved
    assert repo.delete_item(state["completed"])
    assert repo.get_variant_outcome(state["batch"], state["slot"], 1)["item_id"] is None
    assert repo.get_variant_outcome(state["batch"], state["slot"], 1)["outcome_status"] == "completed"
    assert repo.get_item(state["child"])["parent_item_id"] is None
    assert repo.count_items_referencing_generated_file(71) == 1
    for name in ("failed", "cancelled"):
        assert repo.delete_item(state[name])
    assert repo.count_items_for_generation(state["pack"]) == 3
    assert repo.get_batch(state["batch"])["completed_count"] == 1
    assert repo.get_batch(state["batch"])["failed_count"] == 1
    assert repo.get_batch(state["cancelled_batch"])["cancelled_count"] == 1


@pytest.mark.integration
def test_recipe_index_initializes_after_legacy_columns_and_preserves_recipe_rows(tmp_path: Path) -> None:
    """Legacy tables gain missing columns before index DDL, retaining base rows."""
    database = CharactersRAGDB(tmp_path / "legacy-recipes.db", client_id="recipe-index-legacy")
    try:
        repo = VNAssetPacksRepository.initialized(database)
        _seed(repo)
        connection = database.get_connection()
        base_query = "SELECT batch_id, slot_id, variant_index, recipe_json, created_at FROM vn_asset_generation_recipes"
        before = [tuple(row) for row in connection.execute(base_query)]
        with database.transaction() as connection:
            connection.execute("DROP INDEX IF EXISTS idx_vn_asset_generation_recipes_item_outcome")
            connection.execute("ALTER TABLE vn_asset_generation_recipes DROP COLUMN claim_token")
            connection.execute("ALTER TABLE vn_asset_generation_recipes DROP COLUMN claim_lease_id")
            connection.execute("ALTER TABLE vn_asset_generation_recipes DROP COLUMN item_id")
            connection.execute("ALTER TABLE vn_asset_generation_recipes DROP COLUMN outcome_status")
        database.close_connection()
        database = CharactersRAGDB(tmp_path / "legacy-recipes.db", client_id="recipe-index-legacy-reopen")
        repo = VNAssetPacksRepository.initialized(database)
        repo.initialize_schema()
        connection = database.get_connection()
        assert [tuple(row) for row in connection.execute(base_query)] == before
        columns = [row[2] for row in connection.execute("PRAGMA index_info(idx_vn_asset_generation_recipes_item_outcome)")]
        assert columns == ["item_id", "outcome_status"]
    finally:
        database.close_connection()


@pytest.mark.integration
@pytest.mark.parametrize("existing", [False, True])
def test_recipe_index_native_creation_error_rolls_back_and_propagates(tmp_path: Path, existing: bool) -> None:
    """Native CREATE INDEX denial propagates, rolls back, and allows later retry."""
    database = CharactersRAGDB(tmp_path / "index-denial.db", client_id="recipe-index-error")
    try:
        repo = VNAssetPacksRepository(database)
        if existing:
            repo.initialize_schema()
            _seed(repo)
            with database.transaction() as connection:
                connection.execute("DROP INDEX IF EXISTS idx_vn_asset_generation_recipes_item_outcome")
        connection = database.get_connection()
        before = _rows(connection)

        def deny_index(
            action: int, name: str | None, _table: str | None, _db_name: str | None, _trigger: str | None,
        ) -> int:
            """Deny only this schema-owned index through SQLite's native authorizer."""
            if action == sqlite3.SQLITE_CREATE_INDEX and name == INDEX_NAME:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(deny_index)
        try:
            with pytest.raises(sqlite3.DatabaseError) as raised:
                repo.initialize_schema()
            assert raised.value.sqlite_errorcode == sqlite3.SQLITE_AUTH
        finally:
            connection.set_authorizer(None)
        assert not connection.in_transaction
        assert _rows(connection) == before
        repo.initialize_schema()
        assert connection.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
                                  (INDEX_NAME,)).fetchone()[0] == INDEX_NAME
        if existing:
            assert _rows(connection) == before
    finally:
        database.close_connection()
