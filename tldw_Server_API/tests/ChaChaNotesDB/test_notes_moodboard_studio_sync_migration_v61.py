"""SQLite schema-v61 contracts for scoped moodboards and Studio sidecars."""

from __future__ import annotations

import inspect
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.moodboard_sync_store import (
    MoodboardSyncStore,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_store import NoteStore
from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    SchemaError,
)
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    diagram_render_hash,
    placement_object_id,
)

pytestmark = pytest.mark.unit

OWNER = "moodboard-studio-v61-owner"
OTHER_OWNER = "moodboard-studio-v61-other"
LOCAL_UNBOUND = "local-unbound"
DATASET_A = "dataset-a"
DATASET_B = "dataset-b"
BOARD_NOTE_ID = "11111111-1111-4111-8111-111111111111"
STUDIO_NOTE_ID = "22222222-2222-4222-8222-222222222222"
SOURCE_NOTE_ID = "33333333-3333-4333-8333-333333333333"

EXPECTED_AUTHORITY_COLUMNS = (
    "owner_user_id",
    "dataset_id",
    "task_graph_bound",
    "moodboard_graph_bound",
    "studio_graph_bound",
)
EXPECTED_MOODBOARD_COLUMNS = (
    "id",
    "owner_user_id",
    "dataset_id",
    "sync_id",
    "name",
    "description",
    "smart_rule_json",
    "canvas_json",
    "created_at",
    "last_modified",
    "deleted",
    "client_id",
    "version",
    "canonical_revision",
    "canonical_hash",
    "source_diagnostic_code",
    "source_diagnostic_hash",
)
EXPECTED_PLACEMENT_COLUMNS = (
    "owner_user_id",
    "dataset_id",
    "moodboard_id",
    "note_id",
    "placement_id",
    "x",
    "y",
    "width",
    "height",
    "order_index",
    "display_json",
    "created_at",
    "last_modified",
    "deleted",
    "version",
    "canonical_revision",
    "canonical_hash",
    "source_diagnostic_code",
    "source_diagnostic_hash",
)
EXPECTED_STUDIO_COLUMNS = (
    "owner_user_id",
    "dataset_id",
    "note_id",
    "payload_json",
    "template_type",
    "handwriting_mode",
    "source_note_id",
    "excerpt_snapshot",
    "excerpt_hash",
    "diagram_manifest_json",
    "companion_content_hash",
    "render_version",
    "note_revision",
    "note_hash",
    "accepted_provenance_json",
    "created_at",
    "last_modified",
    "deleted",
    "version",
    "canonical_revision",
    "canonical_hash",
    "source_diagnostic_code",
    "source_diagnostic_hash",
)
EXPECTED_INDEXES = {
    "idx_moodboards_scope_page",
    "idx_moodboards_scope_sync_id",
    "idx_moodboard_notes_scope_board_page",
    "idx_moodboard_notes_scope_note",
    "idx_moodboard_notes_scope_placement",
    "idx_note_studio_documents_scope_page",
    "idx_note_studio_documents_scope_note",
    "idx_note_studio_documents_scope_source",
}


def _table_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(str(row[1]) for row in conn.execute(f"PRAGMA table_xinfo({table})"))  # nosec B608


def _database_dump(db_path: Path) -> tuple[str, ...]:
    with sqlite3.connect(db_path) as conn:
        return tuple(conn.iterdump())


def _schema_version(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return int(
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name=?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
        )


def _prepare_v60_product_database(
    db_path: Path,
    *,
    board_count: int = 1,
) -> list[int]:
    """Create the exact preceding schema with a representative legacy graph."""

    original = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = 60
    try:
        db = CharactersRAGDB(str(db_path), client_id=OWNER)
        assert db.add_note("Board note", "Pinned content", note_id=BOARD_NOTE_ID) == BOARD_NOTE_ID  # nosec B101
        assert db.add_note("Studio target", "Studio markdown", note_id=STUDIO_NOTE_ID) == STUDIO_NOTE_ID  # nosec B101
        assert db.add_note("Source note", "Source excerpt body", note_id=SOURCE_NOTE_ID) == SOURCE_NOTE_ID  # nosec B101
        with db.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE note_studio_documents(
                  note_id TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
                  payload_json TEXT NOT NULL,
                  template_type TEXT NOT NULL,
                  handwriting_mode TEXT NOT NULL,
                  source_note_id TEXT REFERENCES notes(id) ON DELETE SET NULL ON UPDATE CASCADE,
                  excerpt_snapshot TEXT,
                  excerpt_hash TEXT,
                  diagram_manifest_json TEXT,
                  companion_content_hash TEXT,
                  render_version INTEGER NOT NULL DEFAULT 1,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX idx_note_studio_documents_source_note_id "
                "ON note_studio_documents(source_note_id)"
            )
        board_ids: list[int] = []
        for index in range(board_count):
            board_id = db.add_moodboard(
                name=f"Legacy board {index}",
                description="Legacy description",
                smart_rule={"query": "Pinned", "keyword_tokens": ["Research"]},
            )
            assert board_id is not None  # nosec B101
            board_ids.append(board_id)
        assert db.link_note_to_moodboard(board_ids[0], BOARD_NOTE_ID) is True  # nosec B101
        db.create_note_studio_document(
            note_id=STUDIO_NOTE_ID,
            payload_json={
                "meta": {
                    "title": "Studio target",
                    "source_note_id": SOURCE_NOTE_ID,
                },
                "layout": {
                    "template_type": "lined",
                    "handwriting_mode": "accented",
                    "render_version": 1,
                },
                "sections": [
                    {
                        "id": "section-1",
                        "kind": "notes",
                        "title": "Summary",
                        "content": "Accepted content",
                    }
                ],
            },
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=SOURCE_NOTE_ID,
            excerpt_snapshot="Source excerpt",
            excerpt_hash="legacy-excerpt-hash",
            companion_content_hash="legacy-content-hash",
            render_version=1,
        )
        db.close_all_connections()
        return board_ids
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original


def _legacy_diagram_manifest() -> dict[str, object]:
    diagram = "graph TD; A-->B"
    source_graph = [
        {
            "id": "section-1",
            "kind": "notes",
            "title": "Summary",
            "content": "Accepted content",
        }
    ]
    return {
        "diagram_type": "flowchart",
        "source_section_ids": ["section-1"],
        "source_graph": source_graph,
        "diagram": diagram,
        "format": "mermaid",
        "status": "ready",
        "render_hash": diagram_render_hash(
            diagram_type="flowchart",
            context="Summary\nAccepted content",
            diagram=diagram,
        ),
    }


def _relevant_catalog(db: CharactersRAGDB) -> dict[str, object]:
    with db.transaction() as conn:
        return db._notes_moodboard_studio_catalog_snapshot_sqlite(conn)


def test_sqlite_schema_authority_is_v61_and_version_update_is_last() -> None:
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 61
    source = inspect.getsource(CharactersRAGDB._migrate_from_v60_to_v61_sqlite)
    for operation in (
        "_create_notes_moodboard_studio_schema_v61_sqlite",
        "_copy_notes_moodboard_studio_graph_v61_sqlite",
        "_create_notes_moodboard_studio_indexes_v61_sqlite",
        "_verify_notes_moodboard_studio_schema_sqlite",
    ):
        assert source.index(operation) < source.index("SET version=61")
    assert "WHERE schema_name=? AND version=60" in source


def test_fresh_and_v60_upgrade_have_exact_v61_catalog_parity(tmp_path: Path) -> None:
    upgrade_path = tmp_path / "upgrade.sqlite"
    _prepare_v60_product_database(upgrade_path)
    upgraded = CharactersRAGDB(str(upgrade_path), client_id=OWNER)
    fresh = CharactersRAGDB(str(tmp_path / "fresh.sqlite"), client_id=OWNER)
    try:
        assert _relevant_catalog(upgraded) == _relevant_catalog(fresh)  # nosec B101
        with fresh.transaction() as conn:
            assert fresh._get_db_version(conn) == 61  # nosec B101
            assert _table_columns(conn, "note_task_scope_authority") == EXPECTED_AUTHORITY_COLUMNS  # nosec B101
            assert _table_columns(conn, "moodboards") == EXPECTED_MOODBOARD_COLUMNS  # nosec B101
            assert _table_columns(conn, "moodboard_notes") == EXPECTED_PLACEMENT_COLUMNS  # nosec B101
            assert _table_columns(conn, "note_studio_documents") == EXPECTED_STUDIO_COLUMNS  # nosec B101
            indexes = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
                )
            }
            assert indexes >= EXPECTED_INDEXES  # nosec B101
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []  # nosec B101
    finally:
        upgraded.close_all_connections()
        fresh.close_all_connections()


def test_v61_authority_defaults_and_boolean_storage_are_exact(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "authority-defaults.sqlite"), client_id=OWNER)
    try:
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                (OWNER, DATASET_A),
            )
            row = conn.execute(
                "SELECT * FROM note_task_scope_authority WHERE owner_user_id=?", (OWNER,)
            ).fetchone()
            assert tuple(row) == (OWNER, DATASET_A, 1, 0, 0)  # nosec B101
            for column in (
                "task_graph_bound",
                "moodboard_graph_bound",
                "studio_graph_bound",
            ):
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute(
                        f"UPDATE note_task_scope_authority SET {column}=1.5 WHERE owner_user_id=?",  # nosec B608
                        (OWNER,),
                    )
    finally:
        db.close_all_connections()


def test_v60_upgrade_preserves_rows_in_local_unbound_with_canonical_lineage(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.sqlite"
    _prepare_v60_product_database(db_path)
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with db.transaction() as conn:
            board = dict(conn.execute("SELECT * FROM moodboards").fetchone())
            placement = dict(conn.execute("SELECT * FROM moodboard_notes").fetchone())
            studio = dict(conn.execute("SELECT * FROM note_studio_documents").fetchone())
        parsed_board_id = UUID(str(board["sync_id"]))
        assert parsed_board_id.version == 4  # nosec B101
        assert str(parsed_board_id) == board["sync_id"]  # nosec B101
        assert board["owner_user_id"] == OWNER  # nosec B101
        assert board["dataset_id"] == LOCAL_UNBOUND  # nosec B101
        assert json.loads(board["canvas_json"]) == {  # nosec B101
            "layout_mode": "masonry",
            "metadata": {},
        }
        assert board["canonical_revision"] == board["version"] == 1  # nosec B101
        assert str(board["canonical_hash"]).startswith("sha256:")  # nosec B101

        expected_placement_id = placement_object_id(
            {"moodboard_id": board["sync_id"], "note_id": BOARD_NOTE_ID}
        )
        assert placement["placement_id"] == expected_placement_id  # nosec B101
        assert placement["owner_user_id"] == OWNER  # nosec B101
        assert placement["dataset_id"] == LOCAL_UNBOUND  # nosec B101
        assert (placement["x"], placement["y"], placement["width"], placement["height"]) == (0, 0, 320, 220)  # nosec B101
        assert json.loads(placement["display_json"]) == {}  # nosec B101
        assert placement["canonical_revision"] == 1  # nosec B101
        assert str(placement["canonical_hash"]).startswith("sha256:")  # nosec B101

        assert studio["owner_user_id"] == OWNER  # nosec B101
        assert studio["dataset_id"] == LOCAL_UNBOUND  # nosec B101
        assert studio["note_revision"] == 1  # nosec B101
        assert str(studio["note_hash"]).startswith("sha256:")  # nosec B101
        assert str(studio["canonical_hash"]).startswith("sha256:")  # nosec B101
        provenance = json.loads(studio["accepted_provenance_json"])
        assert provenance["kind"] == "legacy_bootstrap"  # nosec B101
        assert provenance["attestation"] == "trusted_bootstrap_v1"  # nosec B101
        assert "prompt" not in studio["accepted_provenance_json"].lower()  # nosec B101
    finally:
        db.close_all_connections()


def test_v61_legacy_studio_accepts_equal_aliases_cache_and_partial_layout(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-studio-aliases.sqlite"
    _prepare_v60_product_database(db_path)
    manifest = _legacy_diagram_manifest()
    manifest.update(
        {
            "canonical_source": manifest["source_graph"],
            "generation_status": "ready",
            "cached_svg": "<svg>legacy cache is not canonical</svg>",
        }
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE note_studio_documents SET payload_json=?,diagram_manifest_json=?",
            (
                json.dumps(
                    {
                        "meta": {"source_note_id": SOURCE_NOTE_ID},
                        "layout": {"render_version": 1},
                        "sections": [
                            {
                                "id": "section-1",
                                "kind": "notes",
                                "title": "Summary",
                                "content": "Accepted content",
                            }
                        ],
                    }
                ),
                json.dumps(manifest),
            ),
        )

    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        row = db.get_note_studio_document(STUDIO_NOTE_ID)
        assert row is not None  # nosec B101
        assert row["source_diagnostic_code"] is None  # nosec B101
        assert set(row["payload_json"]) == {"sections"}  # nosec B101
        assert row["diagram_manifest_json"] == _legacy_diagram_manifest()  # nosec B101
        provenance = row["accepted_provenance_json"]
        assert provenance["kind"] == "legacy_bootstrap"  # nosec B101
        assert provenance["source_revision"] == 1  # nosec B101
        assert str(provenance["source_hash"]).startswith("sha256:")  # nosec B101
    finally:
        db.close_all_connections()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("payload_json", {"meta": {"title": "wrong"}, "sections": []}),
        ("payload_json", {"layout": {"render_version": 2}, "sections": []}),
        (
            "diagram_manifest_json",
            {**_legacy_diagram_manifest(), "canonical_source": []},
        ),
        (
            "diagram_manifest_json",
            {**_legacy_diagram_manifest(), "generation_status": "error"},
        ),
        ("excerpt_snapshot", "not present in source"),
    ),
)
def test_v61_legacy_studio_mismatches_become_bounded_binding_blockers(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    db_path = tmp_path / f"legacy-studio-mismatch-{field}.sqlite"
    _prepare_v60_product_database(db_path)
    stored = json.dumps(value) if isinstance(value, dict) else value
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"UPDATE note_studio_documents SET {field}=?",  # nosec B608 - fixed parametrized column names.
            (stored,),
        )

    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        row = db.get_note_studio_document(STUDIO_NOTE_ID)
        assert row is not None  # nosec B101
        assert row["source_diagnostic_code"] == "legacy_studio_payload_invalid"  # nosec B101
        assert len(row["source_diagnostic_hash"]) == 71  # nosec B101
        with pytest.raises(ConflictError, match="canonical readiness proof"):
            db.bind_local_studio_graph_to_dataset(
                owner_user_id=OWNER,
                target_dataset_id=DATASET_A,
            )
    finally:
        db.close_all_connections()


def test_v61_legacy_studio_oversized_envelope_is_blocked_before_binding(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-studio-oversized.sqlite"
    _prepare_v60_product_database(db_path)
    sections = [
        {
            "id": f"section-{index}",
            "kind": "notes",
            "title": f"Summary {index}",
            "content": "x" * 65_500,
        }
        for index in range(4)
    ]
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE note_studio_documents SET payload_json=?",
            (json.dumps({"sections": sections}),),
        )
    before = _database_dump(db_path)

    with pytest.raises(CharactersRAGDBError):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    assert _schema_version(db_path) == 60  # nosec B101
    assert _database_dump(db_path) == before  # nosec B101


@pytest.mark.parametrize("collection_state", ("unknown", "cross_owner", "ambiguous"))
def test_v61_smart_rule_portable_collection_ids_must_resolve_uniquely_per_owner(
    tmp_path: Path,
    collection_state: str,
) -> None:
    db_path = tmp_path / f"legacy-collection-{collection_state}.sqlite"
    board_id = _prepare_v60_product_database(db_path)[0]
    portable_id = "44444444-4444-4444-8444-444444444444"
    with sqlite3.connect(db_path) as conn:
        if collection_state == "ambiguous":
            conn.execute("DROP INDEX idx_keyword_collections_sync_id_unique")
        if collection_state in {"cross_owner", "ambiguous"}:
            conn.execute(
                "INSERT INTO keyword_collections(sync_id,name,deleted,client_id,version) "
                "VALUES (?,?,0,?,1)",
                (portable_id, "Other collection", OTHER_OWNER),
            )
        if collection_state == "ambiguous":
            conn.execute(
                "INSERT INTO keyword_collections(sync_id,name,deleted,client_id,version) "
                "VALUES (?,?,1,?,1)",
                (portable_id, "Owner collection", OWNER),
            )
        conn.execute(
            "UPDATE moodboards SET smart_rule_json=? WHERE id=?",
            (json.dumps({"collection_sync_ids": [portable_id]}), board_id),
        )

    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        board = db.get_moodboard_by_id(board_id)
        assert board is not None  # nosec B101
        assert board["source_diagnostic_code"] == "legacy_moodboard_rule_invalid"  # nosec B101
        with pytest.raises(ConflictError, match="canonical readiness proof"):
            db.bind_local_moodboard_graph_to_dataset(
                owner_user_id=OWNER,
                target_dataset_id=DATASET_A,
            )
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("deleted", (0, 1))
def test_v61_smart_rule_allows_same_owner_live_or_tombstoned_portable_collection(
    tmp_path: Path,
    deleted: int,
) -> None:
    db_path = tmp_path / f"legacy-collection-valid-{deleted}.sqlite"
    board_id = _prepare_v60_product_database(db_path)[0]
    portable_id = "44444444-4444-4444-8444-444444444444"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO keyword_collections(sync_id,name,deleted,client_id,version) "
            "VALUES (?,?,?,?,1)",
            (portable_id, "Owner collection", deleted, OWNER),
        )
        conn.execute(
            "UPDATE moodboards SET smart_rule_json=? WHERE id=?",
            (json.dumps({"collection_sync_ids": [portable_id]}), board_id),
        )

    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        board = db.get_moodboard_by_id(board_id)
        assert board is not None  # nosec B101
        assert board["source_diagnostic_code"] is None  # nosec B101
        assert board["smart_rule"]["collection_sync_ids"] == [portable_id]  # nosec B101
    finally:
        db.close_all_connections()


def test_v61_scoped_unique_keys_parent_consistency_and_storage_classes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "constraints.sqlite"
    _prepare_v60_product_database(db_path)
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with db.transaction() as conn:
            board = dict(conn.execute("SELECT * FROM moodboards").fetchone())
            placement = dict(conn.execute("SELECT * FROM moodboard_notes").fetchone())
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE moodboards SET deleted=1.5 WHERE id=?", (board["id"],)
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE moodboard_notes SET width=1.5 WHERE placement_id=?",
                    (placement["placement_id"],),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE note_studio_documents SET version=1.5 WHERE note_id=?",
                    (STUDIO_NOTE_ID,),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE moodboard_notes SET owner_user_id=? WHERE placement_id=?",
                    (OTHER_OWNER, placement["placement_id"]),
                )
    finally:
        db.close_all_connections()


@pytest.mark.parametrize(
    "invalid_sync_id",
    (
        "-1111111-1111-4111-8111-111111111111",
        "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
        "aaaaaaaa-aaaa-3aaa-8aaa-aaaaaaaaaaaa",
        "aaaaaaaa-aaaa-4aaa-7aaa-aaaaaaaaaaaa",
        "gaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    ),
)
def test_v61_sqlite_moodboard_sync_id_check_is_exact_uuidv4(
    tmp_path: Path,
    invalid_sync_id: str,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "uuid-check.sqlite"), client_id=OWNER)
    try:
        board_id = db.add_moodboard("UUID check")
        assert board_id is not None  # nosec B101
        with db.transaction() as conn:
            conn.execute(
                "UPDATE moodboards SET sync_id=? WHERE id=?",
                ("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", board_id),
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "UPDATE moodboards SET sync_id=? WHERE id=?",
                    (invalid_sync_id, board_id),
                )
    finally:
        db.close_all_connections()


def test_v61_legacy_diagnostic_is_bounded_and_privacy_safe(tmp_path: Path) -> None:
    db_path = tmp_path / "diagnostic.sqlite"
    _prepare_v60_product_database(db_path)
    secret = "secret-user-content-that-must-not-be-diagnostic"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE moodboards SET smart_rule_json=?", (json.dumps({"unknown": secret}),)
        )

    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT source_diagnostic_code,source_diagnostic_hash FROM moodboards"
            ).fetchone()
        assert row["source_diagnostic_code"] == "legacy_moodboard_rule_invalid"  # nosec B101
        assert len(row["source_diagnostic_code"]) <= 64  # nosec B101
        assert str(row["source_diagnostic_hash"]).startswith("sha256:")  # nosec B101
        assert secret not in str(row)  # nosec B101
        with pytest.raises(ConflictError, match="readiness proof"):
            db.bind_local_moodboard_graph_to_dataset(
                owner_user_id=OWNER,
                target_dataset_id=DATASET_A,
            )
    finally:
        db.close_all_connections()


@pytest.mark.parametrize(
    "stage",
    ("create", "copy", "copy_verify", "index", "verify", "version"),
)
def test_v61_injected_failure_rolls_back_complete_v60_catalog_data_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    db_path = tmp_path / f"rollback-{stage}.sqlite"
    _prepare_v60_product_database(db_path)
    before = _database_dump(db_path)

    def fail_at_stage(self: CharactersRAGDB, current_stage: str) -> None:
        if current_stage == stage:
            raise SchemaError(f"injected v61 {stage} failure")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_notes_moodboard_studio_v61_migration_checkpoint",
        fail_at_stage,
        raising=False,
    )
    with pytest.raises(CharactersRAGDBError, match=f"injected v61 {stage} failure"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    assert _schema_version(db_path) == 60  # nosec B101
    assert _database_dump(db_path) == before  # nosec B101


@pytest.mark.parametrize(
    "corrupt_copy",
    (
        "DELETE FROM note_studio_documents_v61",
        "UPDATE moodboards_v61 SET description='corrupted-after-copy'",
    ),
)
def test_v61_rejects_post_copy_data_loss_or_corruption_and_restores_exact_v60(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_copy: str,
) -> None:
    db_path = tmp_path / "copy-proof-corruption.sqlite"
    _prepare_v60_product_database(db_path)
    before = _database_dump(db_path)
    original_copy = CharactersRAGDB._copy_notes_moodboard_studio_graph_v61_sqlite

    def copy_then_corrupt(
        self: CharactersRAGDB,
        conn: sqlite3.Connection,
    ) -> object:
        proof = original_copy(self, conn)
        conn.execute(corrupt_copy)
        return proof

    monkeypatch.setattr(
        CharactersRAGDB,
        "_copy_notes_moodboard_studio_graph_v61_sqlite",
        copy_then_corrupt,
    )

    with pytest.raises(CharactersRAGDBError, match="copy verification"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    assert _schema_version(db_path) == 60  # nosec B101
    assert _database_dump(db_path) == before  # nosec B101


def test_v61_rejects_ambiguous_placement_owner_without_any_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "ambiguous-owner.sqlite"
    _prepare_v60_product_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE moodboards SET client_id=?", (OTHER_OWNER,)
        )
    before = _database_dump(db_path)

    with pytest.raises(CharactersRAGDBError, match="placement owner"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    assert _schema_version(db_path) == 60  # nosec B101
    assert _database_dump(db_path) == before  # nosec B101


def test_v61_rejects_duplicate_generated_portable_ids_without_any_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "duplicate-sync-id.sqlite"
    _prepare_v60_product_database(db_path, board_count=2)
    before = _database_dump(db_path)

    import tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB as chacha_module

    class _FixedUuid:
        def __str__(self) -> str:
            return "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    monkeypatch.setattr(chacha_module.uuid, "uuid4", lambda: _FixedUuid())
    with pytest.raises(CharactersRAGDBError, match="portable moodboard identity"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    assert _schema_version(db_path) == 60  # nosec B101
    assert _database_dump(db_path) == before  # nosec B101


def test_v61_rejects_task_authority_mismatch_before_blessing_existing_row(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "task-authority-mismatch.sqlite"
    original = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = 60
    try:
        db = CharactersRAGDB(str(db_path), client_id=OWNER)
        note_id = db.add_note("Task parent", "Body")
        now = db._get_current_utc_timestamp_iso()
        task_hash = db._note_task_v60_hash({"id": "task-1", "dataset_id": DATASET_A})
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO note_tasks(
                  owner_user_id,dataset_id,id,note_id,text,status,metadata_json,
                  projection_status,deleted,created_at,updated_at,completed_at,
                  client_id,version,canonical_revision,canonical_hash,
                  source_diagnostic_code,source_diagnostic_hash
                ) VALUES (?,?,?,?,?,'open','{}','live',0,?,?,NULL,?,1,1,?,NULL,NULL)
                """,
                (OWNER, DATASET_A, "task-1", note_id, "Task", now, now, OWNER, task_hash),
            )
            conn.execute(
                "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
                (OWNER, DATASET_B),
            )
        db.close_all_connections()
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original
    before = _database_dump(db_path)

    with pytest.raises(CharactersRAGDBError, match="task graph.*authority"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    assert _schema_version(db_path) == 60  # nosec B101
    assert _database_dump(db_path) == before  # nosec B101


def test_v61_rejects_nonlocal_task_graph_without_an_authority_row(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "task-without-authority.sqlite"
    _prepare_v60_product_database(db_path)
    with sqlite3.connect(db_path) as conn:
        now = "2026-08-25T00:00:00Z"
        conn.execute(
            """
            INSERT INTO note_tasks(
              owner_user_id,dataset_id,id,note_id,text,status,metadata_json,
              projection_status,deleted,created_at,updated_at,completed_at,
              client_id,version,canonical_revision,canonical_hash,
              source_diagnostic_code,source_diagnostic_hash
            ) VALUES (?,?,?,?,?,'open','{}','live',0,?,?,NULL,?,1,1,?,NULL,NULL)
            """,
            (
                OWNER,
                DATASET_A,
                "task-without-authority",
                BOARD_NOTE_ID,
                "Task",
                now,
                now,
                OWNER,
                "sha256:" + ("1" * 64),
            ),
        )
    before = _database_dump(db_path)

    with pytest.raises(CharactersRAGDBError, match="task graph.*authority"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    assert _schema_version(db_path) == 60  # nosec B101
    assert _database_dump(db_path) == before  # nosec B101


def test_row_presence_with_task_flag_false_never_implies_task_binding(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "task-unbound.sqlite"), client_id=OWNER)
    try:
        db.execute_query(
            "INSERT INTO note_task_scope_authority("
            "owner_user_id,dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound"
            ") VALUES (?,?,?,?,?)",
            (OWNER, DATASET_A, 0, 1, 0),
        )
        assert db.resolve_task_compatibility_dataset_id(owner_user_id=OWNER) == LOCAL_UNBOUND  # nosec B101
        counts = db.bind_local_task_graph_to_dataset(
            owner_user_id=OWNER,
            target_dataset_id=DATASET_A,
        )
        assert all(value == 0 for value in counts.values())  # nosec B101
        assert db.resolve_task_compatibility_dataset_id(owner_user_id=OWNER) == DATASET_A  # nosec B101
        with db.transaction() as conn:
            row = conn.execute(
                "SELECT task_graph_bound,moodboard_graph_bound,studio_graph_bound "
                "FROM note_task_scope_authority WHERE owner_user_id=?",
                (OWNER,),
            ).fetchone()
        assert tuple(row) == (1, 1, 0)  # nosec B101
    finally:
        db.close_all_connections()


def test_every_new_binder_insert_supplies_all_graph_flags_explicitly() -> None:
    for method in (
        TaskStore.bind_local_task_graph_to_dataset,
        MoodboardSyncStore._bind_graph,
    ):
        source = inspect.getsource(method)
        assert "task_graph_bound,moodboard_graph_bound,studio_graph_bound" in source


def test_empty_moodboard_and_studio_graph_binding_is_transactional_and_immutable(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "empty-bind.sqlite"), client_id=OWNER)
    try:
        assert db.bind_local_moodboard_graph_to_dataset(
            owner_user_id=OWNER, target_dataset_id=DATASET_A
        ) == {"moodboards": 0, "moodboard_notes": 0}
        assert db.bind_local_studio_graph_to_dataset(
            owner_user_id=OWNER, target_dataset_id=DATASET_A
        ) == {"note_studio_documents": 0}
        assert db.resolve_moodboard_compatibility_dataset_id(owner_user_id=OWNER) == DATASET_A  # nosec B101
        assert db.resolve_studio_compatibility_dataset_id(owner_user_id=OWNER) == DATASET_A  # nosec B101
        with pytest.raises(ConflictError, match="immutable"):
            db.bind_local_studio_graph_to_dataset(
                owner_user_id=OWNER, target_dataset_id=DATASET_B
            )
    finally:
        db.close_all_connections()


def test_bootstrap_pages_require_the_exact_bound_nonlocal_graph_scope(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "bootstrap-scope.sqlite"), client_id=OWNER)
    try:
        board_id = db.add_moodboard("Local board")
        assert board_id is not None  # nosec B101
        with pytest.raises(InputError, match="non-sentinel"):
            db.page_moodboards_for_sync_bootstrap(
                owner_user_id=OWNER,
                dataset_id=LOCAL_UNBOUND,
            )
        with pytest.raises(ConflictError, match="not bound"):
            db.page_moodboards_for_sync_bootstrap(
                owner_user_id=OWNER,
                dataset_id=DATASET_A,
            )

        db.bind_local_moodboard_graph_to_dataset(
            owner_user_id=OWNER,
            target_dataset_id=DATASET_A,
        )
        rows = db.page_moodboards_for_sync_bootstrap(
            owner_user_id=OWNER,
            dataset_id=DATASET_A,
        )
        assert [row["id"] for row in rows] == [board_id]  # nosec B101
        with pytest.raises(ConflictError, match="not bound"):
            db.page_moodboards_for_sync_bootstrap(
                owner_user_id=OWNER,
                dataset_id=DATASET_B,
            )
    finally:
        db.close_all_connections()


def test_same_dataset_moodboard_replay_reproves_bound_canonical_state(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "moodboard-replay-proof.sqlite"), client_id=OWNER)
    try:
        board_id = db.add_moodboard("Replay board")
        assert board_id is not None  # nosec B101
        db.bind_local_moodboard_graph_to_dataset(
            owner_user_id=OWNER,
            target_dataset_id=DATASET_A,
        )
        db.execute_query(
            "UPDATE moodboards SET canonical_hash=? WHERE id=?",
            ("sha256:" + "0" * 64, board_id),
        )

        with pytest.raises(ConflictError, match="canonical readiness proof"):
            db.bind_local_moodboard_graph_to_dataset(
                owner_user_id=OWNER,
                target_dataset_id=DATASET_A,
            )
    finally:
        db.close_all_connections()


def test_same_dataset_studio_replay_reproves_bound_canonical_state(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "studio-replay-proof.sqlite"), client_id=OWNER)
    try:
        note_id = db.add_note("Companion", "accepted markdown")
        assert note_id is not None  # nosec B101
        db.create_note_studio_document(
            note_id=note_id,
            payload_json={"sections": []},
            template_type="lined",
            handwriting_mode="accented",
            render_version=1,
        )
        db.bind_local_studio_graph_to_dataset(
            owner_user_id=OWNER,
            target_dataset_id=DATASET_A,
        )
        db.execute_query(
            "UPDATE note_studio_documents SET canonical_hash=? WHERE note_id=?",
            ("sha256:" + "0" * 64, note_id),
        )

        with pytest.raises(ConflictError, match="canonical readiness proof"):
            db.bind_local_studio_graph_to_dataset(
                owner_user_id=OWNER,
                target_dataset_id=DATASET_A,
            )
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("mutated_note", ("companion", "source"))
def test_studio_binder_reproves_current_note_and_source_heads(
    tmp_path: Path,
    mutated_note: str,
) -> None:
    db = CharactersRAGDB(str(tmp_path / f"studio-head-{mutated_note}.sqlite"), client_id=OWNER)
    try:
        source_id = db.add_note("Source", "source excerpt remains present")
        companion_id = db.add_note("Companion", "accepted markdown")
        assert source_id is not None and companion_id is not None  # nosec B101
        db.create_note_studio_document(
            note_id=companion_id,
            payload_json={
                "sections": [
                    {
                        "id": "notes-1",
                        "kind": "notes",
                        "title": "Notes",
                        "content": "accepted markdown",
                    }
                ]
            },
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=source_id,
            excerpt_snapshot="source excerpt",
            excerpt_hash="sha256:" + "0" * 64,
            companion_content_hash="sha256:" + "1" * 64,
            render_version=1,
        )
        changed_id = companion_id if mutated_note == "companion" else source_id
        changed_content = (
            "changed accepted markdown"
            if mutated_note == "companion"
            else "changed source excerpt remains present"
        )
        assert db.update_note(  # nosec B101
            changed_id,
            {"content": changed_content},
            expected_version=1,
        )

        with pytest.raises(ConflictError, match="canonical readiness proof"):
            db.bind_local_studio_graph_to_dataset(
                owner_user_id=OWNER,
                target_dataset_id=DATASET_A,
            )
        with db.transaction() as conn:
            authority = conn.execute(
                "SELECT * FROM note_task_scope_authority WHERE owner_user_id=?",
                (OWNER,),
            ).fetchone()
            scope = conn.execute(
                "SELECT dataset_id FROM note_studio_documents WHERE note_id=?",
                (companion_id,),
            ).fetchone()[0]
        assert authority is None  # nosec B101
        assert scope == LOCAL_UNBOUND  # nosec B101
    finally:
        db.close_all_connections()


def test_deleted_parent_studio_tombstone_binds_bootstraps_and_restores_in_place(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "studio-lifecycle-bootstrap.sqlite"), client_id=OWNER)
    try:
        note_id = db.add_note("Companion", "accepted markdown")
        assert note_id is not None  # nosec B101
        before = db.create_note_studio_document(
            note_id=note_id,
            payload_json={"sections": []},
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=None,
            excerpt_snapshot=None,
            excerpt_hash=None,
            companion_content_hash="sha256:" + "0" * 64,
            render_version=1,
        )
        assert db.soft_delete_note(note_id, expected_version=1)  # nosec B101

        assert db.bind_local_studio_graph_to_dataset(  # nosec B101
            owner_user_id=OWNER,
            target_dataset_id=DATASET_A,
        ) == {"note_studio_documents": 1}
        tombstone = db.page_studio_documents_for_sync_bootstrap(
            owner_user_id=OWNER,
            dataset_id=DATASET_A,
        )[0]
        assert tombstone["deleted"] == 1  # nosec B101
        assert json.loads(tombstone["payload_json"]) == before["payload_json"]  # nosec B101
        assert json.loads(tombstone["accepted_provenance_json"]) == before["accepted_provenance_json"]  # nosec B101

        assert db.restore_note(note_id, expected_version=2)  # nosec B101
        restored = db.page_studio_documents_for_sync_bootstrap(
            owner_user_id=OWNER,
            dataset_id=DATASET_A,
        )[0]
        assert restored["deleted"] == 0  # nosec B101
        assert json.loads(restored["payload_json"]) == before["payload_json"]  # nosec B101
        assert json.loads(restored["accepted_provenance_json"]) == before["accepted_provenance_json"]  # nosec B101
        assert restored["canonical_revision"] == tombstone["canonical_revision"] + 1  # nosec B101
        assert restored["canonical_hash"] != tombstone["canonical_hash"]  # nosec B101
    finally:
        db.close_all_connections()


def test_interleaved_first_enrollment_has_one_dataset_winner_without_partial_scope(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "first-enrollment-race.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    board_id = db.add_moodboard("Race board")
    assert board_id is not None  # nosec B101
    db.close_all_connections()
    barrier = threading.Barrier(2)

    def bind(target: str) -> tuple[str, object]:
        contender = CharactersRAGDB(str(db_path), client_id=OWNER)
        try:
            barrier.wait(timeout=10)
            return (
                "ok",
                contender.bind_local_moodboard_graph_to_dataset(
                    owner_user_id=OWNER,
                    target_dataset_id=target,
                ),
            )
        except ConflictError as exc:
            return ("conflict", str(exc))
        finally:
            contender.close_all_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(bind, (DATASET_A, DATASET_B)))

    assert sorted(kind for kind, _result in outcomes) == ["conflict", "ok"]  # nosec B101
    reopened = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with reopened.transaction() as conn:
            authority = conn.execute(
                "SELECT dataset_id,moodboard_graph_bound FROM note_task_scope_authority "
                "WHERE owner_user_id=?",
                (OWNER,),
            ).fetchone()
            board_scopes = {
                str(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT dataset_id FROM moodboards WHERE owner_user_id=?",
                    (OWNER,),
                )
            }
        assert authority["moodboard_graph_bound"] == 1  # nosec B101
        assert board_scopes == {authority["dataset_id"]}  # nosec B101
    finally:
        reopened.close_all_connections()


def test_current_v61_studio_helper_verifies_drift_instead_of_repairing(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "studio-drift.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    db.close_all_connections()
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_note_studio_documents_scope_source")

    with pytest.raises(CharactersRAGDBError, match="Studio.*catalog drifted"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    source = inspect.getsource(CharactersRAGDB._ensure_note_studio_schema_sqlite)
    assert "CREATE TABLE IF NOT EXISTS note_studio_documents" not in source
    assert "_verify_notes_moodboard_studio_schema_sqlite" in source


def test_postgres_v60_keeps_legacy_product_and_authority_sql_without_v61_columns() -> None:
    statements: list[str] = []

    class _Cursor:
        def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
            self._rows = rows or []

        def fetchone(self) -> dict[str, object] | None:
            return self._rows[0] if self._rows else None

        def fetchall(self) -> list[dict[str, object]]:
            return list(self._rows)

    class _Backend:
        backend_type = BackendType.POSTGRESQL

        def execute(
            self,
            statement: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object | None = None,
        ) -> _Cursor:
            del params, connection
            statements.append(statement)
            return _Cursor()

    db = object.__new__(CharactersRAGDB)
    db._backend = _Backend()
    db._uses_shared_content_backend = False
    db._local = threading.local()
    db.client_id = OWNER
    db._get_schema_version_postgres = lambda _conn: 60
    db._ensure_note_studio_schema_postgres(object())
    assert any("CREATE TABLE IF NOT EXISTS note_studio_documents" in sql for sql in statements)  # nosec B101
    assert all("owner_user_id" not in sql and "dataset_id" not in sql for sql in statements)  # nosec B101

    queries: list[str] = []

    def execute_query(query: str, params: tuple[object, ...] = ()) -> _Cursor:
        del params
        queries.append(query)
        if "note_task_scope_authority" in query:
            return _Cursor([{"owner_user_id": OWNER, "dataset_id": DATASET_A}])
        return _Cursor()

    db.execute_query = execute_query
    db.resolve_moodboard_compatibility_dataset_id = lambda **_kwargs: pytest.fail(
        "PostgreSQL v60 moodboard reads must not resolve v61 scope flags"
    )
    db.resolve_studio_compatibility_dataset_id = lambda **_kwargs: pytest.fail(
        "PostgreSQL v60 Studio reads must not resolve v61 scope flags"
    )

    assert db.get_moodboard_by_id(1) is None  # nosec B101
    assert NoteStore(db).get_note_studio_document(STUDIO_NOTE_ID) is None  # nosec B101
    assert TaskStore(db).resolve_task_compatibility_dataset_id(
        owner_user_id=OWNER
    ) == DATASET_A
    issued = "\n".join(queries)
    assert "task_graph_bound" not in issued  # nosec B101
    assert "moodboard_graph_bound" not in issued  # nosec B101
    assert "studio_graph_bound" not in issued  # nosec B101
    assert "owner_user_id=? AND dataset_id=?" not in issued  # nosec B101
