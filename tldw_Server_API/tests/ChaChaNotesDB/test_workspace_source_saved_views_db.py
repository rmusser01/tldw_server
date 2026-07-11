from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)

OWNER_A = "owner-a"
OWNER_B = "owner-b"
WORKSPACE_A = "workspace-a"
WORKSPACE_B = "workspace-b"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "saved-views.db"


@pytest.fixture
def db(db_path: Path) -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(db_path=db_path, client_id=OWNER_A)
    database.upsert_workspace(WORKSPACE_A, "Workspace A")
    try:
        yield database
    finally:
        database.close_all_connections()


def _create(
    database: CharactersRAGDB,
    *,
    owner: str = OWNER_A,
    workspace_id: str = WORKSPACE_A,
    name: str = "My view",
    schema_version: int = 1,
    state_json: str = '{"schema_version":1}',
) -> dict[str, Any]:
    return database.create_workspace_source_saved_view(
        owner,
        workspace_id,
        name=name,
        schema_version=schema_version,
        state_json=state_json,
    )


def _assert_saved_view_error(
    exc_info: pytest.ExceptionInfo[CharactersRAGDBError],
    code: str,
    metadata: dict[str, Any],
) -> None:
    assert exc_info.value.code == code
    assert exc_info.value.metadata == metadata


def test_saved_view_schema_has_portable_keys_named_uniqueness_and_cascade(
    db: CharactersRAGDB,
) -> None:
    row = db.execute_query(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("workspace_source_saved_views",),
    ).fetchone()

    assert row is not None
    normalized_sql = " ".join(row["sql"].split()).lower()
    assert "primary key (workspace_id, id)" in normalized_sql
    assert "constraint uq_workspace_source_saved_views_owner_name" in normalized_sql
    assert "unique (owner_user_id, workspace_id, name_key)" in normalized_sql
    assert "foreign key (workspace_id) references workspaces(id) on delete cascade" in normalized_sql


def test_create_list_get_update_and_delete_round_trip_raw_state(db: CharactersRAGDB) -> None:
    created = _create(db, name="  Reading queue  ", state_json="not-json-yet")

    assert created["id"]
    assert created["workspace_id"] == WORKSPACE_A
    assert created["owner_user_id"] == OWNER_A
    assert created["name"] == "Reading queue"
    assert created["name_key"] == "reading queue"
    assert created["schema_version"] == 1
    assert created["state_json"] == "not-json-yet"
    assert created["version"] == 1
    assert db.get_workspace_source_saved_view(OWNER_A, WORKSPACE_A, created["id"]) == created
    assert db.list_workspace_source_saved_views(OWNER_A, WORKSPACE_A) == [created]

    updated = db.update_workspace_source_saved_view(
        OWNER_A,
        WORKSPACE_A,
        created["id"],
        expected_version=1,
        name="Renamed",
        schema_version=7,
        state_json='{"future":true}',
    )

    assert updated["name"] == "Renamed"
    assert updated["schema_version"] == 7
    assert updated["state_json"] == '{"future":true}'
    assert updated["version"] == 2
    assert updated["created_at"] == created["created_at"]
    db.delete_workspace_source_saved_view(OWNER_A, WORKSPACE_A, created["id"])
    assert db.list_workspace_source_saved_views(OWNER_A, WORKSPACE_A) == []


def test_list_order_is_updated_desc_then_name_key_then_id(db: CharactersRAGDB) -> None:
    older = _create(db, name="Zulu")
    alpha_b = _create(db, name="alpha b")
    alpha_a = _create(db, name="Alpha A")
    tied = sorted((alpha_a, alpha_b), key=lambda row: (row["name_key"], row["id"]))
    db.execute_query(
        "UPDATE workspace_source_saved_views SET updated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00.000Z", older["id"]),
        commit=True,
    )
    db.execute_query(
        "UPDATE workspace_source_saved_views SET updated_at = ? WHERE id IN (?, ?)",
        ("2026-01-02T00:00:00.000Z", alpha_a["id"], alpha_b["id"]),
        commit=True,
    )

    listed = db.list_workspace_source_saved_views(OWNER_A, WORKSPACE_A)

    assert [row["id"] for row in listed] == [tied[0]["id"], tied[1]["id"], older["id"]]


@pytest.mark.parametrize("name", ["", "   ", "x" * 121])
def test_name_must_trim_to_between_one_and_120_characters(
    db: CharactersRAGDB,
    name: str,
) -> None:
    with pytest.raises(InputError, match="name"):
        _create(db, name=name)


def test_name_key_uses_python_nfkc_and_casefold(db: CharactersRAGDB) -> None:
    created = _create(db, name="  Stra\u00dfe \uff21  ")

    assert created["name"] == "Stra\u00dfe \uff21"
    assert created["name_key"] == "strasse a"
    with pytest.raises(CharactersRAGDBError) as exc_info:
        _create(db, name="STRASSE A")
    _assert_saved_view_error(
        exc_info,
        "source_view_name_exists",
        {"view_id": created["id"], "version": 1},
    )


def test_duplicate_update_reports_owned_conflicting_view_and_version(db: CharactersRAGDB) -> None:
    first = _create(db, name="First")
    second = _create(db, name="Second")
    second = db.update_workspace_source_saved_view(
        OWNER_A,
        WORKSPACE_A,
        second["id"],
        expected_version=1,
        state_json="updated raw state",
        schema_version=2,
    )

    with pytest.raises(CharactersRAGDBError) as exc_info:
        db.update_workspace_source_saved_view(
            OWNER_A,
            WORKSPACE_A,
            first["id"],
            expected_version=1,
            name="SECOND",
        )

    _assert_saved_view_error(
        exc_info,
        "source_view_name_exists",
        {"view_id": second["id"], "version": 2},
    )


def test_every_operation_requires_active_workspace_owned_by_supplied_owner(
    db_path: Path,
) -> None:
    owner_a = CharactersRAGDB(db_path=db_path, client_id=OWNER_A)
    owner_b = CharactersRAGDB(db_path=db_path, client_id=OWNER_B)
    owner_a.upsert_workspace(WORKSPACE_A, "Workspace A")
    owner_b.upsert_workspace(WORKSPACE_B, "Workspace B")
    view = _create(owner_a)
    operations: list[Callable[[], object]] = [
        lambda: owner_b.list_workspace_source_saved_views(OWNER_B, WORKSPACE_A),
        lambda: owner_b.get_workspace_source_saved_view(OWNER_B, WORKSPACE_A, view["id"]),
        lambda: _create(owner_b, owner=OWNER_B, workspace_id=WORKSPACE_A, name="Probe"),
        lambda: owner_b.update_workspace_source_saved_view(
            OWNER_B,
            WORKSPACE_A,
            view["id"],
            expected_version=1,
            name="Probe",
        ),
        lambda: owner_b.delete_workspace_source_saved_view(OWNER_B, WORKSPACE_A, view["id"]),
    ]

    try:
        for operation in operations:
            with pytest.raises(CharactersRAGDBError) as exc_info:
                operation()
            _assert_saved_view_error(exc_info, "source_view_not_found", {})
        assert owner_a.get_workspace_source_saved_view(OWNER_A, WORKSPACE_A, view["id"]) == view
        assert owner_b.list_workspace_source_saved_views(OWNER_B, WORKSPACE_B) == []
    finally:
        owner_a.close_all_connections()
        owner_b.close_all_connections()


def test_limit_is_100_per_owner_and_workspace(db: CharactersRAGDB) -> None:
    now = "2026-01-01T00:00:00.000Z"
    rows = [
        (
            str(uuid4()),
            WORKSPACE_A,
            OWNER_A,
            f"View {index}",
            f"view {index}",
            1,
            "{}",
            1,
            now,
            now,
        )
        for index in range(100)
    ]
    db.execute_many(
        """
        INSERT INTO workspace_source_saved_views (
            id, workspace_id, owner_user_id, name, name_key, schema_version,
            state_json, version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
        commit=True,
    )

    with pytest.raises(CharactersRAGDBError) as exc_info:
        _create(db, name="One too many")

    _assert_saved_view_error(exc_info, "source_view_limit_reached", {"limit": 100})


def test_state_json_limit_uses_utf8_bytes(db: CharactersRAGDB) -> None:
    accepted = "\u00e9" * (16 * 1024 // 2)
    created = _create(db, name="At limit", state_json=accepted)

    assert created["state_json"] == accepted
    with pytest.raises(InputError, match="16384"):
        _create(db, name="Over limit", state_json=accepted + "x")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "nul\x00name"),
        ("name", "surrogate\ud800name"),
        ("state_json", "nul\x00state"),
        ("state_json", "surrogate\ud800state"),
    ],
)
def test_saved_view_text_rejects_nul_and_non_utf8_strings(
    db: CharactersRAGDB,
    field: str,
    value: str,
) -> None:
    kwargs = {field: value}

    with pytest.raises(InputError, match=field):
        _create(db, **kwargs)


def test_saved_view_state_accepts_escaped_json_nul(db: CharactersRAGDB) -> None:
    escaped_nul = r'{"value":"\u0000"}'

    created = _create(db, state_json=escaped_nul)

    assert created["state_json"] == escaped_nul


@pytest.mark.parametrize("value", [True, 0, -1, 2_147_483_648, 1.5])
def test_saved_view_schema_version_rejects_nonportable_values(
    db: CharactersRAGDB,
    value: object,
) -> None:
    with pytest.raises(InputError, match="schema_version"):
        _create(db, schema_version=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, 0, -1, 2_147_483_648, 1.5])
def test_saved_view_expected_version_rejects_nonportable_values(
    db: CharactersRAGDB,
    value: object,
) -> None:
    created = _create(db)

    with pytest.raises(InputError, match="expected_version"):
        db.update_workspace_source_saved_view(
            OWNER_A,
            WORKSPACE_A,
            created["id"],
            expected_version=value,  # type: ignore[arg-type]
            name="Updated",
        )


def test_saved_view_integer_validators_accept_postgres_integer_max(
    db: CharactersRAGDB,
) -> None:
    maximum = 2_147_483_647

    created = _create(db, schema_version=maximum)

    assert created["schema_version"] == maximum
    assert db._validate_workspace_source_saved_view_expected_version(maximum) == maximum


def test_stale_update_reports_only_view_id_and_current_version(db: CharactersRAGDB) -> None:
    created = _create(db, state_json="private raw state")
    current = db.update_workspace_source_saved_view(
        OWNER_A,
        WORKSPACE_A,
        created["id"],
        expected_version=1,
        state_json="new private raw state",
        schema_version=2,
    )

    with pytest.raises(CharactersRAGDBError) as exc_info:
        db.update_workspace_source_saved_view(
            OWNER_A,
            WORKSPACE_A,
            created["id"],
            expected_version=1,
            name="Stale",
        )

    _assert_saved_view_error(
        exc_info,
        "source_view_version_conflict",
        {"view_id": created["id"], "current_version": current["version"]},
    )
    assert "private" not in str(exc_info.value)


def test_repeated_unowned_and_missing_deletes_do_not_leak_metadata(db: CharactersRAGDB) -> None:
    created = _create(db)
    db.delete_workspace_source_saved_view(OWNER_A, WORKSPACE_A, created["id"])

    for owner, view_id in (
        (OWNER_A, created["id"]),
        (OWNER_A, str(uuid4())),
        (OWNER_B, created["id"]),
    ):
        with pytest.raises(CharactersRAGDBError) as exc_info:
            db.delete_workspace_source_saved_view(owner, WORKSPACE_A, view_id)
        _assert_saved_view_error(exc_info, "source_view_not_found", {})


def test_soft_deleted_workspace_retains_rows_but_hides_all_saved_view_methods(
    db: CharactersRAGDB,
) -> None:
    created = _create(db)
    workspace = db.get_workspace(WORKSPACE_A)
    assert workspace is not None
    assert db.delete_workspace(WORKSPACE_A, expected_version=workspace["version"]) is True
    operations: list[Callable[[], object]] = [
        lambda: db.list_workspace_source_saved_views(OWNER_A, WORKSPACE_A),
        lambda: db.get_workspace_source_saved_view(OWNER_A, WORKSPACE_A, created["id"]),
        lambda: _create(db, name="After delete"),
        lambda: db.update_workspace_source_saved_view(
            OWNER_A,
            WORKSPACE_A,
            created["id"],
            expected_version=1,
            name="After delete",
        ),
        lambda: db.delete_workspace_source_saved_view(OWNER_A, WORKSPACE_A, created["id"]),
    ]

    for operation in operations:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            operation()
        _assert_saved_view_error(exc_info, "source_view_not_found", {})
    retained = db.execute_query(
        "SELECT COUNT(*) AS total FROM workspace_source_saved_views WHERE workspace_id = ?",
        (WORKSPACE_A,),
    ).fetchone()
    assert retained["total"] == 1


def test_physical_workspace_delete_cascades_saved_views(db: CharactersRAGDB) -> None:
    _create(db)

    db.hard_delete_workspace(WORKSPACE_A)

    row = db.execute_query(
        "SELECT COUNT(*) AS total FROM workspace_source_saved_views WHERE workspace_id = ?",
        (WORKSPACE_A,),
    ).fetchone()
    assert row["total"] == 0


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_saved_view_mutation_then_workspace_soft_delete_serializes(
    db_path: Path,
    operation: str,
) -> None:
    mutation_db = CharactersRAGDB(db_path=db_path, client_id=OWNER_A)
    deletion_db = CharactersRAGDB(db_path=db_path, client_id=OWNER_A)
    mutation_db.upsert_workspace(WORKSPACE_A, "Workspace A")
    existing = _create(mutation_db)
    mutation_finished = threading.Event()
    allow_commit = threading.Event()
    results: dict[str, object] = {}

    def mutate() -> None:
        try:
            with mutation_db.transaction():
                if operation == "create":
                    results["mutation"] = _create(mutation_db, name="Concurrent")
                elif operation == "update":
                    results["mutation"] = mutation_db.update_workspace_source_saved_view(
                        OWNER_A,
                        WORKSPACE_A,
                        existing["id"],
                        expected_version=1,
                        name="Concurrent",
                    )
                else:
                    results["mutation"] = mutation_db.delete_workspace_source_saved_view(
                        OWNER_A,
                        WORKSPACE_A,
                        existing["id"],
                    )
                mutation_finished.set()
                assert allow_commit.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - surfaced below
            results["mutation_error"] = exc

    def soft_delete() -> None:
        assert mutation_finished.wait(timeout=5)
        try:
            workspace = deletion_db.get_workspace(WORKSPACE_A)
            assert workspace is not None
            results["deleted"] = deletion_db.delete_workspace(
                WORKSPACE_A,
                expected_version=workspace["version"],
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - surfaced below
            results["delete_error"] = exc

    mutation_thread = threading.Thread(target=mutate)
    deletion_thread = threading.Thread(target=soft_delete)
    try:
        mutation_thread.start()
        deletion_thread.start()
        assert mutation_finished.wait(timeout=5)
        allow_commit.set()
        mutation_thread.join(timeout=10)
        deletion_thread.join(timeout=10)
        assert not mutation_thread.is_alive()
        assert not deletion_thread.is_alive()
        assert "mutation_error" not in results
        assert "delete_error" not in results
        assert results["deleted"] is True
        with pytest.raises(CharactersRAGDBError) as exc_info:
            mutation_db.list_workspace_source_saved_views(OWNER_A, WORKSPACE_A)
        _assert_saved_view_error(exc_info, "source_view_not_found", {})
    finally:
        allow_commit.set()
        mutation_thread.join(timeout=1)
        deletion_thread.join(timeout=1)
        mutation_db.close_all_connections()
        deletion_db.close_all_connections()


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_workspace_soft_delete_then_saved_view_mutation_fails_after_serialization(
    db_path: Path,
    operation: str,
) -> None:
    deletion_db = CharactersRAGDB(db_path=db_path, client_id=OWNER_A)
    mutation_db = CharactersRAGDB(db_path=db_path, client_id=OWNER_A)
    deletion_db.upsert_workspace(WORKSPACE_A, "Workspace A")
    existing = _create(deletion_db)
    delete_finished = threading.Event()
    allow_commit = threading.Event()
    results: dict[str, object] = {}

    def soft_delete() -> None:
        try:
            with deletion_db.transaction():
                workspace = deletion_db.get_workspace(WORKSPACE_A)
                assert workspace is not None
                results["deleted"] = deletion_db.delete_workspace(
                    WORKSPACE_A,
                    expected_version=workspace["version"],
                )
                delete_finished.set()
                assert allow_commit.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - surfaced below
            results["delete_error"] = exc

    def mutate() -> None:
        assert delete_finished.wait(timeout=5)
        try:
            if operation == "create":
                _create(mutation_db, name="Concurrent")
            elif operation == "update":
                mutation_db.update_workspace_source_saved_view(
                    OWNER_A,
                    WORKSPACE_A,
                    existing["id"],
                    expected_version=1,
                    name="Concurrent",
                )
            else:
                mutation_db.delete_workspace_source_saved_view(
                    OWNER_A,
                    WORKSPACE_A,
                    existing["id"],
                )
        except Exception as exc:  # noqa: BLE001 - thread boundary surfaces the exact error
            results["mutation_error"] = exc

    deletion_thread = threading.Thread(target=soft_delete)
    mutation_thread = threading.Thread(target=mutate)
    try:
        deletion_thread.start()
        mutation_thread.start()
        assert delete_finished.wait(timeout=5)
        allow_commit.set()
        deletion_thread.join(timeout=10)
        mutation_thread.join(timeout=10)
        assert not deletion_thread.is_alive()
        assert not mutation_thread.is_alive()
        assert "delete_error" not in results
        assert results["deleted"] is True
        mutation_error = results.get("mutation_error")
        assert isinstance(mutation_error, CharactersRAGDBError)
        assert mutation_error.code == "source_view_not_found"
        assert mutation_error.metadata == {}
    finally:
        allow_commit.set()
        deletion_thread.join(timeout=1)
        mutation_thread.join(timeout=1)
        deletion_db.close_all_connections()
        mutation_db.close_all_connections()


def test_raw_corrupt_unsupported_and_invalid_v1_rows_remain_retrievable(
    db: CharactersRAGDB,
) -> None:
    now = "2026-01-01T00:00:00.000Z"
    raw_rows = [
        (str(uuid4()), "Corrupt", "corrupt", 1, "{"),
        (str(uuid4()), "Future", "future", 999, '{"future":true}'),
        (str(uuid4()), "Invalid V1", "invalid v1", 1, '{"sort":"not-supported"}'),
    ]
    db.execute_many(
        """
        INSERT INTO workspace_source_saved_views (
            id, workspace_id, owner_user_id, name, name_key, schema_version,
            state_json, version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        [
            (view_id, WORKSPACE_A, OWNER_A, name, name_key, schema_version, state_json, now, now)
            for view_id, name, name_key, schema_version, state_json in raw_rows
        ],
        commit=True,
    )

    listed = db.list_workspace_source_saved_views(OWNER_A, WORKSPACE_A)
    by_id = {row["id"]: row for row in listed}

    for view_id, _name, _key, schema_version, state_json in raw_rows:
        assert by_id[view_id]["schema_version"] == schema_version
        assert by_id[view_id]["state_json"] == state_json
        assert db.get_workspace_source_saved_view(OWNER_A, WORKSPACE_A, view_id)["state_json"] == state_json


def test_real_v53_sqlite_database_migrates_additively_to_v54(db_path: Path) -> None:
    seed = CharactersRAGDB(db_path=db_path, client_id=OWNER_A)
    seed.upsert_workspace(WORKSPACE_A, "Workspace A")
    seed.add_workspace_source(
        WORKSPACE_A,
        {"id": "source-1", "media_id": 42, "title": "Source", "source_type": "pdf"},
    )
    original_table_sql = seed.execute_query(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'workspace_sources'"
    ).fetchone()["sql"]
    with seed.transaction() as conn:
        conn.execute("DROP TABLE workspace_source_saved_views")
        conn.execute(
            "UPDATE db_schema_version SET version = 53 WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        )
    seed.close_all_connections()

    migrated = CharactersRAGDB(db_path=db_path, client_id=OWNER_A)
    try:
        version = migrated.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()["version"]
        migrated_table_sql = migrated.execute_query(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'workspace_sources'"
        ).fetchone()["sql"]
        source = migrated.get_workspace_source(WORKSPACE_A, "source-1")

        assert version == 54
        assert migrated.backend.table_exists("workspace_source_saved_views")
        assert migrated_table_sql == original_table_sql
        assert source is not None and source["title"] == "Source"
    finally:
        migrated.close_all_connections()


def test_named_unique_constraint_is_enforced_by_sqlite(db: CharactersRAGDB) -> None:
    created = _create(db, name="Unique")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO workspace_source_saved_views (
                    id, workspace_id, owner_user_id, name, name_key, schema_version,
                    state_json, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    WORKSPACE_A,
                    OWNER_A,
                    "Unique duplicate",
                    created["name_key"],
                    1,
                    "{}",
                    1,
                    created["created_at"],
                    created["updated_at"],
                ),
            )
