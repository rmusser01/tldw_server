"""Imported collection content must come from ordinary SQLite tables."""

import io
import sqlite3
import zipfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.sqlite_schema_helpers import ordinary_sqlite_table_names
from tldw_Server_API.app.core.Flashcards.apkg_exporter import export_apkg_from_rows
from tldw_Server_API.app.core.Flashcards.apkg_importer import APKGImportError, import_rows_from_apkg_bytes

pytestmark = pytest.mark.unit


def _apkg_with_replaced_table(tmp_path: Path, table: str, kind: str) -> bytes:
    """Replace one real exported table using only valid, harmless SQLite SQL."""
    original = export_apkg_from_rows([{"front": "Question", "back": "Answer", "model_type": "basic"}])
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    path = tmp_path / "collection.anki2"
    path.write_bytes(members["collection.anki2"])
    with sqlite3.connect(path) as conn:
        # Names are fixed by the parametrized test, never supplied externally.
        conn.execute(f'ALTER TABLE "{table}" RENAME TO actual_table')
        if kind == "view":
            conn.execute(f'CREATE VIEW "{table}" AS SELECT * FROM actual_table')  # nosec B608: fixed test names
        elif kind == "virtual":
            columns = [row[1] for row in conn.execute("PRAGMA table_info(actual_table)")]
            names = ", ".join(f'"{name}"' for name in columns)
            conn.execute(f'CREATE VIRTUAL TABLE "{table}" USING fts5({names})')
            conn.execute(f'INSERT INTO "{table}" SELECT * FROM actual_table')  # nosec B608: fixed test names
        elif kind in {"upper", "title"}:
            renamed = table.upper() if kind == "upper" else table.title()
            conn.execute(f'ALTER TABLE actual_table RENAME TO "{renamed}"')
    members["collection.anki2"] = path.read_bytes()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


@pytest.mark.parametrize("table", ["col", "notes", "cards"])
@pytest.mark.parametrize("kind", ["upper", "title"])
def test_ordinary_table_names_remain_case_insensitive(tmp_path: Path, table: str, kind: str) -> None:
    apkg = _apkg_with_replaced_table(tmp_path, table, kind)
    rows, errors = import_rows_from_apkg_bytes(apkg, max_notes=10, max_field_length=8192)
    assert not errors and len(rows) == 1 and rows[0]["front"] == "Question"


@pytest.mark.parametrize("table", ["col", "notes", "cards"])
@pytest.mark.parametrize("kind", ["view", "virtual", "missing"])
def test_apkg_requires_ordinary_source_tables(tmp_path: Path, table: str, kind: str) -> None:
    apkg = _apkg_with_replaced_table(tmp_path, table, kind)
    with pytest.raises(APKGImportError, match=f"required ordinary table: {table}"):
        import_rows_from_apkg_bytes(apkg, max_notes=10, max_field_length=8192)


def test_rejected_schema_does_not_read_imported_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    apkg = _apkg_with_replaced_table(tmp_path, "cards", "view")
    original_connect = sqlite3.connect
    statements: list[str] = []

    def traced_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    with pytest.raises(APKGImportError, match="required ordinary table: cards"):
        import_rows_from_apkg_bytes(apkg, max_notes=10, max_field_length=8192)
    assert not any(sql.lstrip().upper().startswith("SELECT") for sql in statements)


@pytest.mark.parametrize("row_factory", [None, sqlite3.Row])
def test_table_metadata_distinguishes_ordinary_variants_and_shadow_tables(row_factory) -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = row_factory
        conn.executescript(
            """
            CREATE TABLE normal(id INTEGER PRIMARY KEY);
            CREATE TABLE strict_table(id INTEGER PRIMARY KEY) STRICT;
            CREATE TABLE no_rowid(id INTEGER PRIMARY KEY) WITHOUT ROWID;
            CREATE VIRTUAL TABLE indexed_text USING fts5(body);
            CREATE VIEW projected AS SELECT * FROM normal;
            CREATE TEMP TABLE temporary_table(id INTEGER);
            """
        )
        assert ordinary_sqlite_table_names(conn) == {"normal", "strict_table", "no_rowid", "sqlite_schema"}


def test_apkg_fails_closed_without_native_table_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    apkg = export_apkg_from_rows([{"front": "Question", "back": "Answer"}])
    original_connect = sqlite3.connect

    class OldMetadataConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: tuple = ()) -> sqlite3.Cursor:
            if sql == "PRAGMA main.table_list":
                return super().execute("SELECT 1 WHERE 0")
            return super().execute(sql, parameters)

    def old_connect(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=OldMetadataConnection)

    monkeypatch.setattr(sqlite3, "connect", old_connect)
    with pytest.raises(APKGImportError, match="SQLite 3.37 or newer"):
        import_rows_from_apkg_bytes(apkg, max_notes=10, max_field_length=8192)
