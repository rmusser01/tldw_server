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


def test_apkg_metadata_cannot_evaluate_unrelated_fts_config_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing source-read policy evaluates an unrelated uploaded config view."""
    original = export_apkg_from_rows([{"front": "Question", "back": "Answer", "model_type": "basic"}])
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    path = tmp_path / "collection.anki2"
    path.write_bytes(members["collection.anki2"])
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE VIRTUAL TABLE indexed_text USING fts5(body);
            INSERT INTO indexed_text VALUES('ordinary text');
            CREATE VIRTUAL TABLE auxiliary USING fts5(body);
            ALTER TABLE auxiliary_config RENAME TO saved_config;
            CREATE VIEW auxiliary_config AS
                SELECT k, observed_config(v) AS v FROM saved_config
                WHERE EXISTS(SELECT 1 FROM indexed_text WHERE indexed_text MATCH 'ordinary');
            """
        )
    members["collection.anki2"] = path.read_bytes()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    evaluated: list[int] = []
    original_connect = sqlite3.connect

    def observed_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.create_function("observed_config", 1, lambda value: evaluated.append(value) or value)
        return conn

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    rows, errors = import_rows_from_apkg_bytes(output.getvalue(), max_notes=10, max_field_length=8192)
    assert evaluated == []
    assert not errors and len(rows) == 1 and rows[0]["front"] == "Question"


def test_generic_metadata_helper_preserves_caller_authorizer() -> None:
    """Guarding importer-owned connections must not weaken an existing caller policy."""
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE ordinary(id INTEGER)")

        def deny_rows(action, table, column, database, source):
            if action == sqlite3.SQLITE_READ and table == "ordinary":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(deny_rows)
        assert "ordinary" in ordinary_sqlite_table_names(conn)
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("SELECT id FROM ordinary").fetchall()


def test_apkg_closes_source_when_read_policy_installation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed guard must reject the import and release its owned connection."""
    apkg = export_apkg_from_rows([{"front": "Question", "back": "Answer"}])
    original_connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    class UnavailableAuthorizerConnection(sqlite3.Connection):
        def set_authorizer(self, callback):
            raise sqlite3.DatabaseError("source read policy unavailable")

    def unavailable_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs, factory=UnavailableAuthorizerConnection)
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", unavailable_connect)
    with pytest.raises(APKGImportError):
        import_rows_from_apkg_bytes(apkg, max_notes=10, max_field_length=8192)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


def test_apkg_nested_config_views_cannot_search_fts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An inner view must not run MATCH while the outer view is being prepared."""
    original = export_apkg_from_rows([{"front": "Question", "back": "Answer", "model_type": "basic"}])
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    path = tmp_path / "collection.anki2"
    path.write_bytes(members["collection.anki2"])
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE VIRTUAL TABLE seed USING fts5(body);
            INSERT INTO seed VALUES('ordinary text');
            CREATE VIRTUAL TABLE indexed_text USING fts5(body);
            INSERT INTO indexed_text VALUES('ordinary text');
            ALTER TABLE indexed_text_config RENAME TO saved_indexed_config;
            CREATE VIEW indexed_text_config AS SELECT k, v FROM saved_indexed_config
                WHERE EXISTS(SELECT 1 FROM seed WHERE seed MATCH 'ordinary');
            CREATE VIRTUAL TABLE auxiliary USING fts5(body);
            ALTER TABLE auxiliary_config RENAME TO saved_config;
            CREATE VIEW auxiliary_config AS SELECT k, v FROM saved_config
                WHERE EXISTS(SELECT 1 FROM indexed_text WHERE indexed_text MATCH 'ordinary');
            """
        )
    members["collection.anki2"] = path.read_bytes()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    statements: list[str] = []
    original_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced_connect)
    rows, errors = import_rows_from_apkg_bytes(output.getvalue(), max_notes=10, max_field_length=8192)
    assert not any("seed_idx" in statement for statement in statements)
    assert not errors and len(rows) == 1 and rows[0]["front"] == "Question"


@pytest.mark.parametrize("failure", ["error", "ignored", "unavailable"])
def test_apkg_closes_source_if_schema_trust_cannot_be_disabled(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """A rejected or unsupported safety setting must not permit an import."""
    apkg = export_apkg_from_rows([{"front": "Question", "back": "Answer"}])
    original_connect = sqlite3.connect
    connections: list[sqlite3.Connection] = []

    class UntrustedSchemaUnavailableConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql == "PRAGMA trusted_schema = OFF":
                if failure == "error":
                    raise sqlite3.DatabaseError("schema trust setting unavailable")
                if failure == "ignored":
                    return super().execute("SELECT 1 WHERE 0")
            if sql == "PRAGMA trusted_schema" and failure == "unavailable":
                return super().execute("SELECT 1 WHERE 0")
            return super().execute(sql, parameters)

    def unavailable_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs, factory=UntrustedSchemaUnavailableConnection)
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", unavailable_connect)
    with pytest.raises(APKGImportError, match="Invalid APKG collection schema"):
        import_rows_from_apkg_bytes(apkg, max_notes=10, max_field_length=8192)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
