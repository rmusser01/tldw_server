"""Benign SQLite compatibility checks; run explicitly in a backend image.

Usage: python -I -B /checks/verify_sqlite_runtime.py
Only the standard library is imported. No application or model is started.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path


class SQLiteRuntimeChecks(unittest.TestCase):
    """Exercise the SQLite operations used by imports and raw restoration."""

    def test_native_library_contains_vendor_fix(self) -> None:
        """Reject the affected native library even if Python itself is current."""
        self.assertGreaterEqual(sqlite3.sqlite_version_info, (3, 53, 2))

    def test_backup_restores_content_and_search_index(self) -> None:
        """A raw backup retains ordinary rows and a working FTS5 index."""
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.db"
            target_path = Path(directory) / "restored.db"
            with closing(sqlite3.connect(source_path)) as source:
                source.executescript(
                    "CREATE TABLE notes(id INTEGER PRIMARY KEY, body TEXT);"
                    "INSERT INTO notes VALUES (1, 'orchard apples');"
                    "CREATE VIRTUAL TABLE search USING fts5(body);"
                    "INSERT INTO search(rowid, body) SELECT id, body FROM notes;"
                )
                with closing(sqlite3.connect(target_path)) as target:
                    target.execute("BEGIN EXCLUSIVE")
                    target.rollback()
                    source.backup(target, pages=256)
            with closing(sqlite3.connect(target_path)) as restored:
                self.assertEqual(
                    restored.execute(
                        "SELECT notes.body FROM notes JOIN search ON notes.id = search.rowid WHERE search MATCH ?",
                        ("orchard",),
                    ).fetchall(),
                    [("orchard apples",)],
                )

    def test_json_each_remains_available_with_untrusted_schema(self) -> None:
        """The safe import setting preserves OpenWebUI's JSON row expansion."""
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute("PRAGMA trusted_schema=OFF")
            self.assertEqual(connection.execute("PRAGMA trusted_schema").fetchone(), (0,))
            self.assertEqual(
                connection.execute("SELECT value FROM json_each(?)", ('["a", "b"]',)).fetchall(),
                [("a",), ("b",)],
            )

    def test_wal_commits_are_visible_to_a_second_connection(self) -> None:
        """The packaged library retains the runtime's WAL concurrency mode."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wal.db"
            with closing(sqlite3.connect(path)) as writer, closing(sqlite3.connect(path)) as reader:
                self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone(), ("wal",))
                writer.execute("CREATE TABLE notes(body TEXT)")
                writer.execute("INSERT INTO notes VALUES (?)", ("committed",))
                writer.commit()
                self.assertEqual(reader.execute("SELECT body FROM notes").fetchall(), [("committed",)])


if __name__ == "__main__":
    print(json.dumps({"sqlite_version": sqlite3.sqlite_version, "sqlite_module": sqlite3.__file__}))
    unittest.main(verbosity=2)
