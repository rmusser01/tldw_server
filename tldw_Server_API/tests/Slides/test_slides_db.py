import json
import sqlite3
import stat
import threading
import time

import pytest

from tldw_Server_API.app.core.Slides.slides_db import (
    ConflictError,
    InputError,
    SchemaError,
    SlidesDatabase,
)


def _file_identity(path) -> tuple[int, int, int]:
    metadata = path.stat(follow_symlinks=False)
    return (metadata.st_dev, metadata.st_ino, stat.S_IFMT(metadata.st_mode))


def _sample_slides() -> str:
    slides = [
        {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
        {
            "order": 1,
            "layout": "content",
            "title": "Intro",
            "content": "- A\n- B",
            "speaker_notes": None,
            "metadata": {},
        },
    ]
    return json.dumps(slides)


def _sample_studio_data() -> str:
    return json.dumps(
        {
            "origin": "blank",
            "default_voice": {
                "provider": "openai",
                "voice": "alloy",
            },
            "publish_formats": ["mp4", "webm"],
        }
    )


def _sample_visual_style_snapshot() -> str:
    return json.dumps(
        {
            "id": "timeline",
            "scope": "builtin",
            "name": "Timeline",
            "version": 1,
            "description": "Chronology-first slides",
            "generation_rules": {"chronology_bias": "high"},
            "artifact_preferences": ["timeline", "stat_group"],
            "fallback_policy": {"mode": "ordered-bullets"},
            "resolution": {
                "base_theme": "beige",
                "resolved_theme": "beige",
                "resolved_marp_theme": None,
                "style_pack": "editorial_print",
                "style_pack_version": 1,
                "token_overrides": {"surface": "#f5efe6"},
                "resolved_settings": {"controls": False, "progress": False},
            },
        }
    )


def test_slides_db_create_and_get(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    row = db.create_presentation(
        presentation_id=None,
        title="Deck",
        description=None,
        theme="black",
        marp_theme="gaia",
        settings=None,
        studio_data=_sample_studio_data(),
        slides=_sample_slides(),
        slides_text="Deck Intro A B",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )
    fetched = db.get_presentation_by_id(row.id)
    assert fetched.id == row.id
    assert fetched.title == "Deck"
    assert fetched.marp_theme == "gaia"
    assert json.loads(fetched.studio_data) == json.loads(_sample_studio_data())
    assert fetched.content_kind == "structured_slides"
    assert fetched.html_document is None
    assert fetched.generation_job_uuid is None
    db.close_connection()


def test_slides_db_template_id(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    row = db.create_presentation(
        presentation_id=None,
        title="Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        template_id="clean-dark",
        slides=_sample_slides(),
        slides_text="Deck Intro A B",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )
    fetched = db.get_presentation_by_id(row.id)
    assert fetched.template_id == "clean-dark"
    db.close_connection()


def test_slides_db_visual_style_snapshot_round_trip(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    row = db.create_presentation(
        presentation_id=None,
        title="History Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=_sample_slides(),
        slides_text="Deck Intro A B",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
        visual_style_id="timeline",
        visual_style_scope="builtin",
        visual_style_name="Timeline",
        visual_style_version=1,
        visual_style_snapshot=_sample_visual_style_snapshot(),
    )
    fetched = db.get_presentation_by_id(row.id)
    assert fetched.visual_style_id == "timeline"
    assert fetched.visual_style_scope == "builtin"
    assert fetched.visual_style_name == "Timeline"
    assert fetched.visual_style_version == 1
    snapshot = json.loads(fetched.visual_style_snapshot)
    assert snapshot["id"] == "timeline"
    assert "custom_css" not in snapshot
    assert snapshot["resolution"]["resolved_theme"] == "beige"
    db.close_connection()


def test_slides_db_update_conflict(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    row = db.create_presentation(
        presentation_id=None,
        title="Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=_sample_studio_data(),
        slides=_sample_slides(),
        slides_text="Deck Intro A B",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )
    updated = db.update_presentation(
        presentation_id=row.id,
        update_fields={"title": "Updated", "studio_data": json.dumps({"origin": "extension_capture"})},
        expected_version=row.version,
    )
    assert updated.version == row.version + 1
    assert json.loads(updated.studio_data) == {"origin": "extension_capture"}
    with pytest.raises(ConflictError):
        db.update_presentation(
            presentation_id=row.id,
            update_fields={"title": "Conflict"},
            expected_version=row.version,
        )
    db.close_connection()


def test_slides_db_search(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    _ = db.create_presentation(
        presentation_id=None,
        title="Search Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=_sample_slides(),
        slides_text="alpha beta gamma",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )
    rows, total = db.search_presentations(query="alpha", limit=10, offset=0, include_deleted=False)
    assert total == 1
    assert rows[0].title == "Search Deck"
    db.close_connection()


def test_slides_db_search_rejects_malformed_fts_query(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    db.create_presentation(
        presentation_id=None,
        title="Search Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=_sample_slides(),
        slides_text="alpha beta gamma",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )

    with pytest.raises(InputError):
        db.search_presentations(query='"unterminated', limit=10, offset=0, include_deleted=False)

    db.close_connection()


def test_slides_db_search_rejects_unknown_fts_column(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    db.create_presentation(
        presentation_id=None,
        title="Search Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=_sample_slides(),
        slides_text="alpha beta gamma",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )

    with pytest.raises(InputError):
        db.search_presentations(query="unknown_column:alpha", limit=10, offset=0, include_deleted=False)

    db.close_connection()


def test_slides_db_search_does_not_map_non_fts_operational_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")

    class LockedConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "get_connection", lambda: LockedConnection())

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        db.search_presentations(query="alpha", limit=10, offset=0, include_deleted=False)

    db.close_connection()


def test_create_presentation_rolls_back_when_sync_log_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")

    def _fail_sync_log(*args, **kwargs):
        raise RuntimeError("sync log failed")

    monkeypatch.setattr(SlidesDatabase, "_insert_sync_log", _fail_sync_log)

    with pytest.raises(RuntimeError, match="sync log failed"):
        db.create_presentation(
            presentation_id="pres_sync_atomic",
            title="Deck",
            description=None,
            theme="black",
            marp_theme=None,
            settings=None,
            studio_data=None,
            slides=_sample_slides(),
            slides_text="Deck Intro A B",
            source_type="manual",
            source_ref=None,
            source_query=None,
            custom_css=None,
        )

    with pytest.raises(KeyError):
        db.get_presentation_by_id("pres_sync_atomic", include_deleted=True)

    db.close_connection()


def test_slides_db_schema_initialization_serializes_column_migrations(tmp_path, monkeypatch):
    db_path = tmp_path / "Slides.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE presentations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                theme TEXT DEFAULT 'black',
                settings TEXT,
                slides TEXT NOT NULL,
                slides_text TEXT NOT NULL,
                source_type TEXT,
                source_ref TEXT,
                source_query TEXT,
                custom_css TEXT,
                created_at DATETIME NOT NULL,
                last_modified DATETIME NOT NULL,
                deleted INTEGER DEFAULT 0,
                client_id TEXT NOT NULL,
                version INTEGER DEFAULT 1
            )
            """
        )

    def _slow_marp_theme_migration(conn):
        columns = conn.execute("PRAGMA table_info(presentations)").fetchall()
        if any(col["name"] == "marp_theme" for col in columns):
            return
        time.sleep(0.05)
        conn.execute("ALTER TABLE presentations ADD COLUMN marp_theme TEXT")

    monkeypatch.setattr(
        SlidesDatabase,
        "_ensure_marp_theme_column",
        staticmethod(_slow_marp_theme_migration),
    )

    errors: list[BaseException] = []

    def _open_database(client_id: str) -> None:
        try:
            db = SlidesDatabase(db_path=db_path, client_id=client_id)
            db.close_connection()
        except BaseException as exc:  # noqa: BLE001  # pragma: no cover - assertion reports details below
            errors.append(exc)

    threads = [threading.Thread(target=_open_database, args=(f"tester-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []


def test_slides_db_soft_delete_restore(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    row = db.create_presentation(
        presentation_id=None,
        title="Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=_sample_slides(),
        slides_text="Deck Intro A B",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )
    deleted = db.soft_delete_presentation(row.id, expected_version=row.version)
    assert deleted.deleted == 1
    rows, total = db.list_presentations(
        limit=10, offset=0, include_deleted=False, sort_column="created_at", sort_direction="DESC"
    )
    assert total == 0
    restored = db.restore_presentation(row.id, expected_version=deleted.version)
    assert restored.deleted == 0
    db.close_connection()


def test_slides_db_version_snapshots(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="tester")
    row = db.create_presentation(
        presentation_id=None,
        title="Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=_sample_studio_data(),
        slides=_sample_slides(),
        slides_text="Deck Intro A B",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
        visual_style_id="timeline",
        visual_style_scope="builtin",
        visual_style_name="Timeline",
        visual_style_version=1,
        visual_style_snapshot=_sample_visual_style_snapshot(),
    )
    versions, total = db.list_presentation_versions(presentation_id=row.id, limit=10, offset=0)
    assert total == 1
    payload = json.loads(versions[0].payload_json)
    assert payload["title"] == "Deck"
    assert json.loads(payload["studio_data"]) == json.loads(_sample_studio_data())
    assert payload["visual_style_id"] == "timeline"
    payload_snapshot = json.loads(payload["visual_style_snapshot"])
    assert payload_snapshot["id"] == "timeline"
    assert "custom_css" not in payload_snapshot
    assert payload_snapshot["resolution"]["resolved_theme"] == "beige"

    updated = db.update_presentation(
        presentation_id=row.id,
        update_fields={
            "title": "Updated",
            "studio_data": json.dumps({"origin": "workspace_playground"}),
            "visual_style_id": "exam-focused-bullet",
            "visual_style_scope": "builtin",
            "visual_style_name": "Exam-Focused Bullet",
            "visual_style_version": 1,
            "visual_style_snapshot": json.dumps(
                {
                    "id": "exam-focused-bullet",
                    "scope": "builtin",
                    "name": "Exam-Focused Bullet",
                    "version": 1,
                    "resolution": {
                        "base_theme": "black",
                        "resolved_theme": "black",
                        "resolved_marp_theme": None,
                        "style_pack": "brutalist_editorial",
                        "style_pack_version": 1,
                        "token_overrides": {"surface": "#000000"},
                        "resolved_settings": {"controls": True, "progress": True},
                    },
                }
            ),
        },
        expected_version=row.version,
    )
    versions, total = db.list_presentation_versions(presentation_id=row.id, limit=10, offset=0)
    assert total == 2
    assert versions[0].version == updated.version
    latest_payload = json.loads(versions[0].payload_json)
    assert json.loads(latest_payload["studio_data"]) == {"origin": "workspace_playground"}
    assert latest_payload["visual_style_id"] == "exam-focused-bullet"
    latest_snapshot = json.loads(latest_payload["visual_style_snapshot"])
    assert latest_snapshot["id"] == "exam-focused-bullet"
    assert "custom_css" not in latest_snapshot
    assert latest_snapshot["resolution"]["resolved_theme"] == "black"
    db.close_connection()


def test_slides_db_runtime_connections_use_full_shared_policy_after_schema_init(tmp_path):
    db_path = tmp_path / "Slides.db"

    first = SlidesDatabase(db_path=db_path, client_id="first")
    first.close_connection()

    second = SlidesDatabase(db_path=db_path, client_id="second")
    try:
        conn = second.get_connection()
        pragmas = {
            "journal_mode": str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
            "synchronous": int(conn.execute("PRAGMA synchronous").fetchone()[0]),
            "foreign_keys": int(conn.execute("PRAGMA foreign_keys").fetchone()[0]),
            "busy_timeout": int(conn.execute("PRAGMA busy_timeout").fetchone()[0]),
            "temp_store": int(conn.execute("PRAGMA temp_store").fetchone()[0]),
        }
    finally:
        second.close_connection()

    assert pragmas == {
        "journal_mode": "wal",
        "synchronous": 1,
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "temp_store": 2,
    }


def test_open_existing_complete_does_not_create_a_missing_database(tmp_path):
    db_path = tmp_path / "missing" / "Slides.db"

    with pytest.raises(
        SchemaError,
        match="^Slides database is unavailable or incomplete$",
    ):
        SlidesDatabase.open_existing_complete(
            db_path=db_path,
            client_id="tester",
            expected_file_identity=(0, 0, stat.S_IFREG),
        )

    assert not db_path.exists()


def test_open_existing_complete_does_not_repair_an_incomplete_database(tmp_path):
    db_path = tmp_path / "Slides.db"
    initialized = SlidesDatabase(db_path=db_path, client_id="setup")
    initialized.close_connection()
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE sync_log")
    expected_identity = _file_identity(db_path)

    with pytest.raises(
        SchemaError,
        match="^Slides database is unavailable or incomplete$",
    ):
        SlidesDatabase.open_existing_complete(
            db_path=db_path,
            client_id="tester",
            expected_file_identity=expected_identity,
        )

    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as connection:
        sync_log = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sync_log'"
        ).fetchone()
    assert sync_log is None
