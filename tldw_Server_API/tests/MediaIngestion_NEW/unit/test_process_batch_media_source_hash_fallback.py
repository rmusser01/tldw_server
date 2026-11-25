import asyncio
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime.rows import BackendCursorAdapter
from tldw_Server_API.app.core.Ingestion_Media_Processing import (
    persistence as ingestion_persistence,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata_template", "stored_hash", "scenario", "expected_status"),
    [
        pytest.param('{"source_hash":"HASH"}', None, "current", "Skipped", id="compact-legacy-json"),
        pytest.param('{"source_hash": "HASH"}', None, "current", "Skipped", id="spaced-legacy-json"),
        pytest.param('{"source_hash":"HASH"}', "different", "current", "Cancelled", id="stale-legacy-hash"),
        pytest.param('{"source_hash":"different"}', "HASH", "current", "Skipped", id="indexed-hash-match"),
        pytest.param('{"source_hash":"HASH"', None, "current", "Cancelled", id="malformed-legacy-json"),
        pytest.param('{"source_hash":"HASH","other":NaN}', None, "current", "Cancelled", id="non-json-constant"),
        pytest.param('{"source_hash":"HASH"}', None, "blob-metadata", "Cancelled", id="binary-metadata"),
        pytest.param('{"nested":{"source_hash":"HASH"}}', None, "current", "Cancelled", id="nested-legacy-hash"),
        pytest.param('[{"source_hash":"HASH"}]', None, "current", "Cancelled", id="non-object-legacy-json"),
        pytest.param('{"source_hash": "HASH"}', None, "missing-column", "Skipped", id="legacy-schema"),
        pytest.param('{"source_hash":"HASH"}', "HASH", "other-filename", "Cancelled", id="different-filename"),
        pytest.param(
            '{"source_hash":"HASH"}', "different", "inspection-error", "Cancelled", id="schema-inspection-fails",
        ),
        pytest.param(
            '{}', "HASH", "inspection-error", "Skipped", id="indexed-match-after-schema-inspection-error",
        ),
    ],
)
async def test_process_batch_media_source_hash_precheck(
    media_database: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata_template: str,
    stored_hash: str | None,
    scenario: str,
    expected_status: str,
) -> None:
    """Verify duplicate detection before the batch starts transcription."""
    test_path = tmp_path / "staged-audio.wav"
    payload = b"duplicate audio content"
    test_path.write_bytes(payload)
    source_hash = hashlib.sha256(payload).hexdigest()
    source = str(test_path)
    filename = "dup_audio.wav"

    safe_metadata = metadata_template.replace("HASH", source_hash)
    existing_id, _, _ = media_database.add_media_with_keywords(
        url="other_audio.wav" if scenario == "other-filename" else filename,
        title="Existing Audio",
        media_type="audio",
        content="existing content",
        transcription_model="whisper-test",
        safe_metadata=safe_metadata,
        source_hash=source_hash if stored_hash == "HASH" else stored_hash,
    )

    if scenario in {"missing-column", "inspection-error"}:
        monkeypatch.setattr(ingestion_persistence, "create_media_database", lambda *_args, **_kwargs: media_database)
    if scenario == "missing-column":
        media_database.execute_query("DROP INDEX idx_media_source_hash")
        media_database.execute_query("ALTER TABLE Media DROP COLUMN source_hash")
    if scenario == "blob-metadata":
        media_database.execute_query(
            "UPDATE DocumentVersions SET safe_metadata = ?, version = version + 1 WHERE media_id = ?",
            (safe_metadata.encode("utf-16"), existing_id),
        )
    if scenario == "inspection-error":
        def fail_schema_inspection(_table_name: str) -> None:
            raise BackendDatabaseError("schema inspection unavailable")

        monkeypatch.setattr(media_database.backend, "get_table_info", fail_schema_inspection)

    form_data = SimpleNamespace(overwrite_existing=False, transcription_model="whisper-test")
    source_to_ref_map = {source: filename}

    results = await ingestion_persistence.process_batch_media(
        media_type="audio",
        urls=[],
        uploaded_file_paths=[source],
        source_to_ref_map=source_to_ref_map,
        form_data=form_data,
        chunk_options=None,
        loop=asyncio.get_running_loop(),
        db_path=media_database.db_path_str,
        client_id="test_client",
        temp_dir=tmp_path,
        cancel_check=lambda: True,
    )

    assert len(results) == 1
    assert results[0]["status"] == expected_status
    assert results[0]["db_id"] == (existing_id if expected_status == "Skipped" else None)
    assert source_to_ref_map == {source: filename}


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt_text", [False, True], ids=["deep-json", "corrupt-text"])
async def test_process_batch_media_finds_legacy_hash_after_unreadable_version(
    media_database: MediaDatabase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt_text: bool,
) -> None:
    """An unreadable version cannot hide a valid hash in another version."""
    test_path = tmp_path / "audio.wav"
    test_path.write_bytes(b"legacy audio content")
    url = str(test_path)
    source_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
    depth = sys.getrecursionlimit() + 1
    media_id, _, _ = media_database.add_media_with_keywords(
        url=url,
        media_type="audio",
        content="original transcript",
        safe_metadata='{"nested":' + "[" * depth + "]" * depth + "}",
    )
    with media_database.transaction():
        if corrupt_text:
            media_database.execute_query(
                "UPDATE DocumentVersions SET safe_metadata = CAST(? AS TEXT), version = version + 1 WHERE media_id = ?",
                (b'{"payload":"\xff"}', media_id),
            )
        for version_number in range(65):
            media_database.create_document_version(
                media_id=media_id,
                content=f"intermediate transcript {version_number}",
                safe_metadata="{}",
            )
        media_database.create_document_version(
            media_id=media_id,
            content="updated transcript",
            safe_metadata=f'{{"source_hash": "{source_hash}"}}',
        )

    execute_query = media_database.execute_query
    version_batch_sizes: list[int] = []

    def track_version_batches(query: str, params: tuple | None = None) -> BackendCursorAdapter:
        cursor = execute_query(query, params)
        if "JOIN DocumentVersions dv" in query:
            rows = cursor.fetchall()
            version_batch_sizes.append(len(rows))
            monkeypatch.setattr(cursor, "fetchall", lambda: rows)
        return cursor

    monkeypatch.setattr(media_database, "execute_query", track_version_batches)
    monkeypatch.setattr(ingestion_persistence, "create_media_database", lambda *_args, **_kwargs: media_database)

    results = await ingestion_persistence.process_batch_media(
        media_type="audio",
        urls=[],
        uploaded_file_paths=[url],
        source_to_ref_map={url: url},
        form_data=SimpleNamespace(overwrite_existing=False),
        chunk_options=None,
        loop=asyncio.get_running_loop(),
        db_path=media_database.db_path_str,
        client_id="test_client",
        temp_dir=tmp_path,
        cancel_check=lambda: True,
    )

    assert [(result["status"], result["db_id"]) for result in results] == [("Skipped", media_id)]
    assert len(version_batch_sizes) > 1
    assert max(version_batch_sizes) <= 64
