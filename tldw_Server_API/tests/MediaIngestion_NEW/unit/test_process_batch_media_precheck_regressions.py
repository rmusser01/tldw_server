import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import (
    persistence as ingestion_persistence,
)


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [] if self._row is None else [self._row]


class _FakePrecheckDb:
    def __init__(self, *, row=None, error: Exception | None = None, has_source_hash: bool = False):
        self._row = row
        self._error = error
        self.closed = 0
        self.queries: list[str] = []
        self.backend = SimpleNamespace(
            get_table_info=lambda _table: [{"name": "source_hash"}] if has_source_hash else []
        )

    def execute_query(self, query, params):
        self.queries.append(query)
        if self._error is not None:
            raise self._error
        return _FakeCursor(self._row)

    def close_connection(self):
        self.closed += 1


def _patch_audio_processing(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        Audio_Files as audio_files_mod,
    )

    def _fake_process_audio_files(**kwargs):
        src = kwargs["inputs"][0]
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": src,
                    "processing_source": src,
                    "media_type": "audio",
                    "metadata": {},
                    "content": "ok",
                    "analysis_details": {},
                    "warnings": None,
                    "error": None,
                }
            ],
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
        }

    async def _fake_persist_primary_av_item(**kwargs):
        process_result = kwargs["process_result"]
        process_result["db_id"] = None
        process_result["db_message"] = "patched"
        process_result["media_uuid"] = None

    async def _fake_extract_claims_if_requested(*args, **kwargs):
        return None

    monkeypatch.setattr(
        audio_files_mod,
        "process_audio_files",
        _fake_process_audio_files,
        raising=True,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "persist_primary_av_item",
        _fake_persist_primary_av_item,
        raising=True,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "extract_claims_if_requested",
        _fake_extract_claims_if_requested,
        raising=True,
    )


@pytest.mark.asyncio
async def test_process_batch_media_non_hash_precheck_ignores_soft_deleted_rows(
    monkeypatch,
    tmp_path,
):
    source = "https://example.com/audio.mp3"
    _patch_audio_processing(monkeypatch)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
        raising=True,
    )

    fake_db = _FakePrecheckDb(row=None)
    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        lambda client_id, *, db_path=None, **_kwargs: fake_db,
        raising=True,
    )

    form_data = SimpleNamespace(overwrite_existing=False, transcription_model="whisper-test")

    results = await ingestion_persistence.process_batch_media(
        media_type="audio",
        urls=[source],
        uploaded_file_paths=[],
        source_to_ref_map={source: source},
        form_data=form_data,
        chunk_options=None,
        loop=asyncio.get_running_loop(),
        db_path="unused.db",
        client_id="test_client",
        temp_dir=tmp_path,
    )

    assert fake_db.queries
    assert any("deleted = 0" in query.lower() for query in fake_db.queries)
    assert fake_db.closed >= 1
    assert len(results) == 1
    assert results[0]["status"] == "Success"


@pytest.mark.asyncio
async def test_process_batch_media_precheck_closes_db_on_query_error(
    monkeypatch,
    tmp_path,
):
    source = "https://example.com/audio.mp3"
    _patch_audio_processing(monkeypatch)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
        raising=True,
    )

    fake_db = _FakePrecheckDb(error=sqlite3.OperationalError("pre-check query failed"))
    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        lambda client_id, *, db_path=None, **_kwargs: fake_db,
        raising=True,
    )

    form_data = SimpleNamespace(overwrite_existing=False, transcription_model="whisper-test")

    results = await ingestion_persistence.process_batch_media(
        media_type="audio",
        urls=[source],
        uploaded_file_paths=[],
        source_to_ref_map={source: source},
        form_data=form_data,
        chunk_options=None,
        loop=asyncio.get_running_loop(),
        db_path="unused.db",
        client_id="test_client",
        temp_dir=tmp_path,
    )

    assert fake_db.closed >= 1
    assert len(results) == 1
    assert results[0]["status"] == "Success"


@pytest.mark.asyncio
async def test_process_batch_media_source_hash_precheck_uses_factory(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "audio.mp3"
    source_path.write_bytes(b"audio-bytes")
    source = str(source_path)

    _patch_audio_processing(monkeypatch)

    fake_db = _FakePrecheckDb(row=None, has_source_hash=True)
    monkeypatch.setattr(
        ingestion_persistence,
        "create_media_database",
        lambda client_id, *, db_path=None, **_kwargs: fake_db,
        raising=True,
    )

    form_data = SimpleNamespace(overwrite_existing=False, transcription_model="whisper-test")

    results = await ingestion_persistence.process_batch_media(
        media_type="audio",
        urls=[],
        uploaded_file_paths=[source],
        source_to_ref_map={source: source},
        form_data=form_data,
        chunk_options=None,
        loop=asyncio.get_running_loop(),
        db_path="unused.db",
        client_id="test_client",
        temp_dir=tmp_path,
    )

    assert fake_db.queries
    assert any("source_hash = ?" in query.lower() for query in fake_db.queries)
    assert fake_db.closed >= 1
    assert len(results) == 1
    assert results[0]["status"] == "Success"


@pytest.mark.asyncio
async def test_process_batch_media_passes_user_id_to_audio_processor(
    monkeypatch,
    tmp_path,
):
    source = "https://example.com/audio.mp3"
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
        raising=True,
    )

    observed: dict[str, object] = {}

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        Audio_Files as audio_files_mod,
    )

    def _fake_process_audio_files(**kwargs):
        observed["user_id"] = kwargs.get("user_id")
        src = kwargs["inputs"][0]
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": src,
                    "processing_source": src,
                    "media_type": "audio",
                    "metadata": {},
                    "content": "ok",
                    "analysis_details": {},
                    "warnings": None,
                    "error": None,
                }
            ],
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
        }

    async def _fake_persist_primary_av_item(**kwargs):
        process_result = kwargs["process_result"]
        process_result["db_id"] = None
        process_result["db_message"] = "patched"
        process_result["media_uuid"] = None

    async def _fake_extract_claims_if_requested(*args, **kwargs):
        return None

    monkeypatch.setattr(
        audio_files_mod,
        "process_audio_files",
        _fake_process_audio_files,
        raising=True,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "persist_primary_av_item",
        _fake_persist_primary_av_item,
        raising=True,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "extract_claims_if_requested",
        _fake_extract_claims_if_requested,
        raising=True,
    )

    form_data = SimpleNamespace(overwrite_existing=False, transcription_model="whisper-test")

    results = await ingestion_persistence.process_batch_media(
        media_type="audio",
        urls=[source],
        uploaded_file_paths=[],
        source_to_ref_map={source: source},
        form_data=form_data,
        chunk_options=None,
        loop=asyncio.get_running_loop(),
        db_path="unused.db",
        client_id="test_client",
        temp_dir=tmp_path,
        user_id=42,
    )

    assert observed["user_id"] == 42
    assert len(results) == 1
    assert results[0]["status"] == "Success"
