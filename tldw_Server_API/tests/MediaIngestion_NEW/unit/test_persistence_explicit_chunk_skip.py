"""Explicitly skipped chunking is terminal without manufacturing indexed chunks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence
from tldw_Server_API.app.core.Workspaces.status_projection import derive_workspace_source_media_status

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def isolate_external_monitoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real media writes isolated from the unrelated topic-monitor service."""
    from tldw_Server_API.app.core.Monitoring import topic_monitoring_service

    monkeypatch.setattr(
        topic_monitoring_service,
        "get_topic_monitoring_service",
        lambda: SimpleNamespace(schedule_evaluate_and_alert=lambda **_kwargs: None),
    )


async def persist_text(
    db_path: Path,
    *,
    audio: bool,
    extra: bool = False,
    chunking: bool = False,
    overwrite: bool = True,
    strict: bool = False,
) -> dict:
    """Exercise real persistence and repositories with an already extracted local text."""
    text = "The telescope is named Selene. Tours start on Friday at 18:00."
    result = {
        "status": "Success",
        "content": text,
        "transcript": text,
        "summary": None,
        "analysis": None,
        "metadata": {"title": "Rowan skip-chunking source"},
        "warnings": None,
        "error": None,
    }
    if extra:
        result["extra_chunks"] = [{"text": "Recorded OCR evidence.", "chunk_type": "ocr", "metadata": {}}]
    form = SimpleNamespace(
        perform_chunking=chunking,
        chunk_consistency_policy="error" if strict else "warn",
        perform_analysis=False,
        perform_claims_extraction=False,
        generate_embeddings=False,
        keywords=[],
        custom_prompt=None,
        title=None,
        author=None,
        transcription_model=None,
        overwrite_existing=overwrite,
        ingest_attachments=False,
        accept_archives=False,
        accept_mbox=False,
        accept_pst=False,
    )
    common = {
        "form_data": form,
        "media_type": "audio" if audio else "document",
        "chunk_options": {"method": "sentences", "max_size": 500, "overlap": 0} if chunking else None,
        "path_kind": "upload",
        "db_path": str(db_path),
        "client_id": "1",
        "loop": asyncio.get_running_loop(),
        "claims_context": None,
    }
    if audio:
        await persistence.persist_primary_av_item(
            process_result=result, original_input_ref="file:///s04-skip-audio.wav", **common
        )
    else:
        await persistence.persist_doc_item_and_children(
            final_result=result, item_input_ref="file:///s04-skip-document.txt", processing_filename=None, **common
        )
    return result


@pytest.mark.parametrize("audio", [False, True], ids=["document", "audio"])
async def test_explicit_chunk_skip_preserves_searchable_text_without_pending_work(tmp_path: Path, audio: bool) -> None:
    db_path = tmp_path / "media.db"
    db = create_media_database("1", db_path=str(db_path))
    try:
        result = await persist_text(db_path, audio=audio)
        assert result["status"] in {"Success", "Warning"}  # nosec B101 # Pytest checks the real persistence result.
        row = db.get_media_by_id(result["db_id"])
        assert row["chunking_status"] == "completed"  # nosec B101 # Explicit skip must never advertise unfinished work.
        assert db.get_unvectorized_chunk_count(result["db_id"]) == 0  # nosec B101 # No chunks are manufactured.
        status = derive_workspace_source_media_status(row, media_db=db, media_id=result["db_id"])
        assert status["state"] == "partially_queryable"  # nosec B101 # Existing text-only search remains usable.
        assert status["readiness"]["fts_ready"] is True  # nosec B101 # The stored source retains its real text.
    finally:
        db.close_connection()


async def test_explicit_chunk_skip_retains_processor_extra_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "media.db"
    db = create_media_database("1", db_path=str(db_path))
    try:
        result = await persist_text(db_path, audio=True, extra=True)
        assert result["status"] in {"Success", "Warning"}  # nosec B101 # The actual AV persistence path must succeed.
        assert db.get_unvectorized_chunk_count(result["db_id"]) == 1  # nosec B101 # Supplied OCR evidence is preserved.
        assert db.get_media_by_id(result["db_id"])["chunking_status"] == "completed"  # nosec B101 # Existing list semantics remain.
    finally:
        db.close_connection()


async def test_unfinished_repository_chunks_remain_pending(tmp_path: Path) -> None:
    db = create_media_database("1", db_path=str(tmp_path / "media.db"))
    try:
        media_id, _, _ = db.add_media_with_keywords(
            title="Unfinished requested indexing",
            content="The source has extracted text.",
            media_type="document",
            url="file:///s04-unfinished.txt",
            chunks=None,
        )
        row = db.get_media_by_id(media_id)
        assert row["chunking_status"] == "pending"  # nosec B101 # Unfinished work must retain its existing meaning.
        status = derive_workspace_source_media_status(row, media_db=db, media_id=media_id)
        assert (  # nosec B101 -- Pending chunks retain text-only readiness without mutating persistence.
            status["state"], status["status_reason"], status["readiness"]["fts_ready"],
            status["readiness"]["vector_ready"], db.get_media_by_id(media_id),
        ) == ("partially_queryable", "chunking_pending", True, False, row)
    finally:
        db.close_connection()


@pytest.mark.parametrize("audio", [False, True], ids=["document", "audio"])
async def test_requested_chunk_failure_stays_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, audio: bool
) -> None:
    from tldw_Server_API.app.core.Chunking.chunker import Chunker

    def reject_requested_chunks(*_args: object, **_kwargs: object) -> None:
        raise ValueError("declared chunking failure")

    monkeypatch.setattr(Chunker, "chunk_text_hierarchical_flat", reject_requested_chunks)
    db_path = tmp_path / "media.db"
    db = create_media_database("1", db_path=str(db_path))
    try:
        result = await persist_text(db_path, audio=audio, chunking=True)
        assert result["status"] in {"Success", "Warning"}  # nosec B101 # Existing recoverable persistence behavior.
        assert db.get_media_by_id(result["db_id"])["chunking_status"] == "pending"  # nosec B101 # Failed requested work is not a skip.
    finally:
        db.close_connection()


@pytest.mark.parametrize("overwrite", [False, True], ids=["dedupe", "explicit-overwrite"])
async def test_explicit_skip_existing_row_respects_overwrite_policy(tmp_path: Path, overwrite: bool) -> None:
    db_path = tmp_path / "media.db"
    db = create_media_database("1", db_path=str(db_path))
    try:
        media_id, media_uuid, _ = db.add_media_with_keywords(
            title="Rowan skip-chunking source",
            content="The telescope is named Selene. Tours start on Friday at 18:00.",
            media_type="document",
            url="file:///s04-skip-document.txt",
            chunks=None,
            safe_metadata=json.dumps({"title": "Rowan skip-chunking source"}),
        )
        before = db.get_media_by_id(media_id)
        versions = db.get_all_document_versions(media_id, include_content=True)
        result = await persist_text(db_path, audio=False, overwrite=overwrite)
        after = db.get_media_by_id(media_id)
        assert result["db_id"] == media_id and result["media_uuid"] == media_uuid  # nosec B101 # Reuse the existing owned identity.
        assert after["content"] == before["content"]  # nosec B101 # Status repair cannot rewrite extracted text.
        assert after["chunking_status"] == ("completed" if overwrite else "pending")  # nosec B101 # Dedupe is not implicit repair.
        after_versions = db.get_all_document_versions(media_id, include_content=True)
        if overwrite:
            content_before = [
                {key: value for key, value in row.items() if key not in {"version", "last_modified"}}
                for row in versions
            ]
            content_after = [
                {key: value for key, value in row.items() if key not in {"version", "last_modified"}}
                for row in after_versions
            ]
            assert content_after == content_before  # nosec B101 # Plaintext revision identity and payload remain unchanged.
            assert after_versions[0]["version"] > versions[0]["version"]  # nosec B101 # Preserve normal metadata revision accounting.
        else:
            assert after_versions == versions  # nosec B101 # Dedupe cannot rewrite the existing revision.
        if overwrite:
            assert after["version"] > before["version"]  # nosec B101 # Existing optimistic Media version policy remains active.
    finally:
        db.close_connection()


@pytest.mark.parametrize("overwrite", [False, True], ids=["dedupe", "explicit-overwrite"])
async def test_explicit_skip_existing_chunks_respects_overwrite_policy(tmp_path: Path, overwrite: bool) -> None:
    db_path = tmp_path / "media.db"
    db = create_media_database("1", db_path=str(db_path))
    try:
        text = "The telescope is named Selene. Tours start on Friday at 18:00."
        media_id, _, _ = db.add_media_with_keywords(
            title="Rowan skip-chunking source",
            content=text,
            media_type="document",
            url="file:///s04-skip-document.txt",
            safe_metadata=json.dumps({"title": "Rowan skip-chunking source"}),
            chunks=[{"text": text, "chunk_type": "text", "metadata": {}}],
        )
        before = db.get_all_document_versions(media_id, include_content=True)
        result = await persist_text(db_path, audio=False, overwrite=overwrite, strict=True)
        assert result["status"] in {"Success", "Warning", "Skipped"}  # nosec B101 # Legitimate dedupe is not an indexing error.
        assert result["db_id"] == media_id  # nosec B101 # Same-source import preserves its actual identity.
        assert db.get_unvectorized_chunk_count(media_id) == (0 if overwrite else 1)  # nosec B101 # Only explicit overwrite clears old chunks.
        after = db.get_all_document_versions(media_id, include_content=True)
        if overwrite:
            content_before = [
                {key: value for key, value in row.items() if key not in {"version", "last_modified"}} for row in before
            ]
            content_after = [
                {key: value for key, value in row.items() if key not in {"version", "last_modified"}} for row in after
            ]
            assert content_after == content_before  # nosec B101 # Existing text revision identity and payload stay intact.
            assert after[0]["version"] > before[0]["version"]  # nosec B101 # Existing metadata revision advances normally.
        else:
            assert after == before  # nosec B101 # Dedupe preserves both content and revision metadata.
    finally:
        db.close_connection()
