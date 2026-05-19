from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi import status

from tldw_Server_API.app.core.Ingestion_Media_Processing import (
    input_sourcing,
    persistence as ingestion_persistence,
)
from tldw_Server_API.app.core import Storage
from tldw_Server_API.app.services import storage_quota_service


class _FakeStorage:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def store(
        self,
        *,
        user_id: str,
        media_id: int,
        filename: str,
        data: bytes,
        mime_type: str,
    ) -> str:
        payload = data
        if hasattr(data, "read") and not isinstance(data, (bytes, bytearray)):
            payload = data.read()
            if hasattr(data, "seek"):
                data.seek(0)
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        self.calls.append(
            {
                "user_id": user_id,
                "media_id": media_id,
                "filename": filename,
                "data": payload,
                "mime_type": mime_type,
            }
        )
        return f"storage/{media_id}/{filename}"


class _FakeDB:
    def __init__(self) -> None:
        self.db_path_str = ":memory:"
        self.client_id = "test-client"
        self.insert_calls: List[Dict[str, Any]] = []

    def insert_media_file(self, **kwargs: Any) -> None:
        self.insert_calls.append(kwargs)


class _FakeQuotaService:
    async def initialize(self) -> None:
        return None

    async def check_quota(
        self,
        user_id: int,
        new_bytes: int,
        raise_on_exceed: bool = False,
    ) -> tuple[bool, Dict[str, Any]]:
        return True, {"current_usage_mb": 0, "new_size_mb": 0, "quota_mb": 1, "available_mb": 1}


class _FakeUploadQuotaService:
    async def check_quota(
        self,
        user_id: int,
        new_bytes: int,
        raise_on_exceed: bool = False,
    ) -> tuple[bool, Dict[str, Any]]:
        _ = (user_id, new_bytes, raise_on_exceed)
        return True, {"current_usage_mb": 0, "new_size_mb": 0, "quota_mb": 1024, "available_mb": 1024}


@pytest.fixture
def fake_storage() -> _FakeStorage:
    return _FakeStorage()


@pytest.fixture
def fake_db() -> _FakeDB:
    return _FakeDB()


@pytest.fixture(autouse=True)
def _disable_collections_dual_write(monkeypatch: pytest.MonkeyPatch) -> None:
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        ingestion_persistence,
        "sync_media_add_results_to_collections",
        _noop,
    )


@pytest.mark.unit
def test_extract_analysis_extra_chunks_for_indexing_emits_table_and_media_chunks():
    process_result = {
        "analysis_details": {
            "ocr": {
                "structured": {
                    "tables": [
                        {"format": "markdown", "content": "| a | b |"},
                    ],
                    "pages": [
                        {"page": 2, "tables": [{"format": "csv", "content": "x,y"}]},
                    ],
                }
            },
            "vlm": {
                "by_page": [
                    {
                        "page": 4,
                        "detections": [
                            {"label": "table", "caption": "Revenue by quarter"},
                            {"label": "chart", "description": "A line chart"},
                        ],
                    }
                ]
            },
        }
    }

    chunks = ingestion_persistence._extract_analysis_extra_chunks_for_indexing(process_result)
    assert chunks

    table_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "table"]
    media_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "media"]

    assert any((chunk.get("metadata") or {}).get("source") == "ocr_structured_table" for chunk in table_chunks)
    assert any((chunk.get("metadata") or {}).get("source") == "ocr_structured_page_table" for chunk in table_chunks)
    assert any((chunk.get("metadata") or {}).get("source") == "vlm_detection" for chunk in table_chunks)
    assert any((chunk.get("metadata") or {}).get("source") == "vlm_detection" for chunk in media_chunks)


@pytest.mark.unit
def test_extract_analysis_extra_chunks_for_indexing_dedupes_by_content_key():
    process_result = {
        "analysis_details": {
            "vlm": {
                "by_page": [
                    {
                        "page": 1,
                        "detections": [
                            {"label": "chart", "caption": "Duplicated caption"},
                            {"label": "chart", "caption": "Duplicated caption"},
                        ],
                    }
                ]
            }
        }
    }

    chunks = ingestion_persistence._extract_analysis_extra_chunks_for_indexing(process_result)
    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "media"
    assert "Duplicated caption" in str(chunks[0]["text"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_original_storage_uses_processing_source(monkeypatch, fake_db, fake_storage):
    storage = fake_storage
    db = fake_db

    async def fake_save_uploaded_files(_files, temp_dir, **_kwargs):
        file_one = Path(temp_dir) / "stored_one.pdf"
        file_two = Path(temp_dir) / "stored_two.pdf"
        file_one.write_bytes(b"file-one")
        file_two.write_bytes(b"file-two")
        saved = [
            {"path": file_one, "original_filename": "duplicate.pdf"},
            {"path": file_two, "original_filename": "duplicate.pdf"},
        ]
        return saved, []

    async def fake_process_doc_item_fn(
        *,
        item_input_ref: str,
        processing_source: str,
        media_type: Any,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        media_id = 1 if "stored_one.pdf" in str(processing_source) else 2
        return {
            "status": "Success",
            "input_ref": item_input_ref,
            "processing_source": str(processing_source),
            "media_type": media_type,
            "metadata": {},
            "content": "content",
            "analysis": None,
            "summary": None,
            "analysis_details": None,
            "db_id": media_id,
            "db_message": "ok",
        }

    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(ingestion_persistence, "process_document_like_item", fake_process_doc_item_fn)
    monkeypatch.setattr(Storage, "get_storage_backend", lambda: storage)
    monkeypatch.setattr(storage_quota_service, "StorageQuotaService", _FakeQuotaService)

    form_data = SimpleNamespace(
        media_type="pdf",
        urls=[],
        keep_original_file=True,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )

    response = await ingestion_persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[object(), object()],
        db=db,
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
    )

    assert response.status_code == 200
    stored_payloads = {call["data"] for call in storage.calls}
    assert stored_payloads == {b"file-one", b"file-two"}
    assert len(db.insert_calls) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_media_orchestrate_handles_document_exceptions(monkeypatch, tmp_path, fake_db, fake_storage):
    db = fake_db

    async def fake_save_uploaded_files(_files, temp_dir, **_kwargs):
        ok_path = Path(temp_dir) / "ok.txt"
        ok_path.write_text("ok")
        return [{"path": ok_path, "original_filename": "ok.txt"}], []

    async def fake_process_doc_item_fn(
        *,
        item_input_ref: str,
        processing_source: str,
        media_type: Any,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        if str(processing_source).startswith("https://fail.test"):
            raise RuntimeError("boom")
        return {
            "status": "Success",
            "input_ref": item_input_ref,
            "processing_source": str(processing_source),
            "media_type": media_type,
            "metadata": {},
            "content": "content",
            "analysis": None,
            "summary": None,
            "analysis_details": None,
            "db_id": 1,
            "db_message": "ok",
        }

    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(ingestion_persistence, "process_document_like_item", fake_process_doc_item_fn)

    form_data = SimpleNamespace(
        media_type="document",
        urls=["https://fail.test/doc"],
        keep_original_file=False,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )

    response = await ingestion_persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[object()],
        db=db,
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
    )

    assert response.status_code == status.HTTP_207_MULTI_STATUS
    body = json.loads(response.body)
    results = body.get("results") or []

    assert any(
        result.get("status") == "Error"
        and result.get("input_ref") == "https://fail.test/doc"
        for result in results
    )
    assert any(
        result.get("status") == "Success" and result.get("input_ref") == "ok.txt"
        for result in results
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_media_orchestrate_passes_auto_chunk_options_to_document_processor(
    monkeypatch,
    fake_db,
):
    captured: Dict[str, Any] = {}

    def fail_legacy_prepare(_form_data: Any) -> None:
        raise AssertionError("legacy chunk option preparation should not run for Auto mode")

    async def fake_process_doc_item_fn(
        *,
        chunk_options: Dict[str, Any] | None,
        item_input_ref: str,
        processing_source: str,
        media_type: Any,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        captured["chunk_options"] = dict(chunk_options or {})
        return {
            "status": "Success",
            "input_ref": item_input_ref,
            "processing_source": str(processing_source),
            "media_type": media_type,
            "metadata": {},
            "content": "content",
            "analysis": None,
            "summary": None,
            "analysis_details": None,
            "db_id": 1,
            "db_message": "ok",
        }

    monkeypatch.setattr(
        ingestion_persistence,
        "prepare_chunking_options_dict",
        fail_legacy_prepare,
    )
    monkeypatch.setattr(ingestion_persistence, "process_document_like_item", fake_process_doc_item_fn)

    form_data = SimpleNamespace(
        media_type="document",
        urls=["https://example.test/auto.md"],
        keep_original_file=False,
        perform_chunking=True,
        chunking_mode="auto",
        auto_chunking_goal="navigation_summary",
        auto_chunking_use_llm=False,
        chunk_method="words",
        chunk_size=333,
        chunk_overlap=1,
        chunk_language=None,
        transcription_language=None,
        chunking_template_name=None,
        auto_apply_template=True,
        hierarchical_chunking=False,
        hierarchical_template=None,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        custom_chapter_pattern=None,
        enable_contextual_chunking=False,
        contextual_llm_model=None,
        context_window_size=None,
        context_strategy=None,
        context_token_budget=None,
        perform_analysis=False,
        generate_embeddings=False,
        keywords=[],
        title=None,
        author=None,
        overwrite_existing=False,
    )

    response = await ingestion_persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[],
        db=fake_db,
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
    )

    assert response.status_code == status.HTTP_200_OK
    assert captured["chunk_options"]["method"] == "semantic"
    assert captured["chunk_options"]["max_size"] == 1400
    assert captured["chunk_options"]["overlap"] == 100
    assert captured["chunk_options"]["method"] != "words"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_media_orchestrate_passes_auto_chunk_options_to_batch_processor(
    monkeypatch,
    fake_db,
):
    captured: Dict[str, Any] = {}

    def fail_legacy_prepare(_form_data: Any) -> None:
        raise AssertionError("legacy chunk option preparation should not run for Auto mode")

    async def fake_process_batch_media(**kwargs: Any) -> List[Dict[str, Any]]:
        captured["chunk_options"] = dict(kwargs.get("chunk_options") or {})
        return [
            {
                "status": "Success",
                "input_ref": "https://example.test/audio.mp3",
                "processing_source": "https://example.test/audio.mp3",
                "media_type": "audio",
                "metadata": {},
                "content": "transcript",
                "analysis": None,
                "summary": None,
                "analysis_details": None,
                "db_id": 1,
                "db_message": "ok",
            }
        ]

    monkeypatch.setattr(
        ingestion_persistence,
        "prepare_chunking_options_dict",
        fail_legacy_prepare,
    )
    monkeypatch.setattr(ingestion_persistence, "process_batch_media", fake_process_batch_media)

    form_data = SimpleNamespace(
        media_type="audio",
        urls=["https://example.test/audio.mp3"],
        keep_original_file=False,
        perform_chunking=True,
        chunking_mode="auto",
        auto_chunking_goal="qa_search",
        auto_chunking_use_llm=False,
        chunk_method="words",
        chunk_size=333,
        chunk_overlap=1,
        chunk_language=None,
        transcription_language=None,
        chunking_template_name=None,
        auto_apply_template=True,
        hierarchical_chunking=False,
        hierarchical_template=None,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        custom_chapter_pattern=None,
        enable_contextual_chunking=False,
        contextual_llm_model=None,
        context_window_size=None,
        context_strategy=None,
        context_token_budget=None,
        perform_analysis=False,
        generate_embeddings=False,
        keywords=[],
        title=None,
        author=None,
        overwrite_existing=False,
    )

    response = await ingestion_persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[],
        db=fake_db,
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
    )

    assert response.status_code == status.HTTP_200_OK
    assert captured["chunk_options"]["method"] == "sentences"
    assert captured["chunk_options"]["max_size"] == 700
    assert captured["chunk_options"]["overlap"] == 140
    assert captured["chunk_options"]["method"] != "words"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_media_orchestrate_partial_upload_errors_returns_multi_status(monkeypatch, fake_db, fake_storage):
    db = fake_db

    async def fake_save_uploaded_files(_files, temp_dir, **_kwargs):
        ok_path = Path(temp_dir) / "ok.pdf"
        ok_path.write_bytes(b"ok")
        return (
            [{"path": ok_path, "original_filename": "ok.pdf"}],
            [
                {
                    "original_filename": "bad.exe",
                    "input_ref": "bad.exe",
                    "status": "Error",
                    "error": "File type '.exe' is not allowed for security reasons",
                }
            ],
        )

    async def fake_process_doc_item_fn(
        *,
        item_input_ref: str,
        processing_source: str,
        media_type: Any,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "status": "Success",
            "input_ref": item_input_ref,
            "processing_source": str(processing_source),
            "media_type": media_type,
            "metadata": {},
            "content": "content",
            "analysis": None,
            "summary": None,
            "analysis_details": None,
            "db_id": 1,
            "db_message": "ok",
        }

    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(ingestion_persistence, "process_document_like_item", fake_process_doc_item_fn)

    form_data = SimpleNamespace(
        media_type="pdf",
        urls=[],
        keep_original_file=False,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )

    response = await ingestion_persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[object(), object()],
        db=db,
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
    )

    assert response.status_code == status.HTTP_207_MULTI_STATUS
    body = json.loads(response.body)
    results = body.get("results") or []
    assert any(r.get("status") == "Error" and r.get("input_ref") == "bad.exe" for r in results)
    assert any(r.get("status") == "Success" and r.get("input_ref") == "ok.pdf" for r in results)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_media_orchestrate_document_concurrency_limit(monkeypatch, fake_db, fake_storage):
    db = fake_db
    monkeypatch.setenv("DOCUMENT_LIKE_CONCURRENCY", "2")

    async def fake_save_uploaded_files(_files, temp_dir, **_kwargs):
        saved = []
        for idx in range(5):
            path = Path(temp_dir) / f"doc_{idx}.txt"
            path.write_text("ok")
            saved.append({"path": path, "original_filename": f"doc_{idx}.txt"})
        return saved, []

    state = {"current": 0, "max": 0}
    lock = asyncio.Lock()

    async def fake_process_doc_item_fn(
        *,
        item_input_ref: str,
        processing_source: str,
        media_type: Any,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        async with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.05)
        async with lock:
            state["current"] -= 1
        return {
            "status": "Success",
            "input_ref": item_input_ref,
            "processing_source": str(processing_source),
            "media_type": media_type,
            "metadata": {},
            "content": "content",
            "analysis": None,
            "summary": None,
            "analysis_details": None,
            "db_id": 1,
            "db_message": "ok",
        }

    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(ingestion_persistence, "process_document_like_item", fake_process_doc_item_fn)

    form_data = SimpleNamespace(
        media_type="document",
        urls=[],
        keep_original_file=False,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )

    response = await ingestion_persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[object() for _ in range(5)],
        db=db,
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
    )

    assert response.status_code == status.HTTP_200_OK
    assert state["max"] <= 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_media_orchestrate_enforces_rg_media_jobs_limit(monkeypatch, fake_db):
    save_called = {"value": False}

    async def fake_save_uploaded_files(_files, temp_dir, **_kwargs):
        save_called["value"] = True
        path = Path(temp_dir) / "doc.txt"
        path.write_text("ok")
        return [{"path": path, "original_filename": "doc.txt"}], []

    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)

    class _DenyJobsGov:
        async def reserve(self, _req, op_id=None):
            _ = op_id
            decision = SimpleNamespace(
                allowed=False,
                retry_after=7,
                details={"categories": {"jobs": {"limit": 1, "remaining": 0, "retry_after": 7}}},
            )
            return decision, None

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rg_governor=_DenyJobsGov(),
                rg_policy_loader=SimpleNamespace(get_policy=lambda _pid: {"jobs": {"max_concurrent": 1}}),
            )
        ),
        state=SimpleNamespace(rg_policy_id="media.default"),
        url=SimpleNamespace(path="/api/v1/media/add"),
        headers={},
    )

    form_data = SimpleNamespace(
        media_type="document",
        urls=[],
        keep_original_file=False,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )

    with pytest.raises(HTTPException) as exc:
        await ingestion_persistence.add_media_orchestrate(
            background_tasks=BackgroundTasks(),
            form_data=form_data,
            files=[object()],
            db=fake_db,
            current_user=SimpleNamespace(id=1),
            usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
            request=request,
        )

    assert exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "concurrency limit" in str(exc.value.detail).lower()
    assert save_called["value"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_media_orchestrate_enforces_rg_ingestion_bytes_limit_and_releases_slot(
    monkeypatch,
    fake_db,
):
    async def fake_save_uploaded_files(_files, temp_dir, **_kwargs):
        path = Path(temp_dir) / "doc.txt"
        path.write_bytes(b"0123456789ABCDEF")
        return [{"path": path, "original_filename": "doc.txt"}], []

    async def fake_process_doc_item_fn(**_kwargs: Any) -> Dict[str, Any]:
        return {
            "status": "Success",
            "input_ref": "doc.txt",
            "processing_source": "doc.txt",
            "media_type": "document",
            "metadata": {},
            "content": "content",
            "analysis": None,
            "summary": None,
            "analysis_details": None,
            "db_id": 1,
            "db_message": "ok",
        }

    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(ingestion_persistence, "process_document_like_item", fake_process_doc_item_fn)
    monkeypatch.setattr(storage_quota_service, "get_storage_quota_service", lambda: _FakeUploadQuotaService())

    class _Gov:
        def __init__(self) -> None:
            self.released: list[str] = []
            self.checked_units: int | None = None

        async def reserve(self, _req, op_id=None):
            _ = op_id
            decision = SimpleNamespace(allowed=True, retry_after=None, details={"categories": {"jobs": {"limit": 2, "remaining": 1}}})
            return decision, "media-handle-1"

        async def check(self, req):
            self.checked_units = int(req.categories.get("ingestion_bytes", {}).get("units") or 0)
            return SimpleNamespace(
                allowed=False,
                retry_after=30,
                details={
                    "categories": {
                        "ingestion_bytes": {
                            "daily_cap": 10,
                            "daily_used": 10,
                            "daily_remaining": 0,
                            "retry_after": 30,
                        }
                    }
                },
            )

        async def release(self, handle_id):
            self.released.append(str(handle_id))

    gov = _Gov()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rg_governor=gov,
                rg_policy_loader=SimpleNamespace(
                    get_policy=lambda _pid: {
                        "jobs": {"max_concurrent": 2},
                        "ingestion_bytes": {"daily_cap": 10},
                    }
                ),
            )
        ),
        state=SimpleNamespace(rg_policy_id="media.default"),
        url=SimpleNamespace(path="/api/v1/media/add"),
        headers={},
    )

    form_data = SimpleNamespace(
        media_type="document",
        urls=[],
        keep_original_file=False,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )

    with pytest.raises(HTTPException) as exc:
        await ingestion_persistence.add_media_orchestrate(
            background_tasks=BackgroundTasks(),
            form_data=form_data,
            files=[object()],
            db=fake_db,
            current_user=SimpleNamespace(id=1),
            usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
            request=request,
        )

    assert exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "size budget" in str(exc.value.detail).lower()
    assert gov.checked_units == 16
    assert gov.released == ["media-handle-1"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_add_media_orchestrate_records_ingestion_bytes_in_shared_ledger(monkeypatch, fake_db):
    async def fake_save_uploaded_files(_files, temp_dir, **_kwargs):
        path = Path(temp_dir) / "doc.txt"
        path.write_bytes(b"hello-world")
        return [{"path": path, "original_filename": "doc.txt"}], []

    async def fake_process_doc_item_fn(
        *,
        item_input_ref: str,
        processing_source: str,
        media_type: Any,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "status": "Success",
            "input_ref": item_input_ref,
            "processing_source": str(processing_source),
            "media_type": media_type,
            "metadata": {},
            "content": "content",
            "analysis": None,
            "summary": None,
            "analysis_details": None,
            "db_id": 1,
            "db_message": "ok",
        }

    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(ingestion_persistence, "process_document_like_item", fake_process_doc_item_fn)
    monkeypatch.setattr(storage_quota_service, "get_storage_quota_service", lambda: _FakeUploadQuotaService())

    class _AllowGov:
        async def reserve(self, _req, op_id=None):
            _ = op_id
            return SimpleNamespace(allowed=True, retry_after=None, details={}), "media-handle-2"

        async def check(self, _req):
            return SimpleNamespace(
                allowed=True,
                retry_after=None,
                details={
                    "categories": {
                        "ingestion_bytes": {
                            "daily_cap": 1000000,
                            "daily_used": 0,
                            "daily_remaining": 1000000,
                        }
                    }
                },
            )

        async def release(self, _handle_id):
            return None

    recorded: dict[str, Any] = {}

    async def _fake_record(
        *,
        entity_scope: str,
        entity_value: str,
        units: int,
        op_id: str,
    ) -> bool:
        recorded["entity_scope"] = entity_scope
        recorded["entity_value"] = entity_value
        recorded["units"] = units
        recorded["op_id"] = op_id
        return True

    monkeypatch.setattr(
        ingestion_persistence,
        "_record_media_ingestion_bytes_ledger_entry",
        _fake_record,
    )

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                rg_governor=_AllowGov(),
                rg_policy_loader=SimpleNamespace(
                    get_policy=lambda _pid: {
                        "jobs": {"max_concurrent": 2},
                        "ingestion_bytes": {"daily_cap": 1000000},
                    }
                ),
            )
        ),
        state=SimpleNamespace(rg_policy_id="media.default"),
        url=SimpleNamespace(path="/api/v1/media/add"),
        headers={"X-Request-ID": "req-abc"},
    )

    form_data = SimpleNamespace(
        media_type="document",
        urls=[],
        keep_original_file=False,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )

    response = await ingestion_persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[object()],
        db=fake_db,
        current_user=SimpleNamespace(id=42),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
        request=request,
    )

    assert response.status_code == status.HTTP_200_OK
    assert recorded["entity_scope"] == "user"
    assert recorded["entity_value"] == "42"
    assert recorded["units"] == len(b"hello-world")
    assert "req-abc" in recorded["op_id"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_batch_media_test_mode_accepts_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.delenv("TESTING", raising=False)

    captured: dict[str, Any] = {}

    def _fake_evaluate_url_policy(url: str, block_private_override: bool | None = None):
        captured["url"] = url
        captured["block_private_override"] = block_private_override
        return SimpleNamespace(allowed=False, reason="blocked-for-test")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        _fake_evaluate_url_policy,
        raising=True,
    )

    url = "https://example.com/demo.mp3"
    results = await ingestion_persistence.process_batch_media(
        media_type="audio",
        urls=[url],
        uploaded_file_paths=[],
        source_to_ref_map={url: url},
        form_data=SimpleNamespace(overwrite_existing=False, transcription_model=None),
        chunk_options=None,
        loop=asyncio.get_running_loop(),
        db_path=str(tmp_path / "media.db"),
        client_id="test-client",
        temp_dir=tmp_path,
    )

    assert captured.get("url") == url
    assert captured.get("block_private_override") is False
    assert len(results) == 1
    assert "URL blocked by security policy" in str(results[0].get("error"))
