"""Owned-worker cancellation, capacity and fixed process protocol boundaries."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

from tldw_Server_API.app.api.v1.schemas.document_rendering import DocumentRenderRequest
from tldw_Server_API.app.core.Chat.bounded_daemon import DaemonCapacityError
from tldw_Server_API.app.core.Document_Workspace import pdf_service

DRAFT = DocumentRenderRequest(title="Rowan", pages=[{"heading": "Rowan", "text": "Selene"}])


@pytest.mark.asyncio
async def test_cancelled_pdf_request_keeps_capacity_until_real_worker_release(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def held(_payload):
        entered.set()
        if not release.wait(2):
            raise FutureTimeoutError("Owned control did not release")
        return b"complete"

    monkeypatch.setattr(pdf_service, "_run_pdf", held)
    task = asyncio.create_task(pdf_service.render_document_pdf(DRAFT))
    try:
        await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and pdf_service._PDF_POOL.active_count == 1  # nosec B101 # Caller cancellation cannot abandon the worker or recycle its lease early.
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert pdf_service._PDF_POOL.active_count == 0  # nosec B101 # Release is observable only after the real work exits.


@pytest.mark.asyncio
async def test_pdf_admission_refuses_a_third_live_worker(monkeypatch):
    release = threading.Event()
    entered = threading.Barrier(3)

    def held(_payload):
        entered.wait(timeout=2)
        if not release.wait(2):
            raise FutureTimeoutError("Owned control did not release")
        return b"complete"

    monkeypatch.setattr(pdf_service, "_run_pdf", held)
    tasks = [asyncio.create_task(pdf_service.render_document_pdf(DRAFT)) for _ in range(2)]
    try:
        await asyncio.to_thread(entered.wait, 2)
        with pytest.raises(DaemonCapacityError):
            await pdf_service.render_document_pdf(DRAFT)
        assert pdf_service._PDF_POOL.active_count == 2  # nosec B101 # Rejected work starts no unbounded third operation.
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert pdf_service._PDF_POOL.active_count == 0  # nosec B101 # All admitted work is drained.


def test_pdf_child_failure_does_not_expose_stderr(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        pdf_service.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2, stdout=b"partial", stderr=b"private path /secret/credentials"
        ),
    )
    with pytest.raises(ValueError, match="invalid or unsupported"):
        pdf_service._run_pdf({"operation": "render", "draft": DRAFT.model_dump()})


def test_pdf_child_timeout_has_a_fixed_error(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise pdf_service.subprocess.TimeoutExpired("fixed owned worker", 10)

    monkeypatch.setattr(pdf_service.subprocess, "run", timeout)
    with pytest.raises(TimeoutError, match="Document PDF processing timed out"):
        pdf_service._run_pdf({"operation": "render", "draft": DRAFT.model_dump()})


@pytest.mark.asyncio
async def test_pdf_child_uses_the_api_package_even_from_another_checkout(tmp_path, monkeypatch):
    folder = tmp_path
    for part in ("tldw_Server_API", "app", "core", "Document_Workspace"):
        folder = folder / part
        folder.mkdir()
        (folder / "__init__.py").write_text("")
    (folder / "pdf_worker.py").write_text("import sys; sys.stdout.buffer.write(b'%PDF-SHADOW\\n%%EOF')")
    monkeypatch.chdir(tmp_path)
    result = await pdf_service.render_document_pdf(DRAFT)
    assert b"SHADOW" not in result and len(result) > 1000  # nosec B101 # A different cwd cannot substitute its module for the API's admitted package.


@pytest.mark.asyncio
async def test_export_admission_precedes_source_buffering(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints.media import document_rendering

    release = asyncio.Event()
    reads = 0

    class Storage:
        async def retrieve_stream(self, _key):
            nonlocal reads
            reads += 1
            await release.wait()
            yield b"source"

    db = SimpleNamespace(
        get_media_by_id=lambda *_args, **_kwargs: {"id": 1},
        get_media_file=lambda *_args, **_kwargs: {"storage_path": "owned", "file_size": 6},
    )
    monkeypatch.setattr(document_rendering.Storage, "get_storage_backend", lambda: Storage())
    monkeypatch.setattr(
        document_rendering.DocumentWorkspaceRepository, "list_annotations_for_export", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(document_rendering, "annotate_document_pdf", AsyncMock(return_value=b"%PDF-complete\n%%EOF"))
    tasks = [
        asyncio.create_task(document_rendering.get_annotated_document(media_id=1, db=db, user=SimpleNamespace(id=1)))
        for _ in range(2)
    ]
    third = None
    try:
        for _ in range(100):
            if reads == 2:
                break
            await asyncio.sleep(0.001)
        third = asyncio.create_task(
            document_rendering.get_annotated_document(media_id=1, db=db, user=SimpleNamespace(id=1))
        )
        status = None
        try:
            await asyncio.wait_for(third, 0.1)
        except HTTPException as exc:
            status = exc.status_code
        except TimeoutError:
            pass
        assert (status, reads) == (503, 2)  # nosec B101 # A third request must be refused before materializing another source buffer.
    finally:
        release.set()
        await asyncio.gather(*tasks, *([third] if third else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_oversized_source_closes_stream_without_starting_pdf_child(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints.media import document_rendering
    from tldw_Server_API.app.api.v1.schemas.document_rendering import MAX_PDF_BYTES

    closed = []

    class Storage:
        async def retrieve_stream(self, _key):
            try:
                yield b"x" * MAX_PDF_BYTES
                yield b"extra"
            finally:
                closed.append(True)

    db = SimpleNamespace(
        get_media_by_id=lambda *_args, **_kwargs: {"id": 1},
        get_media_file=lambda *_args, **_kwargs: {"storage_path": "owned"},
    )
    monkeypatch.setattr(document_rendering.Storage, "get_storage_backend", lambda: Storage())
    monkeypatch.setattr(
        document_rendering.DocumentWorkspaceRepository, "list_annotations_for_export", lambda *_args, **_kwargs: []
    )
    worker = AsyncMock()
    monkeypatch.setattr(document_rendering, "annotate_document_pdf", worker)
    with pytest.raises(HTTPException) as error:
        await document_rendering.get_annotated_document(media_id=1, db=db, user=SimpleNamespace(id=1))
    assert (error.value.status_code, closed, worker.call_count, pdf_service._PDF_REQUESTS.active_count) == (
        413,
        [True],
        0,
        0,
    )  # nosec B101 # Rejected source drains its stream and lease without child work.
