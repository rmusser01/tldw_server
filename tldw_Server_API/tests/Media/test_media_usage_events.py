import pytest


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.debug_kwargs: list[dict[str, object]] = []
        self.errors: list[str] = []
        self.error_args: list[tuple[object, ...]] = []
        self.error_kwargs: list[dict[str, object]] = []
        self.warnings: list[str] = []
        self.warning_args: list[tuple[object, ...]] = []
        self.warning_kwargs: list[dict[str, object]] = []

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def log(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append(message.format(*args, **kwargs) if args or kwargs else message)
        self.warning_args.append(args)
        self.warning_kwargs.append(dict(kwargs))

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append(message.format(*args, **kwargs) if args or kwargs else message)
        self.debug_kwargs.append(dict(kwargs))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append(message.format(*args, **kwargs) if args or kwargs else message)
        self.error_args.append(args)
        self.error_kwargs.append(dict(kwargs))


class _DebugFailingLoggerStub(_LoggerStub):
    def debug(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("debug logging exploded at /private/debug.log")


class _AudioWarningDebugFailingLoggerStub(_LoggerStub):
    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        if message.startswith("TEST_MODE: /process-audios returned 207"):
            raise RuntimeError("audio warning formatting exploded at /private/audio.log")
        super().debug(message, *args, **kwargs)


def _assert_sanitized_debug_log(logger: _LoggerStub, expected: str) -> None:
    target_kwargs = [
        kwargs for message, kwargs in zip(logger.debugs, logger.debug_kwargs) if message == expected
    ]
    assert target_kwargs, logger.debugs
    rendered = "\n".join(message for message in logger.debugs if message == expected)
    assert "exploded" not in rendered
    assert "/private/" not in rendered
    assert all(not kwargs for kwargs in target_kwargs)


def _assert_sanitized_error_log(logger: _LoggerStub, expected: str) -> None:
    target_records = [
        (message, args, kwargs)
        for message, args, kwargs in zip(
            logger.errors,
            logger.error_args,
            logger.error_kwargs,
        )
        if message == expected
    ]
    assert target_records, logger.errors
    rendered = "\n".join(message for message, _args, _kwargs in target_records)
    assert "exploded" not in rendered
    assert "/private/" not in rendered
    assert all(not args and not kwargs for _message, args, kwargs in target_records)


def _assert_sanitized_warning_log(logger: _LoggerStub, expected: str) -> None:
    target_records = [
        (message, args, kwargs)
        for message, args, kwargs in zip(
            logger.warnings,
            logger.warning_args,
            logger.warning_kwargs,
        )
        if message.startswith(expected)
    ]
    assert target_records, logger.warnings
    rendered = "\n".join(message for message, _args, _kwargs in target_records)
    assert "exploded" not in rendered
    assert "/private/" not in rendered
    assert all(not args and not kwargs for _message, args, kwargs in target_records)


class _StubQuotaService:
    async def check_quota(self, user_id, size_bytes, raise_on_exceed=False):
        # Always allow in tests
        return True, {
            "current_usage_mb": 0,
            "new_size_mb": float(size_bytes) / (1024 * 1024),
            "quota_mb": 999999,
            "available_mb": 999999,
        }


@pytest.fixture()
def quota_service_stub(monkeypatch):
    # Patch quota service globally to avoid DB access in tests
    import tldw_Server_API.app.services.storage_quota_service as quota_mod
    monkeypatch.setattr(quota_mod, "get_storage_quota_service", lambda: _StubQuotaService())
    yield


def test_ebooks_process_usage_event_logged(client_with_single_user, quota_service_stub, monkeypatch):


    client, usage_logger = client_with_single_user

    # Stub heavy processing to return immediately
    import tldw_Server_API.app.api.v1.endpoints.media as media_mod

    def _stub_process_epub(**kwargs):

        return {
            "status": "Success",
            "content": "",
            "metadata": {"title": "stub-ebook"},
        }

    monkeypatch.setattr(media_mod.books, "process_epub", _stub_process_epub)

    files = [
        ("files", ("sample.epub", b"fake", "application/epub+zip")),
    ]

    r = client.post("/api/v1/media/process-ebooks", files=files)
    assert r.status_code == 200, r.text
    assert any(e[0] == "media.process.ebook" for e in usage_logger.events)


def test_ebooks_process_usage_event_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_ebooks as process_ebooks_mod

    logger_stub = _LoggerStub()

    def _fail_usage_event(*_args, **_kwargs):
        raise RuntimeError("ebook usage logger exploded at /private/usage-events.db")

    def _stub_process_epub(**_kwargs):
        return {
            "status": "Success",
            "content": "Hello ebook content.",
            "metadata": {"title": "stub-ebook"},
        }

    monkeypatch.setattr(process_ebooks_mod, "logger", logger_stub)
    monkeypatch.setattr(usage_logger, "log_event", _fail_usage_event)
    monkeypatch.setattr(media_mod.books, "process_epub", _stub_process_epub)

    response = client.post(
        "/api/v1/media/process-ebooks",
        files=[("files", ("sample.epub", b"fake", "application/epub+zip"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "Ebook process endpoint usage logging failed")


def test_ebooks_process_rechunk_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    import tldw_Server_API.app.core.Chunking as chunking_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_ebooks as process_ebooks_mod

    logger_stub = _LoggerStub()

    def _stub_process_epub(**_kwargs):
        return {
            "status": "Success",
            "content": "Hello ebook content for rechunking.",
            "metadata": {"title": "stub-ebook"},
        }

    def _fail_improved_chunking_process(*_args, **_kwargs):
        raise RuntimeError("ebook rechunk exploded at /private/chunks")

    monkeypatch.setattr(process_ebooks_mod, "logger", logger_stub)
    monkeypatch.setattr(media_mod.books, "process_epub", _stub_process_epub)
    monkeypatch.setattr(chunking_mod, "improved_chunking_process", _fail_improved_chunking_process)

    response = client.post(
        "/api/v1/media/process-ebooks",
        data={"perform_chunking": "true"},
        files=[("files", ("sample.epub", b"fake", "application/epub+zip"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(
        logger_stub,
        "Ebook post-processing re-chunking skipped/failed",
    )


def test_ebooks_process_sanitizes_worker_failure(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod

    def _fail_process_epub(**_kwargs):
        raise RuntimeError("ebook parser failed at /private/book.epub")

    monkeypatch.setattr(media_mod.books, "process_epub", _fail_process_epub)

    files = [
        ("files", ("sample.epub", b"fake", "application/epub+zip")),
    ]

    response = client.post("/api/v1/media/process-ebooks", files=files)

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Ebook processing failed"
    assert "ebook parser failed" not in response.text
    assert "/private/book.epub" not in response.text


def test_ebooks_process_sanitizes_url_download_failure(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    from tldw_Server_API.app.api.v1.endpoints.media import process_ebooks as process_ebooks_mod

    async def _fail_download_url_async(*_args, **_kwargs):
        raise RuntimeError("ebook download failed at /private/cache/book.epub")

    monkeypatch.setattr(process_ebooks_mod, "core_download_url_async", _fail_download_url_async)

    response = client.post(
        "/api/v1/media/process-ebooks",
        data={
            "urls": "https://example.com/book.epub",
            "perform_chunking": "false",
            "perform_analysis": "false",
        },
    )

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Download/preparation failed"
    assert "ebook download failed" not in response.text
    assert "/private/cache/book.epub" not in response.text


def test_ebooks_process_sanitizes_task_execution_failure(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    from tldw_Server_API.app.api.v1.endpoints.media import process_ebooks as process_ebooks_mod

    def _fail_process_single_ebook(**_kwargs):
        raise RuntimeError("executor failed at /private/cache/book.epub")

    monkeypatch.setattr(process_ebooks_mod, "_process_single_ebook", _fail_process_single_ebook)

    files = [
        ("files", ("sample.epub", b"fake", "application/epub+zip")),
    ]

    response = client.post("/api/v1/media/process-ebooks", files=files)

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Ebook processing failed"
    assert "executor failed" not in response.text
    assert "/private/cache/book.epub" not in response.text


def test_documents_process_usage_event_logged(client_with_single_user, quota_service_stub, monkeypatch):


    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod

    def _stub_process_document_content(**kwargs):

        return {
            "status": "Success",
            "content": "Hello",
            "metadata": {"title": "stub-doc"},
        }

    monkeypatch.setattr(media_mod.docs, "process_document_content", _stub_process_document_content)

    files = [
        ("files", ("note.txt", b"hi", "text/plain")),
    ]

    r = client.post("/api/v1/media/process-documents", files=files)
    assert r.status_code == 200, r.text
    assert any(e[0] == "media.process.document" for e in usage_logger.events)


def test_documents_process_usage_event_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_documents as process_documents_mod

    logger_stub = _LoggerStub()

    def _fail_usage_event(*_args, **_kwargs):
        raise RuntimeError("usage logger exploded at /private/usage-events.db")

    def _stub_process_document_content(**_kwargs):
        return {
            "status": "Success",
            "content": "Hello",
            "metadata": {"title": "stub-doc"},
        }

    monkeypatch.setattr(process_documents_mod, "logger", logger_stub)
    monkeypatch.setattr(usage_logger, "log_event", _fail_usage_event)
    monkeypatch.setattr(media_mod.docs, "process_document_content", _stub_process_document_content)

    response = client.post(
        "/api/v1/media/process-documents",
        files=[("files", ("note.txt", b"hi", "text/plain"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "Document process endpoint usage logging failed")


def test_documents_process_rechunk_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    import tldw_Server_API.app.core.Chunking as chunking_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_documents as process_documents_mod

    logger_stub = _LoggerStub()

    def _stub_process_document_content(**_kwargs):
        return {
            "status": "Success",
            "content": "Hello document content for rechunking.",
            "metadata": {"title": "stub-doc"},
        }

    def _fail_improved_chunking_process(*_args, **_kwargs):
        raise RuntimeError("document rechunk exploded at /private/chunks")

    monkeypatch.setattr(process_documents_mod, "logger", logger_stub)
    monkeypatch.setattr(media_mod.docs, "process_document_content", _stub_process_document_content)
    monkeypatch.setattr(chunking_mod, "improved_chunking_process", _fail_improved_chunking_process)

    response = client.post(
        "/api/v1/media/process-documents",
        data={"perform_chunking": "true"},
        files=[("files", ("note.txt", b"hi", "text/plain"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(
        logger_stub,
        "Re-chunking failed during metadata normalization",
    )


def test_documents_process_sanitizes_worker_failure(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod

    def _fail_process_document_content(**_kwargs):
        raise RuntimeError("document parser failed at /private/doc.txt")

    monkeypatch.setattr(
        media_mod.docs,
        "process_document_content",
        _fail_process_document_content,
    )

    files = [
        ("files", ("note.txt", b"hi", "text/plain")),
    ]

    response = client.post("/api/v1/media/process-documents", files=files)

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Document processing failed"
    assert "document parser failed" not in response.text
    assert "/private/doc.txt" not in response.text


def test_documents_process_sanitizes_url_download_failure(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    from tldw_Server_API.app.api.v1.endpoints.media import process_documents as process_documents_mod

    async def _fail_download_url_async(*_args, **_kwargs):
        raise RuntimeError("document download failed at /private/cache/doc.txt")

    monkeypatch.setattr(process_documents_mod, "core_download_url_async", _fail_download_url_async)

    response = client.post(
        "/api/v1/media/process-documents",
        data={
            "urls": "https://example.com/doc.txt",
            "perform_chunking": "false",
            "perform_analysis": "false",
        },
    )

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Download/preparation failed"
    assert "document download failed" not in response.text
    assert "/private/cache/doc.txt" not in response.text


def test_pdfs_process_usage_event_logged(client_with_single_user, quota_service_stub, monkeypatch):


    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod

    captured_kwargs = {}

    async def _stub_process_pdf_task(**kwargs):
        captured_kwargs.update(kwargs)
        return {
            "status": "Success",
            "content": "",
            "metadata": {"title": "stub-pdf"},
        }

    monkeypatch.setattr(media_mod.pdf_lib, "process_pdf_task", _stub_process_pdf_task)

    files = [
        ("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf")),
    ]

    data = {
        "enable_ocr": "true",
        "ocr_backend": "hunyuan",
        "ocr_output_format": "json",
        "ocr_prompt_preset": "json",
    }
    r = client.post("/api/v1/media/process-pdfs", data=data, files=files)
    assert r.status_code == 200, r.text
    assert any(e[0] == "media.process.pdf" for e in usage_logger.events)
    assert captured_kwargs.get("enable_ocr") is True
    assert captured_kwargs.get("ocr_backend") == "hunyuan"
    assert captured_kwargs.get("ocr_output_format") == "json"
    assert captured_kwargs.get("ocr_prompt_preset") == "json"


def test_pdfs_process_usage_event_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_pdfs as process_pdfs_mod

    logger_stub = _LoggerStub()

    def _fail_usage_event(*_args, **_kwargs):
        raise RuntimeError("usage logger exploded at /private/usage-events.db")

    async def _stub_process_pdf_task(**_kwargs):
        return {
            "status": "Success",
            "content": "",
            "metadata": {"title": "stub-pdf"},
        }

    monkeypatch.setattr(process_pdfs_mod, "logger", logger_stub)
    monkeypatch.setattr(usage_logger, "log_event", _fail_usage_event)
    monkeypatch.setattr(media_mod.pdf_lib, "process_pdf_task", _stub_process_pdf_task)

    response = client.post(
        "/api/v1/media/process-pdfs",
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "PDF process endpoint usage logging failed")


def test_pdfs_process_rechunk_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod
    import tldw_Server_API.app.core.Chunking as chunking_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_pdfs as process_pdfs_mod

    logger_stub = _LoggerStub()

    async def _stub_process_pdf_task(**_kwargs):
        return {
            "status": "Success",
            "content": "PDF text content for rechunking.",
            "metadata": {"title": "stub-pdf"},
        }

    def _fail_improved_chunking_process(*_args, **_kwargs):
        raise RuntimeError("pdf rechunk exploded at /private/chunks")

    monkeypatch.setattr(process_pdfs_mod, "logger", logger_stub)
    monkeypatch.setattr(media_mod.pdf_lib, "process_pdf_task", _stub_process_pdf_task)
    monkeypatch.setattr(chunking_mod, "improved_chunking_process", _fail_improved_chunking_process)

    response = client.post(
        "/api/v1/media/process-pdfs",
        data={"perform_chunking": "true"},
        files=[("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(
        logger_stub,
        "PDF process endpoint rechunking failed; returning original result",
    )


def test_pdfs_process_sanitizes_processor_failure(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.api.v1.endpoints.media as media_mod

    async def _fail_process_pdf_task(**_kwargs):
        raise RuntimeError("pdf parser failed at /private/paper.pdf")

    monkeypatch.setattr(media_mod.pdf_lib, "process_pdf_task", _fail_process_pdf_task)

    files = [
        ("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf")),
    ]

    response = client.post("/api/v1/media/process-pdfs", files=files)

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "PDF processing failed"
    assert "pdf parser failed" not in response.text
    assert "/private/paper.pdf" not in response.text


def test_pdfs_process_sanitizes_url_download_failure(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    from tldw_Server_API.app.api.v1.endpoints.media import process_pdfs as process_pdfs_mod

    async def _fail_download_url_async(*_args, **_kwargs):
        raise RuntimeError("pdf download failed at /private/cache/paper.pdf")

    monkeypatch.setattr(process_pdfs_mod, "core_download_url_async", _fail_download_url_async)

    response = client.post(
        "/api/v1/media/process-pdfs",
        data={
            "urls": "https://example.com/paper.pdf",
            "perform_chunking": "false",
            "perform_analysis": "false",
        },
    )

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Download/preparation failed"
    assert "pdf download failed" not in response.text
    assert "/private/cache/paper.pdf" not in response.text


def test_pdfs_process_sanitizes_prepared_file_read_failure(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    from tldw_Server_API.app.api.v1.endpoints.media import process_pdfs as process_pdfs_mod

    def _fail_read_bytes(_self):
        raise RuntimeError("prepared read failed at /private/cache/paper.pdf")

    monkeypatch.setattr(process_pdfs_mod.Path, "read_bytes", _fail_read_bytes)

    files = [
        ("files", ("paper.pdf", b"%PDF-1.4\n", "application/pdf")),
    ]
    response = client.post(
        "/api/v1/media/process-pdfs",
        data={"perform_chunking": "false"},
        files=files,
    )

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Failed to read prepared file"
    assert "prepared read failed" not in response.text
    assert "/private/cache/paper.pdf" not in response.text


def test_audios_process_usage_event_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.core.Ingestion_Media_Processing.audio_batch as audio_batch_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_audios as process_audios_mod

    logger_stub = _LoggerStub()

    def _fail_usage_event(*_args, **_kwargs):
        raise RuntimeError("usage logger exploded at /private/usage-events.db")

    async def _stub_run_audio_batch(**_kwargs):
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [
                {
                    "status": "Success",
                    "input_ref": "clip.mp3",
                    "media_type": "audio",
                    "content": "audio transcript",
                    "metadata": {},
                }
            ],
        }

    monkeypatch.setattr(process_audios_mod, "logger", logger_stub)
    monkeypatch.setattr(usage_logger, "log_event", _fail_usage_event)
    monkeypatch.setattr(audio_batch_mod, "run_audio_batch", _stub_run_audio_batch)

    response = client.post(
        "/api/v1/media/process-audios",
        data={"perform_chunking": "false"},
        files=[("files", ("clip.mp3", b"ID3", "audio/mpeg"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "Audio process endpoint usage logging failed")


def test_videos_process_usage_event_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, usage_logger = client_with_single_user

    import tldw_Server_API.app.core.Ingestion_Media_Processing.video_batch as video_batch_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_videos as process_videos_mod

    logger_stub = _LoggerStub()

    def _fail_usage_event(*_args, **_kwargs):
        raise RuntimeError("usage logger exploded at /private/usage-events.db")

    async def _stub_run_video_batch(**_kwargs):
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [
                {
                    "status": "Success",
                    "input_ref": "clip.mp4",
                    "media_type": "video",
                    "content": "video transcript",
                    "metadata": {},
                }
            ],
            "confabulation_results": None,
        }

    monkeypatch.setattr(process_videos_mod, "logger", logger_stub)
    monkeypatch.setattr(usage_logger, "log_event", _fail_usage_event)
    monkeypatch.setattr(video_batch_mod, "run_video_batch", _stub_run_video_batch)

    response = client.post(
        "/api/v1/media/process-videos",
        data={"perform_chunking": "false"},
        files=[("files", ("clip.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "Video process endpoint usage logging failed")


def test_audios_process_rechunk_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.core.Chunking as chunking_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.audio_batch as audio_batch_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_audios as process_audios_mod

    logger_stub = _LoggerStub()

    async def _stub_run_audio_batch(**_kwargs):
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [
                {
                    "status": "Success",
                    "input_ref": "clip.mp3",
                    "media_type": "audio",
                    "content": "audio transcript for rechunking",
                    "metadata": {},
                }
            ],
        }

    def _fail_improved_chunking_process(*_args, **_kwargs):
        raise RuntimeError("audio rechunk exploded at /private/chunks")

    monkeypatch.setattr(process_audios_mod, "logger", logger_stub)
    monkeypatch.setattr(audio_batch_mod, "run_audio_batch", _stub_run_audio_batch)
    monkeypatch.setattr(chunking_mod, "improved_chunking_process", _fail_improved_chunking_process)

    response = client.post(
        "/api/v1/media/process-audios",
        data={"perform_chunking": "true"},
        files=[("files", ("clip.mp3", b"ID3", "audio/mpeg"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_warning_log(
        logger_stub,
        "Best-effort audio chunking post-processing failed; leaving results unchunked",
    )


def test_audios_process_warning_log_formatting_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.core.Ingestion_Media_Processing.audio_batch as audio_batch_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_audios as process_audios_mod

    logger_stub = _AudioWarningDebugFailingLoggerStub()

    async def _stub_run_audio_batch(**_kwargs):
        return {
            "processed_count": 0,
            "errors_count": 1,
            "errors": ["Download failed: Host could not be resolved"],
            "results": [
                {
                    "status": "Error",
                    "input_ref": "clip.mp3",
                    "media_type": "audio",
                    "error": "Download failed",
                }
            ],
        }

    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setattr(process_audios_mod, "logger", logger_stub)
    monkeypatch.setattr(audio_batch_mod, "run_audio_batch", _stub_run_audio_batch)

    response = client.post(
        "/api/v1/media/process-audios",
        data={"perform_chunking": "false"},
        files=[("files", ("clip.mp3", b"ID3", "audio/mpeg"))],
    )

    assert response.status_code == 207, response.text
    _assert_sanitized_debug_log(
        logger_stub,
        "Audio process endpoint warning log formatting failed",
    )


def test_videos_process_rechunk_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.core.Chunking as chunking_mod
    import tldw_Server_API.app.core.Ingestion_Media_Processing.video_batch as video_batch_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_videos as process_videos_mod

    logger_stub = _LoggerStub()

    async def _stub_run_video_batch(**_kwargs):
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [
                {
                    "status": "Success",
                    "input_ref": "clip.mp4",
                    "media_type": "video",
                    "content": "video transcript for rechunking",
                    "metadata": {},
                }
            ],
            "confabulation_results": None,
        }

    def _fail_improved_chunking_process(*_args, **_kwargs):
        raise RuntimeError("video rechunk exploded at /private/chunks")

    monkeypatch.setattr(process_videos_mod, "logger", logger_stub)
    monkeypatch.setattr(video_batch_mod, "run_video_batch", _stub_run_video_batch)
    monkeypatch.setattr(chunking_mod, "improved_chunking_process", _fail_improved_chunking_process)

    response = client.post(
        "/api/v1/media/process-videos",
        data={"perform_chunking": "true"},
        files=[("files", ("clip.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(
        logger_stub,
        "Video process endpoint rechunking failed; returning original result",
    )


def test_videos_process_debug_logging_failure_log_is_sanitized(
    client_with_single_user,
    quota_service_stub,
    monkeypatch,
):
    client, _ = client_with_single_user

    import tldw_Server_API.app.core.Ingestion_Media_Processing.video_batch as video_batch_mod
    from tldw_Server_API.app.api.v1.endpoints.media import process_videos as process_videos_mod

    logger_stub = _DebugFailingLoggerStub()

    async def _stub_run_video_batch(**_kwargs):
        return {
            "processed_count": 1,
            "errors_count": 0,
            "errors": [],
            "results": [
                {
                    "status": "Success",
                    "input_ref": "clip.mp4",
                    "media_type": "video",
                    "content": "video transcript",
                    "metadata": {},
                }
            ],
            "confabulation_results": None,
        }

    monkeypatch.setattr(process_videos_mod, "logger", logger_stub)
    monkeypatch.setattr(video_batch_mod, "run_video_batch", _stub_run_video_batch)

    response = client.post(
        "/api/v1/media/process-videos",
        data={"perform_chunking": "false"},
        files=[("files", ("clip.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4"))],
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_error_log(
        logger_stub,
        "Video process endpoint debug logging failed",
    )
