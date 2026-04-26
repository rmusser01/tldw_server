import pytest


pytestmark = pytest.mark.unit


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
