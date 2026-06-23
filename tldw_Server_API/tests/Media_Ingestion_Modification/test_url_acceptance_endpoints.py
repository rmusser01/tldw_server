from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urlparse

import pytest

_STUB_TABLE: Dict[str, Tuple[str, Dict[str, str], bytes]] = {}
_CONTENT_TYPE_EXTENSIONS = {
    "application/epub+zip": ".epub",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "application/rtf": ".rtf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/xhtml+xml": ".xhtml",
    "application/xml": ".xml",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "text/xml": ".xml",
}


def _stub_filename_from_headers(headers: Dict[str, str]) -> str | None:
    content_disposition = headers.get("content-disposition") or ""
    if "filename=" not in content_disposition:
        return None
    filename = content_disposition.split("filename=", 1)[1].strip()
    if filename.startswith('"'):
        filename = filename.split('"', 2)[1]
    else:
        filename = filename.split(";", 1)[0].strip()
    return filename or None


def _safe_stub_filename(filename: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in filename) or "downloaded.tmp"


class _URLStub:
    def __init__(self, url: str):
        self._url = url
        try:
            self.path = urlparse(url).path
        except Exception:
            self.path = url

    def __str__(self):

        return self._url


class FakeResponse:
    def __init__(self, final_url: str, headers: Dict[str, str], content: bytes):
        self.url = _URLStub(final_url)
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._content = content or b""
        self.status_code = 200

    def raise_for_status(self):

        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self, chunk_size: int = 8192):
        yield self._content


@pytest.fixture(autouse=True)
def patch_http_fetch(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.download_utils as download_utils
    from tldw_Server_API.app.api.v1.endpoints.media import (
        process_code,
        process_documents,
        process_ebooks,
        process_pdfs,
    )

    async def fake_afetch(*args, **kwargs):
        url = kwargs.get("url")
        if url is None and len(args) > 1:
            url = args[1]
        if url not in _STUB_TABLE:
            raise AssertionError(f"No stub configured for URL: {url}")
        final_url, headers, content = _STUB_TABLE[url]
        return FakeResponse(final_url, headers, content)

    async def fake_download_url_async(*args, **kwargs):
        url = kwargs.get("url")
        if url is None and len(args) > 1:
            url = args[1]
        if url not in _STUB_TABLE:
            raise AssertionError(f"No stub configured for URL: {url}")

        target_dir = kwargs.get("target_dir")
        if target_dir is None and len(args) > 2:
            target_dir = args[2]
        if target_dir is None:
            raise AssertionError("No target_dir provided for URL stub")

        allowed_extensions = {ext.lower() for ext in (kwargs.get("allowed_extensions") or set())}
        check_extension = kwargs.get("check_extension", True)
        disallow_content_types = {value.lower() for value in (kwargs.get("disallow_content_types") or set())}
        max_bytes = kwargs.get("max_bytes")

        final_url, headers, content = _STUB_TABLE[url]
        headers = {key.lower(): value for key, value in (headers or {}).items()}
        content_type = (headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if content_type and content_type in disallow_content_types:
            allowed_list = ", ".join(sorted(allowed_extensions or [])) or "*"
            raise ValueError(
                f"Downloaded file from {url} does not have an allowed extension "
                f"(allowed: {allowed_list}); content-type '{content_type}' unsupported for this endpoint"
            )

        filename = _stub_filename_from_headers(headers)
        if not filename:
            final_path_name = Path(urlparse(final_url).path).name
            url_path_name = Path(urlparse(url).path).name
            filename = final_path_name or url_path_name or "downloaded.tmp"

        suffix = Path(filename).suffix.lower()
        if check_extension and allowed_extensions and suffix not in allowed_extensions:
            mapped_extension = _CONTENT_TYPE_EXTENSIONS.get(content_type)
            final_suffix = Path(urlparse(final_url).path).suffix.lower()
            if mapped_extension in allowed_extensions:
                filename = f"{Path(filename).stem or 'downloaded'}{mapped_extension}"
            elif final_suffix in allowed_extensions:
                filename = Path(urlparse(final_url).path).name
            else:
                allowed_list = ", ".join(sorted(allowed_extensions))
                raise ValueError(
                    f"Downloaded file from {url} does not have an allowed extension "
                    f"(allowed: {allowed_list}); content-type '{content_type}' unsupported for this endpoint"
                )

        if max_bytes and len(content or b"") > int(max_bytes):
            raise ValueError(f"Downloaded file from {url} exceeds maximum allowed size ({max_bytes} bytes).")

        target_path = Path(target_dir) / _safe_stub_filename(filename)
        target_path.write_bytes(content or b"")
        return target_path

    monkeypatch.setattr(download_utils, "_m_afetch", fake_afetch)
    monkeypatch.setattr(process_ebooks, "core_download_url_async", fake_download_url_async)
    monkeypatch.setattr(process_pdfs, "core_download_url_async", fake_download_url_async)
    monkeypatch.setattr(process_documents, "core_download_url_async", fake_download_url_async)
    monkeypatch.setattr(process_code, "download_url_async", fake_download_url_async)
    yield
    _STUB_TABLE.clear()


@pytest.fixture()
def client(client_user_only):
    """
    Use the shared single-user TestClient fixture so that auth and DB handling
    match the rest of the media tests without custom dependency overrides.
    """
    return client_user_only


@pytest.fixture
def dummy_headers():
    return {"token": "dummy"}


def _stub_url(url: str, *, final: str = None, headers: Dict[str, str] = None, body: bytes = None):
    _STUB_TABLE[url] = (final or url, headers or {}, body or b"TEST")


def _error_messages(data: dict) -> list[str]:
    result_errors = [
        str(result.get("error") or "")
        for result in data.get("results", [])
        if isinstance(result, dict)
    ]
    batch_errors = [str(error or "") for error in data.get("errors", [])]
    return [message for message in result_errors + batch_errors if message]


def _assert_error_result(data: dict) -> None:
    assert any("Error" == r.get("status") for r in data.get("results", []))
    assert _error_messages(data)


########################
# EBOOKS
########################

@pytest.mark.parametrize(
    "desc,url,final,headers,expect_status,expect_error",
    [
        ("suffix .epub", "http://t/x.epub", None, {}, 207, None),
        (
            "content-disposition .epub",
            "http://t/download",
            None,
            {"content-disposition": 'attachment; filename="book.epub"'},
            207,
            None,
        ),
        (
            "content-type epub+zip",
            "http://t/any",
            None,
            {"content-type": "application/epub+zip"},
            207,
            None,
        ),
        (
            "reject unknown",
            "http://t/bin",
            None,
            {"content-type": "application/octet-stream"},
            207,
            "allowed extension",
        ),
    ],
)
def test_ebooks_url_acceptance(desc, url, final, headers, expect_status, expect_error, client, dummy_headers, monkeypatch):
    # Stub HTTP
    _stub_url(url, final=final, headers=headers)

    # Stub processing to avoid heavy EPUB parsing
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib as books

    def fake_process_epub(**kwargs):

        return {
            "status": "Success",
            "input_ref": kwargs.get("file_path"),
            "processing_source": kwargs.get("file_path"),
            "media_type": "ebook",
            "content": "ok",
            "metadata": {"title": "t", "author": "a", "raw": {}},
            "chunks": [],
            "analysis": None,
            "keywords": [],
            "warnings": None,
            "error": None,
            "analysis_details": {},
        }

    monkeypatch.setattr(books, "process_epub", fake_process_epub)

    resp = client.post(
        "/api/v1/media/process-ebooks",
        data={"urls": [url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )

    # For suffix/content-disposition/content-type acceptance, downstream processing returns Success.
    # For reject, endpoint returns 207 with Error result.
    if expect_error:
        assert resp.status_code == 207
        data = resp.json()
        _assert_error_result(data)
    else:
        assert resp.status_code in (200, 207)
        data = resp.json()
        # Since we stubbed process_epub to succeed, expect one Success
        assert any("Success" == r.get("status") for r in data.get("results", []))


########################
# PDFs
########################

@pytest.mark.parametrize(
    "desc,url,final,headers,expect_error",
    [
        ("suffix .pdf", "http://t/x.pdf", None, {}, None),
        ("content-disposition .pdf", "http://t/dl", None, {"content-disposition": 'attachment; filename="p.pdf"'}, None),
        ("content-type application/pdf", "http://t/any", None, {"content-type": "application/pdf"}, None),
        ("reject unknown", "http://t/bin", None, {"content-type": "application/octet-stream"}, "allowed extension"),
    ],
)
def test_pdfs_url_acceptance(desc, url, final, headers, expect_error, client, dummy_headers, monkeypatch):
    _stub_url(url, final=final, headers=headers, body=b"%PDF-1.4\n...")

    # Stub processor
    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as pdf_lib

    async def fake_process_pdf_task(**kwargs):
        return {
            "status": "Success",
            "input_ref": kwargs.get("filename"),
            "processing_source": kwargs.get("filename"),
            "media_type": "pdf",
            "parser_used": "pymupdf4llm",
            "content": "ok",
            "metadata": {},
            "chunks": [],
            "analysis": None,
            "keywords": [],
            "warnings": None,
            "error": None,
            "analysis_details": {},
        }

    monkeypatch.setattr(pdf_lib, "process_pdf_task", fake_process_pdf_task)

    resp = client.post(
        "/api/v1/media/process-pdfs",
        data={"urls": [url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )

    if expect_error:
        assert resp.status_code == 207
        data = resp.json()
        _assert_error_result(data)
    else:
        assert resp.status_code in (200, 207)
        data = resp.json()
        assert any("Success" == r.get("status") for r in data.get("results", []))


########################
# Documents
########################

@pytest.mark.parametrize(
    "desc,url,final,headers,expect_error",
    [
        ("suffix .txt", "http://t/x.txt", None, {"content-type": "text/plain"}, None),
        ("content-disposition .md", "http://t/dl", None, {"content-disposition": 'attachment; filename="d.md"'}, None),
        ("content-type text/html", "http://t/any", None, {"content-type": "text/html"}, None),
        ("content-type application/xhtml+xml", "http://t/xhtml", None, {"content-type": "application/xhtml+xml"}, None),
        ("content-type text/xml", "http://t/xml", None, {"content-type": "text/xml"}, None),
        ("content-type application/rtf", "http://t/rtf", None, {"content-type": "application/rtf"}, None),
        ("content-type application/vnd.openxmlformats-officedocument.wordprocessingml.document", "http://t/docx", None, {"content-type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}, None),
        ("reject unknown", "http://t/bin", None, {"content-type": "application/octet-stream"}, "allowed extension"),
    ],
)
def test_documents_url_acceptance(desc, url, final, headers, expect_error, client, dummy_headers, monkeypatch):
    _stub_url(url, final=final, headers=headers, body=b"DATA")

    # Stub processor
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files as docs

    def fake_process_document_content(**kwargs):

        return {
            "status": "Success",
            "input_ref": str(kwargs.get("doc_path")),
            "processing_source": str(kwargs.get("doc_path")),
            "media_type": "document",
            "source_format": Path(str(kwargs.get("doc_path"))).suffix.lstrip("."),
            "content": "ok",
            "metadata": {},
            "chunks": [],
            "analysis": None,
            "analysis_details": {},
            "keywords": [],
            "error": None,
            "warnings": None,
        }

    monkeypatch.setattr(docs, "process_document_content", fake_process_document_content)

    resp = client.post(
        "/api/v1/media/process-documents",
        data={"urls": [url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )

    if expect_error:
        assert resp.status_code == 207
        data = resp.json()
        _assert_error_result(data)
    else:
        assert resp.status_code in (200, 207)
        data = resp.json()
        assert any("Success" == r.get("status") for r in data.get("results", []))


########################
# Code
########################

@pytest.mark.parametrize(
    "desc,url,final,headers,expect_error",
    [
        ("suffix .py", "http://t/x.py", None, {}, None),
        ("content-disposition .ts", "http://t/dl", None, {"content-disposition": 'attachment; filename="f.ts"'}, None),
        ("reject unknown", "http://t/bin", None, {"content-type": "application/octet-stream"}, "allowed extension"),
    ],
)
def test_code_url_acceptance(desc, url, final, headers, expect_error, client, dummy_headers):
    _stub_url(url, final=final, headers=headers, body=b"print('hi')\n")

    resp = client.post(
        "/api/v1/media/process-code",
        data={"urls": [url], "perform_chunking": "false"},
        headers=dummy_headers,
    )

    if expect_error:
        assert resp.status_code == 207
        data = resp.json()
        _assert_error_result(data)
    else:
        assert resp.status_code in (200, 207)
        data = resp.json()
        assert any("Success" == r.get("status") for r in data.get("results", []))


def test_code_url_acceptance_redirect_final_suffix(client, dummy_headers):


    url = "http://t/dl"
    final = "http://t/file.rs"
    _stub_url(url, final=final, headers={}, body=b"fn main() {}\n")

    resp = client.post(
        "/api/v1/media/process-code",
        data={"urls": [url], "perform_chunking": "false"},
        headers=dummy_headers,
    )
    assert resp.status_code in (200, 207)
    assert any(r.get("status") == "Success" for r in resp.json().get("results", []))


def test_code_mixed_urls_multi_status(client, dummy_headers):


    ok_url = "http://t/good.py"
    bad_url = "http://t/unknown"
    _stub_url(ok_url, headers={}, body=b"def x():\n    return 1\n")
    _stub_url(bad_url, headers={"content-type": "application/octet-stream"})

    resp = client.post(
        "/api/v1/media/process-code",
        data={"urls": [ok_url, bad_url], "perform_chunking": "false"},
        headers=dummy_headers,
    )
    assert resp.status_code == 207
    data = resp.json()
    assert any(r.get("status") == "Success" for r in data.get("results", []))
    assert any(r.get("status") == "Error" for r in data.get("results", []))


########################
# Redirect final suffix acceptance
########################

def test_ebooks_url_acceptance_redirect_final_suffix(client, dummy_headers, monkeypatch):

    url = "http://t/dl"
    final = "http://t/file.epub"
    _stub_url(url, final=final, headers={})

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib as books

    def fake_process_epub(**kwargs):

        return {"status": "Success", "input_ref": kwargs.get("file_path"), "processing_source": kwargs.get("file_path"), "media_type": "ebook", "content": "ok", "metadata": {}, "chunks": [], "analysis": None, "keywords": [], "warnings": None, "error": None, "analysis_details": {}}

    monkeypatch.setattr(books, "process_epub", fake_process_epub)

    resp = client.post(
        "/api/v1/media/process-ebooks",
        data={"urls": [url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )
    assert resp.status_code in (200, 207)
    assert any(r.get("status") == "Success" for r in resp.json().get("results", []))


def test_pdfs_url_acceptance_redirect_final_suffix(client, dummy_headers, monkeypatch):


    url = "http://t/dl"
    final = "http://t/file.pdf"
    _stub_url(url, final=final, headers={}, body=b"%PDF-1.4\n...")

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as pdf_lib

    async def fake_process_pdf_task(**kwargs):
        return {"status": "Success", "input_ref": kwargs.get("filename"), "processing_source": kwargs.get("filename"), "media_type": "pdf", "parser_used": "pymupdf4llm", "content": "ok", "metadata": {}, "chunks": [], "analysis": None, "keywords": [], "warnings": None, "error": None, "analysis_details": {}}

    monkeypatch.setattr(pdf_lib, "process_pdf_task", fake_process_pdf_task)

    resp = client.post(
        "/api/v1/media/process-pdfs",
        data={"urls": [url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )
    assert resp.status_code in (200, 207)
    assert any(r.get("status") == "Success" for r in resp.json().get("results", []))


def test_documents_url_acceptance_redirect_final_suffix(client, dummy_headers, monkeypatch):


    url = "http://t/dl"
    final = "http://t/file.html"
    _stub_url(url, final=final, headers={}, body=b"<html>ok</html>")

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files as docs

    def fake_process_document_content(**kwargs):

        p = str(kwargs.get("doc_path"))
        return {"status": "Success", "input_ref": p, "processing_source": p, "media_type": "document", "source_format": Path(p).suffix.lstrip("."), "content": "ok", "metadata": {}, "chunks": [], "analysis": None, "analysis_details": {}, "keywords": [], "error": None, "warnings": None}

    monkeypatch.setattr(docs, "process_document_content", fake_process_document_content)

    resp = client.post(
        "/api/v1/media/process-documents",
        data={"urls": [url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )
    assert resp.status_code in (200, 207)
    assert any(r.get("status") == "Success" for r in resp.json().get("results", []))


########################
# Mixed batches (expect 207)
########################

def test_ebooks_mixed_urls_multi_status(client, dummy_headers, monkeypatch):

    ok_url = "http://t/book"
    bad_url = "http://t/unknown"
    _stub_url(ok_url, headers={"content-disposition": 'attachment; filename="a.epub"'})
    _stub_url(bad_url, headers={"content-type": "application/octet-stream"})

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Books.Book_Processing_Lib as books

    def fake_process_epub(**kwargs):

        return {"status": "Success", "input_ref": kwargs.get("file_path"), "processing_source": kwargs.get("file_path"), "media_type": "ebook", "content": "ok", "metadata": {}, "chunks": [], "analysis": None, "keywords": [], "warnings": None, "error": None, "analysis_details": {}}

    monkeypatch.setattr(books, "process_epub", fake_process_epub)

    resp = client.post(
        "/api/v1/media/process-ebooks",
        data={"urls": [ok_url, bad_url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )
    assert resp.status_code == 207
    data = resp.json()
    assert any(r.get("status") == "Success" for r in data.get("results", []))
    assert any(r.get("status") == "Error" for r in data.get("results", []))


def test_pdfs_mixed_urls_multi_status(client, dummy_headers, monkeypatch):


    ok_url = "http://t/x.pdf"
    bad_url = "http://t/unknown"
    _stub_url(ok_url, headers={"content-type": "application/pdf"}, body=b"%PDF-1.4\n...")
    _stub_url(bad_url, headers={"content-type": "application/octet-stream"})

    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as pdf_lib

    async def fake_process_pdf_task(**kwargs):
        return {"status": "Success", "input_ref": kwargs.get("filename"), "processing_source": kwargs.get("filename"), "media_type": "pdf", "parser_used": "pymupdf4llm", "content": "ok", "metadata": {}, "chunks": [], "analysis": None, "keywords": [], "warnings": None, "error": None, "analysis_details": {}}

    monkeypatch.setattr(pdf_lib, "process_pdf_task", fake_process_pdf_task)

    resp = client.post(
        "/api/v1/media/process-pdfs",
        data={"urls": [ok_url, bad_url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )
    assert resp.status_code == 207
    data = resp.json()
    assert any(r.get("status") == "Success" for r in data.get("results", []))
    assert any(r.get("status") == "Error" for r in data.get("results", []))


def test_documents_mixed_urls_multi_status(client, dummy_headers, monkeypatch):


    ok_url = "http://t/page"
    bad_url = "http://t/unknown"
    _stub_url(ok_url, headers={"content-type": "text/html"}, body=b"<html>ok</html>")
    _stub_url(bad_url, headers={"content-type": "application/octet-stream"})

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files as docs

    def fake_process_document_content(**kwargs):

        p = str(kwargs.get("doc_path"))
        return {"status": "Success", "input_ref": p, "processing_source": p, "media_type": "document", "source_format": Path(p).suffix.lstrip("."), "content": "ok", "metadata": {}, "chunks": [], "analysis": None, "analysis_details": {}, "keywords": [], "error": None, "warnings": None}

    monkeypatch.setattr(docs, "process_document_content", fake_process_document_content)

    resp = client.post(
        "/api/v1/media/process-documents",
        data={"urls": [ok_url, bad_url], "perform_analysis": "false", "perform_chunking": "false"},
        headers=dummy_headers,
    )
    assert resp.status_code == 207
    data = resp.json()
    assert any(r.get("status") == "Success" for r in data.get("results", []))
    assert any(r.get("status") == "Error" for r in data.get("results", []))


########################
# Negative content types (application/msword)
########################

@pytest.mark.parametrize("endpoint", [
    "/api/v1/media/process-ebooks",
    "/api/v1/media/process-pdfs",
    "/api/v1/media/process-documents",
])
def test_reject_msword_content_type(endpoint, client, dummy_headers):
    url = "http://t/msword"
    # application/msword should be rejected by all three endpoints
    _stub_url(url, headers={"content-type": "application/msword"}, body=b"...")

    resp = client.post(endpoint, data={"urls": [url], "perform_analysis": "false", "perform_chunking": "false"}, headers=dummy_headers)
    # Multi-status with an error result
    assert resp.status_code == 207
    data = resp.json()
    _assert_error_result(data)
