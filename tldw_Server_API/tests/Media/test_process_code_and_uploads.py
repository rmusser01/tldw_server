import os
import io
import json
import tarfile
import pytest
from dataclasses import dataclass, field
from fastapi import UploadFile


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.debug_kwargs: list[dict[str, object]] = []
        self.warnings: list[str] = []

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append(message.format(*args, **kwargs) if args or kwargs else message)
        self.debug_kwargs.append(dict(kwargs))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append(message.format(*args, **kwargs) if args or kwargs else message)


class _WarningFailingLoggerStub(_LoggerStub):
    def warning(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("test-mode diagnostics logger exploded at /private/logger")


def _assert_sanitized_debug_log(logger: _LoggerStub, expected: str) -> None:
    target_kwargs = [
        kwargs for message, kwargs in zip(logger.debugs, logger.debug_kwargs) if message == expected
    ]
    assert target_kwargs, logger.debugs
    rendered = "\n".join(message for message in logger.debugs if message == expected)
    assert "exploded" not in rendered
    assert "/private/" not in rendered
    assert all(not kwargs for kwargs in target_kwargs)


def _assert_sanitized_warning_log(logger: _LoggerStub, expected: str) -> None:
    assert expected in logger.warnings
    rendered = "\n".join(logger.warnings)
    assert "exploded" not in rendered
    assert "/private/" not in rendered


def test_process_code_js_lines(client_with_single_user):


    client, _ = client_with_single_user
    code = b"console.log('hi');\nconsole.log('bye');\n"
    files = [("files", ("script.js", code, "application/javascript"))]
    data = {
        "perform_chunking": "true",
        "chunk_method": "lines",
        "chunk_size": "1",
        "chunk_overlap": "0",
    }
    r = client.post("/api/v1/media/process-code", files=files, data=data)
    assert r.status_code in (200, 207), r.text
    payload = r.json()
    assert payload.get("results"), payload
    res0 = payload["results"][0]
    assert res0["status"] in ("Success", "Warning"), res0
    assert isinstance(res0.get("chunks"), list)
    # For lines method with size=1 and 2 lines, expect at least 2 chunks
    assert len(res0["chunks"]) >= 2


def test_process_code_js_codechunk(client_with_single_user):


    client, _ = client_with_single_user
    code = b"function add(a,b){return a+b;}\nexport default add;\n"
    files = [("files", ("lib.js", code, "application/javascript"))]
    data = {
        "perform_chunking": "true",
        "chunk_method": "code",
        "chunk_size": "4000",
        "chunk_overlap": "100",
    }
    r = client.post("/api/v1/media/process-code", files=files, data=data)
    # Even if chunker falls back, endpoint should succeed
    assert r.status_code in (200, 207), r.text
    payload = r.json()
    assert payload.get("results"), payload
    assert payload["results"][0]["status"] in ("Success", "Warning")


def test_process_code_sanitizes_read_failure(client_with_single_user, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import process_code as process_code_mod

    client, _ = client_with_single_user

    def fail_read_text_safe(_path):
        raise RuntimeError("read failed at /private/source.py")

    monkeypatch.setattr(process_code_mod, "read_text_safe", fail_read_text_safe)

    files = [("files", ("script.py", b"print('hi')\n", "text/x-python"))]
    response = client.post(
        "/api/v1/media/process-code",
        files=files,
        data={"perform_chunking": "false"},
    )

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Failed to read code file"
    assert "read failed" not in response.text
    assert "/private/source.py" not in response.text


def test_process_code_sanitizes_url_download_failure(client_with_single_user, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import process_code as process_code_mod

    client, _ = client_with_single_user

    async def fail_download_url_async(*_args, **_kwargs):
        raise RuntimeError("download failed at /private/cache/snippet.py")

    monkeypatch.setattr(process_code_mod, "download_url_async", fail_download_url_async)

    response = client.post(
        "/api/v1/media/process-code",
        data={
            "urls": "https://example.com/snippet.py",
            "perform_chunking": "false",
        },
    )

    assert response.status_code == 207, response.text
    payload = response.json()
    result = payload["results"][0]
    assert result["status"] == "Error"
    assert result["error"] == "Download/preparation failed"
    assert "download failed" not in response.text
    assert "/private/cache/snippet.py" not in response.text


@pytest.mark.asyncio
async def test_save_uploaded_files_extension_candidates_tar_gz(tmp_path, monkeypatch):
    # Call internal helper to validate multi-suffix support (.tar.gz)
    from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
        save_uploaded_files,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    tar_buffer = io.BytesIO()
    payload = b"archive payload\n"
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        member = tarfile.TarInfo("payload.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    tar_buffer.seek(0)
    content = tar_buffer.getvalue()
    up = UploadFile(filename="archive.tar.gz", file=io.BytesIO(content))

    saved, errors = await save_uploaded_files(
        files=[up],
        temp_dir=tmp_path,
        validator=FileValidator(),
        allowed_extensions=[".tar.gz"],
        skip_archive_scanning=True,
    )
    assert not errors, errors
    assert len(saved) == 1
    assert saved[0]["original_filename"] == "archive.tar.gz"
    assert str(saved[0]["path"]).endswith("archive.tar.gz")
    assert "archive.tar.tar.gz" not in str(saved[0]["path"])


def test_process_docs_streaming_respects_validator_limits(client_with_single_user, monkeypatch, tmp_path):


    # Monkeypatch the file_validator_instance to enforce a tiny max size for documents
    from tldw_Server_API.app.api.v1.endpoints import media as media_mod
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    tiny_validator = FileValidator(custom_media_configs={
        "document": {"max_size_mb": 0.0001},  # ~100 bytes
    })
    monkeypatch.setattr(media_mod, "file_validator_instance", tiny_validator)

    client, _ = client_with_single_user
    big = b"x" * 200  # > 100 bytes
    files = [("files", ("note.txt", big, "text/plain"))]
    data = {
        "perform_analysis": "false",
    }
    r = client.post("/api/v1/media/process-documents", files=files, data=data)
    # Expect partial failure with an oversize error or hard 413
    if r.status_code == 413:
        return
    assert r.status_code in (200, 207), r.text
    payload = r.json()
    assert payload.get("errors_count", 0) >= 1


def test_process_code_logs_upload_errors_when_test_mode_is_single_letter_y(
    client_with_single_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints.media import process_code as process_code_mod

    client, _ = client_with_single_user
    monkeypatch.setenv("TEST_MODE", "y")
    logger_stub = _LoggerStub()
    monkeypatch.setattr(process_code_mod, "logger", logger_stub)

    files = [("files", ("bad.exe", b"MZ\x90\x00", "application/octet-stream"))]
    response = client.post("/api/v1/media/process-code", files=files, data={})

    assert response.status_code in (200, 207), response.text
    _assert_sanitized_warning_log(logger_stub, "TEST_MODE: process-code upload errors")


def test_process_code_upload_diagnostic_failure_log_is_sanitized(
    client_with_single_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints.media import process_code as process_code_mod

    client, _ = client_with_single_user
    monkeypatch.setenv("TEST_MODE", "y")
    logger_stub = _WarningFailingLoggerStub()
    monkeypatch.setattr(process_code_mod, "logger", logger_stub)

    files = [("files", ("bad.exe", b"MZ\x90\x00", "application/octet-stream"))]
    response = client.post("/api/v1/media/process-code", files=files, data={})

    assert response.status_code in (200, 207), response.text
    _assert_sanitized_debug_log(logger_stub, "Failed to emit TEST_MODE upload diagnostics")


def test_process_code_read_error_test_mode_log_is_sanitized(
    client_with_single_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints.media import process_code as process_code_mod

    client, _ = client_with_single_user
    monkeypatch.setenv("TEST_MODE", "y")
    logger_stub = _LoggerStub()

    def fail_read_text_safe(_path):
        raise RuntimeError("read failed at /private/source.py")

    monkeypatch.setattr(process_code_mod, "logger", logger_stub)
    monkeypatch.setattr(process_code_mod, "read_text_safe", fail_read_text_safe)

    files = [("files", ("script.py", b"print('hi')\n", "text/x-python"))]
    response = client.post(
        "/api/v1/media/process-code",
        files=files,
        data={"perform_chunking": "false"},
    )

    assert response.status_code == 207, response.text
    _assert_sanitized_warning_log(logger_stub, "TEST_MODE: process-code read error")


def test_process_code_read_error_diagnostic_failure_log_is_sanitized(
    client_with_single_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints.media import process_code as process_code_mod

    client, _ = client_with_single_user
    monkeypatch.setenv("TEST_MODE", "y")
    logger_stub = _WarningFailingLoggerStub()

    def fail_read_text_safe(_path):
        raise RuntimeError("read failed at /private/source.py")

    monkeypatch.setattr(process_code_mod, "logger", logger_stub)
    monkeypatch.setattr(process_code_mod, "read_text_safe", fail_read_text_safe)

    files = [("files", ("script.py", b"print('hi')\n", "text/x-python"))]
    response = client.post(
        "/api/v1/media/process-code",
        files=files,
        data={"perform_chunking": "false"},
    )

    assert response.status_code == 207, response.text
    _assert_sanitized_debug_log(logger_stub, "Failed to emit TEST_MODE read-error diagnostics")


def test_process_code_chunk_line_bounds_failure_log_is_sanitized(
    client_with_single_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints.media import process_code as process_code_mod
    import tldw_Server_API.app.core.Chunking.chunker as chunker_mod

    client, _ = client_with_single_user
    logger_stub = _LoggerStub()

    class BadLineNumber(int):
        def __new__(cls, value: int):
            return int.__new__(cls, value)

        def __lt__(self, _other):
            raise RuntimeError("line bounds exploded at /private/source.py")

    @dataclass
    class FakeMetadata:
        start_line: int | None = None
        end_line: int | None = None
        blocks: list[dict[str, object]] = field(
            default_factory=lambda: [
                {"start_line": BadLineNumber(1)},
                {"start_line": BadLineNumber(2)},
            ]
        )
        options: dict[str, object] = field(default_factory=dict)

    @dataclass
    class FakeChunkResult:
        text: str
        metadata: FakeMetadata

    class FakeChunker:
        def __init__(self, *_args, **_kwargs):
            return None

        def chunk_text_with_metadata(self, *_args, **_kwargs):
            return [FakeChunkResult(text="console.log('hi');", metadata=FakeMetadata())]

    monkeypatch.setattr(process_code_mod, "logger", logger_stub)
    monkeypatch.setattr(chunker_mod, "Chunker", FakeChunker)

    files = [("files", ("script.js", b"console.log('hi');\n", "application/javascript"))]
    response = client.post(
        "/api/v1/media/process-code",
        files=files,
        data={
            "perform_chunking": "true",
            "chunk_method": "code",
            "chunk_size": "4000",
            "chunk_overlap": "0",
        },
    )

    assert response.status_code == 200, response.text
    _assert_sanitized_debug_log(logger_stub, "Failed to derive code chunk line bounds")


@pytest.mark.asyncio
async def test_pdf_analysis_without_explicit_api_key(monkeypatch):
    # Unit level: exercise process_pdf_task so that analysis runs with api_name only
    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as pdf_mod

    # Stub parser and metadata to avoid heavy dependencies and errors
    monkeypatch.setattr(pdf_mod, "pymupdf4llm_parse_pdf", lambda path: "Some extracted content")
    monkeypatch.setattr(pdf_mod, "extract_metadata_from_pdf", lambda path: {})
    # Stub analyze to a quick response
    monkeypatch.setattr(pdf_mod, "analyze", lambda **kwargs: "OK")

    out = await pdf_mod.process_pdf_task(
        file_bytes=b"%PDF-fake",
        filename="paper.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=True,
        api_name="openai",
        api_key=None,
    )
    assert out.get("status") in ("Success", "Warning"), out
    # Analysis should run using api_name only
    assert out.get("analysis") == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parser_name", "patch_attr"),
    [
        ("pymupdf4llm", "pymupdf4llm_parse_pdf"),
        ("pymupdf", "extract_text_and_format_from_pdf"),
        ("docling", "docling_parse_pdf"),
    ],
)
async def test_pdf_text_normalization_applies_to_all_parser_paths(
    monkeypatch,
    parser_name,
    patch_attr,
):
    import importlib.util
    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as pdf_mod

    wrapped_text = (
        "We are not just interested in models that perform well on a\n"
        "single physical task."
    )
    monkeypatch.setattr(pdf_mod, patch_attr, lambda path: wrapped_text)
    if parser_name == "docling":
        original_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: object() if name == "docling.document_converter" else original_find_spec(name),
        )
        monkeypatch.setattr(pdf_mod, "_is_usable_torch_module_for_docling", lambda: True)

    out = await pdf_mod.process_pdf_task(
        file_bytes=b"%PDF-fake",
        filename=f"{parser_name}.pdf",
        parser=parser_name,
        perform_chunking=False,
        perform_analysis=False,
    )
    assert out.get("status") in ("Success", "Warning"), out
    assert "perform well on a single physical task." in (out.get("content") or "")


@pytest.mark.asyncio
async def test_pdf_text_normalization_applies_after_ocr_merge(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as pdf_mod

    monkeypatch.setattr(pdf_mod, "pymupdf4llm_parse_pdf", lambda path: "Parser output")
    monkeypatch.setattr(
        pdf_mod,
        "_ocr_pdf_pages",
        lambda **kwargs: ("OCR line one\nline two", 1, [1], None),
    )

    class _FakeOcrBackend:
        name = "fake-ocr"

    monkeypatch.setattr(pdf_mod, "_get_ocr_backend", lambda _name=None: _FakeOcrBackend())

    out = await pdf_mod.process_pdf_task(
        file_bytes=b"%PDF-fake",
        filename="ocr-path.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_mode="always",
    )
    assert out.get("status") in ("Success", "Warning"), out
    assert out.get("content") == "OCR line one line two"


@pytest.mark.asyncio
async def test_pdf_text_normalization_records_analysis_details(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as pdf_mod

    monkeypatch.setattr(
        pdf_mod,
        "pymupdf4llm_parse_pdf",
        lambda path: "Paragraph starts here\nand continues there.",
    )

    out = await pdf_mod.process_pdf_task(
        file_bytes=b"%PDF-fake",
        filename="normalization-details.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
    )
    assert out.get("status") in ("Success", "Warning"), out
    details = (out.get("analysis_details") or {}).get("text_normalization") or {}
    assert details.get("applied") is True
    assert details.get("mode") == "paragraph_safe"
    assert details.get("chars_before", 0) >= details.get("chars_after", 0)
    assert details.get("line_breaks_before", 0) >= details.get("line_breaks_after", 0)


@pytest.mark.asyncio
async def test_pdf_text_normalization_fails_softly(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.PDF_Processing_Lib as pdf_mod

    raw_text = "Line one\nline two"
    monkeypatch.setattr(pdf_mod, "pymupdf4llm_parse_pdf", lambda path: raw_text)
    monkeypatch.setattr(
        pdf_mod,
        "normalize_pdf_text_for_storage",
        lambda _text: (_ for _ in ()).throw(ValueError("normalize boom")),
    )

    out = await pdf_mod.process_pdf_task(
        file_bytes=b"%PDF-fake",
        filename="normalization-fail-soft.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
    )
    assert out.get("status") in ("Warning", "Success"), out
    warnings = out.get("warnings") or []
    assert any("Text normalization failed" in warning for warning in warnings)
    assert out.get("content") == raw_text
