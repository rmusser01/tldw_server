import os
from io import BytesIO
from pathlib import Path

import pytest
from loguru import logger


from tldw_Server_API.app.api.v1.API_Deps.validations_deps import file_validator_instance
from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
    save_uploaded_files,
)


class DummyUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._bio = BytesIO(content)

    async def read(self, n: int) -> bytes:
        return self._bio.read(n)

    async def close(self) -> None:
        """Mimic FastAPI's UploadFile.close() as an awaitable.
        Ensures production code awaiting file.close() does not raise.
        """
        try:
            self._bio.close()
        except Exception:
            _ = None


@pytest.fixture
def tmp_media_dir(tmp_path: Path) -> Path:
    d = tmp_path / "media_uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.mark.asyncio
async def test_save_uploaded_files_empty_file_cleanup_logs(tmp_media_dir, capsys):
    # Capture loguru logs to stdout for simple assertion
    sink_id = logger.add(lambda m: print(m, end=""))
    try:
        files = [DummyUploadFile("empty.txt", b"")]
        processed, errors = await save_uploaded_files(
            files,
            tmp_media_dir,
            validator=file_validator_instance,
            allowed_extensions=[".txt"],
        )
        assert processed == []
        assert len(errors) == 1
        assert errors[0]["status"] == "Error"
        # Ensure a warning about empty upload was logged and no crash occurred
        out = capsys.readouterr().out
        assert "is empty. Skipping." in out
    finally:
        logger.remove(sink_id)


@pytest.mark.asyncio
async def test_save_uploaded_files_write_failure_cleanup(tmp_media_dir, monkeypatch):
    # Force aiofiles.open to raise OSError to exercise the write-error cleanup path
    import tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing as input_sourcing_mod

    class _FailOpen:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            raise OSError("simulated write open error")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _fake_open(*a, **kw):

        return _FailOpen()

    monkeypatch.setattr(input_sourcing_mod.aiofiles, "open", _fake_open, raising=True)

    files = [DummyUploadFile("code.py", b"print('x')\n")]
    processed, errors = await save_uploaded_files(
        files,
        tmp_media_dir,
        validator=file_validator_instance,
        allowed_extensions=[".py"],
        expected_media_type_key="code",
    )
    assert processed == []
    assert len(errors) == 1
    assert errors[0]["status"] == "Error"
    assert errors[0]["error"] == "Failed to save uploaded file"
    assert "simulated write open error" not in str(errors[0]["error"])


@pytest.mark.asyncio
async def test_save_uploaded_files_sanitizes_unexpected_code_validation_failure(
    tmp_media_dir,
    monkeypatch,
):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing as input_sourcing_mod

    def fail_process_and_validate_file(*_args, **_kwargs):
        raise RuntimeError("validator exploded at /private/cache/upload.py")

    monkeypatch.setattr(
        input_sourcing_mod,
        "process_and_validate_file",
        fail_process_and_validate_file,
        raising=True,
    )

    files = [DummyUploadFile("code.py", b"print('x')\n")]
    processed, errors = await save_uploaded_files(
        files,
        tmp_media_dir,
        validator=file_validator_instance,
        allowed_extensions=[".py"],
        expected_media_type_key="code",
    )

    assert processed == []
    assert len(errors) == 1
    assert errors[0]["status"] == "Error"
    assert errors[0]["error"] == "File validation failed"
    assert "validator exploded" not in str(errors[0]["error"])
    assert "/private/cache/upload.py" not in str(errors[0]["error"])


@pytest.mark.asyncio
async def test_save_uploaded_files_sanitizes_unexpected_document_validation_failure(
    tmp_media_dir,
    monkeypatch,
):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing as input_sourcing_mod

    def _fail_validation(*_args, **_kwargs):
        raise RuntimeError("validator backend exploded at /private/cache/upload.txt")

    monkeypatch.setattr(
        input_sourcing_mod,
        "process_and_validate_file",
        _fail_validation,
        raising=True,
    )

    files = [DummyUploadFile("sample.txt", b"hello\n")]
    processed, errors = await save_uploaded_files(
        files,
        tmp_media_dir,
        validator=file_validator_instance,
        allowed_extensions=[".txt"],
        expected_media_type_key="document",
    )

    assert processed == []
    assert len(errors) == 1
    assert errors[0]["status"] == "Error"
    assert errors[0]["error"] == "File validation failed"
    assert "validator backend exploded" not in str(errors[0]["error"])
    assert "/private/cache/upload.txt" not in str(errors[0]["error"])


@pytest.mark.asyncio
async def test_save_uploaded_files_sanitizes_unexpected_upload_processing_failure(
    tmp_media_dir,
    monkeypatch,
):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing as input_sourcing_mod

    def fail_extension_candidates(_filename):
        raise RuntimeError("extension resolver exploded at /private/cache/upload.txt")

    monkeypatch.setattr(
        input_sourcing_mod,
        "_extension_candidates",
        fail_extension_candidates,
        raising=True,
    )

    files = [DummyUploadFile("sample.txt", b"hello\n")]
    processed, errors = await save_uploaded_files(
        files,
        tmp_media_dir,
        validator=file_validator_instance,
        allowed_extensions=[".txt"],
        expected_media_type_key="document",
    )

    assert processed == []
    assert len(errors) == 1
    assert errors[0]["status"] == "Error"
    assert errors[0]["error"] == "Upload processing failed"
    assert "extension resolver exploded" not in str(errors[0]["error"])
    assert "/private/cache/upload.txt" not in str(errors[0]["error"])
