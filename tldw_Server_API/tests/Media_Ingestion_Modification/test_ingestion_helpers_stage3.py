from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
    TempDirManager,
    save_uploaded_files,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.pipeline import (
    ProcessItem,
    run_batch_processor,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.result_normalization import (
    normalize_process_batch,
)


class _DummyUploadFile:
    """
    Minimal UploadFile-like object for save_uploaded_files tests.

    It provides the `filename` attribute and an async `read()` method, which is
    all the helper relies on.
    """

    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = content
        self._consumed = False

    async def read(self, chunk_size: int = -1) -> bytes:  # pragma: no cover - trivial
        if self._consumed:
            return b""
        self._consumed = True
        return self._content


class _DummyValidator:
    """
    Simple validator stub used to avoid coupling these tests to the full
    Upload_Sink implementation.
    """

    def get_media_config(self, media_key: Optional[str]) -> Dict[str, Any]:
        # Provide a small but non-zero size limit to exercise the path that
        # enforces max size, without actually triggering it in these tests.
        return {"max_size_mb": 10} if media_key else {}


def test_tempdirmanager_creates_and_cleans() -> None:


    mgr = TempDirManager(prefix="test_media_stage3_", cleanup=True)
    with mgr as tmp_dir:
        assert isinstance(tmp_dir, Path)
        assert tmp_dir.is_dir()
        # While inside the context, get_path should return the same directory.
        assert mgr.get_path() == tmp_dir

    # After context exit, the directory should be cleaned up and get_path fails.
    with pytest.raises(RuntimeError):
        mgr.get_path()

    assert not tmp_dir.exists()


@pytest.mark.asyncio
async def test_save_uploaded_files_blocks_dangerous_extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure that any unexpected call into process_and_validate_file fails loudly
    async def _fail_process_and_validate_file(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("process_and_validate_file should not be called for blocked extensions")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing.process_and_validate_file",
        _fail_process_and_validate_file,
        raising=False,
    )

    files: List[_DummyUploadFile] = [
        _DummyUploadFile("malicious.exe", b"dummy-binary"),
    ]
    processed, errors = await save_uploaded_files(
        files=files,
        temp_dir=tmp_path,
        validator=_DummyValidator(),
    )

    assert processed == []
    assert len(errors) == 1
    err = errors[0]
    assert err["status"] == "Error"
    assert "not allowed for security reasons" in err["error"]


@pytest.mark.asyncio
async def test_save_uploaded_files_enforces_fractional_size_limit(tmp_path: Path) -> None:
    class _SizeLimitValidator(_DummyValidator):
        def get_media_config(self, media_key: Optional[str]) -> Dict[str, Any]:
            return {"max_size_mb": 0.5} if media_key else {}

    payload = b"x" * (1024 * 1024 + 1)
    files: List[_DummyUploadFile] = [
        _DummyUploadFile("oversize.txt", payload),
    ]

    processed, errors = await save_uploaded_files(
        files=files,
        temp_dir=tmp_path,
        validator=_SizeLimitValidator(),
    )

    assert processed == []
    assert len(errors) == 1
    assert "exceeds maximum allowed size" in errors[0]["error"]


@pytest.mark.asyncio
async def test_save_uploaded_files_explicit_media_type_overrides_filename_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RecordingValidator(_DummyValidator):
        def __init__(self) -> None:
            self.config_keys: list[Optional[str]] = []

        def get_media_config(self, media_key: Optional[str]) -> Dict[str, Any]:
            self.config_keys.append(media_key)
            return {"max_size_mb": 50}

    validation_media_keys: list[Optional[str]] = []

    def _record_process_and_validate_file(*_args: Any, **kwargs: Any) -> bool:
        validation_media_keys.append(kwargs.get("media_type_key_override"))
        return True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing.process_and_validate_file",
        _record_process_and_validate_file,
        raising=True,
    )

    validator = _RecordingValidator()
    files: List[_DummyUploadFile] = [
        _DummyUploadFile("page.xhtml", b"<html><body>document</body></html>"),
    ]

    processed, errors = await save_uploaded_files(
        files=files,
        temp_dir=tmp_path,
        validator=validator,
        allowed_extensions=[".xhtml"],
        expected_media_type_key="document",
    )

    assert errors == []
    assert len(processed) == 1
    assert validator.config_keys == ["document"]
    assert validation_media_keys == ["document"]


@pytest.mark.asyncio
async def test_run_batch_processor_counts_and_orders(tmp_path: Path) -> None:
    items = [
        ProcessItem(
            input_ref="item-success",
            local_path=tmp_path / "a.txt",
            media_type="document",
            metadata={},
        ),
        ProcessItem(
            input_ref="item-warning",
            local_path=tmp_path / "b.txt",
            media_type="document",
            metadata={},
        ),
    ]

    async def _processor(process_items: List[ProcessItem]) -> List[Dict[str, Any]]:
        # First item succeeds, second yields a warning, and we append one error.
        results: List[Dict[str, Any]] = [
            {
                "status": "Success",
                "input_ref": process_items[0].input_ref,
            },
            {
                "status": "Warning",
                "input_ref": process_items[1].input_ref,
            },
            {
                "status": "Error",
                "input_ref": "bad-item",
            },
        ]
        return results

    base_batch: Dict[str, Any] = {"results": [], "errors": ["pre-existing-error"]}

    batch = await run_batch_processor(items, _processor, base_batch=base_batch)

    # "Success" and "Warning" contribute to processed_count; a single Error contributes to errors_count.
    assert batch["processed_count"] == 2
    assert batch["errors_count"] == 1
    assert batch["errors"] == ["pre-existing-error"]

    statuses = [r["status"] for r in batch["results"]]
    # Success and Warning should be ordered before Error.
    assert statuses[0] in {"Success", "Warning"}
    assert statuses[1] in {"Success", "Warning"}
    assert statuses[-1] == "Error"


def test_normalize_process_batch_orders_and_sets_defaults() -> None:


    batch: Dict[str, Any] = {
        "results": [
            {"status": "Error", "input_ref": "b"},
            {"status": "Success", "input_ref": "a"},
        ]
    }

    normalized = normalize_process_batch(batch)

    # Success result should be first after normalization.
    refs = [r["input_ref"] for r in normalized["results"]]
    assert refs == ["a", "b"]

    # Default counters and errors list should be present.
    assert "processed_count" in normalized
    assert "errors_count" in normalized
    assert "errors" in normalized
