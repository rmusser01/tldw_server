from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import (
    persistence as ingestion_persistence,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files


pytestmark = pytest.mark.unit


class _MetricsCapture:
    def __init__(self) -> None:
        self.increment_calls: list[tuple[str, float, dict[str, Any] | None]] = []

    def increment(
        self,
        metric_name: str,
        value: float = 1,
        labels: dict[str, Any] | None = None,
    ) -> None:
        self.increment_calls.append((metric_name, value, labels))

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@pytest.mark.parametrize(
    "media_type,metadata",
    [
        ("audio", {"title": "Audio Sample", "duration": 12.5}),
        ("video", {"title": "Video Sample", "duration": 42}),
        ("document", {"title": "Doc Sample", "parser_used": "builtin-text"}),
        ("pdf", {"title": "PDF Sample", "parser_used": "pymupdf4llm"}),
        ("ebook", {"title": "Book Sample", "parser_used": "ebooklib"}),
        ("json", {"title": "JSON Sample", "parser_used": "builtin-json"}),
        ("email", {"title": "Email Sample", "email": {"subject": "Hello"}}),
    ],
)
def test_metadata_contract_matrix_accepts_minimum(media_type: str, metadata: dict[str, Any]) -> None:
    issues = ingestion_persistence._evaluate_metadata_contract_issues(
        media_type=media_type,
        metadata=metadata,
    )
    assert issues == []


def test_enforce_metadata_contract_warn_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _MetricsCapture()
    monkeypatch.setattr(ingestion_persistence, "get_metrics_registry", lambda: metrics)
    monkeypatch.setenv("MEDIA_METADATA_CONTRACT_POLICY", "warn")

    result = {
        "status": "Success",
        "input_ref": "missing-metadata-item",
        "metadata": {},
        "warnings": None,
        "error": None,
    }
    ingestion_persistence._enforce_metadata_contract_on_result(
        result=result,
        media_type="document",
        form_data=SimpleNamespace(metadata_contract_policy=None),
        path_kind="upload",
        processor="document_processor",
    )

    assert result["status"] == "Success"
    warnings = result.get("warnings") or []
    assert any("Metadata contract warning" in msg for msg in warnings)
    assert (
        "ingestion_validation_failures_total",
        1,
        {"reason": "metadata_contract", "path_kind": "upload"},
    ) in metrics.increment_calls


def test_enforce_metadata_contract_error_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_METADATA_CONTRACT_POLICY", "error")

    result = {
        "status": "Success",
        "input_ref": "missing-metadata-item",
        "metadata": {},
        "warnings": None,
        "error": None,
        "db_id": 10,
        "db_message": "ok",
        "media_uuid": "uuid-10",
    }
    ingestion_persistence._enforce_metadata_contract_on_result(
        result=result,
        media_type="document",
        form_data=SimpleNamespace(metadata_contract_policy=None),
        path_kind="upload",
        processor="document_processor",
    )

    assert result["status"] == "Error"
    assert "Metadata contract validation failed" in str(result.get("error", ""))
    assert result["db_id"] is None
    assert result["media_uuid"] is None
    assert result["db_message"] == "DB operation skipped (metadata contract failure)."


@pytest.mark.asyncio
async def test_process_document_like_item_applies_metadata_contract_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_process_document_content(**_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "Success",
            "content": "ok",
            "metadata": {},
            "analysis": None,
            "summary": None,
            "analysis_details": {},
            "error": None,
            "warnings": None,
        }

    async def fake_persist_doc_item_and_children(**_kwargs: Any) -> None:
        return None

    async def fake_extract_claims_if_requested(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setenv("MEDIA_METADATA_CONTRACT_POLICY", "warn")
    monkeypatch.setattr(
        ingestion_persistence,
        "persist_doc_item_and_children",
        fake_persist_doc_item_and_children,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "extract_claims_if_requested",
        fake_extract_claims_if_requested,
    )

    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("hello", encoding="utf-8")
    form_data = SimpleNamespace(
        title=None,
        author=None,
        keywords=None,
        perform_chunking=False,
        perform_analysis=False,
        api_name=None,
        custom_prompt=None,
        system_prompt=None,
        summarize_recursively=False,
        metadata_contract_policy="warn",
    )

    result = await ingestion_persistence.process_document_like_item(
        item_input_ref="doc.txt",
        processing_source=str(doc_path),
        media_type="document",
        is_url=False,
        form_data=form_data,
        chunk_options=None,
        temp_dir=tmp_path,
        loop=asyncio.get_running_loop(),
        db_path=":memory:",
        client_id="test-client",
        user_id=None,
        process_document_content=fake_process_document_content,
    )

    assert result.get("status") == "Success"
    warnings = result.get("warnings") or []
    assert any("Metadata contract warning" in msg for msg in warnings)


@pytest.mark.asyncio
async def test_process_batch_media_applies_metadata_contract_warning_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_process_audio_files(**kwargs: Any) -> dict[str, Any]:
        inputs = kwargs.get("inputs") or []
        src = inputs[0] if inputs else "unknown.mp3"
        return {
            "results": [
                {
                    "status": "Success",
                    "input_ref": src,
                    "processing_source": src,
                    "media_type": "audio",
                    "metadata": {},
                    "content": "audio transcript",
                    "transcript": "audio transcript",
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "summary": None,
                    "analysis_details": {},
                    "error": None,
                    "warnings": None,
                    "db_id": None,
                    "db_message": None,
                }
            ],
            "errors_count": 0,
        }

    async def fake_persist_primary_av_item(**_kwargs: Any) -> None:
        return None

    async def fake_extract_claims_if_requested(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setenv("MEDIA_METADATA_CONTRACT_POLICY", "warn")
    monkeypatch.setattr(Audio_Files, "process_audio_files", fake_process_audio_files)
    monkeypatch.setattr(
        ingestion_persistence,
        "persist_primary_av_item",
        fake_persist_primary_av_item,
    )
    monkeypatch.setattr(
        ingestion_persistence,
        "extract_claims_if_requested",
        fake_extract_claims_if_requested,
    )

    source = str(tmp_path / "input.mp3")
    result = await ingestion_persistence.process_batch_media(
        media_type="audio",
        urls=[],
        uploaded_file_paths=[source],
        source_to_ref_map={source: "input.mp3"},
        form_data=SimpleNamespace(
            overwrite_existing=True,
            transcription_model=None,
            metadata_contract_policy="warn",
        ),
        chunk_options=None,
        loop=asyncio.get_running_loop(),
        db_path=":memory:",
        client_id="test-client",
        temp_dir=tmp_path,
    )

    assert len(result) == 1
    warnings = result[0].get("warnings") or []
    assert any("Metadata contract warning" in msg for msg in warnings)
