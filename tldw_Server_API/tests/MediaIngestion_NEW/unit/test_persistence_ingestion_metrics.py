from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import BackgroundTasks, HTTPException

from tldw_Server_API.app.core.Ingestion_Media_Processing import (
    input_sourcing,
    persistence,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.services import storage_quota_service


pytestmark = pytest.mark.unit


class _MetricsCapture:
    def __init__(self) -> None:
        self.increment_calls: list[tuple[str, float, dict[str, Any] | None]] = []
        self.observe_calls: list[tuple[str, float, dict[str, Any] | None]] = []

    def increment(
        self,
        metric_name: str,
        value: float = 1,
        labels: dict[str, Any] | None = None,
    ) -> None:
        self.increment_calls.append((metric_name, value, labels))

    def observe(
        self,
        metric_name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        self.observe_calls.append((metric_name, value, labels))


class _MetricsRaising:
    def increment(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("metrics increment failure")

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("metrics observe failure")


class _FakeDB:
    db_path_str = ":memory:"
    client_id = "test-client"


class _AllowingQuotaService:
    def __init__(self) -> None:
        self.checks: list[tuple[int, int, bool]] = []

    async def check_quota(
        self,
        user_id: int,
        new_bytes: int,
        raise_on_exceed: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        self.checks.append((user_id, new_bytes, raise_on_exceed))
        return True, {
            "current_usage_mb": 0,
            "new_size_mb": 0,
            "quota_mb": 1024,
            "available_mb": 1024,
        }


def _contains_metric_call(
    calls: list[tuple[str, float, dict[str, Any] | None]],
    *,
    metric_name: str,
    labels: dict[str, Any],
) -> bool:
    return any(name == metric_name and metric_labels == labels for name, _value, metric_labels in calls)


def test_ingestion_metrics_are_registered() -> None:
    reg = get_metrics_registry()
    assert "ingestion_requests_total" in reg.metrics
    assert "ingestion_processing_seconds" in reg.metrics
    assert "ingestion_validation_failures_total" in reg.metrics
    assert "ingestion_chunks_total" in reg.metrics
    assert "ingestion_embeddings_enqueue_total" in reg.metrics


def test_ingestion_metric_helpers_are_no_throw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(persistence, "get_metrics_registry", lambda: _MetricsRaising())

    persistence._emit_ingestion_request_metric(media_type="pdf", outcome="success")
    persistence._emit_ingestion_processing_duration_metric(
        media_type="pdf",
        processor="media_add_orchestrate",
        duration_seconds=0.25,
    )
    persistence._emit_ingestion_validation_failure_metric(reason="file_type", path_kind="upload")
    persistence._emit_ingestion_chunks_metric(media_type="pdf", chunk_method="sentences", chunk_count=3)
    persistence._emit_ingestion_embeddings_enqueue_metric(path_kind="jobs", outcome="success")


def test_validate_add_media_inputs_emits_error_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    metrics = _MetricsCapture()
    monkeypatch.setattr(persistence, "get_metrics_registry", lambda: metrics)

    with pytest.raises(HTTPException):
        persistence.validate_add_media_inputs(media_type="document", urls=[], files=[])

    assert _contains_metric_call(
        metrics.increment_calls,
        metric_name="ingestion_requests_total",
        labels={"media_type": "document", "outcome": "error"},
    )
    assert _contains_metric_call(
        metrics.observe_calls,
        metric_name="ingestion_processing_seconds",
        labels={"media_type": "document", "processor": "media_add_orchestrate"},
    )


@pytest.mark.asyncio
async def test_add_media_orchestrate_emits_request_and_duration_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _MetricsCapture()
    quota_service = _AllowingQuotaService()

    async def fake_save_uploaded_files(_files: list[Any], temp_dir: Path, **_kwargs: Any):
        path = Path(temp_dir) / "doc.txt"
        path.write_text("hello", encoding="utf-8")
        return [{"path": path, "original_filename": "doc.txt"}], []

    async def fake_process_doc_item_fn(
        *,
        item_input_ref: str,
        processing_source: str,
        media_type: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "status": "Success",
            "input_ref": item_input_ref,
            "processing_source": str(processing_source),
            "media_type": media_type,
            "metadata": {},
            "content": "content",
            "analysis": None,
            "summary": None,
            "analysis_details": {},
            "db_id": 1,
            "db_message": "ok",
            "media_uuid": "uuid-1",
            "warnings": None,
        }

    monkeypatch.setattr(persistence, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(input_sourcing, "save_uploaded_files", fake_save_uploaded_files)
    monkeypatch.setattr(persistence, "process_document_like_item", fake_process_doc_item_fn)
    monkeypatch.setattr(storage_quota_service, "get_storage_quota_service", lambda: quota_service)

    form_data = SimpleNamespace(
        media_type="document",
        urls=[],
        keep_original_file=False,
        perform_chunking=False,
        perform_analysis=False,
        generate_embeddings=False,
    )
    response = await persistence.add_media_orchestrate(
        background_tasks=BackgroundTasks(),
        form_data=form_data,
        files=[object()],
        db=_FakeDB(),
        current_user=SimpleNamespace(id=1),
        usage_log=SimpleNamespace(log_event=lambda *_args, **_kwargs: None),
    )

    assert response.status_code == 200
    assert quota_service.checks == [(1, len("hello"), False)]
    assert _contains_metric_call(
        metrics.increment_calls,
        metric_name="ingestion_requests_total",
        labels={"media_type": "document", "outcome": "success"},
    )
    assert any(
        metric_name == "ingestion_processing_seconds"
        and labels == {"media_type": "document", "processor": "media_add_orchestrate"}
        and value >= 0
        for metric_name, value, labels in metrics.observe_calls
    )
