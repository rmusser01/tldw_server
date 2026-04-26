from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.responses import Response

from tldw_Server_API.app.api.v1.endpoints import slides as slides_ep
from tldw_Server_API.app.core.Slides.slides_generator import SlidesGenerationError


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.warnings: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append((message, args, kwargs))


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        visual_style_id=None,
        visual_style_scope=None,
        template_id=None,
        theme="black",
        marp_theme=None,
        settings={"controls": True},
        provider="stub-provider",
        model=None,
        title_hint="Deck",
        temperature=None,
        max_tokens=None,
        max_source_tokens=None,
        max_source_chars=None,
        enable_chunking=True,
        chunk_size_tokens=None,
        summary_tokens=None,
        custom_css=None,
    )


def test_generate_presentation_metric_error_log_is_sanitized(monkeypatch):
    class _RaisingMetrics:
        def increment(self, *_args, **_kwargs) -> None:
            raise RuntimeError("metrics backend exploded at /private/slides-metrics.db")

    class _RaisingGenerator:
        def generate_from_text(self, **_kwargs):
            raise SlidesGenerationError("llm backend exploded")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(slides_ep, "logger", logger_stub)
    monkeypatch.setattr(slides_ep, "_resolve_provider", lambda provider: "stub-provider")
    monkeypatch.setattr(slides_ep, "SlidesGenerator", _RaisingGenerator)
    monkeypatch.setattr(slides_ep, "get_metrics_registry", lambda: _RaisingMetrics())

    with pytest.raises(HTTPException) as exc_info:
        slides_ep._generate_presentation(
            response=Response(),
            db=SimpleNamespace(client_id="1"),
            request=_request(),
            source_text="Source text",
            source_type="prompt",
            source_ref=None,
            source_query=None,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to generate presentation"
    assert logger_stub.debugs == [("Failed to record slides generation error metric", (), {})]
    rendered = " ".join([logger_stub.debugs[0][0], *(str(arg) for arg in logger_stub.debugs[0][1])])
    assert "/private/slides-metrics.db" not in rendered
    assert "exploded" not in rendered


def test_generate_presentation_latency_metric_log_is_sanitized(monkeypatch):
    class _RaisingMetrics:
        def observe(self, *_args, **_kwargs) -> None:
            raise RuntimeError("metrics backend exploded at /private/slides-latency.db")

    class _ReturningGenerator:
        def generate_from_text(self, **_kwargs):
            return {
                "title": "Generated deck",
                "slides": [
                    {
                        "order": 0,
                        "layout": "title",
                        "title": "Deck",
                        "content": "Hello",
                        "speaker_notes": None,
                        "metadata": {},
                    }
                ],
            }

    class _PresentationDB:
        client_id = "1"

        def create_presentation(self, **kwargs):
            return SimpleNamespace(
                id="presentation-1",
                title=kwargs["title"],
                description=kwargs["description"],
                theme=kwargs["theme"],
                marp_theme=kwargs["marp_theme"],
                template_id=kwargs["template_id"],
                visual_style_id=kwargs["visual_style_id"],
                visual_style_scope=kwargs["visual_style_scope"],
                visual_style_name=kwargs["visual_style_name"],
                visual_style_version=kwargs["visual_style_version"],
                visual_style_snapshot=kwargs["visual_style_snapshot"],
                settings=kwargs["settings"],
                studio_data=None,
                slides=kwargs["slides"],
                slides_text=kwargs["slides_text"],
                source_type=kwargs["source_type"],
                source_ref=kwargs["source_ref"],
                source_query=kwargs["source_query"],
                custom_css=kwargs["custom_css"],
                created_at="2026-04-26T00:00:00+00:00",
                last_modified="2026-04-26T00:00:00+00:00",
                deleted=0,
                client_id="1",
                version=1,
            )

    logger_stub = _LoggerStub()
    monkeypatch.setattr(slides_ep, "logger", logger_stub)
    monkeypatch.setattr(slides_ep, "_resolve_provider", lambda provider: "stub-provider")
    monkeypatch.setattr(slides_ep, "SlidesGenerator", _ReturningGenerator)
    monkeypatch.setattr(slides_ep, "get_metrics_registry", lambda: _RaisingMetrics())

    response = slides_ep._generate_presentation(
        response=Response(),
        db=_PresentationDB(),
        request=_request(),
        source_text="Source text",
        source_type="prompt",
        source_ref=None,
        source_query=None,
    )

    assert response.id == "presentation-1"
    assert logger_stub.debugs == [("Failed to record slides generation latency metric", (), {})]
    rendered = " ".join([logger_stub.debugs[0][0], *(str(arg) for arg in logger_stub.debugs[0][1])])
    assert "/private/slides-latency.db" not in rendered
    assert "exploded" not in rendered


def test_resolve_media_source_text_document_fallback_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(slides_ep, "logger", logger_stub)
    monkeypatch.setattr(slides_ep, "get_latest_transcription", lambda _db, _media_id: None)

    def _raise_document_lookup(**_kwargs):
        raise RuntimeError("document backend exploded at /private/slides-documents.db")

    monkeypatch.setattr(slides_ep, "get_document_version", _raise_document_lookup, raising=False)

    source_text = slides_ep._resolve_media_source_text(
        media_db=object(),
        media_row={"content": " Stored media content "},
        media_id=99,
    )

    assert source_text == "Stored media content"
    assert logger_stub.debugs == [("Failed to resolve latest document content for slides source media", (), {})]
    rendered = " ".join([logger_stub.debugs[0][0], *(str(arg) for arg in logger_stub.debugs[0][1])])
    assert "/private/slides-documents.db" not in rendered
    assert "exploded" not in rendered
    assert "99" not in rendered


async def test_slides_health_backend_failure_log_is_sanitized(monkeypatch):
    class _RaisingSlidesDB:
        def list_presentations(self, **_kwargs):
            raise RuntimeError("slides db exploded at /private/slides-health.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(slides_ep, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await slides_ep.slides_health(db=_RaisingSlidesDB())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "slides_db_unavailable"
    assert logger_stub.warnings == [("slides health check failed", (), {})]
    rendered = " ".join([logger_stub.warnings[0][0], *(str(arg) for arg in logger_stub.warnings[0][1])])
    assert "/private/slides-health.db" not in rendered
    assert "exploded" not in rendered
