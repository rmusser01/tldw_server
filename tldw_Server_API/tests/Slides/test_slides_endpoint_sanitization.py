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

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))


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
