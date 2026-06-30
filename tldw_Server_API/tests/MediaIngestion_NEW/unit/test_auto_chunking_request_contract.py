from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps import (
    media_add_deps,
    media_processing_deps,
)
from tldw_Server_API.app.api.v1.schemas.media_request_models import (
    AddMediaForm,
    ChunkingOptions,
    IngestWebContentRequest,
    WebScrapingRequest,
)

pytestmark = pytest.mark.unit


def _dependency_kwargs(func: Callable[..., Any], **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name, parameter in inspect.signature(func).parameters.items():
        default = parameter.default
        kwargs[name] = default.default if hasattr(default, "default") else default
    kwargs.update(overrides)
    return kwargs


def _valid_hierarchical_template() -> dict[str, Any]:
    return {
        "boundaries": [
            {
                "kind": "heading",
                "pattern": r"^#{1,3}\s+",
                "flags": "m",
            }
        ]
    }


@pytest.mark.asyncio
async def test_add_media_form_dependency_parses_auto_chunking_and_template_fields():
    form = await media_add_deps.get_add_media_form(
        **_dependency_kwargs(
            media_add_deps.get_add_media_form,
            media_type="document",
            urls=["https://example.test/article.md"],
            transcription_model=None,
            chunking_mode="auto",
            auto_chunking_goal="qa_search",
            auto_chunking_use_llm=True,
            auto_apply_template=True,
            chunking_template_name="article-defaults",
            hierarchical_chunking=True,
            hierarchical_template=_valid_hierarchical_template(),
        )
    )

    assert form.chunking_mode == "auto"
    assert form.auto_chunking_goal == "qa_search"
    assert form.auto_chunking_use_llm is True
    assert form.auto_apply_template is True
    assert form.chunking_template_name == "article-defaults"
    assert form.hierarchical_chunking is True
    assert form.hierarchical_template == _valid_hierarchical_template()


@pytest.mark.asyncio
async def test_add_media_form_dependency_normalizes_empty_template_name():
    form = await media_add_deps.get_add_media_form(
        **_dependency_kwargs(
            media_add_deps.get_add_media_form,
            media_type="document",
            urls=["https://example.test/article.md"],
            transcription_model=None,
            chunking_mode="manual",
            chunking_template_name="",
            hierarchical_template="",
        )
    )

    assert form.chunking_template_name is None
    assert form.hierarchical_template is None


@pytest.mark.asyncio
async def test_add_media_form_dependency_preserves_legacy_missing_chunking_mode():
    form = await media_add_deps.get_add_media_form(
        **_dependency_kwargs(
            media_add_deps.get_add_media_form,
            media_type="document",
            urls=["https://example.test/article.md"],
            transcription_model=None,
        )
    )

    assert form.chunking_mode is None
    assert form.auto_chunking_goal == "balanced"
    assert form.auto_chunking_use_llm is False


@pytest.mark.asyncio
async def test_add_media_form_dependency_rejects_invalid_auto_chunking_values():
    with pytest.raises(HTTPException) as excinfo:
        await media_add_deps.get_add_media_form(
            **_dependency_kwargs(
                media_add_deps.get_add_media_form,
                media_type="document",
                urls=["https://example.test/article.md"],
                transcription_model=None,
                chunking_mode="automatic",
                auto_chunking_goal="summarize_everything",
            )
        )

    assert excinfo.value.status_code == media_add_deps.HTTP_422_UNPROCESSABLE
    rendered_detail = str(excinfo.value.detail)
    assert "chunking_mode" in rendered_detail
    assert "auto_chunking_goal" in rendered_detail


@pytest.mark.asyncio
async def test_form_dependencies_reject_malformed_hierarchical_template_json():
    with pytest.raises(HTTPException) as add_excinfo:
        await media_add_deps.get_add_media_form(
            **_dependency_kwargs(
                media_add_deps.get_add_media_form,
                media_type="document",
                urls=["https://example.test/article.md"],
                transcription_model=None,
                hierarchical_template="{not-json",
            )
        )

    assert add_excinfo.value.status_code == media_add_deps.HTTP_422_UNPROCESSABLE
    assert "hierarchical_template" in str(add_excinfo.value.detail)

    with pytest.raises(HTTPException) as process_excinfo:
        await media_processing_deps.get_process_documents_form(
            **_dependency_kwargs(
                media_processing_deps.get_process_documents_form,
                urls=["https://example.test/article.md"],
                hierarchical_template="{not-json",
            )
        )

    assert process_excinfo.value.status_code == media_processing_deps.HTTP_422_UNPROCESSABLE
    assert "hierarchical_template" in str(process_excinfo.value.detail)


@pytest.mark.parametrize(
    ("dependency", "expected_media_type"),
    [
        (media_processing_deps.get_process_documents_form, "document"),
        (media_processing_deps.get_process_videos_form, "video"),
        (media_processing_deps.get_process_audios_form, "audio"),
        (media_processing_deps.get_process_pdfs_form, "pdf"),
        (media_processing_deps.get_process_ebooks_form, "ebook"),
        (media_processing_deps.get_process_emails_form, "email"),
    ],
)
@pytest.mark.asyncio
async def test_process_form_dependencies_parse_auto_chunking_contract(
    dependency: Callable[..., Any],
    expected_media_type: str,
):
    form = await dependency(
        **_dependency_kwargs(
            dependency,
            urls=["https://example.test/source"],
            chunking_mode="auto",
            auto_chunking_goal="navigation_summary",
            auto_chunking_use_llm=True,
            auto_apply_template=True,
            chunking_template_name="detected-template",
            hierarchical_chunking=True,
            hierarchical_template=_valid_hierarchical_template(),
        )
    )

    assert form.media_type == expected_media_type
    assert form.chunking_mode == "auto"
    assert form.auto_chunking_goal == "navigation_summary"
    assert form.auto_chunking_use_llm is True
    assert form.auto_apply_template is True
    assert form.chunking_template_name == "detected-template"
    assert form.hierarchical_chunking is True
    assert form.hierarchical_template == _valid_hierarchical_template()


def test_auto_chunking_schema_validates_unknown_modes_and_allows_disabled_chunking():
    valid = AddMediaForm(
        media_type="document",
        perform_chunking=False,
        chunking_mode="auto",
        auto_chunking_goal="balanced",
        auto_chunking_use_llm=True,
    )

    assert valid.perform_chunking is False
    assert valid.chunking_mode == "auto"
    assert valid.auto_chunking_goal == "balanced"
    assert valid.auto_chunking_use_llm is True

    with pytest.raises(ValidationError) as excinfo:
        AddMediaForm(
            media_type="document",
            chunking_mode="automatic",
            auto_chunking_goal="summarize_everything",
        )

    rendered_errors = str(excinfo.value)
    assert "chunking_mode" in rendered_errors
    assert "auto_chunking_goal" in rendered_errors


def test_chunking_options_schema_allows_internal_structure_aware_method():
    options = ChunkingOptions(chunk_method="structure_aware")

    assert options.chunk_method == "structure_aware"


def test_web_article_request_models_accept_auto_chunking_fields():
    web_scrape = WebScrapingRequest(
        scrape_method="individual",
        url_input="https://example.test/article",
        perform_chunking=False,
        chunking_mode="auto",
        auto_chunking_goal="qa_search",
        auto_chunking_use_llm=True,
    )
    ingest_web = IngestWebContentRequest(
        urls=["https://example.test/article"],
        perform_chunking=False,
        chunking_mode="auto",
        auto_chunking_goal="navigation_summary",
        auto_chunking_use_llm=True,
    )

    assert web_scrape.chunking_mode == "auto"
    assert web_scrape.auto_chunking_goal == "qa_search"
    assert web_scrape.auto_chunking_use_llm is True
    assert ingest_web.chunking_mode == "auto"
    assert ingest_web.auto_chunking_goal == "navigation_summary"
    assert ingest_web.auto_chunking_use_llm is True


def test_web_article_request_models_reject_invalid_auto_chunking_fields():
    with pytest.raises(ValidationError) as excinfo:
        WebScrapingRequest(
            scrape_method="individual",
            url_input="https://example.test/article",
            chunking_mode="automatic",
            auto_chunking_goal="summarize_everything",
        )

    rendered_errors = str(excinfo.value)
    assert "chunking_mode" in rendered_errors
    assert "auto_chunking_goal" in rendered_errors
