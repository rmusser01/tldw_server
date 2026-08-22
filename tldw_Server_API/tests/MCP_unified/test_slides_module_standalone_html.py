"""Standalone HTML operation-matrix coverage for the MCP Slides module."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.slides_module import (
    SlidesModule,
)
from tldw_Server_API.app.core.Slides.presentation_service import PresentationService, PresentationServiceError
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase
from tldw_Server_API.app.core.Slides.standalone_html_validator import (
    validate_standalone_html,
)


def _valid_html() -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Standalone MCP Deck</title><style>.slide{color:#111}</style></head>"
        '<body><section class="slide"><h1>Visible MCP summary</h1></section>'
        "<script>document.addEventListener('keydown', () => {});</script>"
        "</body></html>"
    )


def _provenance() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_kind": "prompt",
        "source_ref": None,
        "source_snapshot_hmac_sha256": "a" * 64,
        "digest_key_id": "slides-generation-v1",
        "source_bytes": 10,
        "provider": "openai",
        "model": "test-model",
        "adapter_id": "openai_official_chat_v1",
        "endpoint_identity": "https://api.openai.com:443/v1/chat/completions",
        "prompt_sha256": "b" * 64,
    }


class _InlineValidationPool:
    async def validate(self, document: str, **_kwargs: Any):
        return validate_standalone_html(document)


@pytest.fixture()
def standalone_mcp(tmp_path):
    db_path = tmp_path / "Slides.db"
    db = SlidesDatabase(db_path=db_path, client_id="setup")
    try:
        service = PresentationService(db)
        source = _valid_html()
        html_row = service.create_standalone_for_worker(
            presentation_id="html-deck",
            html_document=source,
            validation_result=validate_standalone_html(source),
            generation_job_uuid="job-html-mcp",
            generation_provenance=_provenance(),
        )
        structured_row = db.create_presentation(
            presentation_id="structured-deck",
            title="Structured MCP Deck",
            description=None,
            theme="black",
            marp_theme=None,
            settings=None,
            studio_data=None,
            slides=json.dumps(
                [
                    {
                        "order": 0,
                        "layout": "title",
                        "title": "Structured",
                        "content": "Visible structured content",
                        "speaker_notes": None,
                        "metadata": {},
                    }
                ]
            ),
            slides_text="Visible structured content",
            source_type="manual",
            source_ref=None,
            source_query=None,
            custom_css=None,
        )
    finally:
        db.close_connection()

    module = SlidesModule(ModuleConfig(name="slides"))
    context = SimpleNamespace(
        db_paths={"slides": str(db_path)},
        user_id="1",
        metadata={"_server_standalone_html_validation_pool": _InlineValidationPool()},
    )
    return module, context, html_row, structured_row


def _assert_source_free(value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    assert "html_document" not in serialized
    assert "document.addEventListener" not in serialized
    assert "payload_json" not in serialized


@pytest.mark.asyncio
async def test_html_metadata_operations_are_source_free_and_skip_full_mappers(standalone_mcp, monkeypatch):
    module, context, html_row, _structured_row = standalone_mcp
    original_presentation_mapper = module._presentation_to_dict

    def _full_row_mapper_must_not_run_for_html(row):
        if row.id == html_row.id:
            raise AssertionError("HTML metadata must not use the full-row mapper")
        return original_presentation_mapper(row)

    def _full_version_mapper_must_not_run(_row):
        raise AssertionError("HTML version lists must not deserialize payload_json")

    monkeypatch.setattr(module, "_presentation_to_dict", _full_row_mapper_must_not_run_for_html)
    monkeypatch.setattr(module, "_version_to_dict", _full_version_mapper_must_not_run)

    listed = await module.execute_tool("slides.presentations.list", {}, context=context)
    searched = await module.execute_tool(
        "slides.presentations.search",
        {"query": "Visible"},
        context=context,
    )
    fetched = await module.execute_tool(
        "slides.presentations.get",
        {"presentation_id": html_row.id},
        context=context,
    )
    versions = await module.execute_tool(
        "slides.versions.list",
        {"presentation_id": html_row.id},
        context=context,
    )

    for result in (listed, searched, fetched, versions):
        _assert_source_free(result)
    assert fetched["presentation"]["content_kind"] == "standalone_html"
    assert fetched["presentation"]["html_slide_count"] == 1
    assert versions["versions"][0]["content_kind"] == "standalone_html"

    deleted = await module.execute_tool(
        "slides.presentations.delete",
        {"presentation_id": html_row.id, "expected_version": html_row.version},
        context=context,
    )
    restored = await module.execute_tool(
        "slides.presentations.restore",
        {
            "presentation_id": html_row.id,
            "expected_version": deleted["presentation"]["version"],
        },
        context=context,
    )
    _assert_source_free(deleted)
    _assert_source_free(restored)
    assert deleted["presentation"]["deleted"] is True
    assert restored["presentation"]["deleted"] is False


@pytest.mark.asyncio
async def test_structured_metadata_and_generation_remain_compatible(standalone_mcp, monkeypatch):
    module, context, _html_row, structured_row = standalone_mcp

    fetched = await module.execute_tool(
        "slides.presentations.get",
        {"presentation_id": structured_row.id},
        context=context,
    )
    assert fetched["presentation"]["slides"]

    class _Generator:
        def generate_from_text(self, **_kwargs):
            return {
                "title": "Generated structured deck",
                "slides": [{"order": 0, "title": "Generated", "content": "OK"}],
            }

    monkeypatch.setattr(module, "_get_generator", lambda: _Generator())
    generated = await module.execute_tool(
        "slides.generate.from_prompt",
        {
            "prompt": "Generate a normal structured presentation",
            "generation_mode": "structured_slides",
        },
        context=context,
    )
    assert generated["success"] is True
    assert generated["presentation"]["content_kind"] == "structured_slides"


@pytest.mark.asyncio
async def test_structured_list_search_and_version_list_preserve_legacy_wire_shapes(standalone_mcp):
    module, context, html_row, structured_row = standalone_mcp

    listed = await module.execute_tool("slides.presentations.list", {}, context=context)
    searched = await module.execute_tool(
        "slides.presentations.search",
        {"query": "Visible structured"},
        context=context,
    )
    versions = await module.execute_tool(
        "slides.versions.list",
        {"presentation_id": structured_row.id},
        context=context,
    )

    listed_by_id = {row["id"]: row for row in listed["presentations"]}
    structured = listed_by_id[structured_row.id]
    assert structured["slides"] == structured_row.slides
    assert structured["slides_text"] == structured_row.slides_text
    assert structured["client_id"] == structured_row.client_id
    assert structured["content_kind"] == "structured_slides"
    _assert_source_free(listed_by_id[html_row.id])

    assert searched["presentations"][0]["id"] == structured_row.id
    assert searched["presentations"][0]["slides"] == structured_row.slides
    assert searched["presentations"][0]["client_id"] == structured_row.client_id

    version = versions["versions"][0]
    assert set(version) == {
        "presentation_id",
        "version",
        "payload",
        "created_at",
        "client_id",
    }
    assert version["payload"]["slides"] == structured_row.slides
    assert version["client_id"] == structured_row.client_id


@pytest.mark.asyncio
async def test_stale_standalone_restore_validates_before_undelete(standalone_mcp):
    module, context, html_row, _structured_row = standalone_mcp
    deleted = await module.execute_tool(
        "slides.presentations.delete",
        {"presentation_id": html_row.id, "expected_version": html_row.version},
        context=context,
    )

    class _StaleValidationPool:
        async def validate(self, _document: str, **_kwargs: Any):
            return validate_standalone_html(_valid_html().replace("Standalone MCP Deck", "Changed title"))

    context.metadata["_server_standalone_html_validation_pool"] = _StaleValidationPool()

    with pytest.raises(PresentationServiceError, match="standalone_html_response_invalid"):
        await module.execute_tool(
            "slides.presentations.restore",
            {
                "presentation_id": html_row.id,
                "expected_version": deleted["presentation"]["version"],
            },
            context=context,
        )

    db = SlidesDatabase(db_path=context.db_paths["slides"], client_id="assert-still-deleted")
    try:
        assert db.get_presentation_by_id(html_row.id, include_deleted=True).deleted == 1
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "slides.presentations.create",
            {"title": "Unknown", "slides": "[]", "content_kind": "future_slides"},
        ),
        (
            "slides.generate.from_prompt",
            {"prompt": "Generate an unknown presentation", "generation_mode": "future_slides"},
        ),
        (
            "slides.presentations.update",
            {
                "presentation_id": "structured-deck",
                "expected_version": 1,
                "updates": {"content_kind": "future_slides"},
            },
        ),
        (
            "slides.presentations.update",
            {
                "presentation_id": "structured-deck",
                "expected_version": 1,
                "updates": {"generation_mode": "future_slides"},
            },
        ),
        (
            "slides.presentations.patch",
            {
                "presentation_id": "structured-deck",
                "expected_version": 1,
                "patch": {"content_kind": "future_slides"},
            },
        ),
        (
            "slides.presentations.patch",
            {
                "presentation_id": "structured-deck",
                "expected_version": 1,
                "patch": {"generation_mode": "future_slides"},
            },
        ),
    ],
)
async def test_unknown_shallow_discriminators_fail_closed_before_work(
    standalone_mcp,
    monkeypatch,
    tool_name,
    arguments,
):
    module, context, _html_row, _structured_row = standalone_mcp

    def _work_must_not_start(*_args, **_kwargs):
        raise AssertionError("unknown discriminator must reject before database or provider work")

    async def _rag_must_not_start(*_args, **_kwargs):
        raise AssertionError("unknown discriminator must reject before RAG work")

    monkeypatch.setattr(module, "_open_db", _work_must_not_start)
    monkeypatch.setattr(module, "_get_generator", _work_must_not_start)
    monkeypatch.setattr(module, "_get_rag_content", _rag_must_not_start)

    result = await module.execute_tool(tool_name, arguments, context=context)

    assert result == {
        "success": False,
        "error": {
            "code": "operation_not_supported_for_content_kind",
            "operation": tool_name,
            "content_kind": "unknown",
        },
    }


@pytest.mark.asyncio
async def test_structured_delete_and_restore_preserve_existing_response_shapes(
    standalone_mcp,
):
    module, context, _html_row, structured_row = standalone_mcp

    deleted = await module.execute_tool(
        "slides.presentations.delete",
        {
            "presentation_id": structured_row.id,
            "expected_version": structured_row.version,
        },
        context=context,
    )
    restored = await module.execute_tool(
        "slides.presentations.restore",
        {
            "presentation_id": structured_row.id,
            "expected_version": structured_row.version + 1,
        },
        context=context,
    )

    assert deleted == {
        "presentation_id": structured_row.id,
        "action": "soft_deleted",
        "success": True,
    }
    assert restored["presentation_id"] == structured_row.id
    assert restored["action"] == "restored"
    assert restored["success"] is True
    assert restored["presentation"]["slides"]
    assert restored["presentation"]["content_kind"] == "structured_slides"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "slides.presentations.create",
            {
                "title": "Forbidden",
                "slides": "[]",
                "content_kind": "standalone_html",
            },
        ),
        (
            "slides.presentations.update",
            {
                "presentation_id": "html-deck",
                "updates": {"title": "Forbidden"},
                "expected_version": 1,
            },
        ),
        (
            "slides.presentations.patch",
            {
                "presentation_id": "html-deck",
                "patch": {"title": "Forbidden"},
                "expected_version": 1,
            },
        ),
        (
            "slides.presentations.reorder",
            {
                "presentation_id": "html-deck",
                "slide_order": [0],
                "expected_version": 1,
            },
        ),
        (
            "slides.generate.from_prompt",
            {"prompt": "Generate a forbidden standalone deck", "content_kind": "standalone_html"},
        ),
        (
            "slides.generate.from_media",
            {"media_id": 1, "generation_mode": "standalone_html"},
        ),
        (
            "slides.generate.from_notes",
            {"note_ids": ["note-1"], "content_kind": "standalone_html"},
        ),
        (
            "slides.generate.from_chat",
            {"conversation_id": "chat-1", "generation_mode": "standalone_html"},
        ),
        (
            "slides.generate.from_rag",
            {"query": "forbidden query", "content_kind": "standalone_html"},
        ),
        (
            "slides.versions.get",
            {"presentation_id": "html-deck", "version": 1},
        ),
        (
            "slides.versions.restore",
            {
                "presentation_id": "html-deck",
                "version": 1,
                "expected_current_version": 1,
            },
        ),
        ("slides.export", {"presentation_id": "html-deck", "format": "reveal"}),
        ("slides.export", {"presentation_id": "html-deck", "format": "json"}),
        ("slides.export", {"presentation_id": "html-deck", "format": "markdown"}),
        ("slides.export", {"presentation_id": "html-deck", "format": "pdf"}),
    ],
)
async def test_html_unsupported_operations_return_exact_fixed_error(standalone_mcp, monkeypatch, tool_name, arguments):
    module, context, _html_row, _structured_row = standalone_mcp

    async def _rag_must_not_run(*_args, **_kwargs):
        raise AssertionError("standalone RAG must reject before _get_rag_content")

    monkeypatch.setattr(module, "_get_rag_content", _rag_must_not_run)
    monkeypatch.setattr(
        module,
        "_presentation_to_dict",
        lambda _row: (_ for _ in ()).throw(AssertionError("standalone rejection must happen before full-row mapping")),
    )
    monkeypatch.setattr(
        module,
        "_version_to_dict",
        lambda _row: (_ for _ in ()).throw(
            AssertionError("standalone rejection must happen before version payload mapping")
        ),
    )

    result = await module.execute_tool(tool_name, arguments, context=context)

    assert result == {
        "success": False,
        "error": {
            "code": "operation_not_supported_for_content_kind",
            "operation": tool_name,
            "content_kind": "standalone_html",
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("container", ["updates", "patch"])
async def test_source_fields_reject_before_target_or_source_loading(standalone_mcp, monkeypatch, container):
    module, context, _html_row, structured_row = standalone_mcp
    tool_name = f"slides.presentations.{container.rstrip('s')}"

    def _db_must_not_open(_context):
        raise AssertionError("explicit standalone source must reject before database access")

    monkeypatch.setattr(module, "_open_db", _db_must_not_open)
    result = await module.execute_tool(
        tool_name,
        {
            "presentation_id": structured_row.id,
            container: {"html_document": "TOP-SECRET-SOURCE"},
            "expected_version": structured_row.version,
        },
        context=context,
    )

    assert result["error"] == {
        "code": "operation_not_supported_for_content_kind",
        "operation": tool_name,
        "content_kind": "standalone_html",
    }
    _assert_source_free(result)


@pytest.mark.asyncio
async def test_deleted_html_version_get_rejects_before_version_payload_loading(
    standalone_mcp,
    monkeypatch,
):
    module, context, html_row, _structured_row = standalone_mcp
    await module.execute_tool(
        "slides.presentations.delete",
        {
            "presentation_id": html_row.id,
            "expected_version": html_row.version,
        },
        context=context,
    )
    monkeypatch.setattr(
        module,
        "_version_to_dict",
        lambda _row: (_ for _ in ()).throw(AssertionError("deleted HTML version payload must not be loaded")),
    )

    result = await module.execute_tool(
        "slides.versions.get",
        {"presentation_id": html_row.id, "version": 1},
        context=context,
    )

    assert result == {
        "success": False,
        "error": {
            "code": "operation_not_supported_for_content_kind",
            "operation": "slides.versions.get",
            "content_kind": "standalone_html",
        },
    }
