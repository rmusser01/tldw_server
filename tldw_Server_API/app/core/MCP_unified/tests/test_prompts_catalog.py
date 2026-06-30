from __future__ import annotations

import base64
import builtins
import importlib
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError, PromptsDatabase
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_module import (
    PromptsModule,
)
from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_catalog import (
    CONFIG_PROMPT_PREFIX,
    DEFAULT_PROMPT_PAGE_SIZE,
    LIBRARY_PROMPT_PREFIX,
    MAX_PROMPT_PAGE_SIZE,
    ConfigPromptCatalogSource,
    MCPPromptFormatter,
    PromptCatalogCursor,
    PromptCatalogError,
    UserPromptCatalogSource,
    clamp_prompt_page_size,
    decode_prompt_cursor,
    encode_prompt_cursor,
)

pytestmark = pytest.mark.unit


def test_prompt_catalog_error_uses_core_exception_type() -> None:
    from tldw_Server_API.app.core.exception_types import PromptCatalogError as CorePromptCatalogError
    from tldw_Server_API.app.core.exceptions import PromptCatalogError as ReexportedPromptCatalogError

    assert PromptCatalogError is CorePromptCatalogError
    assert ReexportedPromptCatalogError is CorePromptCatalogError


def test_prompt_cursor_encodes_none_as_none() -> None:
    assert encode_prompt_cursor(None) is None


def test_prompt_cursor_round_trips_without_padding() -> None:
    cursor = PromptCatalogCursor(
        library_after_name="alpha",
        library_after_uuid="11111111-1111-4111-8111-111111111111",
    )

    encoded = encode_prompt_cursor(cursor)

    assert "=" not in encoded
    assert "+" not in encoded
    assert "/" not in encoded
    assert decode_prompt_cursor(encoded) == cursor


def test_prompt_cursor_round_trips_config_segment() -> None:
    cursor = PromptCatalogCursor(library_done=True, config_index=3)

    assert decode_prompt_cursor(encode_prompt_cursor(cursor)) == cursor


def test_prompt_cursor_rejects_bad_payload() -> None:
    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor("not-valid-base64")

    assert excinfo.value.code == "invalid_cursor"


def test_prompt_cursor_rejects_boolean_version() -> None:
    raw = _encode_raw_cursor({"v": True, "library_done": False, "config_index": 0})

    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor(raw)

    assert excinfo.value.code == "invalid_cursor"


def test_prompt_cursor_rejects_partial_library_keyset() -> None:
    raw = _encode_raw_cursor({"v": 1, "library_after_name": "alpha", "config_index": 0})

    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor(raw)

    assert excinfo.value.code == "invalid_cursor"


def test_prompt_cursor_rejects_partial_library_uuid_keyset() -> None:
    raw = _encode_raw_cursor(
        {
            "v": 1,
            "library_after_uuid": "11111111-1111-4111-8111-111111111111",
            "config_index": 0,
        }
    )

    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor(raw)

    assert excinfo.value.code == "invalid_cursor"


def test_prompt_cursor_rejects_config_index_before_library_done() -> None:
    raw = _encode_raw_cursor({"v": 1, "library_done": False, "config_index": 1})

    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor(raw)

    assert excinfo.value.code == "invalid_cursor"


def test_prompt_cursor_rejects_library_keyset_when_library_done() -> None:
    raw = _encode_raw_cursor(
        {
            "v": 1,
            "library_done": True,
            "library_after_name": "alpha",
            "library_after_uuid": "11111111-1111-4111-8111-111111111111",
            "config_index": 0,
        }
    )

    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor(raw)

    assert excinfo.value.code == "invalid_cursor"


def test_prompt_cursor_rejects_null_library_keyset_fields() -> None:
    raw = _encode_raw_cursor(
        {
            "v": 1,
            "library_after_name": None,
            "library_after_uuid": None,
            "config_index": 0,
        }
    )

    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor(raw)

    assert excinfo.value.code == "invalid_cursor"


def test_clamp_prompt_page_size_uses_default_for_invalid_and_clamps_bounds() -> None:
    assert clamp_prompt_page_size("25") == DEFAULT_PROMPT_PAGE_SIZE
    assert clamp_prompt_page_size(True) == DEFAULT_PROMPT_PAGE_SIZE
    assert clamp_prompt_page_size("not-an-int") == DEFAULT_PROMPT_PAGE_SIZE
    assert clamp_prompt_page_size(0) == 1
    assert clamp_prompt_page_size(-5) == 1
    assert clamp_prompt_page_size(MAX_PROMPT_PAGE_SIZE + 1000) == MAX_PROMPT_PAGE_SIZE


def _encode_raw_cursor(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def test_formatter_creates_stable_library_definition() -> None:
    prompt_uuid = str(uuid.uuid4())
    row = {
        "id": 42,
        "uuid": prompt_uuid,
        "name": "Summarizer",
        "details": "Summarize a document without leaking the body.",
        "author": "tester",
        "version": 7,
        "keywords": ["summary", "docs"],
        "system_prompt": "Be concise.",
        "user_prompt": "Summarize {topic}",
        "prompt_definition": None,
    }
    formatter = MCPPromptFormatter(max_rendered_chars=10_000)

    prompt = formatter.library_prompt_definition(row)

    assert prompt["name"] == f"{LIBRARY_PROMPT_PREFIX}{prompt_uuid}"
    assert prompt["title"] == "Summarizer"
    assert prompt["description"] == "Summarize a document without leaking the body."
    assert prompt["arguments"] == [
        {
            "name": "topic",
            "title": "Topic",
            "description": None,
            "required": True,
        }
    ]
    assert prompt["_meta"]["tldw"]["source"] == "library"
    assert prompt["_meta"]["tldw"]["prompt_uuid"] == prompt_uuid
    assert prompt["_meta"]["tldw"]["prompt_id"] == 42
    assert prompt["_meta"]["tldw"]["version"] == 7
    assert prompt["_meta"]["tldw"]["tags"] == ["summary", "docs"]


def test_formatter_folds_system_and_user_into_mcp_user_message() -> None:
    row = {
        "id": 42,
        "uuid": "11111111-1111-4111-8111-111111111111",
        "name": "Summarizer",
        "details": "",
        "version": 1,
        "keywords": [],
        "system_prompt": "Be concise.",
        "user_prompt": "Summarize {topic}",
        "prompt_definition": None,
    }
    formatter = MCPPromptFormatter(max_rendered_chars=10_000)

    result = formatter.render_library_prompt(row, {"topic": "MCP prompts"})

    assert result["description"] == "Summarizer"
    assert result["messages"] == [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": "System instructions:\nBe concise.\n\nUser prompt:\nSummarize MCP prompts",
            },
        }
    ]
    assert result["_meta"]["tldw"]["source"] == "library"


def test_formatter_preserves_assistant_messages_from_structured_prompt() -> None:
    row = {
        "id": 1,
        "uuid": "22222222-2222-4222-8222-222222222222",
        "name": "Few shot",
        "details": "",
        "version": 2,
        "keywords": [],
        "system_prompt": "",
        "user_prompt": "",
        "prompt_definition": {
            "schema_version": 1,
            "format": "structured",
            "variables": [
                {"name": "topic", "label": "Topic", "required": True, "input_type": "text"}
            ],
            "blocks": [
                {
                    "id": "instructions",
                    "name": "Instructions",
                    "role": "system",
                    "content": "Teach clearly.",
                    "enabled": True,
                    "order": 10,
                    "is_template": False,
                },
                {
                    "id": "question",
                    "name": "Question",
                    "role": "user",
                    "content": "Explain {{topic}}",
                    "enabled": True,
                    "order": 20,
                    "is_template": True,
                },
                {
                    "id": "sample",
                    "name": "Sample",
                    "role": "assistant",
                    "content": "Here is a concise answer.",
                    "enabled": True,
                    "order": 30,
                    "is_template": False,
                },
            ],
        },
    }
    formatter = MCPPromptFormatter(max_rendered_chars=10_000)

    result = formatter.render_library_prompt(row, {"topic": "vectors"})

    assert result["messages"] == [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": "System instructions:\nTeach clearly.\n\nUser prompt:\nExplain vectors",
            },
        },
        {
            "role": "assistant",
            "content": {
                "type": "text",
                "text": "Here is a concise answer.",
            },
        },
    ]


def test_formatter_rejects_non_string_arguments() -> None:
    formatter = MCPPromptFormatter(max_rendered_chars=10_000)

    with pytest.raises(PromptCatalogError) as excinfo:
        formatter.validate_arguments({"topic": 123})

    assert excinfo.value.code == "invalid_argument_type"


def test_formatter_accepts_non_empty_non_identifier_argument_names() -> None:
    formatter = MCPPromptFormatter(max_rendered_chars=10_000)

    assert formatter.validate_arguments({"topic-name": "MCP"}) == {"topic-name": "MCP"}


def test_formatter_size_error_does_not_include_argument_value() -> None:
    row = {
        "id": 1,
        "uuid": "33333333-3333-4333-8333-333333333333",
        "name": "Echo",
        "details": "",
        "version": 1,
        "keywords": [],
        "system_prompt": "",
        "user_prompt": "Echo {secret}",
        "prompt_definition": None,
    }
    formatter = MCPPromptFormatter(max_rendered_chars=5)
    sensitive_value = "SENSITIVE" + "_VALUE"

    with pytest.raises(PromptCatalogError) as excinfo:
        formatter.render_library_prompt(row, {"secret": sensitive_value})

    assert excinfo.value.code == "rendered_prompt_too_large"  # nosec B101
    assert sensitive_value not in str(excinfo.value)  # nosec B101


def _context_for_prompts_db(db_path: str, *, scoped_ids: list[str] | None = None) -> SimpleNamespace:
    metadata = {}
    if scoped_ids is not None:
        metadata["persona_scope"] = {"explicit_ids": {"prompt_id": scoped_ids}}
    return SimpleNamespace(
        db_paths={"prompts": db_path},
        metadata=metadata,
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
    )


def _context_without_prompts_db() -> SimpleNamespace:
    return SimpleNamespace(
        db_paths={},
        metadata={},
        logger=SimpleNamespace(debug=lambda *args, **kwargs: None),
    )


def _add_legacy_prompt(db_path: str, *, name: str, deleted: bool = False) -> dict:
    db = PromptsDatabase(db_path=db_path, client_id="test_prompts_catalog")
    try:
        prompt_id, _prompt_uuid, _message = db.add_prompt(
            name=name,
            author="tester",
            details=f"{name} details",
            system_prompt="Be concise.",
            user_prompt="Summarize {topic}",
            keywords=["summary"],
        )
        if deleted:
            db.soft_delete_prompt(prompt_id)
        return db.get_prompt_by_id(prompt_id, include_deleted=True)
    finally:
        db.close_connection()


def test_user_source_lists_non_deleted_library_prompts_with_uuid_names(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    alpha = _add_legacy_prompt(db_path, name="Alpha")
    _add_legacy_prompt(db_path, name="Deleted", deleted=True)
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    result = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=PromptCatalogCursor(),
        limit=50,
    )

    assert [prompt["name"] for prompt in result.prompts] == [f"{LIBRARY_PROMPT_PREFIX}{alpha['uuid']}"]
    assert result.next_cursor is None
    assert result.warnings == []


def test_user_source_filters_by_prompt_id_scope(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    allowed = _add_legacy_prompt(db_path, name="Allowed")
    _add_legacy_prompt(db_path, name="Blocked")
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    result = source.list_prompts(
        context=_context_for_prompts_db(db_path, scoped_ids=[str(allowed["id"])]),
        cursor=PromptCatalogCursor(),
        limit=50,
    )

    assert [prompt["name"] for prompt in result.prompts] == [f"{LIBRARY_PROMPT_PREFIX}{allowed['uuid']}"]


def test_user_source_list_missing_prompts_db_warns_without_leaking() -> None:
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    result = source.list_prompts(
        context=_context_without_prompts_db(),
        cursor=PromptCatalogCursor(),
        limit=50,
    )

    assert result.prompts == []
    assert result.next_cursor is None
    assert result.warnings == [{"source": "library", "code": "prompt_db_unavailable"}]


def test_user_source_list_internal_formatting_failure_warns(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    bad = _add_legacy_prompt(db_path, name="Bad Structured Prompt")

    class InternalFailureFormatter(MCPPromptFormatter):
        def library_prompt_definition(self, row):
            raise PromptCatalogError("invalid_prompt_definition", "Invalid row.", internal=True)

    source = UserPromptCatalogSource(InternalFailureFormatter(max_rendered_chars=10_000))

    result = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=PromptCatalogCursor(),
        limit=50,
    )

    assert result.prompts == []
    assert result.next_cursor is None
    assert result.warnings == [
        {"source": "library", "code": "prompt_unavailable", "prompt_uuid": bad["uuid"]}
    ]


def test_user_source_list_skips_bad_row_and_keeps_cursor_moving(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    bad = _add_legacy_prompt(db_path, name="Alpha Bad")
    good = _add_legacy_prompt(db_path, name="Beta Good")

    class OneBadFormatter(MCPPromptFormatter):
        def library_prompt_definition(self, row):
            if row.get("uuid") == bad["uuid"]:
                raise PromptCatalogError("invalid_prompt_definition", "Invalid row.", internal=True)
            return super().library_prompt_definition(row)

    source = UserPromptCatalogSource(OneBadFormatter(max_rendered_chars=10_000))

    first_page = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=PromptCatalogCursor(),
        limit=1,
    )
    second_page = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=first_page.next_cursor,
        limit=1,
    )

    assert first_page.prompts == []
    assert first_page.next_cursor.library_after_uuid == bad["uuid"]
    assert first_page.warnings == [
        {"source": "library", "code": "prompt_unavailable", "prompt_uuid": bad["uuid"]}
    ]
    assert [prompt["name"] for prompt in second_page.prompts] == [
        f"{LIBRARY_PROMPT_PREFIX}{good['uuid']}"
    ]
    assert second_page.warnings == []


def test_user_source_get_validates_uuid_and_scope(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    allowed = _add_legacy_prompt(db_path, name="Allowed")
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    result = source.get_prompt(
        context=_context_for_prompts_db(db_path, scoped_ids=[str(allowed["id"])]),
        name=f"{LIBRARY_PROMPT_PREFIX}{allowed['uuid']}",
        arguments={"topic": "scoped prompts"},
    )

    assert result["messages"][0]["content"]["text"].endswith("Summarize scoped prompts")


def test_user_source_get_rejects_malformed_library_prompt_name(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    with pytest.raises(PromptCatalogError) as excinfo:
        source.get_prompt(
            context=_context_for_prompts_db(db_path),
            name=f"{LIBRARY_PROMPT_PREFIX}not-a-uuid",
            arguments={"topic": "x"},
        )

    assert excinfo.value.code == "invalid_prompt_name"


def test_user_source_get_rejects_prompt_outside_scope(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    allowed = _add_legacy_prompt(db_path, name="Allowed")
    blocked = _add_legacy_prompt(db_path, name="Blocked")
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    with pytest.raises(PromptCatalogError) as excinfo:
        source.get_prompt(
            context=_context_for_prompts_db(db_path, scoped_ids=[str(allowed["id"])]),
            name=f"{LIBRARY_PROMPT_PREFIX}{blocked['uuid']}",
            arguments={"topic": "x"},
        )

    assert excinfo.value.code == "permission_denied"


def test_user_source_get_sanitizes_db_failure() -> None:
    class FailingDb:
        def get_prompt_by_uuid(self, prompt_uuid, include_deleted=False):
            raise DatabaseError("raw db failure")

        def close_connection(self):
            pass

    class FailingSource(UserPromptCatalogSource):
        def _open_db(self, context):
            return FailingDb()

    source = FailingSource(MCPPromptFormatter(max_rendered_chars=10_000))

    with pytest.raises(PromptCatalogError) as excinfo:
        source.get_prompt(
            context=_context_without_prompts_db(),
            name=f"{LIBRARY_PROMPT_PREFIX}11111111-1111-4111-8111-111111111111",
            arguments={"topic": "x"},
        )

    assert excinfo.value.code == "prompt_db_unavailable"
    assert excinfo.value.internal is True
    assert "raw db failure" not in excinfo.value.message


def test_user_source_get_sanitizes_internal_formatting_failure(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    prompt = _add_legacy_prompt(db_path, name="Bad Render")

    class InternalFailureFormatter(MCPPromptFormatter):
        def render_library_prompt(self, row, arguments):
            raise PromptCatalogError("invalid_prompt_definition", "Invalid row.", internal=True)

    source = UserPromptCatalogSource(InternalFailureFormatter(max_rendered_chars=10_000))

    with pytest.raises(PromptCatalogError) as excinfo:
        source.get_prompt(
            context=_context_for_prompts_db(db_path),
            name=f"{LIBRARY_PROMPT_PREFIX}{prompt['uuid']}",
            arguments={"topic": "x"},
        )

    assert excinfo.value.code == "prompt_db_unavailable"
    assert excinfo.value.internal is True
    assert "Invalid row" not in excinfo.value.message


def test_config_source_lists_only_explicit_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override_path = tmp_path / "rag_retrieval.txt"
    override_path.write_text("Retrieve context for {query}", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_RAG__RETRIEVAL_GUIDANCE", str(override_path))
    source = ConfigPromptCatalogSource(
        MCPPromptFormatter(max_rendered_chars=10_000),
        {
            "enabled": True,
            "entries": [
                {
                    "id": "rag.retrieval_guidance",
                    "module": "rag",
                    "key": "retrieval_guidance",
                    "title": "Retrieval Guidance",
                }
            ],
        },
    )

    result = source.list_prompts(cursor=PromptCatalogCursor(), limit=50)

    assert [prompt["name"] for prompt in result.prompts] == [
        f"{CONFIG_PROMPT_PREFIX}rag.retrieval_guidance"
    ]
    assert result.prompts[0]["arguments"][0]["name"] == "query"


def test_config_source_grouped_render_folds_system_user_and_preserves_assistant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    system_path = tmp_path / "system.txt"
    user_path = tmp_path / "user.txt"
    assistant_path = tmp_path / "assistant.txt"
    system_path.write_text("Use brief answers.", encoding="utf-8")
    user_path.write_text("Summarize {topic}", encoding="utf-8")
    assistant_path.write_text("Example summary.", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_CHAT__SUMMARY_SYSTEM", str(system_path))
    monkeypatch.setenv("TLDW_PROMPT_FILE_CHAT__SUMMARY_USER", str(user_path))
    monkeypatch.setenv("TLDW_PROMPT_FILE_CHAT__SUMMARY_EXAMPLE", str(assistant_path))
    source = ConfigPromptCatalogSource(
        MCPPromptFormatter(max_rendered_chars=10_000),
        {
            "enabled": True,
            "entries": [
                {
                    "id": "chat.summary",
                    "title": "Conversation Summary",
                    "messages": [
                        {"role": "system", "module": "chat", "key": "summary_system"},
                        {"role": "user", "module": "chat", "key": "summary_user"},
                        {"role": "assistant", "module": "chat", "key": "summary_example"},
                    ],
                }
            ],
        },
    )

    result = source.get_prompt(
        name=f"{CONFIG_PROMPT_PREFIX}chat.summary",
        arguments={"topic": "release notes"},
    )

    assert result["messages"] == [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": "System instructions:\nUse brief answers.\n\nUser prompt:\nSummarize release notes",
            },
        },
        {
            "role": "assistant",
            "content": {
                "type": "text",
                "text": "Example summary.",
            },
        },
    ]


def test_config_source_omits_missing_allowlist_entry() -> None:
    source = ConfigPromptCatalogSource(
        MCPPromptFormatter(max_rendered_chars=10_000),
        {
            "enabled": True,
            "entries": [
                {
                    "id": "missing.entry",
                    "module": "missing",
                    "key": "entry",
                    "title": "Missing Entry",
                }
            ],
        },
    )

    result = source.list_prompts(cursor=PromptCatalogCursor(), limit=50)

    assert result.prompts == []
    assert result.warnings == [
        {"source": "config", "code": "config_prompt_unavailable", "id": "missing.entry"}
    ]


def test_missing_config_entry_error_does_not_include_override_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-prompt.txt"
    monkeypatch.setenv("TLDW_PROMPT_FILE_MISSING__ENTRY", str(missing_path))
    source = ConfigPromptCatalogSource(
        MCPPromptFormatter(max_rendered_chars=10_000),
        {
            "enabled": True,
            "entries": [
                {
                    "id": "missing.entry",
                    "module": "missing",
                    "key": "entry",
                    "title": "Missing Entry",
                }
            ],
        },
    )

    with pytest.raises(PromptCatalogError) as excinfo:
        source.get_prompt(name=f"{CONFIG_PROMPT_PREFIX}missing.entry", arguments={})

    assert excinfo.value.code == "config_prompt_unavailable"  # nosec B101
    assert "Config prompt is unavailable." in str(excinfo.value)  # nosec B101
    assert str(missing_path) not in str(excinfo.value)  # nosec B101


def test_config_source_get_respects_disabled_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override_path = tmp_path / "prompt.txt"
    override_path.write_text("Search {query}", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_MCP__SEARCH_KNOWLEDGE", str(override_path))
    source = ConfigPromptCatalogSource(
        MCPPromptFormatter(max_rendered_chars=10_000),
        {
            "enabled": False,
            "entries": [
                {
                    "id": "mcp.search_knowledge",
                    "module": "mcp",
                    "key": "search_knowledge",
                    "title": "Search Knowledge",
                }
            ],
        },
    )

    with pytest.raises(PromptCatalogError) as excinfo:
        source.get_prompt(name=f"{CONFIG_PROMPT_PREFIX}mcp.search_knowledge", arguments={"query": "x"})

    assert excinfo.value.code == "prompt_not_found"


def test_config_source_reports_entries_after_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    override_path = tmp_path / "prompt.txt"
    override_path.write_text("Search {query}", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_MCP__SEARCH_KNOWLEDGE", str(override_path))
    source = ConfigPromptCatalogSource(
        MCPPromptFormatter(max_rendered_chars=10_000),
        {
            "enabled": True,
            "entries": [
                {
                    "id": "mcp.search_knowledge",
                    "module": "mcp",
                    "key": "search_knowledge",
                    "title": "Search Knowledge",
                }
            ],
        },
    )

    assert source.has_entries_after(0) is True
    assert source.has_entries_after(1) is False


def test_user_source_skips_library_when_cursor_is_inside_config_page(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    _add_legacy_prompt(db_path, name="Alpha")
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    result = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=PromptCatalogCursor(library_done=True, config_index=1),
        limit=50,
    )

    assert result.prompts == []
    assert result.next_cursor is None
    assert result.warnings == []


def test_user_source_uses_collate_nocase_keyset_with_raw_name_cursor(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    alpha = _add_legacy_prompt(db_path, name="Alpha")
    beta = _add_legacy_prompt(db_path, name="beta")
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    first_page = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=PromptCatalogCursor(),
        limit=1,
    )
    second_page = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=first_page.next_cursor,
        limit=1,
    )

    assert first_page.prompts[0]["name"] == f"{LIBRARY_PROMPT_PREFIX}{alpha['uuid']}"
    assert first_page.next_cursor.library_after_name == "Alpha"
    assert second_page.prompts[0]["name"] == f"{LIBRARY_PROMPT_PREFIX}{beta['uuid']}"


def test_user_source_keyset_page_is_stable_when_prompt_inserted_before_cursor(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    alpha = _add_legacy_prompt(db_path, name="Alpha")
    charlie = _add_legacy_prompt(db_path, name="Charlie")
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    first_page = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=PromptCatalogCursor(),
        limit=1,
    )
    _add_legacy_prompt(db_path, name="Aardvark")
    second_page = source.list_prompts(
        context=_context_for_prompts_db(db_path),
        cursor=first_page.next_cursor,
        limit=10,
    )

    assert [prompt["name"] for prompt in first_page.prompts] == [  # nosec B101
        f"{LIBRARY_PROMPT_PREFIX}{alpha['uuid']}"
    ]
    assert [prompt["name"] for prompt in second_page.prompts] == [  # nosec B101
        f"{LIBRARY_PROMPT_PREFIX}{charlie['uuid']}"
    ]


def _prompts_module_config(page_size: int) -> ModuleConfig:
    return ModuleConfig(
        name="prompts",
        settings={
            "prompt_list_page_size": page_size,
            "max_rendered_prompt_chars": 10_000,
            "config_prompts": {
                "enabled": True,
                "entries": [
                    {
                        "id": "mcp.search_knowledge",
                        "module": "mcp",
                        "key": "search_knowledge",
                        "title": "Search Knowledge",
                    }
                ],
            },
        },
    )


@pytest.mark.asyncio
async def test_prompts_module_context_list_combines_library_and_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    library_prompt = _add_legacy_prompt(db_path, name="Library Prompt")
    override_path = tmp_path / "search_knowledge.txt"
    override_path.write_text("Search for {query}", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_MCP__SEARCH_KNOWLEDGE", str(override_path))
    module = PromptsModule(_prompts_module_config(page_size=50))
    await module.on_initialize()

    result = await module.get_prompts_for_context(_context_for_prompts_db(db_path), {})

    assert [prompt["name"] for prompt in result["prompts"]] == [
        f"{LIBRARY_PROMPT_PREFIX}{library_prompt['uuid']}",
        f"{CONFIG_PROMPT_PREFIX}mcp.search_knowledge",
    ]
    assert "nextCursor" not in result


@pytest.mark.asyncio
async def test_prompts_module_returns_config_cursor_when_library_exactly_fills_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = str(tmp_path / "prompts.db")
    library_prompt = _add_legacy_prompt(db_path, name="Library Prompt")
    override_path = tmp_path / "search_knowledge.txt"
    override_path.write_text("Search for {query}", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_MCP__SEARCH_KNOWLEDGE", str(override_path))
    module = PromptsModule(_prompts_module_config(page_size=1))
    await module.on_initialize()

    first_page = await module.get_prompts_for_context(_context_for_prompts_db(db_path), {})
    second_page = await module.get_prompts_for_context(
        _context_for_prompts_db(db_path),
        {"cursor": first_page["nextCursor"]},
    )

    assert [prompt["name"] for prompt in first_page["prompts"]] == [
        f"{LIBRARY_PROMPT_PREFIX}{library_prompt['uuid']}"
    ]
    assert first_page["nextCursor"]
    assert [prompt["name"] for prompt in second_page["prompts"]] == [
        f"{CONFIG_PROMPT_PREFIX}mcp.search_knowledge"
    ]
    assert "nextCursor" not in second_page


@pytest.mark.asyncio
async def test_prompts_module_context_get_routes_by_namespace(tmp_path: Path) -> None:
    db_path = str(tmp_path / "prompts.db")
    library_prompt = _add_legacy_prompt(db_path, name="Library Prompt")
    module = PromptsModule(_prompts_module_config(page_size=50))
    await module.on_initialize()

    result = await module.get_prompt_for_context(
        f"{LIBRARY_PROMPT_PREFIX}{library_prompt['uuid']}",
        {"topic": "MCP catalog"},
        _context_for_prompts_db(db_path),
    )

    assert result["messages"][0]["content"]["text"].endswith("Summarize MCP catalog")


@pytest.mark.asyncio
async def test_prompts_module_bad_cursor_raises_catalog_error(tmp_path: Path) -> None:
    module = PromptsModule(_prompts_module_config(page_size=50))
    await module.on_initialize()

    with pytest.raises(PromptCatalogError) as excinfo:
        await module.get_prompts_for_context(
            _context_for_prompts_db(str(tmp_path / "prompts.db")),
            {"cursor": "not-valid-base64"},
        )

    assert excinfo.value.code == "invalid_cursor"


def test_catalog_adapter_does_not_import_prompt_studio(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_catalog"
    blocked_segments = (
        "Prompt_Management.PromptStudio",
        "PromptStudio",
        "prompt_studio",
        "promptstudio",
    )
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if any(segment in name for segment in blocked_segments):
            raise AssertionError(f"prompts_catalog imported Prompt Studio module: {name}")
        return real_import(name, globals, locals, fromlist, level)

    sys.modules.pop(module_name, None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    prompts_catalog = importlib.import_module(module_name)

    assert hasattr(prompts_catalog, "MCPPromptFormatter")
