# MCP Prompt Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Expose user Prompt Library prompts and explicitly allowlisted config prompts through MCP protocol-level `prompts/list` and `prompts/get`.

**Architecture:** Keep `PromptsModule` as the MCP module boundary and add a focused catalog adapter file that owns listing, cursor parsing, config allowlist parsing, argument validation, and MCP-safe message formatting. Update protocol handlers to use context-aware prompt hooks for dynamic prompts, route `library:` and `config:` names directly to `PromptsModule`, and preserve the existing static prompt registry for context-free modules.

**Tech Stack:** FastAPI, Python async/await, SQLite via `PromptsDatabase`, Pydantic structured prompt models, MCP Unified protocol handlers, pytest, Bandit.

---

## Source References

- Spec: `Docs/superpowers/specs/2026-06-22-mcp-prompt-catalog-support-design.md`
- Backlog task: `TASK-2343`
- Follow-up backlog task: `TASK-2341`
- MCP prompt spec: `https://modelcontextprotocol.io/specification/2025-06-18/server/prompts`

## File Structure

- Create `tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py`
  - Owns `PromptCatalogError`, cursor encoding/decoding, MCP prompt definition formatting, user Prompt Library source, and config prompt allowlist source.
  - Keeps prompt body rendering and prompt body validation out of `protocol.py`.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/base.py`
  - Adds backward-compatible context-aware prompt hooks.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_module.py`
  - Initializes the catalog adapter and exposes context-aware prompt hooks.
  - Leaves `prompts.search` and `prompts.get` tool behavior unchanged.
- Modify `tldw_Server_API/app/core/MCP_unified/protocol.py`
  - Advertises MCP-compliant prompt capability.
  - Uses context-aware prompt hooks for listing.
  - Routes `library:` and `config:` prompt names directly to `PromptsModule`.
  - Preserves static module prompt behavior for future context-free modules.
- Modify `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
  - Adds `cursor` query support to list-only `GET /api/v1/mcp/prompts`.
- Modify `tldw_Server_API/Config_Files/mcp_modules.yaml`
  - Registers the `prompts` module with an empty explicit config prompt allowlist.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py`
  - Unit coverage for formatter, cursor, user source, config source, and module hooks.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py`
  - Protocol coverage for capabilities, routing, permissions, invalid params, and warnings.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`
  - Asserts the shipped YAML registers the prompts module with an empty config allowlist.
- Create `tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py`
  - Covers the HTTP convenience `cursor` mapping if an existing MCP endpoint test file is not a better local fit.
- Create `Docs/MCP/mcp_prompts.md`
  - Documents protocol prompt support, stable names, permissions, config allowlisting, and HTTP list-only behavior.
- Modify `Docs/MCP/mcp_tool_catalogs.md`
  - Links to the new prompt catalog doc from the MCP docs area.

## Data Contracts

Add these public contracts in `prompts_catalog.py`:

```python
LIBRARY_PROMPT_PREFIX = "library:"
CONFIG_PROMPT_PREFIX = "config:"
DEFAULT_PROMPT_LIST_PAGE_SIZE = 50
MAX_PROMPT_LIST_PAGE_SIZE = 100
DEFAULT_MAX_RENDERED_PROMPT_CHARS = 100_000


class PromptCatalogError(ValueError):
    """Sanitized prompt catalog failure suitable for JSON-RPC invalid params."""

    def __init__(self, code: str, message: str, *, internal: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.internal = internal


@dataclass(frozen=True)
class PromptCatalogCursor:
    library_after_name: str | None = None
    library_after_uuid: str | None = None
    library_done: bool = False
    config_index: int = 0
```

Context-aware module hooks use this shape:

```python
async def get_prompts_for_context(
    self,
    context: Any,
    params: dict[str, Any],
) -> dict[str, Any]:
    ...


async def get_prompt_for_context(
    self,
    name: str,
    arguments: dict[str, Any],
    context: Any,
) -> dict[str, Any]:
    ...
```

MCP prompt definitions use this shape:

```python
{
    "name": "library:2f5cf2fd-...",
    "title": "Summarizer",
    "description": "Short summary without prompt body",
    "arguments": [
        {
            "name": "topic",
            "title": "Topic",
            "description": "Topic to summarize",
            "required": True,
        }
    ],
    "_meta": {
        "tldw": {
            "source": "library",
            "prompt_uuid": "2f5cf2fd-...",
            "prompt_id": 123,
            "version": 4,
            "tags": ["summary"],
        }
    },
}
```

## Task 1: Add Catalog Adapter Tests First

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py`
- Read: `tldw_Server_API/app/core/DB_Management/Prompts_DB.py`
- Read: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/models.py`
- Read: `tldw_Server_API/app/core/Prompt_Management/structured_prompts/conversion.py`

- [x] **Step 1: Create focused failing tests for cursor handling and formatter behavior**

Create `tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py` with these imports and first tests:

```python
from __future__ import annotations

import base64
import json
import uuid

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_catalog import (
    CONFIG_PROMPT_PREFIX,
    LIBRARY_PROMPT_PREFIX,
    MCPPromptFormatter,
    PromptCatalogCursor,
    PromptCatalogError,
    decode_prompt_cursor,
    encode_prompt_cursor,
)


def test_prompt_cursor_round_trips_without_padding() -> None:
    cursor = PromptCatalogCursor(
        library_after_name="alpha",
        library_after_uuid="11111111-1111-4111-8111-111111111111",
    )

    encoded = encode_prompt_cursor(cursor)

    assert "=" not in encoded
    assert decode_prompt_cursor(encoded) == cursor


def test_prompt_cursor_round_trips_config_segment() -> None:
    cursor = PromptCatalogCursor(library_done=True, config_index=3)

    assert decode_prompt_cursor(encode_prompt_cursor(cursor)) == cursor


def test_prompt_cursor_rejects_bad_payload() -> None:
    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor("not-valid-base64")

    assert excinfo.value.code == "invalid_cursor"


def test_prompt_cursor_rejects_partial_library_keyset() -> None:
    raw = _encode_raw_cursor({"v": 1, "library_after_name": "alpha", "config_index": 0})

    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor(raw)

    assert excinfo.value.code == "invalid_cursor"


def test_prompt_cursor_rejects_config_index_before_library_done() -> None:
    raw = _encode_raw_cursor({"v": 1, "library_done": False, "config_index": 1})

    with pytest.raises(PromptCatalogError) as excinfo:
        decode_prompt_cursor(raw)

    assert excinfo.value.code == "invalid_cursor"


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
```

- [x] **Step 2: Run the new tests to verify imports fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -v
```

Expected: FAIL with `ModuleNotFoundError` or import errors for `prompts_catalog` symbols.

- [x] **Step 3: Add failing user-source and config-source tests**

Append these tests to `test_prompts_catalog.py`:

```python
from types import SimpleNamespace

from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_catalog import (
    ConfigPromptCatalogSource,
    UserPromptCatalogSource,
)


def _context_for_prompts_db(db_path: str, *, scoped_ids: list[str] | None = None) -> SimpleNamespace:
    metadata = {}
    if scoped_ids is not None:
        metadata["persona_scope"] = {"explicit_ids": {"prompt_id": scoped_ids}}
    return SimpleNamespace(
        db_paths={"prompts": db_path},
        metadata=metadata,
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


def test_user_source_lists_non_deleted_library_prompts_with_uuid_names(tmp_path) -> None:
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


def test_user_source_filters_by_prompt_id_scope(tmp_path) -> None:
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


def test_user_source_get_validates_uuid_and_scope(tmp_path) -> None:
    db_path = str(tmp_path / "prompts.db")
    allowed = _add_legacy_prompt(db_path, name="Allowed")
    source = UserPromptCatalogSource(MCPPromptFormatter(max_rendered_chars=10_000))

    result = source.get_prompt(
        context=_context_for_prompts_db(db_path, scoped_ids=[str(allowed["id"])]),
        name=f"{LIBRARY_PROMPT_PREFIX}{allowed['uuid']}",
        arguments={"topic": "scoped prompts"},
    )

    assert result["messages"][0]["content"]["text"].endswith("Summarize scoped prompts")


def test_config_source_lists_only_explicit_entries(monkeypatch, tmp_path) -> None:
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


def test_config_source_grouped_render_folds_system_user_and_preserves_assistant(monkeypatch, tmp_path) -> None:
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


def test_config_source_get_respects_disabled_flag(monkeypatch, tmp_path) -> None:
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


def test_config_source_reports_entries_after_index(monkeypatch, tmp_path) -> None:
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


def test_user_source_skips_library_when_cursor_is_inside_config_page(tmp_path) -> None:
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


def test_user_source_uses_collate_nocase_keyset_with_raw_name_cursor(tmp_path) -> None:
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


def test_catalog_adapter_does_not_import_prompt_studio() -> None:
    import inspect
    import tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_catalog as prompts_catalog

    assert "PromptStudio" not in inspect.getsource(prompts_catalog)
```

- [x] **Step 4: Run catalog tests to verify adapter symbols fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -v
```

Expected: FAIL on missing `prompts_catalog` symbols.

## Task 2: Implement Prompt Catalog Adapter

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py`

- [x] **Step 1: Add cursor, result, and error primitives**

Create `prompts_catalog.py` with this foundation:

```python
"""MCP prompt catalog adapters for user-library and allowlisted config prompts."""

from __future__ import annotations

import base64
import binascii
import json
import re
import uuid
from dataclasses import dataclass
from sqlite3 import Error as SQLiteError
from typing import Any, Mapping

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError, PromptsDatabase
from tldw_Server_API.app.core.MCP_unified.persona_scope import (
    assert_identifier_in_scope,
    get_explicit_scope_ids,
)
from tldw_Server_API.app.core.Prompt_Management.structured_prompts import (
    PromptBlock,
    PromptDefinition,
    PromptVariableDefinition,
    StructuredPromptAssemblyError,
    assemble_prompt_definition,
    convert_legacy_prompt_to_definition,
    extract_legacy_prompt_variables,
    normalize_legacy_prompt_template,
)
from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt

LIBRARY_PROMPT_PREFIX = "library:"
CONFIG_PROMPT_PREFIX = "config:"
DEFAULT_PROMPT_LIST_PAGE_SIZE = 50
MAX_PROMPT_LIST_PAGE_SIZE = 100
DEFAULT_MAX_RENDERED_PROMPT_CHARS = 100_000
_CONFIG_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")

_CATALOG_DB_EXCEPTIONS = (OSError, RuntimeError, SQLiteError, DatabaseError, TypeError, ValueError)


class PromptCatalogError(ValueError):
    """Sanitized prompt catalog failure suitable for JSON-RPC errors."""

    def __init__(self, code: str, message: str, *, internal: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.internal = internal


@dataclass(frozen=True)
class PromptCatalogCursor:
    library_after_name: str | None = None
    library_after_uuid: str | None = None
    library_done: bool = False
    config_index: int = 0


@dataclass(frozen=True)
class PromptCatalogListResult:
    prompts: list[dict[str, Any]]
    next_cursor: PromptCatalogCursor | None
    warnings: list[dict[str, Any]]


def encode_prompt_cursor(cursor: PromptCatalogCursor | None) -> str | None:
    if cursor is None:
        return None
    payload = {
        "v": 1,
        "library_after_name": cursor.library_after_name,
        "library_after_uuid": cursor.library_after_uuid,
        "library_done": cursor.library_done,
        "config_index": cursor.config_index,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_prompt_cursor(raw_cursor: str | None) -> PromptCatalogCursor:
    if not raw_cursor:
        return PromptCatalogCursor()
    try:
        padding = "=" * (-len(raw_cursor) % 4)
        raw = base64.urlsafe_b64decode((raw_cursor + padding).encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor") from exc
    if not isinstance(payload, dict) or payload.get("v") != 1:
        raise PromptCatalogError("invalid_cursor", "Unsupported prompt list cursor")
    library_after_name = payload.get("library_after_name")
    library_after_uuid = payload.get("library_after_uuid")
    library_done = payload.get("library_done", False)
    config_index = payload.get("config_index", 0)
    if library_after_name is not None and not isinstance(library_after_name, str):
        raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor")
    if (library_after_name is None) != (library_after_uuid is None):
        raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor")
    if library_after_uuid is not None:
        if not isinstance(library_after_uuid, str):
            raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor")
        try:
            uuid.UUID(library_after_uuid)
        except ValueError as exc:
            raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor") from exc
    if not isinstance(library_done, bool):
        raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor")
    if library_done and (library_after_name is not None or library_after_uuid is not None):
        raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor")
    if not isinstance(config_index, int) or config_index < 0:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor")
    if not library_done and config_index != 0:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt list cursor")
    return PromptCatalogCursor(
        library_after_name=library_after_name,
        library_after_uuid=library_after_uuid,
        library_done=library_done,
        config_index=config_index,
    )


def clamp_prompt_page_size(raw_value: Any) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = DEFAULT_PROMPT_LIST_PAGE_SIZE
    return min(MAX_PROMPT_LIST_PAGE_SIZE, max(1, value))
```

- [x] **Step 2: Run cursor tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py::test_prompt_cursor_round_trips_without_padding tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py::test_prompt_cursor_rejects_bad_payload -v
```

Expected: PASS.

- [x] **Step 3: Add formatter implementation**

Append this `MCPPromptFormatter` class to `prompts_catalog.py`:

```python
class MCPPromptFormatter:
    """Format tldw prompt records and config entries as MCP prompt payloads."""

    def __init__(self, *, max_rendered_chars: int = DEFAULT_MAX_RENDERED_PROMPT_CHARS) -> None:
        self.max_rendered_chars = max(1, int(max_rendered_chars or DEFAULT_MAX_RENDERED_PROMPT_CHARS))

    def validate_arguments(self, arguments: Mapping[str, Any] | None) -> dict[str, str]:
        if arguments is None:
            return {}
        if not isinstance(arguments, Mapping):
            raise PromptCatalogError("invalid_arguments", "Prompt arguments must be an object")
        validated: dict[str, str] = {}
        for key, value in arguments.items():
            if not isinstance(key, str) or not key:
                raise PromptCatalogError("invalid_argument_name", "Prompt argument names must be non-empty strings")
            if not isinstance(value, str):
                raise PromptCatalogError("invalid_argument_type", f"Prompt argument '{key}' must be a string")
            validated[key] = value
        return validated

    def library_prompt_definition(self, row: Mapping[str, Any]) -> dict[str, Any]:
        prompt_uuid = self._required_uuid(row)
        definition = self._coerce_definition(row)
        if definition is not None:
            arguments = self._arguments_from_definition(definition)
        else:
            arguments = self._legacy_arguments(row.get("system_prompt"), row.get("user_prompt"))
        return {
            "name": f"{LIBRARY_PROMPT_PREFIX}{prompt_uuid}",
            "title": str(row.get("name") or prompt_uuid),
            "description": self._list_description(row),
            "arguments": arguments,
            "_meta": {
                "tldw": {
                    "source": "library",
                    "prompt_uuid": prompt_uuid,
                    "prompt_id": row.get("id"),
                    "version": row.get("version"),
                    "tags": row.get("keywords") or [],
                }
            },
        }

    def render_library_prompt(self, row: Mapping[str, Any], arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        validated_arguments = self.validate_arguments(arguments)
        prompt_uuid = self._required_uuid(row)
        definition = self._coerce_definition(row)
        if definition is None:
            definition = convert_legacy_prompt_to_definition(
                system_prompt=str(row.get("system_prompt") or ""),
                user_prompt=str(row.get("user_prompt") or ""),
            )
        messages = self._assemble_messages(definition, validated_arguments)
        mcp_messages = self._to_mcp_messages(messages)
        self._enforce_rendered_size(mcp_messages)
        return {
            "description": str(row.get("details") or row.get("name") or prompt_uuid),
            "messages": mcp_messages,
            "_meta": {
                "tldw": {
                    "source": "library",
                    "prompt_uuid": prompt_uuid,
                    "prompt_id": row.get("id"),
                    "version": row.get("version"),
                }
            },
        }

    def config_prompt_definition(self, entry: Mapping[str, Any], parts: list[dict[str, str]]) -> dict[str, Any]:
        entry_id = str(entry["id"])
        variables = extract_legacy_prompt_variables(*(part["content"] for part in parts))
        return {
            "name": f"{CONFIG_PROMPT_PREFIX}{entry_id}",
            "title": str(entry.get("title") or entry_id),
            "description": str(entry.get("description") or ""),
            "arguments": [
                {"name": name, "title": name.replace("_", " ").title(), "description": None, "required": True}
                for name in variables
            ],
            "_meta": {
                "tldw": {
                    "source": "config",
                    "config_id": entry_id,
                }
            },
        }

    def render_config_prompt(
        self,
        entry: Mapping[str, Any],
        parts: list[dict[str, str]],
        arguments: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        validated_arguments = self.validate_arguments(arguments)
        variables = [
            PromptVariableDefinition(
                name=name,
                label=name.replace("_", " ").title(),
                required=True,
                input_type="textarea",
            )
            for name in extract_legacy_prompt_variables(*(part["content"] for part in parts))
        ]
        blocks = [
            PromptBlock(
                id=f"config_{idx}",
                name=f"Config Part {idx + 1}",
                role=part["role"],  # type: ignore[arg-type]
                content=normalize_legacy_prompt_template(part["content"]),
                enabled=True,
                order=(idx + 1) * 10,
                is_template=bool(extract_legacy_prompt_variables(part["content"])),
            )
            for idx, part in enumerate(parts)
        ]
        definition = PromptDefinition(variables=variables, blocks=blocks)
        messages = self._assemble_messages(definition, validated_arguments)
        mcp_messages = self._to_mcp_messages(messages)
        self._enforce_rendered_size(mcp_messages)
        return {
            "description": str(entry.get("description") or entry.get("title") or entry["id"]),
            "messages": mcp_messages,
            "_meta": {
                "tldw": {
                    "source": "config",
                    "config_id": str(entry["id"]),
                }
            },
        }

    def _coerce_definition(self, row: Mapping[str, Any]) -> PromptDefinition | None:
        payload = row.get("prompt_definition")
        if not payload:
            return None
        try:
            return PromptDefinition.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise PromptCatalogError("invalid_prompt_definition", "Stored prompt definition is invalid", internal=True) from exc

    def _arguments_from_definition(self, definition: PromptDefinition) -> list[dict[str, Any]]:
        return [
            {
                "name": variable.name,
                "title": variable.label or variable.name.replace("_", " ").title(),
                "description": variable.description,
                "required": bool(variable.required),
            }
            for variable in definition.variables
        ]

    def _legacy_arguments(self, *templates: Any) -> list[dict[str, Any]]:
        return [
            {"name": name, "title": name.replace("_", " ").title(), "description": None, "required": True}
            for name in extract_legacy_prompt_variables(*(str(template or "") for template in templates))
        ]

    def _assemble_messages(self, definition: PromptDefinition, arguments: Mapping[str, str]) -> list[dict[str, str]]:
        try:
            return assemble_prompt_definition(definition, arguments).messages
        except StructuredPromptAssemblyError as exc:
            raise PromptCatalogError(exc.code, str(exc)) from exc

    def _to_mcp_messages(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        pending_system: list[str] = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role in {"system", "developer"}:
                pending_system.append(content)
                continue
            if role == "assistant":
                output.append({"role": "assistant", "content": {"type": "text", "text": content}})
                continue
            if role == "user":
                text = content
                if pending_system:
                    text = "System instructions:\n" + "\n\n".join(pending_system) + "\n\nUser prompt:\n" + content
                    pending_system = []
                output.append({"role": "user", "content": {"type": "text", "text": text}})
                continue
            pending_system.append(content)

        if pending_system:
            output.append(
                {
                    "role": "user",
                    "content": {"type": "text", "text": "System instructions:\n" + "\n\n".join(pending_system)},
                }
            )
        return output

    def _enforce_rendered_size(self, messages: list[dict[str, Any]]) -> None:
        total = 0
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, dict):
                total += len(str(content.get("text") or ""))
        if total > self.max_rendered_chars:
            raise PromptCatalogError("rendered_prompt_too_large", "Rendered prompt exceeds configured size limit")

    def _required_uuid(self, row: Mapping[str, Any]) -> str:
        raw_uuid = str(row.get("uuid") or "").strip()
        try:
            parsed = uuid.UUID(raw_uuid)
        except ValueError as exc:
            raise PromptCatalogError("invalid_prompt_record", "Stored prompt UUID is invalid", internal=True) from exc
        return str(parsed)

    def _list_description(self, row: Mapping[str, Any]) -> str:
        details = str(row.get("details") or "").strip()
        if details:
            return " ".join(details.split())[:500]
        author = str(row.get("author") or "").strip()
        if author:
            return f"Prompt by {author}"
        return ""
```

- [x] **Step 4: Run formatter tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -k "formatter or cursor" -v
```

Expected: PASS.

- [x] **Step 5: Add user and config source implementation**

Append these source classes to `prompts_catalog.py`:

```python
class UserPromptCatalogSource:
    """Read MCP prompts from the authenticated user's regular Prompt Library DB."""

    def __init__(self, formatter: MCPPromptFormatter) -> None:
        self.formatter = formatter

    def list_prompts(
        self,
        *,
        context: Any,
        cursor: PromptCatalogCursor,
        limit: int,
    ) -> PromptCatalogListResult:
        if limit <= 0:
            return PromptCatalogListResult([], cursor, [])
        if cursor.library_done:
            return PromptCatalogListResult([], None, [])
        try:
            db = self._open_db(context)
        except PromptCatalogError:
            return PromptCatalogListResult(
                prompts=[],
                next_cursor=None,
                warnings=[{"source": "library", "code": "prompt_db_unavailable"}],
            )
        except _CATALOG_DB_EXCEPTIONS:
            return PromptCatalogListResult(
                prompts=[],
                next_cursor=None,
                warnings=[{"source": "library", "code": "prompt_db_unavailable"}],
            )
        try:
            rows = self._query_rows(db, context, cursor, limit + 1)
            visible_rows = rows[:limit]
            prompts = [self.formatter.library_prompt_definition(row) for row in visible_rows]
            next_cursor = None
            if len(rows) > limit and visible_rows:
                last = visible_rows[-1]
                next_cursor = PromptCatalogCursor(
                    library_after_name=str(last.get("name") or ""),
                    library_after_uuid=str(last.get("uuid")),
                    library_done=False,
                    config_index=cursor.config_index,
                )
            return PromptCatalogListResult(prompts=prompts, next_cursor=next_cursor, warnings=[])
        except PromptCatalogError as exc:
            if not exc.internal:
                raise
            logger.debug("MCP prompt library list failed with sanitized warning: {}", exc.code)
            return PromptCatalogListResult(
                prompts=[],
                next_cursor=None,
                warnings=[{"source": "library", "code": "prompt_db_unavailable"}],
            )
        except _CATALOG_DB_EXCEPTIONS as exc:
            logger.debug("MCP prompt library list failed with sanitized warning: {}", exc.__class__.__name__)
            return PromptCatalogListResult(
                prompts=[],
                next_cursor=None,
                warnings=[{"source": "library", "code": "prompt_db_unavailable"}],
            )
        finally:
            self._close_db(db)

    def get_prompt(self, *, context: Any, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        prompt_uuid = self._uuid_from_name(name)
        try:
            db = self._open_db(context)
            row = db.get_prompt_by_uuid(prompt_uuid, include_deleted=False)
            if not row:
                raise PromptCatalogError("prompt_not_found", "Prompt not found")
            assert_identifier_in_scope(context, "prompt_id", row.get("id"), label="Prompt")
            return self.formatter.render_library_prompt(row, arguments)
        except PromptCatalogError:
            raise
        except PermissionError as exc:
            raise PromptCatalogError("permission_denied", "Permission denied for prompt") from exc
        except _CATALOG_DB_EXCEPTIONS as exc:
            raise PromptCatalogError("prompt_db_unavailable", "Prompt library is unavailable", internal=True) from exc
        finally:
            if "db" in locals():
                self._close_db(db)

    def _query_rows(
        self,
        db: PromptsDatabase,
        context: Any,
        cursor: PromptCatalogCursor,
        limit: int,
    ) -> list[dict[str, Any]]:
        where_clauses = ["deleted = 0"]
        params: list[Any] = []
        scoped_ids = get_explicit_scope_ids(context, "prompt_id")
        if scoped_ids is not None:
            if not scoped_ids:
                return []
            sorted_ids = sorted(int(value) for value in scoped_ids if str(value).isdigit())
            if not sorted_ids:
                return []
            placeholders = ", ".join("?" for _ in sorted_ids)
            where_clauses.append(f"id IN ({placeholders})")  # nosec B608
            params.extend(sorted_ids)
        if cursor.library_after_name is not None and cursor.library_after_uuid is not None:
            where_clauses.append("(name COLLATE NOCASE > ? OR (name COLLATE NOCASE = ? AND uuid > ?))")
            params.extend([cursor.library_after_name, cursor.library_after_name, cursor.library_after_uuid])
        query = f"""
            SELECT *
            FROM Prompts
            WHERE {' AND '.join(where_clauses)}
            ORDER BY name COLLATE NOCASE ASC, uuid ASC
            LIMIT ?
        """  # nosec B608
        params.append(limit)
        db_cursor = db.execute_query(query, tuple(params))
        return [db._deserialize_prompt_record(dict(row)) for row in db_cursor.fetchall()]

    def _uuid_from_name(self, name: str) -> str:
        if not name.startswith(LIBRARY_PROMPT_PREFIX):
            raise PromptCatalogError("invalid_prompt_name", "Invalid library prompt name")
        raw_uuid = name[len(LIBRARY_PROMPT_PREFIX):]
        try:
            return str(uuid.UUID(raw_uuid))
        except ValueError as exc:
            raise PromptCatalogError("invalid_prompt_name", "Invalid library prompt name") from exc

    def _open_db(self, context: Any) -> PromptsDatabase:
        db_paths = getattr(context, "db_paths", None)
        if not isinstance(db_paths, dict) or not db_paths.get("prompts"):
            raise PromptCatalogError("prompt_db_unavailable", "Prompt library is unavailable", internal=True)
        return PromptsDatabase(db_path=str(db_paths["prompts"]), client_id="mcp_prompt_catalog")

    def _close_db(self, db: PromptsDatabase) -> None:
        try:
            db.close_connection()
        except _CATALOG_DB_EXCEPTIONS as exc:
            logger.debug("Failed to close Prompt Library DB after MCP prompt catalog access: {}", exc.__class__.__name__)


class ConfigPromptCatalogSource:
    """Read MCP prompts from explicit server-config allowlist entries."""

    def __init__(self, formatter: MCPPromptFormatter, config: Mapping[str, Any] | None) -> None:
        self.formatter = formatter
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        raw_entries = config.get("entries") if isinstance(config.get("entries"), list) else []
        self.entries = [entry for entry in raw_entries if isinstance(entry, Mapping)]

    def list_prompts(self, *, cursor: PromptCatalogCursor, limit: int) -> PromptCatalogListResult:
        if not self.enabled or limit <= 0:
            return PromptCatalogListResult([], None, [])
        prompts: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        next_cursor: PromptCatalogCursor | None = None
        for index in range(cursor.config_index, len(self.entries)):
            if len(prompts) >= limit:
                next_cursor = PromptCatalogCursor(library_done=True, config_index=index)
                break
            entry = self.entries[index]
            try:
                parts = self._load_entry_parts(entry)
                prompts.append(self.formatter.config_prompt_definition(entry, parts))
            except PromptCatalogError:
                warnings.append({"source": "config", "code": "config_prompt_unavailable", "id": str(entry.get("id") or "")})
        return PromptCatalogListResult(prompts=prompts, next_cursor=next_cursor, warnings=warnings)

    def has_entries_after(self, config_index: int) -> bool:
        return self.enabled and max(0, config_index) < len(self.entries)

    def get_prompt(self, *, name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        if not self.enabled:
            raise PromptCatalogError("prompt_not_found", "Prompt not found")
        entry_id = self._entry_id_from_name(name)
        for entry in self.entries:
            if str(entry.get("id") or "") == entry_id:
                parts = self._load_entry_parts(entry)
                return self.formatter.render_config_prompt(entry, parts, arguments)
        raise PromptCatalogError("prompt_not_found", "Prompt not found")

    def _entry_id_from_name(self, name: str) -> str:
        if not name.startswith(CONFIG_PROMPT_PREFIX):
            raise PromptCatalogError("invalid_prompt_name", "Invalid config prompt name")
        entry_id = name[len(CONFIG_PROMPT_PREFIX):]
        if not _CONFIG_ID_RE.fullmatch(entry_id):
            raise PromptCatalogError("invalid_prompt_name", "Invalid config prompt name")
        return entry_id

    def _load_entry_parts(self, entry: Mapping[str, Any]) -> list[dict[str, str]]:
        entry_id = str(entry.get("id") or "")
        if not _CONFIG_ID_RE.fullmatch(entry_id):
            raise PromptCatalogError("invalid_config_entry", "Invalid config prompt entry")
        messages = entry.get("messages")
        if isinstance(messages, list):
            return [self._load_part(part) for part in messages if isinstance(part, Mapping)]
        module = str(entry.get("module") or "")
        key = str(entry.get("key") or "")
        if not module or not key:
            raise PromptCatalogError("invalid_config_entry", "Invalid config prompt entry")
        return [self._load_part({"role": "user", "module": module, "key": key})]

    def _load_part(self, part: Mapping[str, Any]) -> dict[str, str]:
        role = str(part.get("role") or "user")
        if role not in {"system", "developer", "user", "assistant"}:
            raise PromptCatalogError("invalid_config_entry", "Invalid config prompt role")
        module = str(part.get("module") or "")
        key = str(part.get("key") or "")
        if not module or not key:
            raise PromptCatalogError("invalid_config_entry", "Invalid config prompt source")
        content = load_prompt(module, key)
        if content is None:
            raise PromptCatalogError("config_prompt_unavailable", "Config prompt is unavailable")
        return {"role": role, "content": content}
```

- [x] **Step 6: Run full catalog tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -v
```

Expected: PASS.

- [x] **Step 7: Commit adapter and tests**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py
git commit -m "feat: add MCP prompt catalog adapter"
```

## Task 3: Add Context-Aware Prompt Hooks To BaseModule And PromptsModule

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/base.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py`

- [x] **Step 1: Add failing module hook tests**

Append these tests to `test_prompts_catalog.py`:

```python
import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_module import PromptsModule


@pytest.mark.asyncio
async def test_prompts_module_context_list_combines_library_and_config(monkeypatch, tmp_path) -> None:
    db_path = str(tmp_path / "prompts.db")
    row = _add_legacy_prompt(db_path, name="Alpha")
    override_path = tmp_path / "config.txt"
    override_path.write_text("Search for {query}", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_MCP__SEARCH_KNOWLEDGE", str(override_path))
    module = PromptsModule(
        ModuleConfig(
            name="prompts",
            settings={
                "prompt_list_page_size": 50,
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
    )
    await module.on_initialize()

    result = await module.get_prompts_for_context(_context_for_prompts_db(db_path), {})

    assert [prompt["name"] for prompt in result["prompts"]] == [
        f"{LIBRARY_PROMPT_PREFIX}{row['uuid']}",
        f"{CONFIG_PROMPT_PREFIX}mcp.search_knowledge",
    ]
    assert "nextCursor" not in result


@pytest.mark.asyncio
async def test_prompts_module_returns_config_cursor_when_library_exactly_fills_page(monkeypatch, tmp_path) -> None:
    db_path = str(tmp_path / "prompts.db")
    row = _add_legacy_prompt(db_path, name="Alpha")
    override_path = tmp_path / "config.txt"
    override_path.write_text("Search for {query}", encoding="utf-8")
    monkeypatch.setenv("TLDW_PROMPT_FILE_MCP__SEARCH_KNOWLEDGE", str(override_path))
    module = PromptsModule(
        ModuleConfig(
            name="prompts",
            settings={
                "prompt_list_page_size": 1,
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
    )
    await module.on_initialize()

    first_page = await module.get_prompts_for_context(_context_for_prompts_db(db_path), {})
    second_page = await module.get_prompts_for_context(
        _context_for_prompts_db(db_path),
        {"cursor": first_page["nextCursor"]},
    )

    assert [prompt["name"] for prompt in first_page["prompts"]] == [f"{LIBRARY_PROMPT_PREFIX}{row['uuid']}"]
    assert [prompt["name"] for prompt in second_page["prompts"]] == [
        f"{CONFIG_PROMPT_PREFIX}mcp.search_knowledge"
    ]


@pytest.mark.asyncio
async def test_prompts_module_context_get_routes_by_namespace(tmp_path) -> None:
    db_path = str(tmp_path / "prompts.db")
    row = _add_legacy_prompt(db_path, name="Alpha")
    module = PromptsModule(ModuleConfig(name="prompts", settings={"max_rendered_prompt_chars": 10_000}))
    await module.on_initialize()

    result = await module.get_prompt_for_context(
        f"{LIBRARY_PROMPT_PREFIX}{row['uuid']}",
        {"topic": "module hooks"},
        _context_for_prompts_db(db_path),
    )

    assert result["messages"][0]["content"]["text"].endswith("Summarize module hooks")


@pytest.mark.asyncio
async def test_prompts_module_bad_cursor_raises_catalog_error(tmp_path) -> None:
    module = PromptsModule(ModuleConfig(name="prompts"))
    await module.on_initialize()

    with pytest.raises(PromptCatalogError) as excinfo:
        await module.get_prompts_for_context(_context_for_prompts_db(str(tmp_path / "prompts.db")), {"cursor": "bad"})

    assert excinfo.value.code == "invalid_cursor"
```

- [x] **Step 2: Run module hook tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -k "prompts_module" -v
```

Expected: FAIL because context-aware hooks are not implemented on `PromptsModule`.

- [x] **Step 3: Add backward-compatible hooks to BaseModule**

Modify `tldw_Server_API/app/core/MCP_unified/modules/base.py` after `get_prompt()`:

```python
    async def get_prompts_for_context(self, context: Optional[Any], params: dict[str, Any]) -> dict[str, Any]:
        """Get prompts for a request context.

        Dynamic modules can override this to honor per-user databases,
        permissions, scopes, and pagination. Static modules inherit the
        existing context-free prompt list behavior.
        """
        return {"prompts": await self.get_prompts()}

    async def get_prompt_for_context(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Optional[Any],
    ) -> dict[str, Any]:
        """Get one prompt for a request context."""
        return await self.get_prompt(name, arguments)
```

- [x] **Step 4: Add catalog setup and hooks to PromptsModule**

Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_module.py` imports:

```python
from .prompts_catalog import (
    CONFIG_PROMPT_PREFIX,
    LIBRARY_PROMPT_PREFIX,
    ConfigPromptCatalogSource,
    MCPPromptFormatter,
    PromptCatalogError,
    UserPromptCatalogSource,
    clamp_prompt_page_size,
    decode_prompt_cursor,
    encode_prompt_cursor,
)
```

Add initialization in `on_initialize()`:

```python
    async def on_initialize(self) -> None:
        logger.info(f"Initializing Prompts module: {self.name}")
        settings = self.config.settings or {}
        self._prompt_list_page_size = clamp_prompt_page_size(settings.get("prompt_list_page_size", 50))
        self._prompt_formatter = MCPPromptFormatter(
            max_rendered_chars=int(settings.get("max_rendered_prompt_chars", 100000) or 100000)
        )
        self._user_prompt_source = UserPromptCatalogSource(self._prompt_formatter)
        self._config_prompt_source = ConfigPromptCatalogSource(
            self._prompt_formatter,
            settings.get("config_prompts") if isinstance(settings.get("config_prompts"), dict) else {},
        )
```

Add context-aware prompt methods before `validate_tool_arguments()`:

```python
    async def get_prompts_for_context(self, context: Any, params: dict[str, Any]) -> dict[str, Any]:
        params = params or {}
        cursor = decode_prompt_cursor(params.get("cursor"))
        page_size = self._prompt_list_page_size
        library_result = await asyncio.to_thread(
            self._user_prompt_source.list_prompts,
            context=context,
            cursor=cursor,
            limit=page_size,
        )
        prompts = list(library_result.prompts)
        warnings = list(library_result.warnings)
        next_cursor = library_result.next_cursor

        if len(prompts) < page_size and library_result.next_cursor is None:
            remaining = page_size - len(prompts)
            config_result = await asyncio.to_thread(
                self._config_prompt_source.list_prompts,
                cursor=cursor,
                limit=remaining,
            )
            prompts.extend(config_result.prompts)
            warnings.extend(config_result.warnings)
            next_cursor = config_result.next_cursor
        elif (
            len(prompts) == page_size
            and library_result.next_cursor is None
            and self._config_prompt_source.has_entries_after(cursor.config_index)
        ):
            next_cursor = PromptCatalogCursor(library_done=True, config_index=cursor.config_index)

        result: dict[str, Any] = {"prompts": prompts}
        encoded_cursor = encode_prompt_cursor(next_cursor)
        if encoded_cursor:
            result["nextCursor"] = encoded_cursor
        if warnings:
            result["_meta"] = {"tldw": {"warnings": warnings}}
        return result

    async def get_prompt_for_context(self, name: str, arguments: dict[str, Any], context: Any) -> dict[str, Any]:
        if name.startswith(LIBRARY_PROMPT_PREFIX):
            return await asyncio.to_thread(
                self._user_prompt_source.get_prompt,
                context=context,
                name=name,
                arguments=arguments,
            )
        if name.startswith(CONFIG_PROMPT_PREFIX):
            return await asyncio.to_thread(
                self._config_prompt_source.get_prompt,
                name=name,
                arguments=arguments,
            )
        raise PromptCatalogError("prompt_not_found", "Prompt not found")
```

- [x] **Step 5: Run module hook tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -k "prompts_module" -v
```

Expected: PASS.

- [x] **Step 6: Run full catalog test file**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py -v
```

Expected: PASS.

- [x] **Step 7: Commit module hooks**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/base.py tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_module.py tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py
git commit -m "feat: wire MCP prompt catalog hooks"
```

## Task 4: Update Protocol Prompt Capability, Listing, Routing, And Errors

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/protocol.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py`

- [x] **Step 1: Create protocol-focused failing tests**

Create `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py`:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_catalog import (
    LIBRARY_PROMPT_PREFIX,
    PromptCatalogError,
)
from tldw_Server_API.app.core.MCP_unified.auth.rbac import Action, Resource
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext


class PromptOnlyRegistry:
    def __init__(self, modules):
        self.modules = modules
        self.find_calls: list[str] = []

    async def get_all_modules(self):
        return self.modules

    async def get_module(self, module_id: str):
        return self.modules.get(module_id)

    async def find_module_for_prompt(self, name: str):
        self.find_calls.append(name)
        return None

    def get_module_id_for_prompt(self, name: str):
        return None


class PromptPermissionPolicy:
    def __init__(self, *, prompt_read: bool, module_read: bool) -> None:
        self.prompt_read = prompt_read
        self.module_read = module_read

    def check_permission(self, user_id, resource, action, resource_id=None):
        if resource == Resource.PROMPT and action == Action.READ:
            return self.prompt_read
        if resource == Resource.MODULE and action == Action.READ:
            return self.module_read
        return False


def _handler_with_registry(registry) -> MCPProtocol:
    handler = MCPProtocol()
    handler.module_registry = registry
    return handler


class ContextPromptModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict]:
        return []

    async def execute_tool(self, tool_name: str, arguments: dict, context=None):
        raise NotImplementedError

    async def get_prompts_for_context(self, context, params):
        return {
            "prompts": [{"name": "library:11111111-1111-4111-8111-111111111111", "title": "A"}],
            "nextCursor": "abc",
            "_meta": {"tldw": {"warnings": [{"source": "library", "code": "prompt_db_unavailable"}]}},
        }

    async def get_prompt_for_context(self, name, arguments, context):
        return {"description": name, "messages": [{"role": "user", "content": {"type": "text", "text": "ok"}}]}


@pytest.mark.asyncio
async def test_initialize_declares_mcp_prompt_capability() -> None:
    handler = _handler_with_registry(PromptOnlyRegistry({}))
    context = RequestContext("req-1", user_id="1")

    result = await handler._handle_initialize({"clientInfo": {"name": "test"}}, context)

    assert result["capabilities"]["prompts"] == {"listChanged": False}


@pytest.mark.asyncio
async def test_prompts_list_uses_context_hook_and_preserves_cursor_and_warnings(monkeypatch) -> None:
    module = ContextPromptModule(ModuleConfig(name="prompts"))
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))
    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", lambda *args, **kwargs: _async_true())

    result = await handler._handle_prompts_list({"cursor": "incoming"}, RequestContext("req-1", user_id="1"))

    assert result["prompts"][0]["module"] == "prompts"
    assert result["nextCursor"] == "abc"
    assert result["_meta"]["tldw"]["warnings"][0]["code"] == "prompt_db_unavailable"


@pytest.mark.asyncio
async def test_prompts_get_dispatches_namespaced_prompt_before_global_registry(monkeypatch) -> None:
    module = ContextPromptModule(ModuleConfig(name="prompts"))
    registry = PromptOnlyRegistry({"prompts": module})
    handler = _handler_with_registry(registry)
    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", lambda *args, **kwargs: _async_true())

    result = await handler._handle_prompts_get(
        {"name": f"{LIBRARY_PROMPT_PREFIX}11111111-1111-4111-8111-111111111111", "arguments": {}},
        RequestContext("req-1", user_id="1"),
    )

    assert result["description"].startswith(LIBRARY_PROMPT_PREFIX)
    assert registry.find_calls == []


@pytest.mark.asyncio
async def test_prompts_get_rejects_non_object_arguments(monkeypatch) -> None:
    module = ContextPromptModule(ModuleConfig(name="prompts"))
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))

    with pytest.raises(Exception) as excinfo:
        await handler._handle_prompts_get(
            {"name": f"{LIBRARY_PROMPT_PREFIX}11111111-1111-4111-8111-111111111111", "arguments": []},
            RequestContext("req-1", user_id="1"),
        )

    assert "Prompt arguments must be an object" in str(excinfo.value)


@pytest.mark.asyncio
async def test_namespaced_prompts_read_does_not_require_module_read() -> None:
    module = ContextPromptModule(ModuleConfig(name="prompts"))
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))
    handler.rbac_policy = PromptPermissionPolicy(prompt_read=True, module_read=False)

    result = await handler._handle_prompts_list({}, RequestContext("req-1", user_id="1"))

    assert result["prompts"][0]["name"].startswith(LIBRARY_PROMPT_PREFIX)


@pytest.mark.asyncio
async def test_namespaced_prompt_denies_when_only_module_read_is_granted() -> None:
    module = ContextPromptModule(ModuleConfig(name="prompts"))
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))
    handler.rbac_policy = PromptPermissionPolicy(prompt_read=False, module_read=True)

    result = await handler._handle_prompts_list({}, RequestContext("req-1", user_id="1"))

    assert result["prompts"] == []


async def _async_true(*args, **kwargs):
    return True
```

- [x] **Step 2: Run protocol tests to verify failures**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py -v
```

Expected: FAIL on capability shape, module permission behavior, and namespace dispatch.

- [x] **Step 3: Import catalog error and prefixes in protocol.py**

Modify imports in `tldw_Server_API/app/core/MCP_unified/protocol.py`:

```python
from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_catalog import (
    CONFIG_PROMPT_PREFIX,
    LIBRARY_PROMPT_PREFIX,
    PromptCatalogError,
)
```

Place this import near other MCP Unified imports. If importing this module introduces a circular import, move only constants and `PromptCatalogError` into a tiny new `tldw_Server_API/app/core/MCP_unified/prompt_catalog_contract.py` and import that contract from both `protocol.py` and `prompts_catalog.py`.

- [x] **Step 4: Fix initialize capability shape**

Replace the current `capabilities` block in `_handle_initialize()`:

```python
        capabilities = {
            "tools": {"available": bool(modules)},
            "resources": {"available": bool(modules)},
            "prompts": {"listChanged": False},
        }
```

- [x] **Step 5: Add namespaced prompt permission helper**

Add this helper near `_has_prompt_permission()` in `tldw_Server_API/app/core/MCP_unified/protocol.py`:

```python
    async def _has_namespaced_prompt_permission(self, context: RequestContext, prompt_name: str) -> bool:
        """Check prompt catalog namespace access without falling back to module permission."""
        if not await self._rbac_check(context.user_id, Resource.PROMPT, Action.READ, prompt_name):
            return False
        if not self._scope_allows(context, Resource.PROMPT.value, prompt_name):
            return False
        return self._api_key_allows(context, is_write=None)
```

- [x] **Step 6: Replace prompts/list handler with context-aware behavior**

Replace `_handle_prompts_list()` with:

```python
    async def _handle_prompts_list(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """List available prompts."""
        prompts: list[Any] = []
        warnings: list[dict[str, Any]] = []
        next_cursor: str | None = None
        modules = await self.module_registry.get_all_modules()

        for module_id, module in modules.items():
            try:
                if module_id != "prompts" and not await self._has_module_permission(context, module_id):
                    continue

                module_result = await module.get_prompts_for_context(context, params or {})
                module_prompts = module_result.get("prompts", []) if isinstance(module_result, dict) else []

                for prompt in module_prompts:
                    name = prompt.get("name") if isinstance(prompt, dict) else None
                    if name and name.startswith((LIBRARY_PROMPT_PREFIX, CONFIG_PROMPT_PREFIX)):
                        if not await self._has_namespaced_prompt_permission(context, name):
                            continue
                    elif name and not await self._has_prompt_permission(context, name, module_id):
                        continue
                    prompt_copy = prompt.copy() if isinstance(prompt, dict) else prompt
                    if isinstance(prompt_copy, dict):
                        prompt_copy["module"] = module_id
                    prompts.append(prompt_copy)

                if isinstance(module_result, dict):
                    if module_result.get("nextCursor"):
                        next_cursor = str(module_result["nextCursor"])
                    module_meta = module_result.get("_meta")
                    if isinstance(module_meta, dict):
                        tldw_meta = module_meta.get("tldw")
                        if isinstance(tldw_meta, dict) and isinstance(tldw_meta.get("warnings"), list):
                            warnings.extend(
                                warning
                                for warning in tldw_meta["warnings"]
                                if isinstance(warning, dict)
                            )

            except PromptCatalogError as exc:
                if exc.internal:
                    context.logger.exception("Error getting prompts from module {}", module_id)
                else:
                    context.logger.debug("Prompt catalog list rejected request for module {}: {}", module_id, exc.code)
                raise InvalidParamsException(str(exc)) from exc
            except _MCP_PROTOCOL_NONCRITICAL_EXCEPTIONS as e:
                context.logger.exception(f"Error getting prompts from module {module_id}: {e}")

        result: dict[str, Any] = {"prompts": prompts}
        if next_cursor:
            result["nextCursor"] = next_cursor
        if warnings:
            result["_meta"] = {"tldw": {"warnings": warnings}}
        return result
```

- [x] **Step 7: Replace prompts/get handler with namespace dispatch and argument validation**

Replace `_handle_prompts_get()` with:

```python
    async def _handle_prompts_get(
        self,
        params: dict[str, Any],
        context: RequestContext
    ) -> dict[str, Any]:
        """Get a specific prompt."""
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise InvalidParamsException("Prompt name is required")

        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise InvalidParamsException("Prompt arguments must be an object")

        if name.startswith((LIBRARY_PROMPT_PREFIX, CONFIG_PROMPT_PREFIX)):
            module = await self.module_registry.get_module("prompts")
            if not module:
                raise InvalidParamsException(f"Prompt not found: {name}")
            if not await self._has_namespaced_prompt_permission(context, name):
                raise PermissionError(f"Permission denied for prompt: {name}")
            try:
                return await module.get_prompt_for_context(name, arguments, context)
            except PromptCatalogError as exc:
                if exc.code == "permission_denied":
                    raise PermissionError(f"Permission denied for prompt: {name}") from exc
                if exc.internal:
                    context.logger.exception("MCP prompt catalog get failed for {}", name.split(":", 1)[0])
                    raise RuntimeError("Failed to get prompt") from exc
                raise InvalidParamsException(str(exc)) from exc

        module = await self.module_registry.find_module_for_prompt(name)
        if not module:
            raise InvalidParamsException(f"Prompt not found: {name}")
        module_id = self.module_registry.get_module_id_for_prompt(name) or getattr(module, "name", None)

        if not await self._has_prompt_permission(context, name, module_id):
            raise PermissionError(f"Permission denied for prompt: {name}")

        return await module.get_prompt_for_context(name, arguments, context)
```

- [x] **Step 8: Run protocol tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py -v
```

Expected: PASS.

- [x] **Step 9: Run existing MCP regression tests around basic protocol behavior**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_registry_iteration_race.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py \
  -v
```

Expected: PASS.

- [x] **Step 10: Commit protocol changes**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/protocol.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py
git commit -m "feat: route MCP prompt catalog protocol calls"
```

## Task 5: Add HTTP Cursor Mapping And Default Module Configuration

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`
- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`
- Create: `tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py`

- [x] **Step 1: Add failing config YAML test**

Append this test to `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`:

```python
from pathlib import Path

import yaml


def test_default_mcp_modules_config_declares_prompts_module_with_empty_config_allowlist() -> None:
    config_path = Path("tldw_Server_API/Config_Files/mcp_modules.yaml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modules = {module["id"]: module for module in data["modules"]}

    prompts_module = modules["prompts"]

    assert prompts_module["enabled"] is True
    assert prompts_module["settings"]["prompt_list_page_size"] == 50
    assert prompts_module["settings"]["max_rendered_prompt_chars"] == 100000
    assert prompts_module["settings"]["config_prompts"] == {"enabled": True, "entries": []}
```

- [x] **Step 2: Run config test to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py::test_default_mcp_modules_config_declares_prompts_module_with_empty_config_allowlist -v
```

Expected: FAIL because the shipped YAML does not have a `prompts` module entry.

- [x] **Step 3: Add prompts module entry to mcp_modules.yaml**

Insert this module entry after the `notes` module entry in `tldw_Server_API/Config_Files/mcp_modules.yaml`:

```yaml
  - id: prompts
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_module:PromptsModule
    enabled: true
    name: Prompts
    version: "1.0.0"
    department: knowledge
    max_concurrent: 10
    settings:
      prompt_list_page_size: 50
      max_rendered_prompt_chars: 100000
      config_prompts:
        enabled: true
        entries: []
        # Example entries; replace entries: [] with explicit entries like these
        # to publish config prompts through MCP.
        # entries:
        #   - id: rag.retrieval_guidance
        #     module: rag
        #     key: retrieval_guidance
        #     title: Retrieval Guidance
        #   - id: chat.summary
        #     title: Conversation Summary
        #     messages:
        #       - role: system
        #         module: chat
        #         key: summary_system
        #       - role: user
        #         module: chat
        #         key: summary_user
        #   - id: mcp.search_knowledge
        #     module: mcp
        #     key: search_knowledge
        #     title: Search Knowledge
```

- [x] **Step 4: Run config test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py::test_default_mcp_modules_config_declares_prompts_module_with_empty_config_allowlist -v
```

Expected: PASS.

- [x] **Step 5: Add HTTP cursor query test**

Create `tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py` with this focused route test.

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint


class _CaptureServer:
    initialized = True

    def __init__(self) -> None:
        self.request = None

    async def handle_http_request(self, request, user_id=None, metadata=None):
        self.request = request
        return type("Response", (), {"error": None, "result": {"prompts": [], "nextCursor": "next"}})()


def test_get_mcp_prompts_maps_cursor_to_protocol_request(monkeypatch) -> None:
    server = _CaptureServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)
    app = FastAPI()
    app.include_router(mcp_unified_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[mcp_unified_endpoint.enforce_http_security] = lambda: None
    app.dependency_overrides[mcp_unified_endpoint.get_mcp_auth_context] = lambda: mcp_unified_endpoint.McpAuthContext(
        user=None,
        principal=None,
        api_key_info=None,
        raw_api_key=None,
    )
    client = TestClient(app)

    response = client.get("/api/v1/mcp/prompts?cursor=abc")

    assert response.status_code == 200
    assert response.json() == {"prompts": [], "nextCursor": "next"}
    assert server.request.method == "prompts/list"
    assert server.request.params == {"cursor": "abc"}
```

- [x] **Step 6: Run HTTP cursor test to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py::test_get_mcp_prompts_maps_cursor_to_protocol_request -v
```

Expected: FAIL because the endpoint does not accept or pass `cursor`.

- [x] **Step 7: Add cursor query parameter to endpoint**

Modify `list_prompts()` in `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py`:

```python
async def list_prompts(
    http_request: Request,
    cursor: str | None = Query(None, description="Opaque MCP prompt list cursor"),
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    _guard: None = Depends(enforce_http_security),
):
```

Replace the current request creation with:

```python
    params = {"cursor": cursor} if cursor else None
    request = MCPRequest(method="prompts/list", params=params, id="http-prompts-list")
```

- [x] **Step 8: Run HTTP cursor test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py::test_get_mcp_prompts_maps_cursor_to_protocol_request -v
```

Expected: PASS.

- [x] **Step 9: Commit config and HTTP changes**

Run:

```bash
git add tldw_Server_API/Config_Files/mcp_modules.yaml tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py
git commit -m "feat: enable MCP prompts module configuration"
```

## Task 6: Add Integration And Security Regression Coverage

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py`

- [x] **Step 1: Add keyset pagination regression test**

Append to `test_prompts_catalog.py`:

```python
def test_user_source_keyset_cursor_survives_prompt_insert_between_pages(tmp_path) -> None:
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

    assert first_page.prompts[0]["name"] == f"{LIBRARY_PROMPT_PREFIX}{alpha['uuid']}"
    assert [prompt["name"] for prompt in second_page.prompts] == [
        f"{LIBRARY_PROMPT_PREFIX}{charlie['uuid']}"
    ]
```

- [x] **Step 2: Add sanitized error regression tests**

Append to `test_prompts_catalog.py`:

```python
def test_formatter_size_error_does_not_include_argument_value() -> None:
    row = {
        "id": 42,
        "uuid": "11111111-1111-4111-8111-111111111111",
        "name": "Large",
        "details": "",
        "version": 1,
        "keywords": [],
        "system_prompt": "",
        "user_prompt": "Echo {secret}",
        "prompt_definition": None,
    }
    formatter = MCPPromptFormatter(max_rendered_chars=5)

    with pytest.raises(PromptCatalogError) as excinfo:
        formatter.render_library_prompt(row, {"secret": "SENSITIVE_VALUE"})

    assert excinfo.value.code == "rendered_prompt_too_large"
    assert "SENSITIVE_VALUE" not in str(excinfo.value)


def test_missing_config_entry_error_does_not_include_override_path(monkeypatch, tmp_path) -> None:
    override_path = tmp_path / "missing.txt"
    monkeypatch.setenv("TLDW_PROMPT_FILE_MISSING__ENTRY", str(override_path))
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

    assert str(override_path) not in str(excinfo.value)
```

- [x] **Step 3: Add protocol error mapping regression tests**

Append to `test_protocol_prompts_catalog.py`:

```python
class ErrorPromptModule(ContextPromptModule):
    def __init__(self, config: ModuleConfig, error: PromptCatalogError) -> None:
        super().__init__(config)
        self.error = error

    async def get_prompt_for_context(self, name, arguments, context):
        raise self.error


@pytest.mark.asyncio
async def test_protocol_maps_catalog_invalid_params_without_body_leak(monkeypatch) -> None:
    module = ErrorPromptModule(ModuleConfig(name="prompts"), PromptCatalogError("missing_required_variable", "Missing required variable: topic"))
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))
    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", lambda *args, **kwargs: _async_true())

    with pytest.raises(Exception) as excinfo:
        await handler._handle_prompts_get(
            {"name": f"{LIBRARY_PROMPT_PREFIX}11111111-1111-4111-8111-111111111111", "arguments": {}},
            RequestContext("req-1", user_id="1"),
        )

    assert "Missing required variable: topic" in str(excinfo.value)


@pytest.mark.asyncio
async def test_protocol_maps_internal_catalog_error_to_internal_message(monkeypatch) -> None:
    module = ErrorPromptModule(ModuleConfig(name="prompts"), PromptCatalogError("prompt_db_unavailable", "Prompt body SENSITIVE", internal=True))
    handler = _handler_with_registry(PromptOnlyRegistry({"prompts": module}))
    monkeypatch.setattr(handler, "_has_namespaced_prompt_permission", lambda *args, **kwargs: _async_true())

    with pytest.raises(Exception) as excinfo:
        await handler._handle_prompts_get(
            {"name": f"{LIBRARY_PROMPT_PREFIX}11111111-1111-4111-8111-111111111111", "arguments": {}},
            RequestContext("req-1", user_id="1"),
        )

    assert "Failed to get prompt" in str(excinfo.value)
    assert "SENSITIVE" not in str(excinfo.value)
```

- [x] **Step 4: Run focused security and regression tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py \
  -v
```

Expected: PASS.

- [x] **Step 5: Commit regression coverage**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py
git commit -m "test: cover MCP prompt catalog edge cases"
```

## Task 7: Add MCP Prompt Documentation

**Files:**
- Create: `Docs/MCP/mcp_prompts.md`
- Modify: `Docs/MCP/mcp_tool_catalogs.md`
- Modify: `backlog/tasks/task-2343 - Plan-MCP-prompt-catalog-implementation.md`

- [x] **Step 1: Create prompt catalog documentation**

Create `Docs/MCP/mcp_prompts.md`:

```markdown
# MCP Prompts

tldw_server exposes selected prompts through MCP protocol-level prompts:

- `prompts/list`
- `prompts/get`

This is separate from the MCP tools `prompts.search` and `prompts.get`. Tools return Prompt Library records for retrieval workflows. Protocol prompts return MCP prompt definitions and rendered MCP prompt messages for MCP clients that understand the prompt capability.

## Capability

The MCP server advertises:

```json
{
  "prompts": {
    "listChanged": false
  }
}
```

`listChanged: false` means clients should call `prompts/list` again when they need a fresh view. The server does not emit `notifications/prompts/list_changed` in this version.

## Prompt Sources

The server lists:

- non-deleted records from the authenticated user's regular Prompt Library
- config prompts explicitly allowlisted in `tldw_Server_API/Config_Files/mcp_modules.yaml`

Prompt Studio prompts are not exposed through MCP prompts in this version.

## Stable Names

Prompt names are stable protocol identifiers:

- `library:<uuid>` for Prompt Library prompts
- `config:<module>.<key-or-group>` for allowlisted config prompts

The human-readable prompt name is returned as `title`. Renaming a Prompt Library prompt does not change its `library:<uuid>` name.

## Config Prompt Allowlist

Config prompt exposure is opt-in. The shipped default allowlist is empty:

```yaml
settings:
  config_prompts:
    enabled: true
    entries: []
```

Publish one config prompt:

```yaml
settings:
  config_prompts:
    enabled: true
    entries:
      - id: rag.retrieval_guidance
        module: rag
        key: retrieval_guidance
        title: Retrieval Guidance
```

Publish a grouped prompt:

```yaml
settings:
  config_prompts:
    enabled: true
    entries:
      - id: chat.summary
        title: Conversation Summary
        messages:
          - role: system
            module: chat
            key: summary_system
          - role: user
            module: chat
            key: summary_user
```

Config roles may be `system`, `developer`, `user`, or `assistant`. MCP only allows `user` and `assistant` prompt message roles, so `system` and `developer` content is folded into labeled `user` text when a prompt is rendered.

## Arguments

Structured Prompt Library prompts use their declared variables. Legacy and config prompts infer variables from placeholders such as:

- `{{topic}}`
- `{context}`
- `$query`
- `<input>`

Argument values must be strings. Missing required arguments and non-string values return JSON-RPC invalid params errors.

## Pagination

`prompts/list` accepts MCP's `cursor` parameter and may return `nextCursor`.

The HTTP convenience route remains list-only:

```http
GET /api/v1/mcp/prompts
GET /api/v1/mcp/prompts?cursor=<nextCursor>
```

Use `/api/v1/mcp/request` for `prompts/get`.

## Permissions

Authenticated MCP callers need `prompts.read`. Protocol-level prompt access does not require `modules.read`.

Prompt Library results are read from the authenticated user's per-user Prompt Library database. Persona prompt scopes still filter Prompt Library list and get operations. Allowlisted config prompts are visible to authenticated callers with `prompts.read`.

## Safety Notes

The server does not include prompt bodies in `prompts/list` descriptions. Prompt bodies, rendered argument values, and prompt override file paths are not included in errors.
```

- [x] **Step 2: Link docs from MCP tool catalog docs**

Open `Docs/MCP/mcp_tool_catalogs.md` and add this line near the top-level related-links or overview section:

```markdown
For protocol-level prompt discovery and rendering, see [MCP Prompts](mcp_prompts.md).
```

- [x] **Step 3: Update Backlog task with plan link**

Use Backlog MCP:

```text
task_edit(project="/Users/macbook-dev/Documents/GitHub/tldw_server2", task_id="TASK-2343", notes_append="Implementation plan created: Docs/superpowers/plans/2026-06-22-mcp-prompt-catalog-implementation-plan.md")
```

- [x] **Step 4: Commit docs**

Run:

```bash
git add Docs/MCP/mcp_prompts.md Docs/MCP/mcp_tool_catalogs.md "backlog/tasks/task-2343 - Plan-MCP-prompt-catalog-implementation.md"
git commit -m "docs: document MCP prompt catalog support"
```

## Task 8: Final Verification

**Files:**
- Verify all touched files from Tasks 1-7.

- [x] **Step 1: Run focused MCP prompt catalog tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_prompts_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_prompts_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py::test_default_mcp_modules_config_declares_prompts_module_with_empty_config_allowlist \
  -v
```

Expected: PASS.

- [x] **Step 2: Run existing MCP Unified smoke and regression tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_registry_iteration_race.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_protocol_catalog_filter.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  -v
```

Expected: PASS.

- [x] **Step 3: Run HTTP MCP prompt endpoint test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MCP_unified/test_mcp_prompts_http.py -v
```

Expected: PASS.

- [x] **Step 4: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_module.py \
  tldw_Server_API/app/core/MCP_unified/modules/base.py \
  tldw_Server_API/app/core/MCP_unified/protocol.py \
  tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py \
  -f json -o /tmp/bandit_mcp_prompt_catalog.json
```

Expected: PASS with no new high or medium findings in touched code.

- [x] **Step 5: Inspect final diff**

Run:

```bash
BASE_SHA=$(git merge-base HEAD origin/dev)
git diff --stat "${BASE_SHA}..HEAD"
git diff "${BASE_SHA}..HEAD" -- tldw_Server_API/app/core/MCP_unified/modules/implementations/prompts_catalog.py
git diff "${BASE_SHA}..HEAD" -- tldw_Server_API/app/core/MCP_unified/protocol.py
```

Expected:

- Catalog logic is isolated in `prompts_catalog.py`.
- `protocol.py` only contains protocol routing and MCP error mapping.
- Existing `prompts.search` and `prompts.get` tools remain unchanged.
- No prompt bodies or argument values appear in log messages or exception messages.

- [x] **Step 6: Update Backlog verification notes**

Use Backlog MCP:

```text
task_edit(project="/Users/macbook-dev/Documents/GitHub/tldw_server2", task_id="TASK-2343", status="Done", notes_append="Verified MCP prompt catalog implementation with focused pytest, MCP regression tests, HTTP cursor test, and touched-scope Bandit. See commit history for task-level commits.")
```

- [x] **Step 7: Final commit if verification notes changed**

Run:

```bash
git add "backlog/tasks/task-2343 - Plan-MCP-prompt-catalog-implementation.md"
git commit -m "chore: finalize MCP prompt catalog task notes"
```

## Review Checklist

- [x] `initialize` returns `{"prompts": {"listChanged": false}}`.
- [x] `prompts/list` supports `cursor` and returns `nextCursor` only when another page exists.
- [x] `prompts/get` resolves `library:` and `config:` before the global prompt registry.
- [x] `prompts.read` is enough for prompt catalog access; `modules.read` is not required for `PromptsModule` protocol prompts.
- [x] Prompt Library prompts are per-user through `RequestContext.db_paths["prompts"]`.
- [x] Prompt Library list excludes deleted rows.
- [x] Persona prompt scopes filter Prompt Library list and get.
- [x] Config prompts require explicit allowlist entries.
- [x] The shipped config allowlist is empty.
- [x] Prompt Studio prompts are absent from the catalog.
- [x] MCP prompt messages use only `user` and `assistant` roles.
- [x] Non-string prompt arguments return invalid params.
- [x] Errors do not include prompt bodies, rendered values, override paths, or DB paths.
- [x] Existing `prompts.search` and `prompts.get` tools still pass existing tests.

## Out Of Scope For This Plan

- Prompt Studio prompt exposure.
- Live `notifications/prompts/list_changed`.
- Shared prompt registry service beyond MCP.
- Frontend prompt catalog UI.
- New HTTP convenience route for `prompts/get`.
