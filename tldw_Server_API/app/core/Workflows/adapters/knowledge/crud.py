"""Knowledge management adapters.

This module includes adapters for knowledge CRUD operations:
- notes: Manage notes (create, get, list, update, delete, search)
- prompts: Manage prompts (get, list, create, update, search)
- collections: Manage collections
- chunking: Chunk text using various strategies
- claims_extract: Extract claims from text
- voice_intent: Voice intent detection
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._common import resolve_context_user_id
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.knowledge._config import (
    ChunkingConfig,
    ClaimsExtractConfig,
    CollectionsConfig,
    NotesConfig,
    PromptsConfig,
    VoiceIntentConfig,
)

_KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
)


@contextlib.contextmanager
def _workflow_media_db(user_id: str) -> Iterator[Any]:
    """Yield a per-user Media DB handle for short-lived workflow knowledge reads."""
    with managed_media_database(
        client_id=f"workflow_engine:{user_id}",
        db_path=str(DatabasePaths.get_media_db_path(int(user_id))),
        initialize=False,
    ) as media_db:
        yield media_db


def _resolve_workflow_user_id(context: dict[str, Any]) -> str | None:
    try:
        return resolve_context_user_id(context)
    except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
        return None


def _create_workflow_claim_extractor(api_name: str) -> Any:
    from tldw_Server_API.app.core.Claims_Extraction.claims_engine import LLMBasedClaimExtractor
    from tldw_Server_API.app.core.Claims_Extraction.claims_service import _fva_claims_analyze_call

    provider = str(api_name or "").strip()

    def _analyze(
        api_endpoint: str | None,
        input_data: Any,
        prompt: str | None,
        api_key: str | None,
        system_message: str | None,
        temp: float | None = None,
        streaming: bool = False,
        recursive_summarization: bool = False,
        chunked_summarization: bool = False,
        chunk_options: Any = None,
        model_override: str | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return _fva_claims_analyze_call(
            provider or api_endpoint,
            input_data,
            prompt,
            api_key,
            system_message,
            temp=temp,
            streaming=streaming,
            recursive_summarization=recursive_summarization,
            chunked_summarization=chunked_summarization,
            chunk_options=chunk_options,
            model_override=model_override,
            response_format=response_format,
            **kwargs,
        )

    return LLMBasedClaimExtractor(_analyze)


@registry.register(
    "notes",
    category="knowledge",
    description="Manage notes",
    parallelizable=True,
    tags=["knowledge", "notes"],
    config_model=NotesConfig,
)
async def run_notes_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Manage notes within a workflow step.

    Config:
      - action: Literal["create", "get", "list", "update", "delete", "search"]
      - note_id: Optional[str] (for get/update/delete)
      - title: Optional[str] (templated, for create/update)
      - content: Optional[str] (templated, for create/update)
      - query: Optional[str] (templated, for search)
      - limit: int = 100
      - offset: int = 0
      - expected_version: Optional[int] (for update/delete)
    Output:
      - {"note": {...}, "notes": [...], "success": bool}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    user_id = _resolve_workflow_user_id(context)
    if not user_id:
        try:
            user_id = str(DatabasePaths.get_single_user_id())
        except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    action = str(config.get("action") or "").strip().lower()
    if not action:
        return {"error": "missing_action"}

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    # Test mode simulation
    if is_test_mode():
        if action == "create":
            return {"note": {"id": "test-note-id", "title": _render(config.get("title")), "content": _render(config.get("content"))}, "success": True, "simulated": True}
        if action == "get":
            return {"note": {"id": config.get("note_id"), "title": "Test Note", "content": "Test content"}, "simulated": True}
        if action == "list":
            return {"notes": [], "count": 0, "simulated": True}
        if action == "update":
            return {"note": {"id": config.get("note_id")}, "success": True, "simulated": True}
        if action == "delete":
            return {"success": True, "simulated": True}
        if action == "search":
            return {"notes": [], "count": 0, "simulated": True}
        return {"error": f"unknown_action:{action}", "simulated": True}

    try:
        from tldw_Server_API.app.core.Notes.Notes_Library import NotesInteropService

        # Resolve notes DB directory
        try:
            notes_base_dir = DatabasePaths.get_user_base_directory(int(user_id))
        except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
            notes_base_dir = Path("Databases") / "user_databases"

        service = NotesInteropService(base_db_directory=notes_base_dir, api_client_id="workflow_engine")

        if action == "create":
            title = _render(config.get("title") or "")
            content = _render(config.get("content") or "")
            if not title:
                return {"error": "missing_title"}
            note_id = service.add_note(user_id=user_id, title=title, content=content)
            note = service.get_note_by_id(user_id=user_id, note_id=note_id)
            return {"note": note, "success": True}

        if action == "get":
            note_id = str(config.get("note_id") or "").strip()
            if not note_id:
                return {"error": "missing_note_id"}
            note = service.get_note_by_id(user_id=user_id, note_id=note_id)
            if note is None:
                return {"error": "note_not_found", "note_id": note_id}
            return {"note": note}

        if action == "list":
            limit = int(config.get("limit") or 100)
            offset = int(config.get("offset") or 0)
            notes = service.list_notes(user_id=user_id, limit=limit, offset=offset)
            return {"notes": notes, "count": len(notes)}

        if action == "update":
            note_id = str(config.get("note_id") or "").strip()
            if not note_id:
                return {"error": "missing_note_id"}
            expected_version = config.get("expected_version")
            if expected_version is None:
                # Fetch current version
                current = service.get_note_by_id(user_id=user_id, note_id=note_id)
                if current is None:
                    return {"error": "note_not_found", "note_id": note_id}
                expected_version = current.get("version", 1)
            update_data: dict[str, Any] = {}
            title = config.get("title")
            if title is not None:
                update_data["title"] = _render(title)
            content = config.get("content")
            if content is not None:
                update_data["content"] = _render(content)
            if not update_data:
                return {"error": "no_update_fields"}
            service.update_note(user_id=user_id, note_id=note_id, update_data=update_data, expected_version=int(expected_version))
            updated = service.get_note_by_id(user_id=user_id, note_id=note_id)
            return {"note": updated, "success": True}

        if action == "delete":
            note_id = str(config.get("note_id") or "").strip()
            if not note_id:
                return {"error": "missing_note_id"}
            expected_version = config.get("expected_version")
            if expected_version is None:
                current = service.get_note_by_id(user_id=user_id, note_id=note_id)
                if current is None:
                    return {"error": "note_not_found", "note_id": note_id}
                expected_version = current.get("version", 1)
            success = service.soft_delete_note(user_id=user_id, note_id=note_id, expected_version=int(expected_version))
            return {"success": success}

        if action == "search":
            query = _render(config.get("query") or "")
            if not query:
                return {"error": "missing_query"}
            limit = int(config.get("limit") or 100)
            offset = int(config.get("offset") or 0)
            notes = service.search_notes(user_id=user_id, query=query, limit=limit, offset=offset)
            return {"notes": notes, "count": len(notes)}

        return {"error": f"unknown_action:{action}"}

    except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
        logger.exception("Notes adapter error")
        return {"error": "notes_error"}


@registry.register(
    "prompts",
    category="knowledge",
    description="Manage prompts",
    parallelizable=True,
    tags=["knowledge", "prompts"],
    config_model=PromptsConfig,
)
async def run_prompts_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Manage prompts within a workflow step.

    Config:
      - action: Literal["get", "list", "create", "update", "search"]
      - prompt_id: Optional[int]
      - name: Optional[str] (templated)
      - content: Optional[str] (templated)
      - query: Optional[str] (templated, for search)
    Output:
      - {"prompt": {...}, "prompts": [...], "success": bool}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    action = str(config.get("action") or "").strip().lower()
    if not action:
        return {"error": "missing_action"}

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    # Test mode simulation
    if is_test_mode():
        if action == "create":
            return {"prompt": {"id": 1, "name": _render(config.get("name"))}, "success": True, "simulated": True}
        if action == "get":
            return {"prompt": {"id": config.get("prompt_id"), "name": "Test Prompt", "prompt": "Test content"}, "simulated": True}
        if action == "list":
            return {"prompts": [], "total": 0, "simulated": True}
        if action == "update":
            return {"prompt": {"id": config.get("prompt_id")}, "success": True, "simulated": True}
        if action == "search":
            return {"prompts": [], "total": 0, "simulated": True}
        return {"error": f"unknown_action:{action}", "simulated": True}

    try:
        from tldw_Server_API.app.core.Prompt_Management import Prompts_Interop as prompts_interop

        # Ensure interop is initialized
        if not prompts_interop.is_initialized():
            try:
                prompts_db_path = DatabasePaths.get_prompts_db_path()
            except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
                prompts_db_path = Path("Databases") / "prompts.db"
            prompts_interop.initialize_interop(db_path=str(prompts_db_path), client_id="workflow_engine")

        if action == "get":
            prompt_id = config.get("prompt_id")
            prompt_name = config.get("name")
            prompt_uuid = config.get("uuid")
            if prompt_id is not None:
                prompt = prompts_interop.get_prompt_by_id(int(prompt_id))
            elif prompt_uuid:
                prompt = prompts_interop.get_prompt_by_uuid(str(prompt_uuid))
            elif prompt_name:
                prompt = prompts_interop.get_prompt_by_name(_render(prompt_name))
            else:
                return {"error": "missing_prompt_identifier"}
            if prompt is None:
                return {"error": "prompt_not_found"}
            return {"prompt": prompt}

        if action == "list":
            page = int(config.get("page") or 1)
            per_page = int(config.get("limit") or config.get("per_page") or 50)
            prompts_list, total_prompts, total_pages, current_page = prompts_interop.list_prompts(
                page=page, per_page=per_page
            )
            return {"prompts": prompts_list, "total": total_prompts, "total_pages": total_pages, "page": current_page}

        if action == "create":
            name = _render(config.get("name") or "")
            if not name:
                return {"error": "missing_name"}
            author = _render(config.get("author") or "")
            details = _render(config.get("details") or "")
            system_prompt = _render(config.get("system_prompt") or config.get("system") or "")
            user_prompt = _render(config.get("user_prompt") or config.get("prompt") or "")
            keywords = config.get("keywords") or config.get("tags") or []
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            prompt_id, prompt_uuid, msg = prompts_interop.add_prompt(
                name=name,
                author=author or None,
                details=details or None,
                system_prompt=system_prompt or None,
                user_prompt=user_prompt or None,
                keywords=keywords if keywords else None,
                overwrite=bool(config.get("overwrite", False)),
            )
            return {"prompt": {"id": prompt_id, "uuid": prompt_uuid, "name": name}, "message": msg, "success": prompt_id is not None}

        if action == "update":
            prompt_id = config.get("prompt_id")
            prompt_name = config.get("name")
            if prompt_id is None and not prompt_name:
                return {"error": "missing_prompt_identifier"}
            # Fetch existing to update
            if prompt_id is not None:
                existing = prompts_interop.get_prompt_by_id(int(prompt_id))
            else:
                existing = prompts_interop.get_prompt_by_name(_render(prompt_name))
            if existing is None:
                return {"error": "prompt_not_found"}
            name = _render(config.get("new_name") or existing.get("name") or "")
            author = _render(config.get("author")) if config.get("author") is not None else existing.get("author")
            details = _render(config.get("details")) if config.get("details") is not None else existing.get("details")
            system_prompt = _render(config.get("system_prompt")) if config.get("system_prompt") is not None else existing.get("system_prompt")
            user_prompt = _render(config.get("user_prompt") or config.get("prompt")) if (config.get("user_prompt") or config.get("prompt")) is not None else existing.get("user_prompt")
            keywords = config.get("keywords") or config.get("tags")
            if keywords is None:
                keywords = existing.get("keywords") or []
            elif isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",") if k.strip()]
            pid, puuid, msg = prompts_interop.add_prompt(
                name=name,
                author=author,
                details=details,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                keywords=keywords if keywords else None,
                overwrite=True,
            )
            return {"prompt": {"id": pid, "uuid": puuid, "name": name}, "message": msg, "success": pid is not None}

        if action == "search":
            query = _render(config.get("query") or "")
            if not query:
                return {"error": "missing_query"}
            page = int(config.get("page") or 1)
            per_page = int(config.get("limit") or 50)
            search_fields = config.get("search_fields")
            results, total = prompts_interop.search_prompts(
                search_query=query,
                search_fields=search_fields,
                page=page,
                results_per_page=per_page,
            )
            return {"prompts": results, "total": total}

        return {"error": f"unknown_action:{action}"}

    except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
        logger.exception("Prompts adapter error")
        return {"error": "prompts_error"}


@registry.register(
    "collections",
    category="knowledge",
    description="Manage collections",
    parallelizable=True,
    tags=["knowledge", "collections"],
    config_model=CollectionsConfig,
)
async def run_collections_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Manage reading list collections within a workflow step.

    Config:
      - action: Literal["save", "update", "list", "get", "delete", "search"]
      - url: Optional[str] (for save)
      - item_id: Optional[int] (for get/update/delete)
      - status: Optional[Literal["saved", "reading", "read", "archived"]]
      - tags: Optional[List[str]]
      - query: Optional[str] (for search, templated)
      - favorite: Optional[bool]
      - limit: int = 50
      - page: int = 1
    Output:
      - {"item": {...}, "items": [...], "count": int}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    user_id = _resolve_workflow_user_id(context)
    if not user_id:
        try:
            user_id = str(DatabasePaths.get_single_user_id())
        except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    action = str(config.get("action") or "").strip().lower()
    if not action:
        return {"error": "missing_action"}

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    # Test mode simulation
    if is_test_mode():
        if action == "save":
            return {
                "item": {"id": 1, "url": _render(config.get("url")), "title": "Test Item", "status": "saved"},
                "created": True,
                "simulated": True,
            }
        if action == "get":
            return {
                "item": {"id": config.get("item_id"), "url": "https://example.com", "title": "Test Item"},
                "simulated": True,
            }
        if action == "list":
            return {"items": [], "count": 0, "total": 0, "simulated": True}
        if action == "update":
            return {"item": {"id": config.get("item_id")}, "success": True, "simulated": True}
        if action == "delete":
            return {"success": True, "simulated": True}
        if action == "search":
            return {"items": [], "count": 0, "simulated": True}
        return {"error": f"unknown_action:{action}", "simulated": True}

    try:
        from tldw_Server_API.app.core.Collections.reading_service import ReadingService

        service = ReadingService(user_id=int(user_id))

        if action == "save":
            url = _render(config.get("url") or "")
            if not url:
                return {"error": "missing_url"}
            tags = config.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            status = config.get("status") or "saved"
            favorite = bool(config.get("favorite", False))
            title_override = _render(config.get("title")) if config.get("title") else None
            notes = _render(config.get("notes")) if config.get("notes") else None

            result = await service.save_url(
                url=url,
                tags=tags,
                status=status,
                favorite=favorite,
                title_override=title_override,
                notes=notes,
            )
            return {
                "item": {
                    "id": result.item.id,
                    "url": result.item.url,
                    "title": result.item.title,
                    "status": result.item.status,
                    "canonical_url": result.item.canonical_url,
                },
                "created": result.created,
                "media_id": result.media_id,
            }

        if action == "get":
            item_id = config.get("item_id")
            if item_id is None:
                return {"error": "missing_item_id"}
            try:
                item = service.get_item(int(item_id))
                return {
                    "item": {
                        "id": item.id,
                        "url": item.url,
                        "title": item.title,
                        "status": item.status,
                        "favorite": item.favorite,
                        "summary": item.summary,
                        "notes": item.notes,
                    }
                }
            except KeyError:
                return {"error": "item_not_found", "item_id": item_id}

        if action == "list":
            page = int(config.get("page") or 1)
            limit = int(config.get("limit") or config.get("size") or 50)
            status = config.get("status")
            status_list = [status] if status and isinstance(status, str) else status
            tags = config.get("tags")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            favorite = config.get("favorite")
            if favorite is not None:
                favorite = bool(favorite)

            items, total = service.list_items(
                status=status_list,
                tags=tags,
                favorite=favorite,
                page=page,
                size=limit,
            )
            return {
                "items": [
                    {"id": i.id, "url": i.url, "title": i.title, "status": i.status, "favorite": i.favorite}
                    for i in items
                ],
                "count": len(items),
                "total": total,
                "page": page,
            }

        if action == "update":
            item_id = config.get("item_id")
            if item_id is None:
                return {"error": "missing_item_id"}
            status = config.get("status")
            favorite = config.get("favorite")
            if favorite is not None:
                favorite = bool(favorite)
            tags = config.get("tags")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            notes = _render(config.get("notes")) if config.get("notes") else None
            title = _render(config.get("title")) if config.get("title") else None

            try:
                item = service.update_item(
                    int(item_id),
                    status=status,
                    favorite=favorite,
                    tags=tags,
                    notes=notes,
                    title=title,
                )
                return {
                    "item": {"id": item.id, "url": item.url, "title": item.title, "status": item.status},
                    "success": True,
                }
            except KeyError:
                return {"error": "item_not_found", "item_id": item_id}

        if action == "delete":
            item_id = config.get("item_id")
            if item_id is None:
                return {"error": "missing_item_id"}
            try:
                service.delete_item(int(item_id))
                return {"success": True}
            except KeyError:
                return {"error": "item_not_found", "item_id": item_id}

        if action == "search":
            query = _render(config.get("query") or "")
            if not query:
                return {"error": "missing_query"}
            page = int(config.get("page") or 1)
            limit = int(config.get("limit") or 50)

            items, total = service.list_items(q=query, page=page, size=limit)
            return {
                "items": [
                    {"id": i.id, "url": i.url, "title": i.title, "status": i.status}
                    for i in items
                ],
                "count": len(items),
                "total": total,
            }

        return {"error": f"unknown_action:{action}"}

    except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
        logger.error("Collections adapter error")
        return {"error": "collections_error"}


@registry.register(
    "chunking",
    category="knowledge",
    description="Chunk text content",
    parallelizable=True,
    tags=["knowledge", "chunking"],
    config_model=ChunkingConfig,
)
async def run_chunking_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Chunk text using various strategies.

    Config:
      - text: Optional[str] (templated, defaults to last.text or last.content)
      - method: Literal["words", "sentences", "tokens", "structure_aware", "fixed_size"] = "sentences"
      - max_size: int = 400
      - overlap: int = 50
      - language: Optional[str]
    Output:
      - {"chunks": [...], "count": int, "text": str}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    # Resolve text input
    text = config.get("text")
    if text is not None:
        text = _render(text)
    else:
        # Try to get from context (last step output)
        last = context.get("last") or {}
        text = last.get("text") or last.get("content") or last.get("summary") or "" if isinstance(last, dict) else ""
    text = str(text) if text else ""

    if not text.strip():
        return {"chunks": [], "count": 0, "text": ""}

    method = str(config.get("method") or "sentences").strip().lower()
    max_size = int(config.get("max_size") or config.get("max_tokens") or 400)
    overlap = int(config.get("overlap") or 50)
    language = config.get("language")

    # Validate method
    valid_methods = {"words", "sentences", "tokens", "structure_aware", "fixed_size"}
    if method not in valid_methods:
        return {"error": f"invalid_method:{method}", "valid_methods": list(valid_methods)}

    # Test mode simulation
    if is_test_mode():
        # Simple mock chunking
        words = text.split()
        chunk_size = max(1, max_size // 5)  # Rough word-based simulation
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return {"chunks": chunks, "count": len(chunks), "text": text, "method": method, "simulated": True}

    try:
        from tldw_Server_API.app.core.Chunking import Chunker

        chunker = Chunker()
        chunks_result = chunker.chunk_text(
            text=text,
            method=method,
            max_size=max_size,
            overlap=overlap,
            language=language,
        )

        return {
            "chunks": chunks_result,
            "count": len(chunks_result),
            "text": text,
            "method": method,
            "max_size": max_size,
            "overlap": overlap,
        }

    except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
        logger.exception("Chunking adapter error")
        return {"error": "chunking_error"}


@registry.register(
    "claims_extract",
    category="knowledge",
    description="Extract claims from text",
    parallelizable=True,
    tags=["knowledge", "extraction"],
    config_model=ClaimsExtractConfig,
)
async def run_claims_extract_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract and search claims from text within a workflow step.

    Config:
      - action: Literal["extract", "search", "list"]
      - text: Optional[str] (templated, defaults to last.text) - for extract
      - media_id: Optional[int] - associate claims with media item
      - query: Optional[str] (templated) - for search
      - limit: int = 50
      - offset: int = 0
      - api_name: Optional[str] - LLM provider for extraction
    Output:
      - {"claims": [{claim_text, source_span, confidence, metadata}], "count": int}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    user_id = _resolve_workflow_user_id(context)
    if not user_id:
        try:
            user_id = str(DatabasePaths.get_single_user_id())
        except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
            return {"error": "missing_user_id"}
    user_id = str(user_id)

    action = str(config.get("action") or "").strip().lower()
    if not action:
        return {"error": "missing_action"}

    def _render(value: Any) -> Any:
        if isinstance(value, str):
            return apply_template_to_string(value, context) or value
        return value

    # Test mode simulation
    if is_test_mode():
        if action == "extract":
            return {
                "claims": [
                    {"id": "claim-1", "text": "Test claim extracted from text", "span": [0, 30], "confidence": 0.9}
                ],
                "count": 1,
                "simulated": True,
            }
        if action == "search":
            return {
                "claims": [],
                "count": 0,
                "query": _render(config.get("query") or ""),
                "simulated": True,
            }
        if action == "list":
            return {"claims": [], "count": 0, "simulated": True}
        return {"error": f"unknown_action:{action}", "simulated": True}

    try:
        if action == "extract":
            # Resolve text input
            text = config.get("text")
            if text is not None:
                text = _render(text)
            else:
                last = context.get("last") or {}
                text = last.get("text") or last.get("content") or "" if isinstance(last, dict) else ""

            if not text:
                return {"error": "missing_text_for_extraction"}

            api_name = _render(config.get("api_name") or "openai")
            max_claims = int(config.get("max_claims") or 25)

            extractor = _create_workflow_claim_extractor(api_name)
            claims = await extractor.extract(text, max_claims=max_claims)

            # Format claims for output
            claims_list = []
            for claim in claims:
                claims_list.append({
                    "id": claim.id,
                    "text": claim.text,
                    "span": list(claim.span) if claim.span else None,
                })

            return {
                "claims": claims_list,
                "count": len(claims_list),
                "text": text[:500] if len(text) > 500 else text,
            }

        if action == "search":
            query = _render(config.get("query") or "")
            if not query:
                return {"error": "missing_query"}

            limit = int(config.get("limit") or 50)
            offset = int(config.get("offset") or 0)

            with _workflow_media_db(user_id) as media_db:
                results = media_db.search_claims(query=query, limit=limit, offset=offset)

            claims_list = []
            for r in results:
                claims_list.append({
                    "id": r.get("id"),
                    "text": r.get("claim_text") or r.get("text"),
                    "media_id": r.get("media_id"),
                    "relevance_score": r.get("relevance_score"),
                })

            return {
                "claims": claims_list,
                "count": len(claims_list),
                "query": query,
            }

        if action == "list":
            limit = int(config.get("limit") or 50)
            offset = int(config.get("offset") or 0)
            media_id = config.get("media_id")

            with _workflow_media_db(user_id) as media_db:
                if media_id is not None:
                    results = media_db.list_claims_for_media(media_id=int(media_id), limit=limit, offset=offset)
                else:
                    results = media_db.list_claims(limit=limit, offset=offset)

            claims_list = []
            for r in results:
                claims_list.append({
                    "id": r.get("id"),
                    "text": r.get("claim_text") or r.get("text"),
                    "media_id": r.get("media_id"),
                })

            return {
                "claims": claims_list,
                "count": len(claims_list),
            }

        return {"error": f"unknown_action:{action}"}

    except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
        logger.error("Claims extract adapter error")
        return {"error": "claims_extract_error"}


@registry.register(
    "voice_intent",
    category="knowledge",
    description="Voice intent detection",
    parallelizable=False,
    tags=["knowledge", "voice"],
    config_model=VoiceIntentConfig,
)
async def run_voice_intent_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Parse voice/text input into actionable intents.

    Config:
      - text: str (templated, typically from STT output - required)
      - llm_enabled: bool (default: True - enable LLM fallback for complex queries)
      - awaiting_confirmation: bool (default: False - if expecting yes/no response)
      - conversation_history: List[Dict] (optional - for context)
    Output:
      - {intent, action_type, action_config, entities, confidence, requires_confirmation, match_method, alternatives, processing_time_ms}
    """
    # Check cancellation
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    # Resolve user_id
    user_id = _resolve_workflow_user_id(context)
    if not user_id:
        # Voice intent can work without user_id, default to 0
        user_id_int = 0
    else:
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            user_id_int = 0

    # Get and template text
    text_t = config.get("text")
    if not text_t:
        # Try to get from last.text
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                text_t = last.get("text") or last.get("transcript") or last.get("content") or ""
        except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
            text_t = ""

    if not text_t:
        return {
            "error": "missing_text",
            "intent": "",
            "action_type": "custom",
            "action_config": {"action": "empty_input"},
            "entities": {},
            "confidence": 0.0,
            "requires_confirmation": False,
            "match_method": "empty",
            "alternatives": [],
            "processing_time_ms": 0.0,
        }

    text = apply_template_to_string(str(text_t), context) or str(text_t)
    text = text.strip()

    if not text:
        return {
            "error": "empty_text",
            "intent": "",
            "action_type": "custom",
            "action_config": {"action": "empty_input"},
            "entities": {},
            "confidence": 0.0,
            "requires_confirmation": False,
            "match_method": "empty",
            "alternatives": [],
            "processing_time_ms": 0.0,
        }

    # Get config options
    llm_enabled = config.get("llm_enabled")
    llm_enabled = True if llm_enabled is None else bool(llm_enabled)

    awaiting_confirmation = bool(config.get("awaiting_confirmation"))

    conversation_history = config.get("conversation_history")
    if not isinstance(conversation_history, list):
        conversation_history = None

    # Test mode simulation
    if is_test_mode():
        # Simulate basic intent parsing
        text_lower = text.lower()

        # Check for confirmation responses
        if awaiting_confirmation:
            if any(w in text_lower for w in ["yes", "yeah", "yep", "sure", "ok", "okay", "confirm"]):
                return {
                    "intent": "confirmation",
                    "action_type": "custom",
                    "action_config": {"action": "confirmation", "confirmed": True},
                    "entities": {},
                    "confidence": 1.0,
                    "requires_confirmation": False,
                    "match_method": "confirmation",
                    "alternatives": [],
                    "processing_time_ms": 1.0,
                    "simulated": True,
                }
            elif any(w in text_lower for w in ["no", "nope", "cancel", "stop", "abort"]):
                return {
                    "intent": "confirmation",
                    "action_type": "custom",
                    "action_config": {"action": "confirmation", "confirmed": False},
                    "entities": {},
                    "confidence": 1.0,
                    "requires_confirmation": False,
                    "match_method": "confirmation",
                    "alternatives": [],
                    "processing_time_ms": 1.0,
                    "simulated": True,
                }

        # Check for search-related patterns
        if any(w in text_lower for w in ["search", "find", "look for", "look up"]):
            # Extract query
            query = text
            for prefix in ["search for", "search", "find", "look for", "look up"]:
                if text_lower.startswith(prefix):
                    query = text[len(prefix):].strip()
                    break

            return {
                "intent": "search",
                "action_type": "mcp_tool",
                "action_config": {"tool_name": "media.search", "query": query},
                "entities": {"query": query},
                "confidence": 0.8,
                "requires_confirmation": False,
                "match_method": "pattern",
                "alternatives": [],
                "processing_time_ms": 5.0,
                "simulated": True,
            }

        # Check for note-related patterns
        if any(w in text_lower for w in ["note", "remember", "take a note"]):
            content = text
            for prefix in ["take a note", "note that", "note", "remember that", "remember"]:
                if text_lower.startswith(prefix):
                    content = text[len(prefix):].strip()
                    break

            return {
                "intent": "create_note",
                "action_type": "mcp_tool",
                "action_config": {"tool_name": "notes.create", "content": content},
                "entities": {"content": content},
                "confidence": 0.8,
                "requires_confirmation": False,
                "match_method": "pattern",
                "alternatives": [],
                "processing_time_ms": 5.0,
                "simulated": True,
            }

        # Default: treat as chat
        return {
            "intent": "chat",
            "action_type": "llm_chat",
            "action_config": {"message": text},
            "entities": {},
            "confidence": 0.5,
            "requires_confirmation": False,
            "match_method": "default",
            "alternatives": [],
            "processing_time_ms": 10.0,
            "simulated": True,
        }

    # Production mode: use the intent parser
    try:
        from tldw_Server_API.app.core.VoiceAssistant.intent_parser import get_intent_parser

        parser = get_intent_parser()

        # Save original LLM setting to restore after parsing (avoid mutating singleton state)
        original_llm_enabled = parser.llm_enabled

        try:
            # Override LLM setting if specified
            if not llm_enabled:
                parser.llm_enabled = False

            # Build context dict
            parse_context: dict[str, Any] = {}
            if awaiting_confirmation:
                parse_context["awaiting_confirmation"] = True
            if conversation_history:
                parse_context["conversation_history"] = conversation_history

            # Parse the intent
            result = await parser.parse(
                text=text,
                user_id=user_id_int,
                context=parse_context if parse_context else None,
            )
        finally:
            # Restore original LLM setting to avoid side effects on subsequent calls
            parser.llm_enabled = original_llm_enabled

        # Extract action_type value (convert enum to string)
        action_type_str = result.intent.action_type.value if hasattr(result.intent.action_type, 'value') else str(result.intent.action_type)

        # Build alternatives list
        alternatives_out = []
        for alt in result.alternatives:
            alt_action_type = alt.action_type.value if hasattr(alt.action_type, 'value') else str(alt.action_type)
            alternatives_out.append({
                "command_id": alt.command_id,
                "action_type": alt_action_type,
                "action_config": alt.action_config,
                "entities": alt.entities,
                "confidence": alt.confidence,
                "requires_confirmation": alt.requires_confirmation,
            })

        return {
            "intent": result.intent.command_id or action_type_str,
            "action_type": action_type_str,
            "action_config": result.intent.action_config,
            "entities": result.intent.entities,
            "confidence": result.intent.confidence,
            "requires_confirmation": result.intent.requires_confirmation,
            "match_method": result.match_method,
            "alternatives": alternatives_out,
            "processing_time_ms": result.processing_time_ms,
        }

    except _KNOWLEDGE_CRUD_NONCRITICAL_EXCEPTIONS:
        logger.error("Voice intent adapter error")
        return {
            "error": "voice_intent_error",
            "intent": "",
            "action_type": "custom",
            "action_config": {"action": "error"},
            "entities": {},
            "confidence": 0.0,
            "requires_confirmation": False,
            "match_method": "error",
            "alternatives": [],
            "processing_time_ms": 0.0,
        }
