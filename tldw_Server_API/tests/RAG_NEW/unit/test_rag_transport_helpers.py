from __future__ import annotations

from types import SimpleNamespace

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.transport import (
    build_source_health_payload,
    build_unified_pipeline_kwargs,
)


def test_build_unified_pipeline_kwargs_preserves_search_agent_defaults() -> None:
    request = UnifiedRAGRequest(query="default behavior check")

    kwargs = build_unified_pipeline_kwargs(
        request=request,
        db_paths={
            "media_db_path": None,
            "notes_db_path": None,
            "character_db_path": None,
            "kanban_db_path": None,
        },
        media_db=None,
        chacha_db=None,
        current_user=None,
        search_agent_setting_fn=lambda env_key, config_key: {  # noqa: ARG005
            "SEARCH_QUERY_CLASSIFICATION": "true",
        }.get(env_key),
    )

    assert kwargs["enable_query_classification"] is True  # nosec B101
    assert kwargs["sources"] == ["media_db"]  # nosec B101


def test_build_source_health_payload_uses_existing_paths_without_leaking_paths() -> None:
    payload = build_source_health_payload(
        current_user=SimpleNamespace(id=1, id_int=1),
        existing_source_db_paths_fn=lambda *_args, **_kwargs: {"media_db": "/secret/media.db"},
        media_db_uses_non_file_storage_fn=lambda: False,
    )

    assert [entry.source_id for entry in payload.sources][:2] == ["media_db", "notes"]  # nosec B101
    assert "/secret" not in str(payload)  # nosec B101
