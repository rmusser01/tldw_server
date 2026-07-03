from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
from tldw_Server_API.app.core.Billing import enforcement as billing_enforcement
from tldw_Server_API.app.core.RAG.rag_service.transport import (
    build_source_health_payload,
    build_unified_pipeline_kwargs,
    enforce_rag_query_limit_for_org_context,
    resolve_org_id_for_rag_context,
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


@pytest.mark.asyncio
async def test_resolve_org_id_for_rag_context_uses_request_metadata_and_state() -> None:
    metadata_context = SimpleNamespace(metadata={"org_id": "42"})
    state_context = SimpleNamespace(state=SimpleNamespace(org_ids=[7]))

    assert await resolve_org_id_for_rag_context(request_like=metadata_context) == 42  # nosec B101
    assert await resolve_org_id_for_rag_context(request_like=state_context) == 7  # nosec B101


@pytest.mark.asyncio
async def test_enforce_rag_query_limit_for_org_context_uses_rag_daily_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeEnforcer:
        async def check_limit(self, org_id, category, *, requested_units):  # noqa: ANN001
            captured.update(
                {
                    "org_id": org_id,
                    "category": category,
                    "requested_units": requested_units,
                }
            )
            return SimpleNamespace(should_block=False, message=None)

    monkeypatch.setattr(billing_enforcement, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(billing_enforcement, "get_billing_enforcer", lambda: FakeEnforcer())

    await enforce_rag_query_limit_for_org_context(
        request_like=SimpleNamespace(metadata={"org_id": 55}),
        units=3,
    )

    assert captured == {  # nosec B101
        "org_id": 55,
        "category": billing_enforcement.LimitCategory.RAG_QUERIES_DAY,
        "requested_units": 3,
    }


@pytest.mark.asyncio
async def test_enforce_rag_query_limit_for_org_context_raises_when_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingEnforcer:
        async def check_limit(self, org_id, category, *, requested_units):  # noqa: ANN001
            del org_id, category, requested_units
            return SimpleNamespace(should_block=True, message="RAG query daily limit exceeded")

    monkeypatch.setattr(billing_enforcement, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(billing_enforcement, "get_billing_enforcer", lambda: BlockingEnforcer())

    with pytest.raises(PermissionError, match="daily limit"):
        await enforce_rag_query_limit_for_org_context(
            request_like=SimpleNamespace(metadata={"org_id": 55}),
            units=1,
        )
