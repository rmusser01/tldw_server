"""RAG sensitivity filtering must use trusted authority without bypassing ACLs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.DB_Management.scope_context import ScopeContext
from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as up
from tldw_Server_API.app.core.RAG.rag_service.security_filters import SecurityFilters
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_user", [None, 99])
@pytest.mark.parametrize(
    ("scope_user", "is_admin", "request_user", "max_sensitivity", "deny_acl", "restricted", "expected"),
    [
        (1, True, "1", "internal", False, False, ["proposal"]),
        (1, True, "1", "public", False, False, []),
        (1, False, "1", "internal", False, False, []),
        (None, True, "1", "internal", False, False, []),
        ("absent", False, "1", "internal", False, False, []),
        (2, True, "1", "internal", False, False, []),
        (1, "true", "1", "internal", False, False, []),
        (1, True, "1", "internal", True, False, []),
        (1, True, "anonymous", "internal", False, False, []),
        (1, True, "1", "internal", False, True, []),
        (1, True, "1", "restricted", False, True, ["proposal"]),
    ],
)
async def test_security_filter_honors_only_matching_trusted_admin_authority(
    monkeypatch, runtime_user, scope_user, is_admin, request_user, max_sensitivity, deny_acl, restricted, expected
):
    """An authenticated admin can read INTERNAL prose within the requested ceiling."""
    scope = ScopeContext(scope_user, [], [], None, None, is_admin=is_admin)
    if scope_user == "absent":
        scope = None
    monkeypatch.setattr(up, "get_scope", lambda: scope)
    security = SecurityFilters(enable_audit_logging=False)
    if deny_acl:
        security.access_controller.set_document_acl("proposal", ["different-user"], [])
    monkeypatch.setattr(up, "SecurityFilter", lambda: security)
    retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=[
                Document(
                    id="proposal",
                    content=(
                        "Restricted: classified evidence."
                        if restricted
                        else "The public proposal would create an education savings account."
                    ),
                    metadata={},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]
        )
    )
    monkeypatch.setattr(up, "MultiDatabaseRetriever", lambda *args, **kwargs: retriever)

    result = await up.unified_rag_pipeline(
        query="What does the proposal create?",
        user_id=request_user,
        metadata={"is_admin": True, "role": "admin"},
        credential_runtime=SimpleNamespace(_user_id=runtime_user) if runtime_user else None,
        enable_security_filter=True,
        sensitivity_level=max_sensitivity,
        detect_pii=False,
        redact_pii=False,
        enable_generation=False,
        enable_cache=False,
        enable_reranking=False,
    )

    assert [doc["id"] for doc in result.documents] == (expected if runtime_user is None else [])
    assert not result.errors
