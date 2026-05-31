from __future__ import annotations

from tldw_Server_API.app.api.v1.endpoints import acp_permissions
from tldw_Server_API.app.services import admin_acp_sessions_service


class _ExplodingStore:
    def get_db(self):
        raise RuntimeError("ACP store unavailable at /private/acp.db")


def test_get_acp_db_falls_back_when_shared_store_get_db_fails(monkeypatch):
    fallback_db = object()

    class _FallbackACPSessionsDB:
        def __new__(cls):
            return fallback_db

    monkeypatch.setattr(admin_acp_sessions_service, "_store", _ExplodingStore())
    monkeypatch.setattr(acp_permissions, "ACPSessionsDB", _FallbackACPSessionsDB)

    assert acp_permissions._get_acp_db() is fallback_db
