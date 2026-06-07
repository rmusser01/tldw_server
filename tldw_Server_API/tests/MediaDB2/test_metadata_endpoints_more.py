import importlib.machinery
import json
import os
import sys
import types

import pytest
from httpx import ASGITransport, AsyncClient

# Stub heavy modules before importing the full app
torch_stub = types.ModuleType("torch")
torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
torch_stub.Tensor = object
torch_stub.nn = types.SimpleNamespace(Module=object)
sys.modules.setdefault('torch', torch_stub)

dill_stub = types.ModuleType("dill")
dill_stub.__spec__ = None
sys.modules.setdefault('dill', dill_stub)


def _auth_headers():
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    key = get_settings().SINGLE_USER_API_KEY or os.getenv("SINGLE_USER_API_KEY", "test-api-key-12345")
    return {"X-API-KEY": key}


class _FakeConn:
    def execute(self, *args, **kwargs):
        return None
    def commit(self):
        return None


class _FakeDB:
    def __init__(self):
        self.last_filters = None
        self.last_search_kwargs = None
    def transaction(self):
        class _Tx:
            def __enter__(self_inner):
                return _FakeConn()
            def __exit__(self_inner, exc_type, exc, tb):
                return False
        return _Tx()
    def get_connection(self):
        return _FakeConn()
    def search_by_safe_metadata(self, filters=None, match_all=True, page=1, per_page=20, group_by_media=True, **kwargs):
        self.last_filters = filters
        self.last_search_kwargs = {
            "match_all": match_all,
            "page": page,
            "per_page": per_page,
            "group_by_media": group_by_media,
            **kwargs,
        }
        return [], 0


@pytest.mark.asyncio
async def test_metadata_search_normalizes_doi(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.main import app

    fake_db = _FakeDB()
    app.dependency_overrides[get_media_db_for_user] = lambda: fake_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers=_auth_headers(),
    ) as client:
        r = await client.get(
            "/api/v1/media/metadata-search",
            params={"field": "doi", "op": "eq", "value": "10.1234/ABC-123"},
        )
        assert r.status_code == 200
        assert fake_db.last_filters and fake_db.last_filters[0]["value"] == "10.1234/ABC-123"

    app.dependency_overrides.pop(get_media_db_for_user, None)


@pytest.mark.asyncio
async def test_metadata_search_invalid_doi_returns_400(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.main import app

    fake_db = _FakeDB()
    app.dependency_overrides[get_media_db_for_user] = lambda: fake_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers=_auth_headers(),
    ) as client:
        r = await client.get(
            "/api/v1/media/metadata-search",
            params={"field": "doi", "op": "eq", "value": "bad"},
        )
        assert r.status_code == 400

    app.dependency_overrides.pop(get_media_db_for_user, None)


@pytest.mark.asyncio
async def test_metadata_search_forwards_standard_constraints(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.main import app

    fake_db = _FakeDB()
    app.dependency_overrides[get_media_db_for_user] = lambda: fake_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers=_auth_headers(),
    ) as client:
        r = await client.get(
            "/api/v1/media/metadata-search",
            params={
                "filters": json.dumps([{"field": "doi", "op": "eq", "value": "10.1000/xyz"}]),
                "match_mode": "all",
                "group_by_media": True,
                "page": 2,
                "per_page": 10,
                "q": "nature medicine",
                "media_types": "document,pdf",
                "must_have": "biology,review",
                "must_not_have": "private",
                "date_start": "2026-01-01T00:00:00.000Z",
                "date_end": "2026-01-31T23:59:59.999Z",
                "sort_by": "date_desc",
            },
        )
        assert r.status_code == 200
        assert fake_db.last_search_kwargs is not None
        assert fake_db.last_search_kwargs.get("text_query") == "nature medicine"
        assert fake_db.last_search_kwargs.get("media_types") == ["document", "pdf"]
        assert fake_db.last_search_kwargs.get("must_have_keywords") == ["biology", "review"]
        assert fake_db.last_search_kwargs.get("must_not_have_keywords") == ["private"]
        assert fake_db.last_search_kwargs.get("date_start") == "2026-01-01T00:00:00.000Z"
        assert fake_db.last_search_kwargs.get("date_end") == "2026-01-31T23:59:59.999Z"
        assert fake_db.last_search_kwargs.get("sort_by") == "date_desc"

    app.dependency_overrides.pop(get_media_db_for_user, None)


@pytest.mark.asyncio
async def test_by_identifier_invalid_doi_returns_400(monkeypatch):
    from tldw_Server_API.app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get(
            "/api/v1/media/by-identifier",
            params={"doi": "nope"},
        )
        assert r.status_code == 400
