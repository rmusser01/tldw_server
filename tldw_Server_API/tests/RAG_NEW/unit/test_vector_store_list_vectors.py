import json as _json
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


class _FakeCollection:
    def __init__(self):
        self._count = 3

    def count(self):  # noqa: D401
        return self._count

    def get(self, limit=10, offset=0, include=None, where=None):  # noqa: ANN001
        # Return a static set; ignore where filter for this fake
        ids = ["v1", "v2", "v3"]
        docs = ["a", "b", "c"]
        metas = [
            {"genre": "a", "score": 2},
            {"genre": "a", "score": 1},
            {"genre": "b", "score": 3},
        ]
        # Apply limit/offset
        ids = ids[offset:offset + limit]
        docs = docs[offset:offset + limit]
        metas = metas[offset:offset + limit]
        return {"ids": ids, "documents": docs, "metadatas": metas}


class _FakeManager:
    def get_or_create_collection(self, name):  # noqa: D401, ANN001
        return _FakeCollection()


class _FakeAdapter:
    def __init__(self):
        self._initialized = True
        self.manager = _FakeManager()

    async def initialize(self):  # pragma: no cover
        self._initialized = True


@pytest.mark.unit
def test_list_vectors_parses_filter_and_sorts(monkeypatch, disable_heavy_startup, admin_user):
     # Patch adapter factory to return our fake adapter
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapter()

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)

    client = TestClient(app)
    # JSON filter is parsed but ignored by fake; order_by should sort client-side
    params = {
        "limit": 3,
        "offset": 0,
        "filter": _json.dumps({"genre": "a"}),
        "order_by": "metadata.score",
        "order_dir": "desc",
    }
    r = client.get("/api/v1/vector_stores/store-xyz/vectors", params=params)
    assert r.status_code == 200
    body = r.json()
    items = body.get("data") or []
    # Expect sorted by score desc: v3 (3), v1 (2), v2 (1) after filtering logic ignored by fake
    ids_in_order = [it.get("id") for it in items]
    assert ids_in_order == ["v3", "v1", "v2"]


@pytest.mark.unit
def test_list_vectors_invalid_filter_returns_400(monkeypatch, disable_heavy_startup, admin_user):
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapter()

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)

    client = TestClient(app)
    # Invalid JSON should return 400 with error
    params = {
        "limit": 3,
        "offset": 0,
        "filter": "{not-json}",
    }
    r = client.get("/api/v1/vector_stores/store-xyz/vectors", params=params)
    assert r.status_code == 400
    assert "invalid_filter" in r.text


class _FakeAdapterPaginated:
    def __init__(self, total: int):
        self._initialized = True
        self._total = int(total)

    async def initialize(self):  # pragma: no cover
        self._initialized = True

    async def list_vectors_paginated(self, store_id: str, limit: int, offset: int, filter=None, order_by=None, order_dir=None):  # noqa: ANN001
        # Build a deterministic range of items based on total/offset
        start = int(offset)
        end = min(self._total, start + int(limit))
        items = []
        for i in range(start, end):
            items.append({
                "id": f"v{i+1}",
                "content": f"c{i+1}",
                "metadata": {"idx": i+1}
            })
        # Apply adapter-side ordering for metadata.idx
        if order_by and isinstance(order_by, str) and order_by.startswith("metadata."):
            key = order_by.split(".", 1)[1]
            reverse = str(order_dir or "asc").lower() == "desc"
            items.sort(key=lambda it: (it.get("metadata") or {}).get(key, 0), reverse=reverse)
        elif order_by == "id":
            reverse = str(order_dir or "asc").lower() == "desc"
            def _id_index(s: str) -> int:
                try:
                    return int(s.lstrip("v"))
                except Exception:
                    return 0
            items.sort(key=lambda it: _id_index(it.get("id", "v0")), reverse=reverse)
        return {"items": items, "total": self._total}


@pytest.mark.unit
def test_list_vectors_pagination_next_offset_chroma_fallback(monkeypatch, disable_heavy_startup, admin_user):
     # Fake adapter without list_vectors_paginated triggers Chroma fallback path
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapter()

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)

    client = TestClient(app)

    # total is 3 in _FakeCollection; with limit 2, offset 0 → next_offset 2
    r1 = client.get("/api/v1/vector_stores/store-xyz/vectors", params={"limit": 2, "offset": 0})
    assert r1.status_code == 200
    p1 = r1.json().get("pagination", {})
    assert p1.get("next_offset") == 2

    # offset 2, limit 2 → returned=1 != limit → no next_offset
    r2 = client.get("/api/v1/vector_stores/store-xyz/vectors", params={"limit": 2, "offset": 2})
    assert r2.status_code == 200
    p2 = r2.json().get("pagination", {})
    assert p2.get("next_offset") is None


@pytest.mark.unit
def test_list_vectors_pagination_next_offset_adapter_path(monkeypatch, disable_heavy_startup, admin_user) -> None:
    """Adapter pagination should include canonical offset metadata."""
    # Adapter with list_vectors_paginated sets total explicitly
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapterPaginated(total=5)

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)

    client = TestClient(app)

    # offset 0, limit 2 → returned 2, next_offset 2
    r1 = client.get("/api/v1/vector_stores/store-abc/vectors", params={"limit": 2, "offset": 0})
    assert r1.status_code == 200
    p1 = r1.json().get("pagination", {})
    assert p1.get("mode") == "offset"
    assert p1.get("has_more") is True
    assert p1.get("next_offset") == 2

    # offset 3, limit 2 → returned 2, offset+returned == total → next_offset None
    r2 = client.get("/api/v1/vector_stores/store-abc/vectors", params={"limit": 2, "offset": 3})
    assert r2.status_code == 200
    p2 = r2.json().get("pagination", {})
    assert p2.get("mode") == "offset"
    assert p2.get("has_more") is False
    assert p2.get("next_offset") is None


@pytest.mark.unit
def test_list_vectors_adapter_ordering_asc_desc(monkeypatch, disable_heavy_startup, admin_user):
     # Ensure adapter receives order_by and order_dir and sorts accordingly
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapterPaginated(total=4)

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)

    client = TestClient(app)

    # Ascending order → v1, v2
    r1 = client.get(
        "/api/v1/vector_stores/store-abc/vectors",
        params={"limit": 2, "offset": 0, "order_by": "metadata.idx", "order_dir": "asc"},
    )
    assert r1.status_code == 200
    ids1 = [it.get("id") for it in (r1.json().get("data") or [])]
    assert ids1 == ["v1", "v2"]

    # Descending order → v2, v1 on the same window
    r2 = client.get(
        "/api/v1/vector_stores/store-abc/vectors",
        params={"limit": 2, "offset": 0, "order_by": "metadata.idx", "order_dir": "desc"},
    )
    assert r2.status_code == 200
    ids2 = [it.get("id") for it in (r2.json().get("data") or [])]
    assert ids2 == ["v2", "v1"]


@pytest.mark.unit
def test_list_vectors_invalid_order_by_returns_400(monkeypatch, disable_heavy_startup, admin_user):
     # Invalid order_by should be rejected with 400
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapter()

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)

    client = TestClient(app)
    r = client.get(
        "/api/v1/vector_stores/store-xyz/vectors",
        params={"order_by": "name", "order_dir": "asc"},
    )
    assert r.status_code == 400
    assert "invalid_order_by" in r.text


@pytest.mark.unit
def test_list_vectors_invalid_order_dir_returns_422(monkeypatch, disable_heavy_startup, admin_user):
     # order_dir is validated by pydantic pattern → invalid value yields 422
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapter()

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)

    client = TestClient(app)
    r = client.get(
        "/api/v1/vector_stores/store-xyz/vectors",
        params={"order_dir": "down"},
    )
    assert r.status_code == 422


@pytest.mark.unit
def test_list_vectors_negative_offset_and_limit_one(monkeypatch, disable_heavy_startup, admin_user):
     # offset < 0 and limit == 1 should be rejected by validation
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapter()

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)

    client = TestClient(app)
    r1 = client.get(
        "/api/v1/vector_stores/store-xyz/vectors",
        params={"offset": -1},
    )
    assert r1.status_code == 422

    r2 = client.get(
        "/api/v1/vector_stores/store-xyz/vectors",
        params={"limit": 1},
    )
    assert r2.status_code == 422


@pytest.mark.unit
def test_list_vectors_boundary_offsets_adapter_path(monkeypatch, disable_heavy_startup, admin_user):
     # When offset equals total or exceeds it, items should be empty and next_offset None
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapterPaginated(total=5)

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)
    client = TestClient(app)

    # offset == total
    r1 = client.get(
        "/api/v1/vector_stores/store-abc/vectors",
        params={"limit": 2, "offset": 5},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1.get("data") == []
    assert body1.get("pagination", {}).get("next_offset") is None

    # offset > total
    r2 = client.get(
        "/api/v1/vector_stores/store-abc/vectors",
        params={"limit": 2, "offset": 6},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2.get("data") == []
    assert body2.get("pagination", {}).get("next_offset") is None


@pytest.mark.unit
def test_list_vectors_order_by_id_adapter_path(monkeypatch, disable_heavy_startup, admin_user):
     # Ensure ordering by id works in adapter path
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapterPaginated(total=4)

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)
    client = TestClient(app)

    r1 = client.get(
        "/api/v1/vector_stores/store-abc/vectors",
        params={"limit": 2, "offset": 0, "order_by": "id", "order_dir": "asc"},
    )
    assert r1.status_code == 200
    ids1 = [it.get("id") for it in (r1.json().get("data") or [])]
    assert ids1 == ["v1", "v2"]

    r2 = client.get(
        "/api/v1/vector_stores/store-abc/vectors",
        params={"limit": 2, "offset": 0, "order_by": "id", "order_dir": "desc"},
    )
    assert r2.status_code == 200
    ids2 = [it.get("id") for it in (r2.json().get("data") or [])]
    assert ids2 == ["v2", "v1"]


def _assert_envelope(resp_json):


    assert isinstance(resp_json, dict)
    assert "data" in resp_json and isinstance(resp_json["data"], list)
    assert "pagination" in resp_json and isinstance(resp_json["pagination"], dict)
    p = resp_json["pagination"]
    # Required numeric fields
    assert isinstance(p.get("limit"), int)
    assert isinstance(p.get("offset"), int)
    # Optional fields
    assert (p.get("next_offset") is None) or isinstance(p.get("next_offset"), int)
    assert isinstance(p.get("total"), int)


@pytest.mark.unit
def test_envelope_pagination_contract_adapter_path(monkeypatch, disable_heavy_startup, admin_user):
     # Generic envelope contract validation on adapter path
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapterPaginated(total=7)

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)
    client = TestClient(app)

    # Page 1
    r1 = client.get("/api/v1/vector_stores/store-e/vectors", params={"limit": 3, "offset": 0})
    assert r1.status_code == 200
    j1 = r1.json()
    _assert_envelope(j1)
    assert j1["pagination"]["next_offset"] == 3
    assert len(j1["data"]) == 3

    # Page 2
    r2 = client.get("/api/v1/vector_stores/store-e/vectors", params={"limit": 3, "offset": 3})
    assert r2.status_code == 200
    j2 = r2.json()
    _assert_envelope(j2)
    assert j2["pagination"]["next_offset"] == 6
    assert len(j2["data"]) == 3

    # Page 3 (tail)
    r3 = client.get("/api/v1/vector_stores/store-e/vectors", params={"limit": 3, "offset": 6})
    assert r3.status_code == 200
    j3 = r3.json()
    _assert_envelope(j3)
    assert j3["pagination"]["next_offset"] is None
    assert len(j3["data"]) == 1

    # Beyond total
    r4 = client.get("/api/v1/vector_stores/store-e/vectors", params={"limit": 3, "offset": 9})
    assert r4.status_code == 200
    j4 = r4.json()
    _assert_envelope(j4)
    assert j4["pagination"]["next_offset"] is None
    assert len(j4["data"]) == 0


@pytest.mark.unit
def test_envelope_pagination_contract_fallback_path(monkeypatch, disable_heavy_startup, admin_user):
     # Generic envelope contract validation on Chroma fallback path
    from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod

    async def _fake_get_adapter_for_user(user, dim):  # noqa: ANN001
        return _FakeAdapter()

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user, raising=True)
    client = TestClient(app)

    # total=3 (see _FakeCollection); page 1
    r1 = client.get("/api/v1/vector_stores/store-f/vectors", params={"limit": 2, "offset": 0})
    assert r1.status_code == 200
    j1 = r1.json()
    _assert_envelope(j1)
    assert j1["pagination"]["next_offset"] == 2
    assert len(j1["data"]) == 2

    # page 2 (tail)
    r2 = client.get("/api/v1/vector_stores/store-f/vectors", params={"limit": 2, "offset": 2})
    assert r2.status_code == 200
    j2 = r2.json()
    _assert_envelope(j2)
    assert j2["pagination"]["next_offset"] is None
    assert len(j2["data"]) == 1
