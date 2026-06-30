import types

import pytest

from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vector_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


pytestmark = pytest.mark.unit


_SENSITIVE_MARKERS = (
    "vector backend exploded",
    "/private/vector.db",
)


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def warning(self, *args: object, **kwargs: object) -> None:
        self.warnings.append((args, kwargs))


class _FakeCollection:
    def __init__(self) -> None:
        self.ids = ["vec_1"]
        self.documents = ["stored content"]
        self.metadatas = [{"source": "test"}]
        self.metadata: dict[str, object] = {}

    def count(self) -> int:
        return len(self.ids)

    def get(self, limit=100, offset=0, include=None, where=None):
        del where
        end = min(offset + limit, len(self.ids))
        idxs = list(range(offset, end))
        data: dict[str, list[object]] = {"ids": [self.ids[i] for i in idxs]}
        if include and "documents" in include:
            data["documents"] = [self.documents[i] for i in idxs]
        if include and "metadatas" in include:
            data["metadatas"] = [self.metadatas[i] for i in idxs]
        return data


class _FakeAdapter:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(embedding_dim=8)
        self.collection = _FakeCollection()
        self.collections: dict[str, _FakeCollection] = {"vs_existing": self.collection}
        self.manager = types.SimpleNamespace(get_or_create_collection=self.get_or_create_collection)

    async def initialize(self) -> None:
        return None

    async def create_collection(self, name, metadata=None) -> None:
        collection = self.get_or_create_collection(name)
        if metadata:
            collection.metadata.update(metadata)

    async def list_collections(self):
        return list(self.collections.keys())

    async def get_collection_stats(self, name):
        collection = self.get_or_create_collection(name)
        return {
            "dimension": self.config.embedding_dim,
            "metadata": collection.metadata,
        }

    async def delete_collection(self, name) -> None:
        self.collections.pop(name, None)

    async def list_vectors_paginated(self, *_args, **_kwargs):
        raise RuntimeError("vector backend exploded /private/vector.db")

    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = _FakeCollection()
        return self.collections[name]


@pytest.fixture()
def logger_stub(monkeypatch):
    stub = _LoggerStub()
    monkeypatch.setattr(vector_ep, "logger", stub)
    return stub


@pytest.fixture()
def user() -> User:
    return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)


def _sensitive_error(*_args, **_kwargs):
    raise RuntimeError("vector backend exploded /private/vector.db")


def _assert_warning_sanitized(logger_stub: _LoggerStub, expected_message: str) -> None:
    messages = [args[0] for args, _kwargs in logger_stub.warnings if args]
    assert expected_message in messages
    rendered = repr(logger_stub.warnings)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.mark.asyncio
async def test_create_vector_store_sanitizes_meta_uniqueness_fallback_log(
    monkeypatch,
    logger_stub,
    user,
):
    fake_adapter = _FakeAdapter()

    async def fake_get_adapter_for_user(_user, embedding_dim):
        fake_adapter.config.embedding_dim = embedding_dim
        return fake_adapter

    monkeypatch.setattr(vector_ep, "_get_adapter_for_user", fake_get_adapter_for_user)
    monkeypatch.setattr(vector_ep, "resolve_user_id_for_request", lambda *_args, **_kwargs: "1")
    monkeypatch.setattr(vector_ep, "init_meta_db", lambda _uid: None)
    monkeypatch.setattr(vector_ep, "meta_find_store_by_name", _sensitive_error)
    monkeypatch.setattr(vector_ep, "meta_register_store", lambda *_args, **_kwargs: None)

    result = await vector_ep.create_vector_store(
        vector_ep.VectorStoreCreate(name="Docs", dimensions=8),
        current_user=user,
    )

    assert result.name == "Docs"
    _assert_warning_sanitized(
        logger_stub,
        "Meta DB uniqueness check failed; falling back to adapter scan",
    )


@pytest.mark.asyncio
async def test_create_vector_store_sanitizes_meta_registration_fallback_log(
    monkeypatch,
    logger_stub,
    user,
):
    fake_adapter = _FakeAdapter()

    async def fake_get_adapter_for_user(_user, embedding_dim):
        fake_adapter.config.embedding_dim = embedding_dim
        return fake_adapter

    monkeypatch.setattr(vector_ep, "_get_adapter_for_user", fake_get_adapter_for_user)
    monkeypatch.setattr(vector_ep, "resolve_user_id_for_request", lambda *_args, **_kwargs: "1")
    monkeypatch.setattr(vector_ep, "init_meta_db", lambda _uid: None)
    monkeypatch.setattr(vector_ep, "meta_find_store_by_name", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vector_ep, "meta_register_store", _sensitive_error)

    result = await vector_ep.create_vector_store(
        vector_ep.VectorStoreCreate(name="Docs", dimensions=8),
        current_user=user,
    )

    assert result.name == "Docs"
    _assert_warning_sanitized(
        logger_stub,
        "Failed to register vector store in meta DB",
    )


@pytest.mark.asyncio
async def test_update_vector_store_sanitizes_meta_name_persistence_fallback_log(
    monkeypatch,
    logger_stub,
    user,
):
    fake_adapter = _FakeAdapter()
    fake_adapter.collection.metadata.update(
        {
            "openai_id": "vs_existing",
            "name": "Existing",
            "created_at": 123,
        }
    )

    async def fake_get_adapter_for_user(_user, embedding_dim):
        fake_adapter.config.embedding_dim = embedding_dim
        return fake_adapter

    monkeypatch.setattr(vector_ep, "_get_adapter_for_user", fake_get_adapter_for_user)
    monkeypatch.setattr(vector_ep, "resolve_user_id_for_request", lambda *_args, **_kwargs: "1")
    monkeypatch.setattr(vector_ep, "init_meta_db", lambda _uid: None)
    monkeypatch.setattr(vector_ep, "meta_find_store_by_name", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vector_ep, "meta_list_stores", _sensitive_error)

    result = await vector_ep.update_vector_store(
        store_id="vs_existing",
        payload=vector_ep.VectorStoreUpdate(name="Renamed"),
        current_user=user,
    )

    assert result.name == "Renamed"
    _assert_warning_sanitized(
        logger_stub,
        "Failed to persist vector store meta name update",
    )


@pytest.mark.asyncio
async def test_delete_vector_store_sanitizes_meta_delete_fallback_log(
    monkeypatch,
    logger_stub,
    user,
):
    fake_adapter = _FakeAdapter()

    async def fake_get_adapter_for_user(_user, embedding_dim):
        fake_adapter.config.embedding_dim = embedding_dim
        return fake_adapter

    monkeypatch.setattr(vector_ep, "_get_adapter_for_user", fake_get_adapter_for_user)
    monkeypatch.setattr(vector_ep, "resolve_user_id_for_request", lambda *_args, **_kwargs: "1")
    monkeypatch.setattr(vector_ep, "meta_delete_store", _sensitive_error)

    result = await vector_ep.delete_vector_store(
        store_id="vs_existing",
        current_user=user,
    )

    assert result == {"id": "vs_existing", "deleted": True}
    _assert_warning_sanitized(
        logger_stub,
        "Failed to delete vector store from meta DB",
    )


@pytest.mark.asyncio
async def test_list_vector_stores_sanitizes_meta_db_fallback_log(
    monkeypatch,
    logger_stub,
    user,
):
    fake_adapter = _FakeAdapter()
    fake_adapter.collection.metadata.update(
        {
            "openai_id": "vs_existing",
            "name": "Existing",
            "created_at": 123,
        }
    )

    async def fake_get_adapter_for_user(_user, embedding_dim):
        fake_adapter.config.embedding_dim = embedding_dim
        return fake_adapter

    monkeypatch.setattr(vector_ep, "_get_adapter_for_user", fake_get_adapter_for_user)
    monkeypatch.setattr(vector_ep, "resolve_user_id_for_request", lambda *_args, **_kwargs: "1")
    monkeypatch.setattr(vector_ep, "init_meta_db", _sensitive_error)

    result = await vector_ep.list_vector_stores(current_user=user)

    assert result["data"][0]["id"] == "vs_existing"
    assert result["data"][0]["name"] == "Existing"
    _assert_warning_sanitized(
        logger_stub,
        "Meta DB list failed; falling back to Chroma-only",
    )


@pytest.mark.asyncio
async def test_list_vectors_sanitizes_adapter_pagination_fallback_log(
    monkeypatch,
    logger_stub,
    user,
):
    fake_adapter = _FakeAdapter()

    async def fake_get_adapter_for_user(_user, embedding_dim):
        fake_adapter.config.embedding_dim = embedding_dim
        return fake_adapter

    monkeypatch.setattr(vector_ep, "_get_adapter_for_user", fake_get_adapter_for_user)

    result = await vector_ep.list_vectors(
        store_id="vs_existing",
        limit=10,
        offset=0,
        filter=None,
        order_by="id",
        order_dir="asc",
        current_user=user,
    )

    assert result["data"][0]["id"] == "vec_1"
    assert result["pagination"]["total"] == 1
    _assert_warning_sanitized(
        logger_stub,
        "Adapter list_vectors_paginated failed; falling back to Chroma path",
    )
