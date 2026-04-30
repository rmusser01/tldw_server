import os
import types
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


@pytest.fixture(autouse=True)
def testing_env(monkeypatch, tmp_path):
    os.environ['TESTING']='true'
    from tldw_Server_API.app.core import config as cfg
    monkeypatch.setitem(cfg.settings, 'USER_DB_BASE_DIR', tmp_path)
    yield
    os.environ.pop('TESTING', None)
    app.dependency_overrides.clear()


@pytest.fixture()
def client(monkeypatch):
    # Fake adapter minimal
    class FakeCol:
        def __init__(self):
            self.data={'ids':[], 'embeddings':[], 'documents':[], 'metadatas':[]}
        def count(self):
            return len(self.data['ids'])
        def get(self, **kw):
            return {'ids': []}
    class FakeAdapter:
        def __init__(self):
            self._initialized=False
            self.config=types.SimpleNamespace(embedding_dim=32)
            self._col = FakeCol()
            self.manager = types.SimpleNamespace(get_or_create_collection=lambda name: self._col)
        async def initialize(self):
            self._initialized=True
        async def get_collection_stats(self, name):
            return {'dimension': self.config.embedding_dim, 'metadata':{}}
        async def upsert_vectors(self, collection_name, ids, vectors, documents, metadatas):
            self._col.data['ids'] += ids
            self._col.data['embeddings'] += vectors
            self._col.data['documents'] += documents
            self._col.data['metadatas'] += metadatas
        async def create_collection(self, name, metadata=None):
            col = self.manager.get_or_create_collection(name)
            if metadata:
                if not hasattr(col, 'metadata') or not isinstance(col.metadata, dict):
                    col.metadata = {}
                col.metadata.update(metadata)

    fake = FakeAdapter()
    import tldw_Server_API.app.api.v1.endpoints.vector_stores_openai as vs
    async def fake_adapter_for_user(user, embedding_dim):
        fake.config.embedding_dim = embedding_dim
        return fake
    monkeypatch.setattr(vs, '_adapter_for_user', fake_adapter_for_user)
    def fake_create_embeddings_batch(texts, app_config, model_id):
        return [[0.0]*fake.config.embedding_dim for _ in texts]
    monkeypatch.setattr(vs, 'create_embeddings_batch', fake_create_embeddings_batch)

    # Stub Chunker to produce desired number of chunks regardless of method
    class StubChunker:
        def __init__(self, *a, **k): pass
        def chunk_text(self, text, method=None, max_size=None, overlap=None):
            # pretend to create 3 chunks
            return ['c1','c2','c3']
    monkeypatch.setattr(vs, 'Chunker', StubChunker)

    async def override_user():
        return User(id=1, username='tester', email='t@e.com', is_active=True, is_admin=True)
    app.dependency_overrides[get_request_user]=override_user
    with TestClient(app) as c:
        yield c


def test_tokens_method_flow(client, monkeypatch):
    # Fake DB returning one media item
    class FakeDB:
        def get_media_by_id(self, mid, include_deleted=False, include_trash=False):
            return {'id': mid, 'title': 'T', 'content': 'dummy text'}
    from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import get_media_db_for_user
    app.dependency_overrides[get_media_db_for_user] = lambda: FakeDB()

    store = client.post('/api/v1/vector_stores', json={'name':'TokStore','dimensions':32}).json()
    body = {
        'store_name':'ignore','dimensions':32,'media_ids':[1],
        'chunk_size':10,'chunk_overlap':2,'chunk_method':'tokens',
        'update_existing_store_id': store['id']
    }
    r = client.post('/api/v1/vector_stores/create_from_media', json=body)
    assert r.status_code == 200, r.text
    assert r.json()['upserted'] == 3


def test_semantic_method_flow(client):
    # Same as tokens; chunker stub returns 3 chunks
    class FakeDB:
        def get_media_by_id(self, mid, include_deleted=False, include_trash=False):
            return {'id': mid, 'title': 'T', 'content': 'dummy text'}
    from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import get_media_db_for_user
    app.dependency_overrides[get_media_db_for_user] = lambda: FakeDB()

    store = client.post('/api/v1/vector_stores', json={'name':'SemStore','dimensions':32}).json()
    body = {
        'store_name':'ignore','dimensions':32,'media_ids':[1],
        'chunk_size':10,'chunk_overlap':2,'chunk_method':'semantic',
        'update_existing_store_id': store['id']
    }
    r = client.post('/api/v1/vector_stores/create_from_media', json=body)
    assert r.status_code == 200, r.text
    assert r.json()['upserted'] == 3


@pytest.mark.asyncio
async def test_create_from_media_sanitizes_embedding_failure(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.vector_stores_openai as vs

    class FakeAdapter:
        def __init__(self):
            self._initialized = False
            self.config = types.SimpleNamespace(embedding_dim=32)

        async def initialize(self):
            self._initialized = True

        async def upsert_vectors(self, *args, **kwargs):
            raise AssertionError("upsert_vectors should not be reached after embedding failure")

    class StubChunker:
        def __init__(self, *args, **kwargs):
            pass

        def chunk_text(self, text, method=None, max_size=None, overlap=None):
            return ["c1", "c2", "c3"]

    def exploding_embeddings(_texts, _app_config, _model_id):
        raise RuntimeError("embedding backend exploded")

    fake_adapter = FakeAdapter()

    async def fake_get_adapter_for_user(_user, embedding_dim):
        fake_adapter.config.embedding_dim = embedding_dim
        return fake_adapter

    monkeypatch.setattr(vs, "Chunker", StubChunker)
    monkeypatch.setattr(vs, "_get_adapter_for_user", fake_get_adapter_for_user)
    monkeypatch.setattr(vs, "_get_embeddings_fn", lambda: exploding_embeddings)
    monkeypatch.setattr(vs, "get_media_by_id", lambda _db, mid: {"id": mid, "title": "T", "content": "dummy text"})
    monkeypatch.setattr(vs, "_allowed_providers", lambda: None)
    monkeypatch.setattr(vs, "_allowed_models", lambda: None)
    monkeypatch.setattr(vs, "_count_tokens", lambda _text, _model_id: 1)
    monkeypatch.setattr(vs, "init_batches_db", lambda _uid: None)
    monkeypatch.setattr(vs, "db_create_batch", lambda *args, **kwargs: None)
    updates = []
    monkeypatch.setattr(vs, "db_update_batch", lambda *args, **kwargs: updates.append(kwargs))

    payload = vs.CreateFromMediaRequest(
        store_name="ignore",
        dimensions=32,
        media_ids=[1],
        chunk_size=10,
        chunk_overlap=2,
        chunk_method="tokens",
        update_existing_store_id="vs_demo",
    )
    user = User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    with pytest.raises(HTTPException) as excinfo:
        await vs.create_store_from_media(payload, current_user=user, db=object())

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to generate embeddings for media content"
    assert updates[-1]["error"] == "Failed to generate embeddings for media content"
    assert "embedding backend exploded" not in str(updates)


@pytest.mark.asyncio
async def test_upsert_vectors_batch_sanitizes_unexpected_failure(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.vector_stores_openai as vs

    class LoggerStub:
        def __init__(self):
            self.errors = []

        def error(self, *args, **kwargs):
            self.errors.append((args, kwargs))

    async def _fail_upsert_vectors(*_args, **_kwargs):
        raise RuntimeError("vector batch backend exploded at /private/chroma/batches")

    updates = []
    logger_stub = LoggerStub()

    monkeypatch.setattr(vs, "resolve_user_id_for_request", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(vs, "init_batches_db", lambda _uid: None)
    monkeypatch.setattr(vs, "db_create_batch", lambda *args, **kwargs: None)
    monkeypatch.setattr(vs, "db_update_batch", lambda *args, **kwargs: updates.append(kwargs))
    monkeypatch.setattr(vs, "upsert_vectors", _fail_upsert_vectors)
    monkeypatch.setattr(vs, "logger", logger_stub)
    vs._BATCH_STATUS.clear()

    payload = vs.UpsertVectorsRequest(
        records=[vs.VectorRecord(id="vec_1", values=[0.1, 0.2], metadata={})],
    )
    user = User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    with pytest.raises(RuntimeError):
        await vs.upsert_vectors_batch("vs_demo", payload, current_user=user)

    batch = next(iter(vs._BATCH_STATUS.values()))
    assert batch["status"] == "failed"
    assert batch["error"] == "Vector batch upsert failed"
    assert updates[-1]["error"] == "Vector batch upsert failed"
    assert "vector batch backend exploded" not in str(batch)
    assert "/private/chroma/batches" not in str(batch)
    assert logger_stub.errors
    assert logger_stub.errors[-1][0][0] == "Vector batch upsert failed for batch {}"
    assert logger_stub.errors[-1][1].get("exc_info") is None
    assert "vector batch backend exploded" not in repr(logger_stub.errors)
    assert "/private/chroma/batches" not in repr(logger_stub.errors)


@pytest.mark.asyncio
async def test_vector_stores_health_sanitizes_adapter_failure(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.vector_stores_openai as vs

    class FakeAdapter:
        async def initialize(self):
            return None

        async def health(self):
            raise RuntimeError("vector health exploded at /private/chroma")

    async def fake_get_adapter_for_user(_user, _embedding_dim):
        return FakeAdapter()

    monkeypatch.setattr(vs, "_get_adapter_for_user", fake_get_adapter_for_user)
    user = User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    response = await vs.vector_stores_health(current_user=user)

    assert response == {"ok": False, "error": "Vector store health check failed"}
