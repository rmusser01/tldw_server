import os
import pytest
pytestmark = pytest.mark.unit
import types
import pytest
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
            self.config=types.SimpleNamespace(embedding_dim=16)
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
    async def override_user():
        return User(id=1, username='tester', email='t@e.com', is_active=True, is_admin=True)
    app.dependency_overrides[get_request_user]=override_user
    with TestClient(app) as c:
        yield c


def test_create_from_media_chunking_words_language(client, monkeypatch):
    # Fake DB returning one item with English content
    class FakeDB:
        def get_media_by_id(self, mid, **kwargs):
            return {'id': mid, 'title': 'T', 'content': 'This is a sentence. And another.'}
    from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import get_media_db_for_user
    app.dependency_overrides[get_media_db_for_user] = lambda: FakeDB()

    # Create target store
    s = client.post('/api/v1/vector_stores', json={'name':'ChunkStore','dimensions':16}).json()
    body = {
        'store_name':'ignore', 'dimensions':16, 'media_ids':[99],
        'chunk_size':3, 'chunk_overlap':1, 'chunk_method':'words', 'language':'en',
        'update_existing_store_id': s['id']
    }
    r = client.post('/api/v1/vector_stores/create_from_media', json=body)
    assert r.status_code == 200, r.text
    assert r.json()['upserted'] > 0
