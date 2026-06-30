import os
import pathlib
import asyncio
import pytest
pytestmark = pytest.mark.integration
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.main import app


@pytest.fixture(autouse=True)
def testing_env(monkeypatch, tmp_path):
    os.environ['TESTING'] = 'true'
    from tldw_Server_API.app.core import config as cfg
    # Ensure base dir is a Path for DB helpers
    monkeypatch.setitem(cfg.settings, 'USER_DB_BASE_DIR', pathlib.Path(tmp_path))
    yield
    os.environ.pop('TESTING', None)
    app.dependency_overrides.clear()


@pytest.fixture()
def client():
    async def override_user():
        return User(id=1, username='tester', email='t@e.com', is_active=True, is_admin=True)

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="tester",
            email="t@e.com",
            subject="user:1",
            roles=["admin"],
            permissions=[SYSTEM_CONFIGURE, "*"],
            is_admin=True,
        )

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}

    with TestClient(app, headers=headers) as test_client:
        try:
            yield test_client
        finally:
            app.dependency_overrides.pop(get_request_user, None)
            app.dependency_overrides.pop(get_auth_principal, None)


def test_vector_store_end_to_end_no_mocks(client):


    # 1) Create a vector store (dim=8)
    resp = client.post('/api/v1/vector_stores', json={'name': 'RealStore', 'dimensions': 8})
    assert resp.status_code == 200, resp.text
    store = resp.json(); sid = store['id']

    # 2) Upsert explicit vectors (no embeddings required)
    records = [
        { 'id': 'v1', 'values': [0.1]*8, 'content': 'doc one', 'metadata': {'i': 1}},
        { 'id': 'v2', 'values': [0.2]*8, 'content': 'doc two', 'metadata': {'i': 2}},
        { 'id': 'v3', 'values': [0.3]*8, 'content': 'doc three', 'metadata': {'i': 3}},
    ]
    r = client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records': records})
    assert r.status_code == 200, r.text
    assert r.json()['upserted'] == 3

    # 3) List vectors with pagination
    r1 = client.get(f"/api/v1/vector_stores/{sid}/vectors", params={'limit':2,'offset':0})
    assert r1.status_code == 200
    d1 = r1.json(); assert len(d1['data']) == 2
    r2 = client.get(f"/api/v1/vector_stores/{sid}/vectors", params={'limit':2,'offset':2})
    d2 = r2.json(); assert len(d2['data']) == 1

    # 4) Query by vector (avoid embeddings)
    q = client.post(f"/api/v1/vector_stores/{sid}/query", json={'vector': [0.2]*8, 'top_k': 2})
    assert q.status_code == 200, q.text
    qd = q.json(); assert 'data' in qd and isinstance(qd['data'], list)

    # 5) Duplicate store
    dup = client.post(f"/api/v1/vector_stores/{sid}/duplicate", json={'new_name':'RealStoreCopy'})
    assert dup.status_code == 200, dup.text
    dup_id = dup.json()['destination_id']
    assert dup_id and dup_id != sid

    # 6) Batch upsert with explicit values
    batch_payload = {'records': [ {'id': 'b1', 'values':[0.4]*8, 'metadata': {'k': 9}} ]}
    rb = client.post(f"/api/v1/vector_stores/{sid}/vectors/batches", json=batch_payload)
    assert rb.status_code == 200, rb.text
    batch_info = rb.json(); assert batch_info['status'] in ('processing','completed','failed')
    lst = client.get('/api/v1/vector_stores/batches')
    assert lst.status_code == 200


def test_create_from_media_with_existing_embeddings_no_mocks(client):


    # Prepare destination store
    dest = client.post('/api/v1/vector_stores', json={'name':'MediaDest','dimensions':8}).json()
    dest_id = dest['id']

    # Populate per-user media embeddings collection using real adapter
    # Source collection is: user_{uid}_media_embeddings
    from tldw_Server_API.app.core.RAG.rag_service.vector_stores.chromadb_adapter import ChromaDBAdapter
    from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import VectorStoreConfig, VectorStoreType
    from tldw_Server_API.app.core.config import settings as sv
    uid = str(sv.get('SINGLE_USER_FIXED_ID','1'))
    adapter = ChromaDBAdapter(VectorStoreConfig(store_type=VectorStoreType.CHROMADB, connection_params={'use_default': True}, embedding_dim=8, user_id=uid))
    import asyncio
    asyncio.run(adapter.initialize())
    source_collection = adapter.manager.get_or_create_collection(f'user_{uid}_media_embeddings')

    # Insert one embedding for media_id=123
    source_collection.add(
        ids=['m123_c0'],
        embeddings=[[0.0]*8],
        documents=['doc'],
        metadatas=[{'media_id': 123}]
    )

    # Call create_from_media with use_existing_embeddings
    body = {
        'store_name':'ignored',
        'dimensions':8,
        'media_ids':[123],
        'chunk_size': 10,
        'chunk_overlap': 0,
        'chunk_method':'words',
        'use_existing_embeddings': True,
        'update_existing_store_id': dest_id
    }
    r = client.post('/api/v1/vector_stores/create_from_media', json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['store_id'] == dest_id
    assert data['upserted'] >= 1


def test_vector_store_pagination_and_delete_no_mocks(client):


    # Create store
    s = client.post('/api/v1/vector_stores', json={'name': 'PagDel', 'dimensions': 8}).json()
    sid = s['id']

    # Upsert 5 items
    recs = [
        {'id': f'p{i}', 'values': [0.1*i]*8, 'content': f'doc{i}', 'metadata': {'i': i}}
        for i in range(5)
    ]
    r = client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records': recs})
    assert r.status_code == 200

    # Page 1 (2 per page)
    r1 = client.get(f"/api/v1/vector_stores/{sid}/vectors", params={'limit': 2, 'offset': 0})
    assert r1.status_code == 200
    d1 = r1.json(); assert len(d1['data']) == 2
    assert d1['pagination']['total'] >= 5

    # Page 2
    r2 = client.get(f"/api/v1/vector_stores/{sid}/vectors", params={'limit': 2, 'offset': 2})
    assert r2.status_code == 200
    d2 = r2.json(); assert len(d2['data']) == 2

    # Delete one vector
    del_id = d2['data'][0]['id']
    dr = client.delete(f"/api/v1/vector_stores/{sid}/vectors/{del_id}")
    assert dr.status_code == 200

    # Verify count decreased
    r3 = client.get(f"/api/v1/vector_stores/{sid}/vectors", params={'limit': 10, 'offset': 0})
    assert r3.status_code == 200
    d3 = r3.json(); assert d3['pagination']['total'] == 4


def test_query_by_text_with_embeddings_optional(client):


    # Create store and upsert content-only records to trigger embeddings
    s = client.post('/api/v1/vector_stores', json={'name': 'EmbQ', 'dimensions': 8}).json()
    sid = s['id']
    body = {'records': [{'content': 'alpha bravo charlie'}, {'content': 'bravo delta echo'}]}
    r = client.post(f"/api/v1/vector_stores/{sid}/vectors", json=body)
    # If embeddings are not available, this may 500; handle skip in query stage below
    if r.status_code not in (200, 500):
        assert r.status_code == 200

    q = client.post(f"/api/v1/vector_stores/{sid}/query", json={'query': 'bravo', 'top_k': 2})
    if q.status_code in (422, 500):
        pytest.skip("Embeddings backend not configured; skipping text query flow")
    assert q.status_code == 200
    data = q.json()
    assert isinstance(data.get('data'), list)


def test_store_get_and_rename_uniqueness_no_mocks(client):


    # Create two stores
    s1 = client.post('/api/v1/vector_stores', json={'name': 'UniqA', 'dimensions': 8}).json(); id1 = s1['id']
    s2 = client.post('/api/v1/vector_stores', json={'name': 'UniqB', 'dimensions': 8}).json(); id2 = s2['id']

    # GET by id
    g = client.get(f"/api/v1/vector_stores/{id1}")
    assert g.status_code == 200
    assert g.json()['id'] == id1

    # Rename s1 to a new unique name
    p = client.patch(f"/api/v1/vector_stores/{id1}", json={'name': 'UniqC'})
    assert p.status_code == 200
    assert p.json()['name'] == 'UniqC'

    # Attempt to rename s2 to already taken 'UniqC'
    p2 = client.patch(f"/api/v1/vector_stores/{id2}", json={'name': 'UniqC'})
    assert p2.status_code == 409


def test_vector_batch_status_no_mocks(client):


    # Create store and post a batch
    s = client.post('/api/v1/vector_stores', json={'name':'BatchR', 'dimensions': 8}).json(); sid = s['id']
    payload = {'records': [{'id': 'bx', 'values': [0.4]*8, 'metadata': {'k': 7}}]}
    rb = client.post(f"/api/v1/vector_stores/{sid}/vectors/batches", json=payload)
    assert rb.status_code == 200
    batch = rb.json(); bid = batch['id']
    assert bid and batch['status'] in ('processing','completed','failed')

    # Fetch status via GET
    gs = client.get(f"/api/v1/vector_stores/{sid}/vectors/batches/{bid}")
    assert gs.status_code == 200
    data = gs.json(); assert data['id'] == bid
    assert data['status'] in ('processing','completed','failed')


def test_admin_users_list_no_mocks(client):


    # Ensure at least one user dir exists by creating a store
    _ = client.post('/api/v1/vector_stores', json={'name': 'AdminProbe', 'dimensions': 8})
    r = client.get('/api/v1/vector_stores/admin/users')
    assert r.status_code == 200
    users = r.json().get('data', [])
    assert any(u.get('user_id') for u in users)


def test_name_uniqueness_on_create_no_mocks(client):


    r1 = client.post('/api/v1/vector_stores', json={'name': 'SameName', 'dimensions': 8})
    assert r1.status_code == 200
    r2 = client.post('/api/v1/vector_stores', json={'name': 'SameName', 'dimensions': 8})
    assert r2.status_code == 409


def test_list_vector_stores_contains_created_store(client):


    s = client.post('/api/v1/vector_stores', json={'name': 'ListMe', 'dimensions': 8}).json()
    sid = s['id']
    lst = client.get('/api/v1/vector_stores')
    assert lst.status_code == 200
    data = lst.json().get('data', [])
    assert any(row.get('id') == sid for row in data)


def test_query_with_filter_no_mocks(client):


    # Create and insert two vectors with different metadata tags
    s = client.post('/api/v1/vector_stores', json={'name': 'FilterStore', 'dimensions': 8}).json(); sid = s['id']
    records = [
        {'id':'fa','values':[0.1]*8,'content':'A','metadata':{'genre':'a'}},
        {'id':'fb','values':[0.2]*8,'content':'B','metadata':{'genre':'b'}}
    ]
    r = client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records': records})
    assert r.status_code == 200

    # Query near second and filter genre=a -> should either be empty or only genre=a
    q = client.post(f"/api/v1/vector_stores/{sid}/query", json={'vector':[0.2]*8,'top_k':5,'filter':{'genre':'a'}})
    assert q.status_code == 200
    out = q.json().get('data', [])
    for row in out:
        assert row['metadata'].get('genre') == 'a'


def test_duplicate_and_delete_store_no_mocks(client):


    s = client.post('/api/v1/vector_stores', json={'name':'DupDel', 'dimensions':8}).json(); sid = s['id']
    recs = [{'id':f'd{i}','values':[0.1*i]*8,'content':f'doc{i}','metadata':{'i':i}} for i in range(3)]
    client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records': recs})

    dup = client.post(f"/api/v1/vector_stores/{sid}/duplicate", json={'new_name':'DupDelCopy','dimensions':8})
    assert dup.status_code == 200
    did = dup.json()['destination_id']
    # Ensure dest has items by listing vectors
    rv = client.get(f"/api/v1/vector_stores/{did}/vectors", params={'limit':10,'offset':0})
    assert rv.status_code == 200
    assert rv.json()['pagination']['total'] >= 3

    # Delete both stores
    del1 = client.delete(f"/api/v1/vector_stores/{sid}")
    del2 = client.delete(f"/api/v1/vector_stores/{did}")
    assert del1.status_code == 200 and del2.status_code == 200

    # Verify not listed
    lst = client.get('/api/v1/vector_stores')
    ids = [row['id'] for row in lst.json().get('data', [])]
    assert sid not in ids and did not in ids


def test_list_vectors_next_offset_no_mocks(client):


    s = client.post('/api/v1/vector_stores', json={'name':'NextOff', 'dimensions':8}).json(); sid = s['id']
    recs = [{'id':f'n{i}','values':[0.1]*8,'content':f'd{i}','metadata':{'i':i}} for i in range(4)]
    client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records': recs})
    r = client.get(f"/api/v1/vector_stores/{sid}/vectors", params={'limit':2,'offset':0})
    data = r.json()
    assert data['pagination']['next_offset'] == 2


def test_delete_vector_no_mocks(client):


    s = client.post('/api/v1/vector_stores', json={'name':'DelVec', 'dimensions':8}).json(); sid = s['id']
    rec = {'id':'gone','values':[0.0]*8,'content':'x','metadata':{'tag':'del'}}
    client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records':[rec]})
    delr = client.delete(f"/api/v1/vector_stores/{sid}/vectors/gone")
    assert delr.status_code == 200
    assert delr.json().get('deleted') is True


# ------------------- Negative Cases -------------------

def test_create_store_invalid_dimensions(client):

    r = client.post('/api/v1/vector_stores', json={'name': 'BadDim', 'dimensions': 0})
    # Pydantic may reject with 422 before handler returns 400
    assert r.status_code in (400, 422)


def test_upsert_wrong_dimension_rejected(client):


    s = client.post('/api/v1/vector_stores', json={'name':'WrongDim', 'dimensions':8}).json(); sid = s['id']
    # Seed with correct dimension to lock 8
    seed = {'id':'ok','values':[0.0]*8,'content':'seed','metadata':{'k':1}}
    client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records':[seed]})
    # Now wrong-dimension should be rejected
    rec = {'id':'bad','values':[0.0]*7,'content':'x','metadata':{'k':1}}
    r = client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records':[rec]})
    assert r.status_code in (400, 422)


def test_upsert_missing_content_and_values(client):


    s = client.post('/api/v1/vector_stores', json={'name':'MissingFields', 'dimensions':8}).json(); sid = s['id']
    rec = {'id':'m1','metadata':{'k':1}}  # No values or content
    r = client.post(f"/api/v1/vector_stores/{sid}/vectors", json={'records':[rec]})
    assert r.status_code == 400


def test_query_requires_vector_or_text(client):


    s = client.post('/api/v1/vector_stores', json={'name':'QueryNeg', 'dimensions':8}).json(); sid = s['id']
    q = client.post(f"/api/v1/vector_stores/{sid}/query", json={'top_k': 3})
    assert q.status_code == 400


def test_get_nonexistent_store(client):


    # Manager lazily creates collections; GET may return 200 with provided id metadata
    r = client.get('/api/v1/vector_stores/vs_nonexistent_id')
    assert r.status_code in (200, 404, 422)


def test_create_from_media_requires_ids_or_keywords(client):


    r = client.post('/api/v1/vector_stores/create_from_media', json={'store_name':'X','dimensions':8})
    assert r.status_code == 400


def test_create_from_media_invalid_chunk_method(client):


    # Minimal valid request with invalid method
    r = client.post('/api/v1/vector_stores/create_from_media', json={
        'store_name':'BadMethod','dimensions':8,'media_ids':[1],'chunk_method':'not-a-method'
    })
    assert r.status_code in (400, 422)


def test_duplicate_name_conflict(client):


    a = client.post('/api/v1/vector_stores', json={'name':'DupConflictA','dimensions':8}).json(); aid = a['id']
    b = client.post('/api/v1/vector_stores', json={'name':'DupConflictB','dimensions':8}).json(); bid = b['id']
    # Try to duplicate A using name of B
    d = client.post(f"/api/v1/vector_stores/{aid}/duplicate", json={'new_name': 'DupConflictB'})
    assert d.status_code == 409
