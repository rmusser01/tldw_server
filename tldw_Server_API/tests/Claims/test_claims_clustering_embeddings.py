import os
import shutil
import tempfile

from tldw_Server_API.app.core.Claims_Extraction.claims_clustering import rebuild_claim_clusters_embeddings
from tldw_Server_API.app.core.Claims_Extraction.claims_embeddings import claim_embedding_id
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


class _StubCollection:
    def __init__(self, embedding_map):
        self._embedding_map = embedding_map

    def get(self, ids=None, include=None):  # noqa: D401 - stub for chroma
        result_ids = []
        result_embeddings = []
        for item_id in ids or []:
            if item_id in self._embedding_map:
                result_ids.append(item_id)
                result_embeddings.append(self._embedding_map[item_id])
        return {"ids": result_ids, "embeddings": result_embeddings}


class _StubManager:
    def __init__(self, user_id, user_embedding_config):
        self._collection = user_embedding_config.get("_test_collection")

    def get_or_create_collection(self, name):  # noqa: D401 - stub for chroma
        return self._collection


def _stub_manager_for(collection):
    class _BoundStubManager:
        def __init__(self, user_id, user_embedding_config):
            del user_id, user_embedding_config
            self._collection = collection

        def get_or_create_collection(self, name):  # noqa: D401 - stub for chroma
            del name
            return self._collection

    return _BoundStubManager


def _restore_setting(settings, key, original):
    if original is None:
        settings.pop(key, None)
    else:
        settings[key] = original


def _seed_claims_db():


    tmpdir = tempfile.mkdtemp(prefix="claims_cluster_embed_")
    db_path = os.path.join(tmpdir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()
    content = "Cats are great. Cats are great. Different topic."
    media_id, _, _ = db.add_media_with_keywords(title="Doc", media_type="text", content=content, keywords=None)
    db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "Cats are great.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": "hash1",
            },
            {
                "media_id": media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "Cats are very great.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": "hash1",
            },
            {
                "media_id": media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "Different topic.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": "hash1",
            },
        ]
    )
    rows = db.execute_query(
        "SELECT id, claim_text, chunk_index FROM Claims WHERE media_id = ? AND deleted = 0 ORDER BY id ASC",
        (media_id,),
    ).fetchall()
    claim_rows = [dict(r) for r in rows]
    db.close_connection()
    return db_path, media_id, claim_rows, tmpdir


def test_rebuild_claim_clusters_embeddings(monkeypatch):


    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.core.Claims_Extraction import claims_clustering

    db_path, media_id, claim_rows, tmpdir = _seed_claims_db()
    base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
    orig_user_db = app_settings.get("USER_DB_BASE_DIR")
    orig_embedding_cfg = app_settings.get("EMBEDDING_CONFIG")
    app_settings["USER_DB_BASE_DIR"] = base_dir

    db = None
    try:
        embeddings = {}
        for row in claim_rows:
            claim_text = row["claim_text"]
            embed_id = claim_embedding_id(media_id, row["chunk_index"], claim_text)
            if "Different" in claim_text:
                embeddings[embed_id] = [0.0, 1.0]
            elif "very" in claim_text:
                embeddings[embed_id] = [0.99, 0.01]
            else:
                embeddings[embed_id] = [1.0, 0.0]

        stub_collection = _StubCollection(embeddings)
        monkeypatch.setattr(claims_clustering, "ChromaDBManager", _stub_manager_for(stub_collection))
        app_settings["EMBEDDING_CONFIG"] = {}

        db = MediaDatabase(db_path=db_path, client_id="1")
        db.initialize_db()
        result = rebuild_claim_clusters_embeddings(
            db=db,
            user_id="1",
            min_size=2,
            similarity_threshold=0.9,
        )
        assert result.get("clusters_created") == 1
        assert result.get("claims_assigned") == 2

        clusters = db.list_claim_clusters("1", limit=10, offset=0)
        assert len(clusters) == 1
        cluster_id = int(clusters[0]["id"])
        members = db.list_claim_cluster_members(cluster_id, limit=10, offset=0)
        assert len(members) == 2
    finally:
        if db is not None:
            try:
                db.close_connection()
            except Exception:
                _ = None
        _restore_setting(app_settings, "USER_DB_BASE_DIR", orig_user_db)
        _restore_setting(app_settings, "EMBEDDING_CONFIG", orig_embedding_cfg)
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None
        try:
            shutil.rmtree(base_dir, ignore_errors=True)
        except Exception:
            _ = None


def test_rebuild_claim_clusters_embeddings_without_embeddings(monkeypatch):


    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.core.Claims_Extraction import claims_clustering

    db_path, _, _, tmpdir = _seed_claims_db()
    base_dir = tempfile.mkdtemp(prefix="chroma_user_base_")
    orig_user_db = app_settings.get("USER_DB_BASE_DIR")
    orig_embedding_cfg = app_settings.get("EMBEDDING_CONFIG")
    app_settings["USER_DB_BASE_DIR"] = base_dir

    db = None
    try:
        stub_collection = _StubCollection({})
        monkeypatch.setattr(claims_clustering, "ChromaDBManager", _stub_manager_for(stub_collection))
        app_settings["EMBEDDING_CONFIG"] = {}

        db = MediaDatabase(db_path=db_path, client_id="1")
        db.initialize_db()
        result = rebuild_claim_clusters_embeddings(
            db=db,
            user_id="1",
            min_size=2,
            similarity_threshold=0.9,
        )
        assert result.get("clusters_created") == 0
        assert result.get("status") == "no_embeddings"

        clusters = db.list_claim_clusters("1", limit=10, offset=0)
        assert clusters == []
    finally:
        if db is not None:
            try:
                db.close_connection()
            except Exception:
                _ = None
        _restore_setting(app_settings, "USER_DB_BASE_DIR", orig_user_db)
        _restore_setting(app_settings, "EMBEDDING_CONFIG", orig_embedding_cfg)
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            _ = None
        try:
            shutil.rmtree(base_dir, ignore_errors=True)
        except Exception:
            _ = None
