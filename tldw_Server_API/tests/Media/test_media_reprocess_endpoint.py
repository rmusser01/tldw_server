import asyncio
from collections.abc import AsyncGenerator

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import MediaDbLike
from tldw_Server_API.tests.test_utils import create_test_media


class _LoggerStub:
    def __init__(self):
        self.error_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.exception_calls = []

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "chunker backend leaked",
    "/private/chunker/token",
    "embedding backend leaked",
    "/private/vector/token",
)


def _patch_reprocess_logger(monkeypatch) -> _LoggerStub:
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint

    logger_stub = _LoggerStub()
    monkeypatch.setattr(reprocess_endpoint, "logger", logger_stub, raising=True)
    return logger_stub


def _assert_sanitized_error_log(
    logger_stub: _LoggerStub,
    expected_message: str,
) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.error_calls if args]
    assert expected_message in matching_messages

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _assert_sanitized_warning_log(
    logger_stub: _LoggerStub,
    expected_message: str,
) -> None:
    assert logger_stub.warning_calls

    matching_messages = [args[0] for args, _kwargs in logger_stub.warning_calls if args]
    assert expected_message in matching_messages

    rendered_calls = repr(logger_stub.warning_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


def _principal_override():

    async def _override(request=None) -> AuthPrincipal:
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",
            jti=None,
            roles=["admin"],
            permissions=["media.update"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    return _override


def test_reprocess_rebuilds_chunks(tmp_path, monkeypatch, managed_test_media_db):

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    monkeypatch.setenv("TEST_MODE", "1")

    db_path = tmp_path / "media.db"
    with managed_test_media_db(
        "test_client",
        db_path=str(db_path),
        initialize=False,
    ) as seed_db:
        media_id = create_test_media(seed_db, title="Test Doc", content="One two three four five.")

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    async def _override_db() -> AsyncGenerator[MediaDbLike, None]:
        with managed_test_media_db(
            "test_client",
            db_path=str(db_path),
            initialize=False,
        ) as override_db:
            yield override_db

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                f"/api/v1/media/{media_id}/reprocess",
                json={
                    "perform_chunking": True,
                    "chunk_method": "sentences",
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                    "generate_embeddings": False,
                },
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["media_id"] == media_id
            assert data["status"] == "completed"
            assert isinstance(data["chunks_created"], int)
            assert data["chunks_created"] >= 1
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    with managed_test_media_db(
        "test_client",
        db_path=str(db_path),
        initialize=False,
    ) as check_db:
        row = check_db.execute_query(
            "SELECT count(*) AS c FROM UnvectorizedMediaChunks WHERE media_id = ?",
            (media_id,),
        ).fetchone()
    count_val = row["c"] if isinstance(row, dict) else row[0]
    assert count_val >= 1


def test_reprocess_missing_media_returns_404(tmp_path, managed_test_media_db):

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    db_path = tmp_path / "media.db"
    with managed_test_media_db(
        "test_client",
        db_path=str(db_path),
        initialize=False,
    ):
        pass

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    async def _override_db() -> AsyncGenerator[MediaDbLike, None]:
        with managed_test_media_db(
            "test_client",
            db_path=str(db_path),
            initialize=False,
        ) as override_db:
            yield override_db

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                "/api/v1/media/9999/reprocess",
                json={"perform_chunking": True},
            )
            assert resp.status_code == 404, resp.text
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_reprocess_chunking_returns_400_for_input_error(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    monkeypatch.setattr(reprocess_endpoint, "apply_chunking_template_if_any", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        reprocess_endpoint,
        "improved_chunking_process",
        lambda _content, _options: [{"text": "Chunk one", "chunk_index": 0}],
    )

    class _InputErrorChunkingDb:
        def get_media_by_id(self, media_id: int) -> dict:
            return {
                "id": media_id,
                "content": "Chunk me.",
                "type": "document",
                "title": "Doc",
                "url": None,
                "filename": None,
            }

        def clear_unvectorized_chunks(self, media_id: int) -> int:
            raise InputError(f"invalid reprocess target for media {media_id}")

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = lambda: _InputErrorChunkingDb()

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                "/api/v1/media/101/reprocess",
                json={
                    "perform_chunking": True,
                    "chunk_method": "sentences",
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                    "generate_embeddings": False,
                },
            )
            assert resp.status_code == 400, resp.text
            assert resp.json()["detail"] == "invalid reprocess target for media 101"
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_reprocess_chunking_database_error_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    monkeypatch.setattr(reprocess_endpoint, "apply_chunking_template_if_any", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        reprocess_endpoint,
        "improved_chunking_process",
        lambda _content, _options: [{"text": "Chunk one", "chunk_index": 0}],
    )

    class _DatabaseErrorChunkingDb:
        def get_media_by_id(self, media_id: int) -> dict:
            return {
                "id": media_id,
                "content": "Chunk me.",
                "type": "document",
                "title": "Doc",
                "url": None,
                "filename": None,
            }

        def clear_unvectorized_chunks(self, media_id: int) -> int:
            _ = media_id
            raise DatabaseError("sqlite driver leaked while clearing chunks")

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = lambda: _DatabaseErrorChunkingDb()

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                "/api/v1/media/102/reprocess",
                json={
                    "perform_chunking": True,
                    "chunk_method": "sentences",
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                    "generate_embeddings": False,
                },
            )
            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to update media chunks"
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_reprocess_chunking_failure_sanitizes_log(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    logger_stub = _patch_reprocess_logger(monkeypatch)

    monkeypatch.setattr(reprocess_endpoint, "apply_chunking_template_if_any", lambda *args, **kwargs: {})

    def _raise_chunking_failure(_content, _options):
        raise RuntimeError("chunker backend leaked /private/chunker/token")

    monkeypatch.setattr(
        reprocess_endpoint,
        "improved_chunking_process",
        _raise_chunking_failure,
    )

    class _ChunkingFailureDb:
        def get_media_by_id(self, media_id: int) -> dict:
            return {
                "id": media_id,
                "content": "Chunk me.",
                "type": "document",
                "title": "Doc",
                "url": None,
                "filename": None,
            }

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = lambda: _ChunkingFailureDb()

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                "/api/v1/media/103/reprocess",
                json={
                    "perform_chunking": True,
                    "chunk_method": "sentences",
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                    "generate_embeddings": False,
                },
            )
            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to re-chunk media content."
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    _assert_sanitized_error_log(
        logger_stub,
        "Chunking failed for media {}",
    )


def test_reprocess_embeddings_only_returns_400_for_input_error():
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    class _InputErrorEmbeddingsDb:
        def get_media_by_id(self, media_id: int) -> dict:
            return {
                "id": media_id,
                "content": "Embed me.",
                "type": "document",
                "title": "Doc",
                "url": None,
                "filename": None,
            }

        def update_media_reprocess_state(self, media_id: int, **_kwargs) -> None:
            raise InputError(f"media {media_id} cannot be reprocessed")

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = lambda: _InputErrorEmbeddingsDb()

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                "/api/v1/media/202/reprocess",
                json={
                    "perform_chunking": False,
                    "generate_embeddings": True,
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                },
            )
            assert resp.status_code == 400, resp.text
            assert resp.json()["detail"] == "media 202 cannot be reprocessed"
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_reprocess_embeddings_state_database_error_is_sanitized():
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    class _DatabaseErrorEmbeddingsDb:
        def get_media_by_id(self, media_id: int) -> dict:
            return {
                "id": media_id,
                "content": "Embed me.",
                "type": "document",
                "title": "Doc",
                "url": None,
                "filename": None,
            }

        def update_media_reprocess_state(self, media_id: int, **_kwargs) -> None:
            _ = media_id
            raise DatabaseError("sqlite driver leaked while updating reprocess state")

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = lambda: _DatabaseErrorEmbeddingsDb()

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                "/api/v1/media/203/reprocess",
                json={
                    "perform_chunking": False,
                    "generate_embeddings": True,
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                },
            )
            assert resp.status_code == 500, resp.text
            assert resp.json()["detail"] == "Failed to update media reprocess state"
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_reprocess_force_regenerate_embeddings_delete_failure_sanitizes_log(tmp_path, monkeypatch):
    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint

    logger_stub = _patch_reprocess_logger(monkeypatch)
    monkeypatch.setenv("TEST_MODE", "1")

    def _raise_delete_embeddings_for_media(media_id: int, user_id: str) -> None:
        _ = media_id, user_id
        raise RuntimeError("embedding backend leaked /private/vector/token")

    async def _fake_generate_embeddings_for_media(**_kwargs):
        return {"status": "success", "embedding_count": 1, "chunks_processed": 1}

    monkeypatch.setattr(
        reprocess_endpoint,
        "_delete_embeddings_for_media",
        _raise_delete_embeddings_for_media,
    )
    monkeypatch.setattr(
        reprocess_endpoint.embeddings_endpoint,
        "generate_embeddings_for_media",
        _fake_generate_embeddings_for_media,
    )
    monkeypatch.setattr(reprocess_endpoint, "invalidate_rag_caches", lambda *_, **__: None)

    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    media_id = create_test_media(seed_db, title="Embeddings Doc", content="Embeddings should still regenerate.")
    seed_db.close_connection()

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        try:
            yield override_db
        finally:
            override_db.close_connection()

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                f"/api/v1/media/{media_id}/reprocess",
                json={
                    "perform_chunking": False,
                    "generate_embeddings": True,
                    "force_regenerate_embeddings": True,
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "completed"
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    _assert_sanitized_warning_log(
        logger_stub,
        "Failed to delete embeddings before regeneration",
    )


def test_generate_embeddings_sanitizes_persisted_error(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint

    logger_stub = _patch_reprocess_logger(monkeypatch)

    async def _raise_generate_embeddings_for_media(**_kwargs):
        raise RuntimeError("embedding backend leaked /private/vector/token")

    monkeypatch.setattr(
        reprocess_endpoint.embeddings_endpoint,
        "generate_embeddings_for_media",
        _raise_generate_embeddings_for_media,
    )

    class _EmbeddingsErrorDb:
        def __init__(self) -> None:
            self.errors: list[tuple[int, str]] = []

        def mark_embeddings_error(self, media_id: int, error_detail: str) -> None:
            self.errors.append((media_id, error_detail))

    db = _EmbeddingsErrorDb()
    request = reprocess_endpoint.ReprocessMediaRequest(
        perform_chunking=False,
        generate_embeddings=True,
    )

    try:
        asyncio.run(
            reprocess_endpoint._generate_embeddings(
                media_id=321,
                media_payload={"content": "Embed me."},
                request=request,
                user_id="1",
                db=db,
                cache_namespaces=[],
            )
        )
    except RuntimeError as exc:
        assert "embedding backend leaked" in str(exc)
    else:
        raise AssertionError("expected embedding regeneration failure")

    assert db.errors == [(321, "Embeddings regeneration failed")]
    _assert_sanitized_error_log(
        logger_stub,
        "Embeddings regeneration failed",
    )


def test_generate_embeddings_sanitizes_error_status_update_failure(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint

    logger_stub = _patch_reprocess_logger(monkeypatch)

    async def _raise_generate_embeddings_for_media(**_kwargs):
        raise RuntimeError("embedding backend leaked /private/vector/token")

    monkeypatch.setattr(
        reprocess_endpoint.embeddings_endpoint,
        "generate_embeddings_for_media",
        _raise_generate_embeddings_for_media,
    )

    class _EmbeddingsErrorUpdateFailureDb:
        def mark_embeddings_error(self, media_id: int, error_detail: str) -> None:
            raise RuntimeError("embedding status backend leaked /private/vector/token")

    request = reprocess_endpoint.ReprocessMediaRequest(
        perform_chunking=False,
        generate_embeddings=True,
    )

    try:
        asyncio.run(
            reprocess_endpoint._generate_embeddings(
                media_id=321,
                media_payload={"content": "Embed me."},
                request=request,
                user_id="1",
                db=_EmbeddingsErrorUpdateFailureDb(),
                cache_namespaces=[],
            )
        )
    except RuntimeError as exc:
        assert "embedding backend leaked" in str(exc)
    else:
        raise AssertionError("expected embedding regeneration failure")

    _assert_sanitized_error_log(
        logger_stub,
        "Embeddings regeneration failed",
    )
    _assert_sanitized_error_log(
        logger_stub,
        "Failed to mark embeddings error",
    )


def test_delete_embeddings_sanitizes_where_delete_fallback_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint

    logger_stub = _patch_reprocess_logger(monkeypatch)

    class _FallbackDeleteCollection:
        def __init__(self) -> None:
            self.get_calls = 0

        def delete(self, **kwargs) -> None:
            if "where" in kwargs:
                raise RuntimeError("embedding backend leaked /private/vector/token")

        def get(self, **_kwargs) -> dict:
            self.get_calls += 1
            if self.get_calls == 1:
                return {"ids": ["embedding-1"]}
            return {"ids": []}

    collection = _FallbackDeleteCollection()

    class _FakeChromaDBManager:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_or_create_collection(self, collection_name: str):
            return collection

    monkeypatch.setattr(reprocess_endpoint.embeddings_endpoint, "_user_embedding_config", lambda: {})
    monkeypatch.setattr(reprocess_endpoint.embeddings_endpoint, "ChromaDBManager", _FakeChromaDBManager)

    reprocess_endpoint._delete_embeddings_for_media(media_id=321, user_id="1")

    assert collection.get_calls == 2
    _assert_sanitized_warning_log(
        logger_stub,
        "Where-delete failed for media embeddings, falling back to id delete",
    )


def test_delete_embeddings_sanitizes_verify_failure_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint

    logger_stub = _patch_reprocess_logger(monkeypatch)

    class _VerifyFailureCollection:
        def delete(self, **_kwargs) -> None:
            return None

        def get(self, **_kwargs) -> dict:
            raise RuntimeError("embedding backend leaked /private/vector/token")

    class _FakeChromaDBManager:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_or_create_collection(self, collection_name: str):
            return _VerifyFailureCollection()

    monkeypatch.setattr(reprocess_endpoint.embeddings_endpoint, "_user_embedding_config", lambda: {})
    monkeypatch.setattr(reprocess_endpoint.embeddings_endpoint, "ChromaDBManager", _FakeChromaDBManager)

    reprocess_endpoint._delete_embeddings_for_media(media_id=321, user_id="1")

    _assert_sanitized_warning_log(
        logger_stub,
        "Failed to verify embeddings delete",
    )


def test_reprocess_embeddings_marks_vector_processed(tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    monkeypatch.setenv("TEST_MODE", "1")

    async def _fake_generate_embeddings_for_media(**_kwargs):
        return {"status": "success", "embedding_count": 1, "chunks_processed": 1}

    monkeypatch.setattr(
        reprocess_endpoint.embeddings_endpoint,
        "generate_embeddings_for_media",
        _fake_generate_embeddings_for_media,
    )
    monkeypatch.setattr(reprocess_endpoint, "invalidate_rag_caches", lambda *_, **__: None)

    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    media_id = create_test_media(seed_db, title="Embeddings Doc", content="Embeddings should flip ready state.")
    seed_db.close_connection()

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        try:
            yield override_db
        finally:
            override_db.close_connection()

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                f"/api/v1/media/{media_id}/reprocess",
                json={
                    "perform_chunking": False,
                    "generate_embeddings": True,
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "completed"
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    check_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    row = check_db.execute_query(
        "SELECT vector_processing FROM Media WHERE id = ?",
        (media_id,),
    ).fetchone()
    check_db.close_connection()
    vector_status = row["vector_processing"] if isinstance(row, dict) else row[0]
    assert vector_status == 1


def test_reprocess_embeddings_retries_mark_processed_conflicts(tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.media import reprocess as reprocess_endpoint
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    from tldw_Server_API.app.main import app as fastapi_app

    monkeypatch.setenv("TEST_MODE", "1")

    async def _fake_generate_embeddings_for_media(**_kwargs):
        return {"status": "success", "embedding_count": 1, "chunks_processed": 1}

    monkeypatch.setattr(
        reprocess_endpoint.embeddings_endpoint,
        "generate_embeddings_for_media",
        _fake_generate_embeddings_for_media,
    )
    monkeypatch.setattr(reprocess_endpoint, "invalidate_rag_caches", lambda *_, **__: None)

    original_mark_media_as_processed = reprocess_endpoint.mark_media_as_processed
    attempts = {"processed": 0}

    def _flaky_mark_media_as_processed(*, db_instance, media_id):  # noqa: ANN001
        attempts["processed"] += 1
        if attempts["processed"] == 1:
            raise ConflictError("Media", media_id)
        return original_mark_media_as_processed(db_instance=db_instance, media_id=media_id)

    monkeypatch.setattr(
        reprocess_endpoint,
        "mark_media_as_processed",
        _flaky_mark_media_as_processed,
    )

    db_path = tmp_path / "media.db"
    seed_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    media_id = create_test_media(
        seed_db, title="Embeddings Conflict Doc", content="Embeddings should survive a mark conflict."
    )
    seed_db.close_connection()

    async def _override_user() -> User:
        return User(id=1, username="tester", email=None, is_active=True)

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        try:
            yield override_db
        finally:
            override_db.close_connection()

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            resp = client.post(
                f"/api/v1/media/{media_id}/reprocess",
                json={
                    "perform_chunking": False,
                    "generate_embeddings": True,
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "completed"
    finally:
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)

    check_db = MediaDatabase(db_path=str(db_path), client_id="test_client")
    row = check_db.execute_query(
        "SELECT vector_processing, chunking_status FROM Media WHERE id = ?",
        (media_id,),
    ).fetchone()
    check_db.close_connection()
    vector_status = row["vector_processing"] if isinstance(row, dict) else row[0]
    chunking_status = row["chunking_status"] if isinstance(row, dict) else row[1]
    assert attempts["processed"] == 2
    assert vector_status == 1
    assert not str(chunking_status).startswith("embeddings_error: ConflictError")
