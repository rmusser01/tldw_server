import hashlib
import os
import tempfile
import time
from queue import Queue
from typing import Any, AsyncGenerator

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import get_claims_rebuild_service
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext


def _setup_db_with_claims() -> tuple[str, int]:


    tmpdir = tempfile.mkdtemp(prefix="claims_env_")
    db_path = os.path.join(tmpdir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="test_client")
    db.initialize_db()
    content = "S1. S2. S3."
    media_id, _, _ = db.add_media_with_keywords(title="Doc", media_type="text", content=content, keywords=None)
    chunk_hash = hashlib.sha256(content.encode()).hexdigest()
    rows = []
    for i, txt in enumerate(["C1", "C2", "C3"]):
        rows.append({
            "media_id": media_id,
            "chunk_index": i,
            "span_start": None,
            "span_end": None,
            "claim_text": txt,
            "confidence": 0.9,
            "extractor": "heuristic",
            "extractor_version": "v1",
            "chunk_hash": chunk_hash,
        })
    db.upsert_claims(rows)
    db.close_connection()
    return db_path, media_id


def _setup_db_for_policies() -> tuple[str, dict[str, int]]:


    tmpdir = tempfile.mkdtemp(prefix="claims_policy_")
    db_path = os.path.join(tmpdir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="test_client")
    db.initialize_db()

    def _add_media_with_claim(text: str) -> int:
        mid, _, _ = db.add_media_with_keywords(title=text, media_type="text", content=text, keywords=None)
        chunk_hash = hashlib.sha256(text.encode()).hexdigest()
        db.upsert_claims([{
            "media_id": mid,
            "chunk_index": 0,
            "span_start": None,
            "span_end": None,
            "claim_text": text,
            "confidence": 0.9,
            "extractor": "heuristic",
            "extractor_version": "v1",
            "chunk_hash": chunk_hash,
        }])
        return mid

    with_claims = _add_media_with_claim("already claimed")

    missing_mid, _, _ = db.add_media_with_keywords(
        title="missing", media_type="text", content="No claims yet.", keywords=None
    )

    stale_mid = _add_media_with_claim("stale claims")
    cur = db.execute_query("SELECT version FROM Media WHERE id = ?", (stale_mid,))
    row = cur.fetchone()
    if row is None:
        current_version = 1
    else:
        try:
            current_version = int(row["version"])  # type: ignore[index]
        except (TypeError, KeyError):
            current_version = int(row[0])  # type: ignore[index]
    db.execute_query(
        "UPDATE Media SET last_modified = ?, version = ? WHERE id = ?",
        ("9999-01-01T00:00:00Z", current_version + 1, stale_mid),
        commit=True,
    )

    db.close_connection()
    return db_path, {"with_claims": with_claims, "missing": missing_mid, "stale": stale_mid}


def _principal_override(is_admin: bool):
    async def _override(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="admin",
            token_type="access",
            jti=None,
            roles=["admin"] if is_admin else ["user"],
            permissions=["claims.status"],
            is_admin=is_admin,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception:
                _ = None
        return principal

    return _override


def _reset_claims_rebuild_service():


    svc = get_claims_rebuild_service()
    svc.stop()
    svc._queue = Queue()  # type: ignore[attr-defined]
    with svc._stats_lock:  # type: ignore[attr-defined]
        svc._stats = {"enqueued": 0, "processed": 0, "failed": 0}  # type: ignore[attr-defined]
    svc.start()
    return svc


def _wait_for_service_completion(expected_processed: int, timeout: float = 5.0, poll: float = 0.05) -> None:
    svc = get_claims_rebuild_service()
    deadline = time.time() + timeout
    while time.time() < deadline:
        stats = svc.get_stats()
        processed = stats.get("processed", 0) + stats.get("failed", 0)
        if processed >= expected_processed:
            unfinished = getattr(svc._queue, "unfinished_tasks", 0)  # type: ignore[attr-defined]
            if svc.get_queue_length() == 0 and unfinished == 0:
                return
        time.sleep(poll)
    raise AssertionError("Claims rebuild worker did not finish within timeout")


def test_claims_status_admin_ok():


    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    class _Admin:
        def __init__(self) -> None:
            self.id = 1
            self.username = "admin"
            self.is_admin = True

    async def _override_user():
        return _Admin()

    # DB not required for status, but keep consistent override
    db_path, _ = _setup_db_with_claims()

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=db_path, client_id="test_client")
        try:
            yield override_db
        finally:
            try:
                override_db.close_connection()
            except Exception:
                _ = None

    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override(True)
    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            r = client.get("/api/v1/claims/status")
            assert r.status_code == 200, r.text
            data = r.json()
            assert data.get("status") == "ok"
            assert isinstance(data.get("stats", {}), dict)
            assert isinstance(data.get("queue_length"), int)
            # workers may be None or int depending on initialized state
            assert data.get("workers") is None or isinstance(data.get("workers"), int)
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_claims_envelope_pagination_absolute_link():


    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    class _Admin:
        def __init__(self) -> None:
            self.id = 1
            self.username = "admin"
            self.is_admin = True

    db_path, media_id = _setup_db_with_claims()

    async def _override_user():
        return _Admin()

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=db_path, client_id="test_client")
        try:
            yield override_db
        finally:
            try:
                override_db.close_connection()
            except Exception:
                _ = None

    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    with TestClient(fastapi_app) as client:
        r1 = client.get(f"/api/v1/claims/{media_id}", params={"limit": 1, "offset": 0, "envelope": True, "absolute_links": True})
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        assert isinstance(body1.get("total"), int)
        assert isinstance(body1.get("total_pages"), int)
        assert body1.get("pagination") == {
            "mode": "offset",
            "limit": 1,
            "offset": 0,
            "total": body1.get("total"),
            "has_more": body1.get("next_offset") is not None,
            "next_offset": body1.get("next_offset"),
        }
        next_link = body1.get("next_link")
        if body1.get("next_offset") is not None:
            assert isinstance(next_link, str) and next_link.startswith("http")
            r2 = client.get(next_link)
            assert r2.status_code == 200, r2.text
            body2 = r2.json()
            assert body2.get("items"), "Second page items missing"


def test_claims_status_forbidden_non_admin():


    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    class _User:
        def __init__(self) -> None:
            self.id = 1
            self.username = "u"
            self.is_admin = False

    async def _override_user():
        return _User()

    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override(False)
    fastapi_app.dependency_overrides[get_request_user] = _override_user

    try:
        with TestClient(fastapi_app) as client:
            r = client.get("/api/v1/claims/status")
            assert r.status_code == 403
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)


def test_claims_status_reports_queue_activity():


    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    class _Admin:
        def __init__(self) -> None:
            self.id = 1
            self.username = "admin"
            self.is_admin = True

    svc = _reset_claims_rebuild_service()
    db_path, media_id = _setup_db_with_claims()

    async def _override_user():
        return _Admin()

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=db_path, client_id="test_client")
        try:
            yield override_db
        finally:
            try:
                override_db.close_connection()
            except Exception:
                _ = None

    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override(True)
    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            initial = client.get("/api/v1/claims/status")
            assert initial.status_code == 200
            init_body = initial.json()
            assert init_body.get("stats", {}).get("enqueued", 0) == 0

            r = client.post(f"/api/v1/claims/{media_id}/rebuild")
            assert r.status_code == 200, r.text

            _wait_for_service_completion(expected_processed=1)

            status = client.get("/api/v1/claims/status")
            assert status.status_code == 200, status.text
            data = status.json()
            stats = data.get("stats", {})
            assert stats.get("enqueued") == 1
            assert stats.get("processed") == 1
            assert data.get("queue_length") == 0
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
        svc.stop()
        svc.start()


def test_claims_rebuild_policies_enqueue_expected_media():


    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user

    class _Admin:
        def __init__(self) -> None:
            self.id = 1
            self.username = "admin"
            self.is_admin = True

    async def _override_user():
        return _Admin()

    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override(True)
    fastapi_app.dependency_overrides[get_request_user] = _override_user

    policies = ["missing", "stale", "all"]

    try:
        for policy in policies:
            db_path, media_ids = _setup_db_for_policies()
            total_media = len(media_ids)
            expected_map = {
                "missing": 1,
                "stale": len({media_ids["missing"], media_ids["stale"]}),
                "all": total_media,
            }
            expected = expected_map[policy]

            async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
                override_db = MediaDatabase(db_path=db_path, client_id="test_client")
                try:
                    yield override_db
                finally:
                    try:
                        override_db.close_connection()
                    except Exception:
                        _ = None

            fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

            svc = _reset_claims_rebuild_service()

            with TestClient(fastapi_app) as client:
                response = client.post("/api/v1/claims/rebuild/all", params={"policy": policy})
                assert response.status_code == 200, response.text
                body = response.json()
                assert body.get("status") == "accepted"
                assert body.get("enqueued") == expected
                assert body.get("policy") == policy

                _wait_for_service_completion(expected_processed=expected)

                status = client.get("/api/v1/claims/status")
                assert status.status_code == 200, status.text
                stats = status.json().get("stats", {})
                assert stats.get("enqueued") == expected
                assert stats.get("processed") == expected

            svc.stop()
            svc.start()
    finally:
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
