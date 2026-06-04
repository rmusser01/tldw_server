from contextlib import contextmanager

from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_service
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.config import settings


def test_claims_rebuild_health_reads_persisted(monkeypatch, tmp_path):
    base_dir = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("CLAIMS_MONITORING_SYSTEM_USER_ID", "1")
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setitem(settings, "CLAIMS_MONITORING_SYSTEM_USER_ID", 1)
    monkeypatch.setitem(claims_service.settings, "USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setitem(claims_service.settings, "CLAIMS_MONITORING_SYSTEM_USER_ID", 1)

    @contextmanager
    def _managed_media_database(client_id, *, db_path=None, initialize=True, **_kwargs):
        db = MediaDatabase(db_path=db_path, client_id=client_id)
        try:
            if initialize:
                db.initialize_db()
            yield db
        finally:
            db.close_connection()

    monkeypatch.setattr(
        claims_service,
        "managed_media_database",
        _managed_media_database,
        raising=False,
    )
    monkeypatch.setattr(
        claims_service,
        "create_media_database",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy raw factory should not be used")),
        raising=False,
    )

    db_path = get_user_media_db_path(1)
    monkeypatch.setattr(
        claims_service,
        "get_user_media_db_path",
        lambda user_id: db_path if int(user_id) == 1 else get_user_media_db_path(user_id),
        raising=False,
    )

    db = MediaDatabase(db_path=db_path, client_id="test")
    db.initialize_db()
    db.upsert_claims_monitoring_health(
        user_id="1",
        queue_size=7,
        worker_count=2,
        last_worker_heartbeat="2024-01-01T00:00:00.000Z",
        last_processed_at="2024-01-01T00:00:05.000Z",
        last_failure_at="2024-01-01T00:00:10.000Z",
        last_failure_reason="boom",
    )
    db.close_connection()

    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        subject="admin",
        roles=["admin"],
        permissions=[CLAIMS_ADMIN],
        is_admin=True,
    )
    payload = claims_service.claims_rebuild_health(principal)
    assert payload["queue_length"] == 7
    assert payload["workers"] == 2
    assert payload["last_worker_heartbeat"] == "2024-01-01T00:00:00.000Z"
    assert payload["last_failure"]["error"] == "boom"
