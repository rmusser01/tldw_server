"""Study Suggestions do not cross owners in shared PostgreSQL storage."""

from contextlib import contextmanager
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgresql"])
def suggestion_owners(request, tmp_path):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgresql"
        else None
    )
    owner = CharactersRAGDB(tmp_path / "2.db", client_id="2", backend=backend)
    foreign = CharactersRAGDB(tmp_path / "3.db", client_id="3", backend=backend)
    snapshot_id = owner.create_suggestion_snapshot(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"display_label": "Private citrine lesson"}]},
    )
    owner.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id="pending:sel",
        selection_fingerprint="sel",
    )
    role = None
    role_created = False
    try:
        if backend is not None:
            role = backend.escape_identifier(f"suggestions_scope_{uuid4().hex[:12]}")
            with backend.transaction() as conn:
                backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
                backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
                backend.execute(
                    f"GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO {role}", connection=conn
                )
                backend.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}", connection=conn)
                backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
            role_created = True

        @contextmanager
        def restricted(db):
            with chacha_operation(independent=True), db.transaction() as conn:
                if role:
                    conn.execute(f"SET LOCAL ROLE {role}")
                    flags = dict(
                        conn.execute(
                            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user"
                        ).fetchone()
                    )
                    assert flags == {"rolsuper": False, "rolbypassrls": False}
                yield

        yield owner, foreign, snapshot_id, restricted
    finally:
        owner.close_connection()
        foreign.close_connection()
        if backend is not None:
            try:
                if role_created:
                    with backend.transaction() as conn:
                        backend.execute(f"DROP OWNED BY {role}", connection=conn)
                        backend.execute(f"DROP ROLE {role}", connection=conn)
            finally:
                backend.get_pool().close_all()
        else:
            owner.close_all_connections()
            foreign.close_all_connections()


@pytest.mark.parametrize(
    "operation",
    [
        "get",
        "list",
        "find",
        "find_fingerprint",
        "refresh",
        "create_link",
        "replace_link",
        "finalize",
        "delete",
        "release",
    ],
)
def test_foreign_suggestion_reads_and_writes_preserve_owner_data(suggestion_owners, operation):
    owner, foreign, snapshot_id, restricted = suggestion_owners
    link = {
        "snapshot_id": snapshot_id,
        "target_service": "flashcards",
        "target_type": "deck",
        "selection_fingerprint": "sel",
    }
    with restricted(foreign):
        if operation == "get":
            assert foreign.get_suggestion_snapshot(snapshot_id) is None
        elif operation == "list":
            assert foreign.list_suggestion_snapshots_for_anchor("flashcard_review_session", 101) == []
        elif operation == "find":
            assert foreign.find_suggestion_generation_link(**link, target_id="pending:sel") is None
        elif operation == "find_fingerprint":
            assert foreign.find_suggestion_generation_link_by_fingerprint(**link) is None
        elif operation == "refresh":
            with pytest.raises(ConflictError):
                foreign.create_suggestion_snapshot(
                    service="flashcards",
                    activity_type="flashcard_review_session",
                    anchor_type="flashcard_review_session",
                    anchor_id=101,
                    suggestion_type="study_suggestions",
                    payload_json={},
                    refreshed_from_snapshot_id=snapshot_id,
                )
        elif operation in {"create_link", "replace_link"}:
            error = InputError if foreign.backend_type is BackendType.POSTGRESQL else CharactersRAGDBError
            with pytest.raises(error):
                getattr(foreign, f"{operation.split('_')[0]}_suggestion_generation_link")(
                    **{**link, "selection_fingerprint": "foreign"},
                    target_id="foreign-deck",
                )
        elif operation == "finalize":
            assert foreign.finalize_suggestion_generation_link(**link, final_target_id="foreign-deck") == 0
        elif operation == "delete":
            assert foreign.soft_delete_suggestion_generation_link(**link) == 0
        elif operation == "release":
            foreign.release_suggestion_generation_link_reservation(**link)
    with restricted(owner):
        assert owner.get_suggestion_snapshot(snapshot_id)["client_id"] == "2"
        saved = owner.find_suggestion_generation_link_by_fingerprint(**link)
        assert saved is not None
        assert saved["target_id"] == "pending:sel"
        assert saved["client_id"] == "2"


def test_owned_snapshot_lineage_and_link_lifecycle(suggestion_owners):
    owner, _, snapshot_id, restricted = suggestion_owners
    link = {
        "snapshot_id": snapshot_id,
        "target_service": "flashcards",
        "target_type": "deck",
        "selection_fingerprint": "sel",
    }
    with restricted(owner):
        child = owner.create_suggestion_snapshot(
            service="flashcards",
            activity_type="flashcard_review_session",
            anchor_type="flashcard_review_session",
            anchor_id=101,
            suggestion_type="study_suggestions",
            payload_json={},
            refreshed_from_snapshot_id=snapshot_id,
        )
        assert owner.get_suggestion_snapshot(child)["refreshed_from_snapshot_id"] == snapshot_id
        assert len(owner.list_suggestion_snapshots_for_anchor("flashcard_review_session", 101)) == 2
        assert owner.finalize_suggestion_generation_link(**link, final_target_id="owned-deck") == 1
        owner.replace_suggestion_generation_link(**link, target_id="replacement")
        assert owner.find_suggestion_generation_link(**link, target_id="replacement") is not None
        assert owner.soft_delete_suggestion_generation_link(**link) == 1
        assert owner.find_suggestion_generation_link_by_fingerprint(**link) is None


def test_suggestion_http_owner_and_foreign_boundaries(suggestion_owners, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.endpoints import study_suggestions as endpoints
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    owner, foreign, snapshot_id, restricted = suggestion_owners
    actor = {"id": 3, "db": foreign}
    jobs = JobManager(db_path=tmp_path / "jobs.db")
    app = FastAPI()
    app.include_router(endpoints.router, prefix="/api/v1")

    async def database():
        with restricted(actor["db"]):
            yield actor["db"]

    async def principal():
        return AuthPrincipal(kind="user", user_id=actor["id"], roles=[], permissions=[], is_admin=False)

    async def user():
        return User(id=actor["id"], username="fixture", email="fixture@example.com", is_active=True)

    app.dependency_overrides[get_chacha_db_for_user] = database
    app.dependency_overrides[get_auth_principal] = principal
    app.dependency_overrides[get_request_user] = user
    app.dependency_overrides[endpoints.get_job_manager] = lambda: jobs
    url = f"/api/v1/study-suggestions/snapshots/{snapshot_id}"
    with TestClient(app) as client:
        assert client.get(url).status_code == 404
        assert client.post(url + "/refresh", json={}).status_code == 404
        assert (
            client.post(
                url + "/actions",
                json={
                    "target_service": "quiz",
                    "target_type": "quiz",
                    "action_kind": "follow_up_quiz",
                },
            ).status_code
            == 404
        )
        status = client.get("/api/v1/study-suggestions/anchors/flashcard_review_session/101/status")
        assert status.status_code == 200
        assert status.json()["status"] == "none"
        actor.update(id=2, db=owner)
        response = client.get(url)
        assert response.status_code == 200, response.text
        assert response.json()["snapshot"]["payload"]["topics"][0]["display_label"] == "Private citrine lesson"
        assert isinstance(response.json()["snapshot"]["created_at"], str)


@pytest.mark.parametrize("suggestion_owners", ["postgresql"], indirect=True)
@pytest.mark.parametrize("mismatch", ["foreign-parent", "foreign-link", "deleted-parent"])
def test_legacy_links_require_both_owners_and_live_parent(suggestion_owners, mismatch):
    owner, foreign, snapshot_id, restricted = suggestion_owners
    link = {
        "snapshot_id": snapshot_id,
        "target_service": "flashcards",
        "target_type": "deck",
        "selection_fingerprint": "sel",
    }
    actor = owner
    # Commit the legacy state before the independent restricted-role operation.
    with chacha_operation(independent=True), owner.transaction() as conn:
        if mismatch == "deleted-parent":
            conn.execute("UPDATE suggestion_snapshots SET deleted = TRUE WHERE id = ?", (snapshot_id,))
        else:
            conn.execute("UPDATE suggestion_generation_links SET client_id = ? WHERE snapshot_id = ?", ("3", snapshot_id))
            if mismatch == "foreign-parent":
                actor = foreign
    with restricted(actor):
        assert actor.find_suggestion_generation_link(**link, target_id="pending:sel") is None
        assert actor.find_suggestion_generation_link_by_fingerprint(**link) is None
        assert actor.finalize_suggestion_generation_link(**link, final_target_id="changed") == 0
        assert actor.soft_delete_suggestion_generation_link(**link) == 0
        actor.release_suggestion_generation_link_reservation(**link)
    row = owner.execute_query(
        "SELECT target_id, deleted FROM suggestion_generation_links WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone()
    assert row["target_id"] == "pending:sel"
    assert not row["deleted"]


def test_sqlite_snapshot_and_links_survive_device_identity_change(tmp_path):
    path = tmp_path / "legacy.db"
    previous = CharactersRAGDB(path, client_id="study-suggestions-worker-2")
    current = CharactersRAGDB(path, client_id="2")
    try:
        snapshot_id = previous.create_suggestion_snapshot(
            service="flashcards",
            activity_type="flashcard_review_session",
            anchor_type="flashcard_review_session",
            anchor_id=1,
            suggestion_type="study_suggestions",
            payload_json={},
        )
        link = {
            "snapshot_id": snapshot_id,
            "target_service": "flashcards",
            "target_type": "deck",
            "selection_fingerprint": "old",
        }
        previous.create_suggestion_generation_link(**link, target_id="pending:old")
        assert current.get_suggestion_snapshot(snapshot_id) is not None
        assert current.finalize_suggestion_generation_link(**link, final_target_id="deck-1") == 1
        assert current.find_suggestion_generation_link(**link, target_id="deck-1") is not None
    finally:
        previous.close_all_connections()
        current.close_all_connections()
