"""Selected-owner StudyPack and provenance reads on real storage and routes."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_Server_API.app.api.v1.endpoints import flashcards as endpoint
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.tests.StudyPacks.test_study_pack_response_timestamps import pack_api as pack_api


def _seed(db, *, membership=True, source_id=None):
    with chacha_operation(independent=True):
        note = source_id or db.add_note(title="Private pack source", content="Synthetic owner-only citation")
        deck = db.add_deck(f"Private pack deck {db.client_id}")
        card = db.add_flashcard({"deck_id": deck, "front": "Private question", "back": "Private answer"})
        pack = db.create_study_pack(
            title="Private pack",
            workspace_id=None,
            deck_id=deck,
            source_bundle_json={"items": [{"source_type": "note", "source_id": note}]},
            generation_options_json={"deck_mode": "new"},
        )
        if membership:
            db.add_study_pack_cards(pack, [card])
        db.add_flashcard_citations(
            card, [{"source_type": "note", "source_id": note, "citation_text": "Synthetic owner-only citation"}]
        )
        return {"note": note, "deck": deck, "card": card, "pack": pack}


def _actor(client, db, actor):
    client.app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: db
    client.app.dependency_overrides[endpoint.get_request_user] = lambda: SimpleNamespace(id=actor)
    client.app.dependency_overrides[endpoint.get_auth_principal] = lambda: AuthPrincipal(
        kind="user", user_id=actor, roles=[], permissions=[]
    )


@pytest.fixture
def pack_owners(pack_api, tmp_path):
    db, jobs, client, kind = pack_api
    foreign = CharactersRAGDB(
        tmp_path / "foreign.db", client_id="3", backend=db.backend if kind == "postgres" else None
    )
    try:
        yield db, foreign, jobs, client, kind
    finally:
        foreign.close_connection()


@pytest.mark.integration
@pytest.mark.parametrize("route", ["detail", "regenerate"])
def test_private_pack_http_hides_foreign_pack(pack_owners, route):
    owner, foreign, _jobs, client, _kind = pack_owners
    ids = _seed(owner)
    path = f"/api/v1/flashcards/study-packs/{ids['pack']}"
    own = client.get(path)
    assert own.status_code == 200 and own.json()["client_id"] == "2"
    _actor(client, foreign, 3)
    response = client.get(path) if route == "detail" else client.post(path + "/regenerate")
    assert response.status_code == 404, response.json()
    _actor(client, owner, 2)
    assert client.get(path).json() == own.json()


@pytest.mark.integration
@pytest.mark.parametrize(
    "method", ["get_study_pack", "list_study_pack_cards", "list_flashcard_citations", "get_study_pack_for_flashcard"]
)
def test_selected_store_hides_foreign_pack_and_provenance(pack_owners, method):
    owner, foreign, _jobs, _client, kind = pack_owners
    ids = _seed(owner)
    key = ids["pack"] if method in {"get_study_pack", "list_study_pack_cards"} else ids["card"]
    with chacha_operation(independent=True):
        assert getattr(owner, method)(key)
        result = getattr(foreign, method)(key)
    assert result == ([] if method.startswith("list_") else None)


@pytest.mark.integration
@pytest.mark.parametrize("child", ["citation", "membership"])
def test_owned_card_assistant_filters_foreign_child_metadata(pack_api, tmp_path, child):
    owner, _jobs, client, kind = pack_api
    ids = _seed(owner, membership=False)
    # PostgreSQL instances select separate owners in one shared DB. On SQLite,
    # two labels on this same file are sync devices and remain mutually visible.
    other = CharactersRAGDB(owner.db_path, client_id="3", backend=owner.backend if kind == "postgres" else None)
    try:
        other_ids = _seed(other, membership=False)
        # Historical malformed children remain possible after writer validation.
        # The original RED using public writer methods is retained in the packet.
        with chacha_operation(independent=True), other.transaction() as conn:
            now = other._get_current_utc_timestamp_iso()
            if child == "citation":
                conn.execute(
                    "INSERT INTO flashcard_citations(flashcard_uuid, source_type, source_id, citation_text, ordinal, created_at, last_modified, deleted, client_id, version) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ids["card"], "note", other_ids["note"], "Other owner metadata", 1, now, now, False, "3", 1),
                )
            else:
                conn.execute(
                    "INSERT INTO study_pack_cards(study_pack_id, flashcard_uuid, created_at, last_modified, deleted, client_id, version) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (other_ids["pack"], ids["card"], now, now, False, "3", 1),
                )
        response = client.get(f"/api/v1/flashcards/{ids['card']}/assistant")
        assert response.status_code == 200
        body = response.json()
        if kind == "postgres":
            assert all(row["client_id"] == "2" for row in body["citations"])
            assert body["study_pack"] is None
        elif child == "citation":
            assert {row["client_id"] for row in body["citations"]} == {"2", "3"}
        else:
            assert body["study_pack"]["id"] == other_ids["pack"]
    finally:
        other.close_connection()


def test_sqlite_same_file_keeps_study_pack_sync_device_access(tmp_path):
    path = tmp_path / "device-owned.db"
    first = CharactersRAGDB(path, client_id="device-a")
    second = CharactersRAGDB(path, client_id="device-b")
    try:
        ids = _seed(first, source_id="synthetic-device-source")
        assert second.get_study_pack(ids["pack"])["client_id"] == "device-a"
        assert second.list_study_pack_cards(ids["pack"])[0]["flashcard_uuid"] == ids["card"]
        assert second.list_flashcard_citations(ids["card"])[0]["client_id"] == "device-a"
        assert second.get_study_pack_for_flashcard(ids["card"])["id"] == ids["pack"]
        assert (
            second.replace_flashcard_citations_and_source_reference_summary(
                ids["card"],
                [{"source_type": "note", "source_id": "device-source"}],
                source_ref_type="note",
                source_ref_id="device-source",
            )
            == 1
        )
        assert first.get_flashcard(ids["card"])["client_id"] == "device-b"
        assert second.soft_delete_study_pack(ids["pack"], expected_version=1)
        assert first.get_study_pack(ids["pack"]) is None
    finally:
        first.close_all_connections()
        second.close_all_connections()


@pytest.mark.integration
@pytest.mark.parametrize(
    "operation",
    [
        "delete-pack",
        "supersede-pack",
        "add-membership",
        "add-citation",
        "replace-citations",
        "replace-and-summary",
        "set-summary",
    ],
)
def test_foreign_study_pack_provenance_mutation_cannot_change_owner_rows(pack_owners, operation):
    owner, foreign, _jobs, _client, kind = pack_owners
    ids = _seed(owner)
    other = _seed(foreign)

    def snapshot():
        with chacha_operation(independent=True):
            return (
                owner.get_study_pack(ids["pack"]),
                owner.get_flashcard(ids["card"]),
                owner.list_study_pack_cards(ids["pack"], include_deleted=True),
                owner.list_flashcard_citations(ids["card"], include_deleted=True),
            )

    before = snapshot()
    citation = {"source_type": "note", "source_id": other["note"], "citation_text": "Attempted foreign mutation"}
    with chacha_operation(independent=True):
        try:
            if operation == "delete-pack":
                foreign.soft_delete_study_pack(ids["pack"], expected_version=1)
            elif operation == "supersede-pack":
                foreign.supersede_study_pack(ids["pack"], superseded_by_pack_id=other["pack"], expected_version=1)
            elif operation == "add-membership":
                foreign.add_study_pack_cards(ids["pack"], [other["card"]])
            elif operation == "add-citation":
                foreign.add_flashcard_citations(ids["card"], [citation])
            elif operation == "replace-citations":
                foreign.replace_flashcard_citations(ids["card"], [citation])
            elif operation == "replace-and-summary":
                foreign.replace_flashcard_citations_and_source_reference_summary(
                    ids["card"], [citation], source_ref_type="note", source_ref_id=other["note"]
                )
            else:
                foreign.set_flashcard_source_reference_summary(
                    ids["card"], source_ref_type="note", source_ref_id=other["note"]
                )
        except (CharactersRAGDBError, InputError) as exc:
            if kind == "postgres":
                assert isinstance(exc, (ConflictError, InputError))
    assert snapshot() == before


@pytest.mark.integration
@pytest.mark.parametrize("parent", ["deck", "card", "replacement-pack", "initial-replacement-pack"])
def test_postgres_study_pack_cannot_reference_foreign_parent(pack_owners, parent):
    owner, foreign, _jobs, _client, kind = pack_owners
    ids = _seed(owner)
    other = _seed(foreign, membership=False)
    rejected = False
    with chacha_operation(independent=True):
        try:
            if parent == "deck":
                foreign.create_study_pack(
                    title="Foreign destination",
                    workspace_id=None,
                    deck_id=ids["deck"],
                    source_bundle_json={"items": []},
                    generation_options_json={},
                )
            elif parent == "card":
                foreign.add_study_pack_cards(other["pack"], [ids["card"]])
            elif parent == "replacement-pack":
                foreign.supersede_study_pack(other["pack"], superseded_by_pack_id=ids["pack"], expected_version=1)
            else:
                foreign.create_study_pack(
                    title="Foreign successor",
                    workspace_id=None,
                    deck_id=other["deck"],
                    source_bundle_json={},
                    generation_options_json={},
                    superseded_by_pack_id=ids["pack"],
                )
        except (CharactersRAGDBError, InputError) as exc:
            if kind == "postgres":
                assert isinstance(exc, (ConflictError, InputError))
            rejected = True
    if kind == "postgres":
        assert rejected
    with chacha_operation(independent=True):
        assert owner.get_study_pack(ids["pack"])["client_id"] == "2"
        assert owner.get_flashcard(ids["card"])["client_id"] == "2"


def test_owned_pack_lifecycle_preserves_workspace_versions_and_empty_inputs(pack_api):
    db, _jobs, client, _kind = pack_api
    ids = _seed(db)
    with chacha_operation(independent=True):
        db.upsert_workspace("owned-workspace", "Owned workspace")
        deck = db.add_deck("Owned workspace destination", workspace_id="owned-workspace")
        replacement = db.create_study_pack(
            title="Owned successor",
            workspace_id="owned-workspace",
            deck_id=deck,
            source_bundle_json={"items": []},
            generation_options_json={"deck_mode": "new"},
        )
        assert db.get_study_pack(replacement)["workspace_id"] == "owned-workspace"
        assert db.add_study_pack_cards(ids["pack"], []) == 0
        assert db.add_flashcard_citations(ids["card"], []) == 0
        assert db.add_study_pack_cards(ids["pack"], [ids["card"], ids["card"]]) == 0
        assert db.replace_flashcard_citations(ids["card"], []) == 0
        assert db.list_flashcard_citations(ids["card"]) == []
        assert db.list_flashcard_citations(ids["card"], include_deleted=True)
        citation = {"source_type": "note", "source_id": ids["note"], "citation_text": "Owned replacement"}
        assert (
            db.replace_flashcard_citations_and_source_reference_summary(
                ids["card"], [citation], source_ref_type="note", source_ref_id=ids["note"]
            )
            == 1
        )
        assert db.set_flashcard_source_reference_summary(ids["card"], source_ref_type=None, source_ref_id=None)
        with pytest.raises(ConflictError):
            db.supersede_study_pack(ids["pack"], superseded_by_pack_id=replacement, expected_version=99)
        assert db.supersede_study_pack(ids["pack"], superseded_by_pack_id=replacement, expected_version=1)
        assert db.get_study_pack(ids["pack"])["version"] == 2
        with pytest.raises(ConflictError):
            db.soft_delete_study_pack(ids["pack"], expected_version=1)
        assert db.soft_delete_study_pack(ids["pack"], expected_version=2)
        assert db.soft_delete_study_pack(ids["pack"], expected_version=2)
        assert db.get_study_pack(ids["pack"]) is None
    assert client.get(f"/api/v1/flashcards/study-packs/{ids['pack']}").status_code == 404
    assert client.get("/api/v1/flashcards/study-packs/9999999").status_code == 404


def test_membership_denial_preserves_prior_caller_work_and_rolls_back_all_writes(pack_owners):
    db, foreign, _jobs, _client, kind = pack_owners
    ids = _seed(db, membership=False)
    other = _seed(foreign)
    with chacha_operation(independent=True):
        before_card = db.get_flashcard(ids["card"])
        before_cites = db.list_flashcard_citations(ids["card"], include_deleted=True)
        before_sync = db.execute_query("SELECT COUNT(*) AS total FROM sync_log", read_only=True).fetchone()["total"]
        with pytest.raises(RuntimeError, match="caller rollback"):
            with db.transaction():
                marker = db.add_deck("Prior caller work")
                if kind == "postgres":
                    # PG owner validation precedes every membership insert. SQLite
                    # has no owner predicates and retains its outer rollback policy.
                    with pytest.raises(ConflictError):
                        db.add_study_pack_cards(ids["pack"], [ids["card"], other["card"]])
                assert db.get_deck(marker) is not None
                assert db.list_study_pack_cards(ids["pack"]) == []
                db.add_study_pack_cards(ids["pack"], [ids["card"]])
                db.replace_flashcard_citations_and_source_reference_summary(
                    ids["card"], [], source_ref_type=None, source_ref_id=None
                )
                raise RuntimeError("caller rollback")
        assert db.get_deck_by_name("Prior caller work") is None
        assert db.list_study_pack_cards(ids["pack"]) == []
        assert db.get_flashcard(ids["card"]) == before_card
        assert db.list_flashcard_citations(ids["card"], include_deleted=True) == before_cites
        assert (
            db.execute_query("SELECT COUNT(*) AS total FROM sync_log", read_only=True).fetchone()["total"]
            == before_sync
        )


def test_membership_write_failure_propagates_to_existing_caller_rollback(pack_api, monkeypatch):
    db, _jobs, _client, _kind = pack_api
    ids = _seed(db, membership=False)
    execute_many = db.execute_many

    def fail_after_write(query, params, **kwargs):
        execute_many(query, params, **kwargs)
        raise RuntimeError("injected write failure")

    with chacha_operation(independent=True):
        before_sync = db.execute_query("SELECT COUNT(*) AS total FROM sync_log", read_only=True).fetchone()["total"]
        with pytest.raises(RuntimeError, match="injected write failure"), db.transaction():
            db.add_deck("Rolled back caller marker")
            monkeypatch.setattr(db, "execute_many", fail_after_write)
            db.add_study_pack_cards(ids["pack"], [ids["card"]])
        assert db.get_deck_by_name("Rolled back caller marker") is None
        assert db.list_study_pack_cards(ids["pack"]) == []
        assert (
            db.execute_query("SELECT COUNT(*) AS total FROM sync_log", read_only=True).fetchone()["total"]
            == before_sync
        )


@pytest.mark.parametrize("atomic", [False, True])
def test_replacing_owned_citations_preserves_foreign_historical_rows(pack_api, atomic):
    db, _jobs, _client, kind = pack_api
    ids = _seed(db)
    with chacha_operation(independent=True), db.transaction() as conn:
        now = db._get_current_utc_timestamp_iso()
        conn.execute(
            "INSERT INTO flashcard_citations(flashcard_uuid, source_type, source_id, citation_text, ordinal, created_at, last_modified, deleted, client_id, version) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ids["card"], "note", "other-owner-source", "Historical private child", 1, now, now, False, "3", 1),
        )
    with chacha_operation(independent=True):
        before = dict(
            db.execute_query(
                "SELECT * FROM flashcard_citations WHERE source_id = ?", ("other-owner-source",), read_only=True
            ).fetchone()
        )
        replacement = [{"source_type": "note", "source_id": ids["note"], "citation_text": "Replacement"}]
        if atomic:
            db.replace_flashcard_citations_and_source_reference_summary(
                ids["card"], replacement, source_ref_type="note", source_ref_id=ids["note"]
            )
        else:
            db.replace_flashcard_citations(ids["card"], replacement)
        after = dict(
            db.execute_query(
                "SELECT * FROM flashcard_citations WHERE source_id = ?", ("other-owner-source",), read_only=True
            ).fetchone()
        )
        if kind == "postgres":
            assert after == before
        else:
            assert after["deleted"] and after["client_id"] == "2" and after["version"] == 2
        assert [row["citation_text"] for row in db.list_flashcard_citations(ids["card"])] == ["Replacement"]


def test_later_owned_membership_survives_earlier_foreign_history(pack_api):
    db, _jobs, client, kind = pack_api
    ids = _seed(db, membership=False)
    other = CharactersRAGDB(db.db_path, client_id="3", backend=db.backend if kind == "postgres" else None)
    try:
        other_ids = _seed(other, membership=False)
        with chacha_operation(independent=True), db.transaction() as conn:
            now = db._get_current_utc_timestamp_iso()
            conn.execute(
                "INSERT INTO study_pack_cards(study_pack_id, flashcard_uuid, created_at, last_modified, deleted, client_id, version) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (other_ids["pack"], ids["card"], now, now, False, "3", 1),
            )
            second_card = db.add_flashcard({"deck_id": ids["deck"], "front": "Second", "back": "Second answer"})
            ordered_cards = sorted([ids["card"], second_card], reverse=True)
            db.add_study_pack_cards(ids["pack"], ordered_cards)
        with chacha_operation(independent=True):
            assert [row["flashcard_uuid"] for row in db.list_study_pack_cards(ids["pack"])] == ordered_cards
            expected_pack = ids["pack"] if kind == "postgres" else other_ids["pack"]
            assert db.get_study_pack_for_flashcard(ids["card"])["id"] == expected_pack
        response = client.get(f"/api/v1/flashcards/{ids['card']}/assistant")
        assert response.status_code == 200
        assert response.json()["study_pack"]["id"] == expected_pack
    finally:
        other.close_connection()


@pytest.mark.parametrize("parent", ["missing-pack", "deleted-pack", "missing-card", "deleted-card"])
def test_membership_parent_validation_keeps_backend_deletion_contract(pack_api, parent):
    db, _jobs, _client, kind = pack_api
    ids = _seed(db, membership=False)
    pack, card = ids["pack"], ids["card"]
    with chacha_operation(independent=True):
        if parent == "missing-pack":
            pack = 9999999
        elif parent == "deleted-pack":
            db.soft_delete_study_pack(pack, expected_version=1)
        elif parent == "missing-card":
            card = "missing-card"
        else:
            db.soft_delete_flashcard(card, expected_version=1)
        if kind == "postgres":
            with pytest.raises(ConflictError):
                db.add_study_pack_cards(pack, [card])
        elif parent.startswith("missing"):
            with pytest.raises(CharactersRAGDBError):
                db.add_study_pack_cards(pack, [card])
        else:
            assert db.add_study_pack_cards(pack, [card]) == 1


def test_foreign_tombstones_are_hidden_even_when_deleted_rows_requested(pack_owners):
    db, foreign, _jobs, _client, _kind = pack_owners
    ids = _seed(db)
    with chacha_operation(independent=True), db.transaction() as conn:
        conn.execute("UPDATE study_pack_cards SET deleted=? WHERE study_pack_id=?", (True, ids["pack"]))
        conn.execute("UPDATE flashcard_citations SET deleted=? WHERE flashcard_uuid=?", (True, ids["card"]))
    with chacha_operation(independent=True):
        assert db.list_study_pack_cards(ids["pack"], include_deleted=True)
        assert db.list_flashcard_citations(ids["card"], include_deleted=True)
        assert foreign.list_study_pack_cards(ids["pack"], include_deleted=True) == []
        assert foreign.list_flashcard_citations(ids["card"], include_deleted=True) == []


def test_restricted_role_study_pack_access_is_owner_scoped(pack_owners):
    db, foreign, _jobs, _client, kind = pack_owners
    ids = _seed(db)
    other = _seed(foreign)
    if kind == "sqlite":
        assert foreign.get_study_pack(ids["pack"])["client_id"] == "3"
        return
    backend = db.backend
    role = backend.escape_identifier(f"uat222_reader_{uuid4().hex[:12]}")
    created = False
    try:
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {role}", connection=conn)
            backend.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}", connection=conn
            )
            backend.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}", connection=conn)
            backend.execute(f"GRANT {role} TO CURRENT_USER", connection=conn)
        created = True
        with chacha_operation(independent=True), foreign.transaction() as conn:
            conn.execute(f"SET LOCAL ROLE {role}")
            conn.execute("SELECT set_config('app.current_user_id', ?, true)", ("3",))
            flags = conn.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
            assert not flags["rolsuper"] and not flags["rolbypassrls"]
            assert foreign.get_study_pack(ids["pack"]) is None
            assert foreign.list_study_pack_cards(ids["pack"]) == []
            assert foreign.list_flashcard_citations(ids["card"]) == []
            assert foreign.get_study_pack_for_flashcard(ids["card"]) is None
            assert foreign.get_study_pack(other["pack"])["client_id"] == "3"
    finally:
        if created:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {role}", connection=conn)
                backend.execute(f"DROP ROLE {role}", connection=conn)
