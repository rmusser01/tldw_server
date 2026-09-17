"""Selected-owner StudyPack and provenance reads on real storage and routes."""

from types import SimpleNamespace

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
        with chacha_operation(independent=True):
            if child == "citation":
                other.add_flashcard_citations(
                    ids["card"],
                    [{"source_type": "note", "source_id": other_ids["note"], "citation_text": "Other owner metadata"}],
                )
            else:
                other.add_study_pack_cards(other_ids["pack"], [ids["card"]])
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
@pytest.mark.parametrize("parent", ["deck", "card", "replacement-pack"])
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
            else:
                foreign.supersede_study_pack(other["pack"], superseded_by_pack_id=ids["pack"], expected_version=1)
        except (CharactersRAGDBError, InputError) as exc:
            if kind == "postgres":
                assert isinstance(exc, (ConflictError, InputError))
            rejected = True
    if kind == "postgres":
        assert rejected
    with chacha_operation(independent=True):
        assert owner.get_study_pack(ids["pack"])["client_id"] == "2"
        assert owner.get_flashcard(ids["card"])["client_id"] == "2"
