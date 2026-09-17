from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import flashcards as flashcards_endpoints
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Flashcards.study_assistant import build_flashcard_assistant_context


def _load_provenance_module():
    try:
        return import_module("tldw_Server_API.app.core.StudyPacks.provenance")
    except ModuleNotFoundError as exc:
        pytest.fail(f"StudyPacks provenance module is missing: {exc}")
    except ImportError as exc:
        pytest.fail(f"StudyPacks provenance imports are not yet usable: {exc}")


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-packs-provenance.db"), client_id="study-pack-provenance-tests")
    chacha.upsert_workspace("ws-1", "Workspace 1")
    try:
        yield chacha
    finally:
        chacha.close_connection()


def _store(db: CharactersRAGDB):
    module = _load_provenance_module()
    return module.FlashcardProvenanceStore(db)


@pytest.fixture
def assistant_client(db: CharactersRAGDB):
    app = FastAPI()
    app.include_router(flashcards_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def _create_card(db: CharactersRAGDB, *, front: str = "What is slow start?") -> str:
    deck_id = db.add_deck("Study Pack Provenance Deck", workspace_id="ws-1")
    return db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": front,
            "back": "A TCP congestion-control phase that grows the window exponentially.",
            "notes": "provenance fixture",
        }
    )


def _create_pack(db: CharactersRAGDB, *, deck_id: int) -> int:
    return db.create_study_pack(
        title="TCP Fundamentals",
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "note-seed"}]},
        generation_options_json={"deck_mode": "new"},
    )


def test_persist_flashcard_citations_assigns_ordinal_zero_to_deterministic_primary(db: CharactersRAGDB):
    card_uuid = _create_card(db)
    store = _store(db)

    provenance = store.persist_flashcard_citations(
        card_uuid,
        [
            {
                "source_type": "message",
                "source_id": "msg-secondary",
                "citation_text": "Congestion grows carefully after loss.",
            },
            {
                "source_type": "media",
                "source_id": "media-primary",
                "citation_text": "At 61 seconds the lecture explains exponential growth.",
                "locator": "00:01:01",
            },
            {
                "source_type": "note",
                "source_id": "note-tertiary",
                "citation_text": "Slow start doubles the congestion window each RTT.",
            },
        ],
    )

    citations = db.list_flashcard_citations(card_uuid)

    assert [citation["ordinal"] for citation in citations] == [0, 1, 2]  # nosec B101
    assert citations[0]["source_type"] == "media"  # nosec B101
    assert citations[0]["source_id"] == "media-primary"  # nosec B101
    assert provenance["primary_citation"]["ordinal"] == 0  # nosec B101
    assert provenance["primary_citation"]["source_id"] == "media-primary"  # nosec B101


def test_resolve_deep_dive_target_prefers_exact_locator_then_workspace_route_then_citation_fallback(db: CharactersRAGDB):
    module = _load_provenance_module()

    exact_locator = module.resolve_deep_dive_target(
        [
            {
                "source_type": "note",
                "source_id": "note-1",
                "ordinal": 0,
            },
            {
                "source_type": "media",
                "source_id": "media-7",
                "locator": {"chunk_id": "chunk-12"},
                "ordinal": 1,
            },
        ]
    )
    workspace_route = module.resolve_deep_dive_target(
        [
            {
                "source_type": "message",
                "source_id": "msg-3",
                "citation_text": "This message cannot route directly without a conversation id.",
                "ordinal": 0,
            },
            {
                "source_type": "note",
                "source_id": "note-2",
                "ordinal": 1,
            },
        ]
    )
    citation_fallback = module.resolve_deep_dive_target(
        [
            {
                "source_type": "message",
                "source_id": "msg-fallback",
                "citation_text": "The quote survives even though no conversation id is available.",
                "ordinal": 3,
            }
        ]
    )

    assert exact_locator == {  # nosec B101
        "source_type": "media",
        "source_id": "media-7",
        "citation_ordinal": 1,
        "route_kind": "exact_locator",
        "route": "/media/media-7?chunk_id=chunk-12",
        "available": True,
        "fallback_reason": None,
    }
    assert workspace_route == {  # nosec B101
        "source_type": "note",
        "source_id": "note-2",
        "citation_ordinal": 1,
        "route_kind": "workspace_route",
        "route": "/notes/note-2",
        "available": True,
        "fallback_reason": None,
    }
    assert citation_fallback == {  # nosec B101
        "source_type": "message",
        "source_id": "msg-fallback",
        "citation_ordinal": 3,
        "route_kind": "citation_only",
        "route": None,
        "available": False,
        "fallback_reason": "message_conversation_id_required",
    }


def test_select_primary_citation_preserves_explicit_zero_ordinal_even_when_not_first(db: CharactersRAGDB):
    module = _load_provenance_module()

    selected = module.select_primary_citation(
        [
            {
                "source_type": "note",
                "source_id": "note-later",
                "ordinal": 4,
            },
            {
                "source_type": "note",
                "source_id": "note-primary",
                "ordinal": 0,
            },
        ]
    )

    assert selected is not None  # nosec B101
    assert selected["source_id"] == "note-primary"  # nosec B101
    assert selected["ordinal"] == 0  # nosec B101


def test_persist_flashcard_citations_mirrors_only_primary_legacy_source_ref(db: CharactersRAGDB):
    card_uuid = _create_card(db, front="What is additive increase?")
    store = _store(db)

    provenance = store.persist_flashcard_citations(
        card_uuid,
        [
            {
                "source_type": "message",
                "source_id": "msg-opaque-locator",
                "citation_text": "The chat excerpt explicitly names additive increase.",
                "locator": "reply-4",
            },
            {
                "source_type": "note",
                "source_id": "note-routeable",
                "citation_text": "AIMD grows linearly between loss events.",
            },
        ],
    )

    flashcard = db.get_flashcard(card_uuid)

    assert flashcard is not None  # nosec B101
    assert flashcard["source_ref_type"] == "note"  # nosec B101
    assert flashcard["source_ref_id"] == "note-routeable"  # nosec B101
    assert provenance["primary_citation"]["source_type"] == "note"  # nosec B101
    assert provenance["primary_citation"]["source_id"] == "note-routeable"  # nosec B101
    assert provenance["deep_dive_target"]["route_kind"] == "workspace_route"  # nosec B101
    assert provenance["deep_dive_target"]["source_id"] == "note-routeable"  # nosec B101


def test_persist_flashcard_citations_replaces_existing_active_citation_set(db: CharactersRAGDB):
    card_uuid = _create_card(db, front="What triggers fast retransmit?")
    store = _store(db)

    first = store.persist_flashcard_citations(
        card_uuid,
        [
            {
                "source_type": "media",
                "source_id": "media-initial",
                "citation_text": "The lecture introduces duplicate ACKs.",
                "locator": {"timestamp": "00:02:10"},
            }
        ],
    )
    second = store.persist_flashcard_citations(
        card_uuid,
        [
            {
                "source_type": "note",
                "source_id": "note-current",
                "citation_text": "Fast retransmit uses duplicate ACKs as a signal.",
            }
        ],
    )

    active_citations = db.list_flashcard_citations(card_uuid)
    all_citations = db.list_flashcard_citations(card_uuid, include_deleted=True)
    flashcard = db.get_flashcard(card_uuid)

    assert first["primary_citation"]["source_id"] == "media-initial"  # nosec B101
    assert second["primary_citation"]["source_id"] == "note-current"  # nosec B101
    assert [citation["source_id"] for citation in active_citations] == ["note-current"]  # nosec B101
    assert [citation["ordinal"] for citation in active_citations] == [0]  # nosec B101
    assert len(all_citations) == 2  # nosec B101
    assert flashcard is not None  # nosec B101
    assert flashcard["source_ref_type"] == "note"  # nosec B101
    assert flashcard["source_ref_id"] == "note-current"  # nosec B101


def test_persist_flashcard_citations_rolls_back_source_summary_when_insert_fails(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    card_uuid = _create_card(db, front="What is selective acknowledgment?")
    store = _store(db)
    flashcard_before = db.get_flashcard(card_uuid)
    assert flashcard_before is not None  # nosec B101

    original_execute_many = db.execute_many

    def fail_flashcard_citation_insert(query: str, params: list[tuple[Any, ...]], commit: bool = True):
        if "INSERT INTO flashcard_citations(" in query:
            raise RuntimeError("forced citation insert failure")
        return original_execute_many(query, params, commit=commit)

    monkeypatch.setattr(db, "execute_many", fail_flashcard_citation_insert)

    with pytest.raises(RuntimeError, match="forced citation insert failure"):
        store.persist_flashcard_citations(
            card_uuid,
            [
                {
                    "source_type": "note",
                    "source_id": "note-sack",
                    "citation_text": "Selective acknowledgment reports non-contiguous received blocks.",
                }
            ],
        )

    flashcard_after = db.get_flashcard(card_uuid)
    assert flashcard_after is not None  # nosec B101
    assert db.list_flashcard_citations(card_uuid) == []  # nosec B101
    assert flashcard_after["source_ref_type"] == flashcard_before["source_ref_type"]  # nosec B101
    assert flashcard_after["source_ref_id"] == flashcard_before["source_ref_id"]  # nosec B101


def test_build_flashcard_assistant_context_includes_pack_and_provenance_payload(db: CharactersRAGDB):
    note_id = db.add_note("Slow Start", "Slow start expands the congestion window rapidly.")
    card_uuid = _create_card(db)
    flashcard = db.get_flashcard(card_uuid)
    assert flashcard is not None  # nosec B101
    pack_id = _create_pack(db, deck_id=int(flashcard["deck_id"]))
    db.add_study_pack_cards(pack_id, [card_uuid])

    _store(db).persist_flashcard_citations(
        card_uuid,
        [
            {
                "source_type": "note",
                "source_id": note_id,
                "citation_text": "Slow start doubles the congestion window each RTT.",
                "locator": "anchor:slow-start",
            },
            {
                "source_type": "media",
                "source_id": "media-88",
                "citation_text": "The lecture revisits congestion avoidance later.",
            },
        ],
    )

    context = build_flashcard_assistant_context(db, card_uuid)

    assert context["study_pack"]["id"] == pack_id  # nosec B101
    assert [citation["source_id"] for citation in context["citations"]] == [note_id, "media-88"]  # nosec B101
    assert context["primary_citation"]["source_id"] == note_id  # nosec B101
    assert context["deep_dive_target"] == {  # nosec B101
        "source_type": "note",
        "source_id": note_id,
        "citation_ordinal": 0,
        "route_kind": "exact_locator",
        "route": f"/notes/{note_id}?locator=anchor%3Aslow-start",
        "available": True,
        "fallback_reason": None,
    }


def test_build_flashcard_assistant_context_returns_empty_citations_for_legacy_cards(db: CharactersRAGDB):
    card_uuid = _create_card(db, front="What is congestion avoidance?")

    context = build_flashcard_assistant_context(db, card_uuid)

    assert context["citations"] == []  # nosec B101
    assert context["primary_citation"] is None  # nosec B101
    assert context["deep_dive_target"] is None  # nosec B101
    assert context["study_pack"] is None  # nosec B101


def test_flashcard_assistant_endpoint_returns_top_level_provenance_fields(
    db: CharactersRAGDB,
    assistant_client: TestClient,
):
    note_id = db.add_note("Congestion Avoidance", "AIMD grows linearly between loss events.")
    card_uuid = _create_card(db, front="What is AIMD?")

    _store(db).persist_flashcard_citations(
        card_uuid,
        [
            {
                "source_type": "note",
                "source_id": note_id,
                "citation_text": "AIMD combines additive increase and multiplicative decrease.",
                "locator": "anchor:aimd",
            }
        ],
    )

    response = assistant_client.get(f"/api/v1/flashcards/{card_uuid}/assistant")

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    assert payload["citations"][0]["source_id"] == note_id  # nosec B101
    assert payload["primary_citation"]["source_id"] == note_id  # nosec B101
    assert payload["deep_dive_target"] == {  # nosec B101
        "source_type": "note",
        "source_id": note_id,
        "citation_ordinal": 0,
        "route_kind": "exact_locator",
        "route": f"/notes/{note_id}?locator=anchor%3Aaimd",
        "available": True,
        "fallback_reason": None,
    }
    assert payload["study_pack"] is None  # nosec B101
