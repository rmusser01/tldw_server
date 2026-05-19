from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackCreateJobRequest, StudyPackSourceSelection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


pytestmark = pytest.mark.asyncio


def _load_generation_modules():
    try:
        types_mod = import_module("tldw_Server_API.app.core.StudyPacks.types")
        service_mod = import_module("tldw_Server_API.app.core.StudyPacks.generation_service")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Study pack generation modules are missing: {exc}")
    except ImportError as exc:
        pytest.fail(f"Study pack generation imports are not yet usable: {exc}")
    return types_mod, service_mod


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-pack-generation.db"), client_id="study-pack-generation-tests")
    chacha.upsert_workspace("ws-1", "Workspace 1")
    try:
        yield chacha
    finally:
        chacha.close_connection()


@pytest.fixture
def study_pack_request() -> StudyPackCreateJobRequest:
    return StudyPackCreateJobRequest(
        title="TCP Fundamentals",
        workspace_id="ws-1",
        source_items=[
            StudyPackSourceSelection(source_type="note", source_id="note-1"),
            StudyPackSourceSelection(source_type="media", source_id="42"),
            StudyPackSourceSelection(source_type="message", source_id="msg-1"),
        ],
    )


@pytest.fixture
def bundle():
    types_mod, _ = _load_generation_modules()
    return types_mod.StudySourceBundle(
        items=[
            types_mod.StudySourceBundleItem(
                source_type="note",
                source_id="note-1",
                label="Slow Start Notes",
                evidence_text="Slow start doubles the congestion window each RTT.",
                locator={"note_id": "note-1"},
            ),
            types_mod.StudySourceBundleItem(
                source_type="media",
                source_id="42",
                label="Lecture 42",
                evidence_text="At 61 seconds the lecture explains exponential growth.",
                locator={"media_id": 42, "timestamp_seconds": 61},
            ),
            types_mod.StudySourceBundleItem(
                source_type="message",
                source_id="msg-1",
                label="Chat message",
                evidence_text="The message ties slow start to packet loss recovery.",
                locator={"conversation_id": "conv-1", "message_id": "msg-1"},
            ),
        ]
    )


def _build_service(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, responses: list[str]):
    _, service_mod = _load_generation_modules()
    service = service_mod.StudyPackGenerationService(
        note_db=db,
        media_db=SimpleNamespace(),
        provider="openai",
        model="gpt-test",
    )

    calls: list[dict[str, Any]] = []

    async def fake_call_generation_model(**kwargs: Any) -> str:
        calls.append(kwargs)
        if not responses:
            raise AssertionError("No fake study-pack generation response remaining")
        return responses.pop(0)

    monkeypatch.setattr(service, "_call_generation_model", fake_call_generation_model)
    return service, calls


def _valid_generation_payload(*, source_id: str = "note-1", source_type: str = "note") -> str:
    return (
        '{"cards":[{"front":"What is TCP slow start?","back":"It doubles the congestion window each RTT.",'
        '"citations":[{"source_type":"'
        + source_type
        + '","source_id":"'
        + source_id
        + '","citation_text":"Slow start doubles the congestion window each RTT."}]}]}'
    )


def _generation_result(types_mod: Any):
    return types_mod.StudyPackGenerationResult(
        cards=[
            types_mod.StudyPackCardDraft(
                front="What is TCP slow start?",
                back="It doubles the congestion window each RTT.",
                citations=[
                    types_mod.StudyCitationDraft(
                        source_type="note",
                        source_id="note-1",
                        citation_text="Slow start doubles the congestion window each RTT.",
                        locator={"note_id": "note-1"},
                    )
                ],
            )
        ]
    )


async def test_generate_validated_cards_rejects_uncited_cards(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    _, service_mod = _load_generation_modules()
    service, _ = _build_service(
        db,
        monkeypatch,
        [
            '{"cards":[{"front":"What is TCP slow start?","back":"It doubles the congestion window each RTT.","citations":[]}]}',
            '{"cards":[{"front":"What is TCP slow start?","back":"It doubles the congestion window each RTT.","citations":[]}]}',
        ],
    )

    with pytest.raises(service_mod.StudyPackValidationError, match="citation"):
        await service.generate_validated_cards(bundle, study_pack_request)


async def test_generate_validated_cards_rejects_citations_outside_source_bundle(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    _, service_mod = _load_generation_modules()
    service, _ = _build_service(
        db,
        monkeypatch,
        [
          _valid_generation_payload(source_id="note-does-not-exist"),
          _valid_generation_payload(source_id="note-does-not-exist"),
        ],
    )

    with pytest.raises(service_mod.StudyPackValidationError, match="source bundle"):
        await service.generate_validated_cards(bundle, study_pack_request)


async def test_generate_validated_cards_repairs_malformed_output_exactly_once(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    service, calls = _build_service(
        db,
        monkeypatch,
        [
            '```json\n{"cards":[{"front":"What is TCP slow start?","back":"It doubles the congestion window each RTT.","citations":[{"source_type":"note","source_id":"note-1","citation_text":"Slow start doubles the congestion window each RTT."}]}\n```',
            _valid_generation_payload(),
        ],
    )

    result = await service.generate_validated_cards(bundle, study_pack_request)

    assert len(calls) == 2  # nosec B101
    assert result.cards[0].front  # nosec B101
    assert result.cards[0].back  # nosec B101
    assert result.cards[0].citations[0].source_id in {"note-1", "42", "msg-1"}  # nosec B101


async def test_generate_validated_cards_repairs_validation_failures_exactly_once(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    service, calls = _build_service(
        db,
        monkeypatch,
        [
            '{"cards":[{"front":"What is TCP slow start?","back":"It doubles the congestion window each RTT.","citations":[]}]}',
            _valid_generation_payload(),
        ],
    )

    result = await service.generate_validated_cards(bundle, study_pack_request)

    assert len(calls) == 2  # nosec B101
    assert result.repair_attempted is True  # nosec B101
    assert result.cards[0].citations[0].source_id == "note-1"  # nosec B101


def test_validate_citation_payload_preserves_bundle_locator_over_conflicting_model_locator(
    db: CharactersRAGDB,
    bundle: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    service, _ = _build_service(db, monkeypatch, [])
    citation = service._validate_citation_payload(
        {
            "source_type": "media",
            "source_id": "42",
            "citation_text": "At 61 seconds the lecture explains exponential growth.",
            "locator": {
                "media_id": 999,
                "timestamp_seconds": 999,
                "chunk_id": "chunk-99",
            },
        },
        bundle_lookup={
            ("media", "42"): bundle.items[1],
        },
    )

    assert citation["locator"]["media_id"] == 42  # nosec B101
    assert citation["locator"]["timestamp_seconds"] == 61  # nosec B101
    assert citation["locator"]["chunk_id"] == "chunk-99"  # nosec B101


async def test_create_study_pack_from_request_uses_collision_safe_deck_suffixing(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    types_mod, service_mod = _load_generation_modules()
    db.add_deck("TCP Fundamentals", workspace_id="ws-1")
    db.add_deck("TCP Fundamentals (2)", workspace_id="ws-1")

    monkeypatch.setattr(service_mod.StudySourceResolver, "resolve", lambda self, selections: bundle)

    async def fake_generate_validated_cards(self, resolved_bundle: Any, generation_request: Any):
        assert resolved_bundle == bundle  # nosec B101
        assert generation_request.title == study_pack_request.title  # nosec B101
        return _generation_result(types_mod)

    monkeypatch.setattr(
        service_mod.StudyPackGenerationService,
        "generate_validated_cards",
        fake_generate_validated_cards,
    )

    result = await service_mod.create_study_pack_from_request(
        note_db=db,
        media_db=SimpleNamespace(),
        request=study_pack_request,
        regenerate_from_pack_id=None,
        provider="openai",
        model="gpt-test",
    )

    deck = db.get_deck(result.deck_id)

    assert deck is not None  # nosec B101
    assert deck["name"] == "TCP Fundamentals (3)"  # nosec B101


async def test_create_study_pack_from_request_rolls_back_empty_deck_on_persistence_failure(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    types_mod, service_mod = _load_generation_modules()
    monkeypatch.setattr(service_mod.StudySourceResolver, "resolve", lambda self, selections: bundle)

    async def fake_generate_validated_cards(self, resolved_bundle: Any, generation_request: Any):
        return _generation_result(types_mod)

    monkeypatch.setattr(
        service_mod.StudyPackGenerationService,
        "generate_validated_cards",
        fake_generate_validated_cards,
    )

    original_add_study_pack_cards = db.add_study_pack_cards

    def fail_membership_insert(study_pack_id: int, flashcard_uuids: list[str]) -> int:
        original_add_study_pack_cards(study_pack_id, flashcard_uuids)
        raise CharactersRAGDBError("membership write failed")

    monkeypatch.setattr(db, "add_study_pack_cards", fail_membership_insert)

    with pytest.raises(CharactersRAGDBError, match="membership write failed"):
        await service_mod.create_study_pack_from_request(
            note_db=db,
            media_db=SimpleNamespace(),
            request=study_pack_request,
            regenerate_from_pack_id=None,
            provider="openai",
            model="gpt-test",
        )

    deck_names = [deck["name"] for deck in db.list_decks(include_workspace_items=True)]
    study_packs = db.execute_query("SELECT COUNT(*) AS count FROM study_packs WHERE deleted = 0").fetchone()

    assert "TCP Fundamentals" not in deck_names  # nosec B101
    assert int(study_packs["count"]) == 0  # nosec B101


async def test_create_study_pack_from_request_only_supersedes_prior_pack_after_replacement_commit(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    types_mod, service_mod = _load_generation_modules()
    original_deck_id = db.add_deck("TCP Fundamentals Original", workspace_id="ws-1")
    original_pack_id = db.create_study_pack(
        title="TCP Fundamentals Original",
        workspace_id="ws-1",
        deck_id=original_deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "note-seed"}]},
        generation_options_json={"deck_mode": "new"},
    )

    monkeypatch.setattr(service_mod.StudySourceResolver, "resolve", lambda self, selections: bundle)

    async def fake_generate_validated_cards(self, resolved_bundle: Any, generation_request: Any):
        return _generation_result(types_mod)

    monkeypatch.setattr(
        service_mod.StudyPackGenerationService,
        "generate_validated_cards",
        fake_generate_validated_cards,
    )

    original_persist = service_mod.FlashcardProvenanceStore.persist_flashcard_citations
    failure_state = {"should_fail": True}

    def fail_once(self, flashcard_uuid: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
        if failure_state["should_fail"]:
            raise CharactersRAGDBError("citation persistence failed")
        return original_persist(self, flashcard_uuid, citations)

    monkeypatch.setattr(service_mod.FlashcardProvenanceStore, "persist_flashcard_citations", fail_once)

    with pytest.raises(CharactersRAGDBError, match="citation persistence failed"):
        await service_mod.create_study_pack_from_request(
            note_db=db,
            media_db=SimpleNamespace(),
            request=study_pack_request,
            regenerate_from_pack_id=original_pack_id,
            provider="openai",
            model="gpt-test",
        )

    failed_original_row = db.execute_query(
        "SELECT status, superseded_by_pack_id FROM study_packs WHERE id = ?",
        (original_pack_id,),
    ).fetchone()
    failed_pack_count = db.execute_query(
        "SELECT COUNT(*) AS count FROM study_packs WHERE deleted = 0",
    ).fetchone()

    assert failed_original_row["status"] == "active"  # nosec B101
    assert failed_original_row["superseded_by_pack_id"] is None  # nosec B101
    assert int(failed_pack_count["count"]) == 1  # nosec B101

    failure_state["should_fail"] = False
    result = await service_mod.create_study_pack_from_request(
        note_db=db,
        media_db=SimpleNamespace(),
        request=study_pack_request,
        regenerate_from_pack_id=original_pack_id,
        provider="openai",
        model="gpt-test",
    )

    original_row = db.execute_query(
        "SELECT status, superseded_by_pack_id FROM study_packs WHERE id = ?",
        (original_pack_id,),
    ).fetchone()
    replacement_pack = db.get_study_pack(result.pack_id)

    assert replacement_pack is not None  # nosec B101
    assert original_row["status"] == "superseded"  # nosec B101
    assert int(original_row["superseded_by_pack_id"]) == result.pack_id  # nosec B101


async def test_create_study_pack_from_request_passes_expected_version_when_superseding(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    types_mod, service_mod = _load_generation_modules()
    original_deck_id = db.add_deck("TCP Fundamentals Original", workspace_id="ws-1")
    original_pack_id = db.create_study_pack(
        title="TCP Fundamentals Original",
        workspace_id="ws-1",
        deck_id=original_deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "note-seed"}]},
        generation_options_json={"deck_mode": "new"},
    )

    monkeypatch.setattr(service_mod.StudySourceResolver, "resolve", lambda self, selections: bundle)

    async def fake_generate_validated_cards(self, resolved_bundle: Any, generation_request: Any):
        return _generation_result(types_mod)

    monkeypatch.setattr(
        service_mod.StudyPackGenerationService,
        "generate_validated_cards",
        fake_generate_validated_cards,
    )

    captured: dict[str, Any] = {}
    original_supersede = db.supersede_study_pack

    def capture_supersede(
        pack_id: int,
        *,
        superseded_by_pack_id: int,
        expected_version: int | None = None,
    ) -> bool:
        captured["pack_id"] = pack_id
        captured["superseded_by_pack_id"] = superseded_by_pack_id
        captured["expected_version"] = expected_version
        return original_supersede(
            pack_id,
            superseded_by_pack_id=superseded_by_pack_id,
            expected_version=expected_version,
        )

    monkeypatch.setattr(db, "supersede_study_pack", capture_supersede)

    await service_mod.create_study_pack_from_request(
        note_db=db,
        media_db=SimpleNamespace(),
        request=study_pack_request,
        regenerate_from_pack_id=original_pack_id,
        provider="openai",
        model="gpt-test",
    )

    assert captured["pack_id"] == original_pack_id  # nosec B101
    assert captured["expected_version"] == 1  # nosec B101
