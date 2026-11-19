from __future__ import annotations

import json
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackCreateJobRequest, StudyPackSourceSelection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.StudyPacks.chapter_browser import browse_deck_cards
from tldw_Server_API.app.core.StudyPacks.jobs import extract_study_pack_source_items
from tldw_Server_API.app.core.StudyPacks.media_sections import plan_media_sections

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


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
            '{"supported":[true]}',
        ],
    )

    result = await service.generate_validated_cards(bundle, study_pack_request)

    assert len(calls) == 3  # nosec B101
    assert result.cards[0].front  # nosec B101
    assert result.cards[0].back  # nosec B101
    assert result.cards[0].citations[0].source_id in {"note-1", "42", "msg-1"}  # nosec B101


@pytest.mark.parametrize(
    ("provider", "model", "expected_max_tokens"),
    [("deepseek", "deepseek-flash", 4_000), ("openai", "gpt-test", 4_000), ("huggingface", "hf-test", 2_048)],
)
async def test_generation_repair_and_verification_send_the_model_output_cap(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model: str,
    expected_max_tokens: int,
):
    _, service_mod = _load_generation_modules()
    service = service_mod.StudyPackGenerationService(
        note_db=db,
        media_db=SimpleNamespace(),
        provider=provider,
        model=model,
    )
    responses = iter(("invalid JSON", _valid_generation_payload(), '{"supported":[true]}'))
    calls = []

    async def fake_chat_call(**kwargs):
        calls.append((kwargs["api_provider"], kwargs["model"], kwargs["max_tokens"]))
        return {"choices": [{"message": {"content": next(responses)}}]}

    monkeypatch.setattr(service_mod, "perform_chat_api_call_async", fake_chat_call)

    result = await service.generate_validated_cards(bundle, study_pack_request)

    assert result.repair_attempted  # nosec B101
    assert calls == [(provider, model, expected_max_tokens)] * 3  # nosec B101


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
            '{"supported":[true]}',
        ],
    )

    result = await service.generate_validated_cards(bundle, study_pack_request)

    assert len(calls) == 3  # nosec B101
    assert result.repair_attempted is True  # nosec B101
    assert result.cards[0].citations[0].source_id == "note-1"  # nosec B101


@pytest.mark.parametrize("verdict", ['{"supported":[false]}', '{"supported":[]}', '{"supported":[1]}'])
async def test_generate_validated_cards_rejects_an_unverified_answer(
    db: CharactersRAGDB,
    bundle: Any,
    study_pack_request: StudyPackCreateJobRequest,
    monkeypatch: pytest.MonkeyPatch,
    verdict: str,
):
    _, service_mod = _load_generation_modules()
    contradictory = _valid_generation_payload().replace(
        "It doubles the congestion window each RTT.", "It halves the congestion window each RTT.",
    )
    service, calls = _build_service(db, monkeypatch, [contradictory, verdict] * 2)

    with pytest.raises(service_mod.StudyPackCitationError):
        await service.generate_validated_cards(bundle, study_pack_request)
    assert len(calls) == 4  # nosec B101
    verification_request = json.loads(calls[1]["user_prompt"])
    assert verification_request[0]["back"] == "It halves the congestion window each RTT."  # nosec B101
    assert verification_request[0]["quotes"] == ["Slow start doubles the congestion window each RTT."]  # nosec B101


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


def test_validate_citation_payload_requires_citation_text_to_match_bundle_evidence(
    db: CharactersRAGDB,
    bundle: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    _, service_mod = _load_generation_modules()
    service, _ = _build_service(db, monkeypatch, [])

    with pytest.raises(service_mod.StudyPackValidationError, match="citation_text"):
        service._validate_citation_payload(
            {
                "source_type": "note",
                "source_id": "note-1",
                "citation_text": "This unsupported citation is not in the note evidence.",
            },
            bundle_lookup={
                ("note", "note-1"): bundle.items[0],
            },
        )


def test_extract_json_payload_rejects_embedded_or_trailing_json(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    _, service_mod = _load_generation_modules()
    service, _ = _build_service(db, monkeypatch, [])

    with pytest.raises(service_mod.StudyPackMalformedResponseError, match="valid JSON"):
        service._extract_json_payload('preface {"cards": []} trailing prose')


def test_destination_deck_name_checks_exact_candidates_beyond_list_window(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    service, _ = _build_service(db, monkeypatch, [])

    class WindowedDeckDb:
        def list_decks(self, **_: Any) -> list[dict[str, str]]:
            return [{"name": f"aaa-{index:05d}"} for index in range(10_000)]

        def get_deck_by_name(self, name: str, *, include_deleted: bool = False) -> dict[str, str] | None:
            if name == "zzzz-existing":
                return {"name": name}
            return None

    service.note_db = WindowedDeckDb()

    assert service._next_destination_deck_name("zzzz-existing") == "zzzz-existing (2)"  # nosec B101


def test_destination_deck_name_fallback_lists_decks_once(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    service, _ = _build_service(db, monkeypatch, [])

    class ListOnlyDeckDb:
        def __init__(self) -> None:
            self.calls = 0

        def list_decks(self, **_: Any) -> list[dict[str, str]]:
            self.calls += 1
            return [
                {"name": "Fallback Pack"},
                {"name": "Fallback Pack (2)"},
            ]

    deck_db = ListOnlyDeckDb()
    service.note_db = deck_db

    assert service._next_destination_deck_name("Fallback Pack") == "Fallback Pack (3)"  # nosec B101
    assert deck_db.calls == 1  # nosec B101


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


@pytest.mark.parametrize("selection_mode", [
    "all", "last", "stale", "empty", "empty_without_revision", "nonstudy_section",
    "bad_repair", "budget", "output_limit", "output_limit_native", "output_limit_budget", "citation_repair", "bad_citation",
])
async def test_whole_media_pack_generates_cited_cards_through_the_end(
    db: CharactersRAGDB,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    selection_mode: str,
):
    _, service_mod = _load_generation_modules()
    media = MediaDatabase(db_path=str(tmp_path / "book.db"), client_id="study-pack-book")
    first_chapter = "Chapter 1: Seedlings need light."
    final_chapter = "Chapter 2: Seed banks preserve biodiversity."
    book = "\n".join([first_chapter, "Plants respond to their environment. " * 450, final_chapter])
    media_id, _, _ = media.add_media_with_keywords(
        title="A study guide to plants",
        content=book,
        media_type="ebook",
        keywords=[],
    )
    service = service_mod.StudyPackGenerationService(note_db=db, media_db=media, provider="openai", model="test")
    (item,) = service.source_resolver.resolve([{"source_type": "media", "source_id": str(media_id)}]).items
    revision, _ = plan_media_sections(item.evidence_text, 12_000)
    locator = {}
    indices = [0, 1]
    if selection_mode in {"last", "stale", "empty", "empty_without_revision"}:
        indices = [] if selection_mode.startswith("empty") else [1]
        locator = {
            "section_indices": [99] if selection_mode == "stale" else indices,
            "section_revision": "stale" if selection_mode == "stale" else revision,
        }
    if selection_mode == "empty_without_revision":
        locator.pop("section_revision")
    request = StudyPackCreateJobRequest(
        title="Plant study guide",
        workspace_id="ws-1",
        source_items=[StudyPackSourceSelection(source_type="media", source_id=str(media_id), locator=locator)],
    )
    passages: list[str] = []

    repair_prompts = []
    capped_passages = []

    async def generate(**kwargs):
        user_prompt = kwargs["messages"][0]["content"]
        system_prompt = kwargs["system_message"]
        if system_prompt.startswith("Verify each flashcard"):
            return {"content": json.dumps({"supported": [True] * len(json.loads(user_prompt))})}
        is_repair = system_prompt.startswith("You repair")
        bundle_json = user_prompt.split("Allowed source bundle:\n", 1)[1].split("\n\n", 1)[0]
        excerpt = json.loads(bundle_json)["items"][0]["evidence_text"]
        is_capped = selection_mode.startswith("output_limit") and len(excerpt) > 6_100
        if is_capped:
            capped_passages.append(excerpt)
        elif is_repair:
            repair_prompts.append((system_prompt, user_prompt))
        else:
            passages.append(excerpt)
        citation = excerpt.splitlines()[0][:96].strip()
        citation_text = citation
        if final_chapter in excerpt and (
            selection_mode == "bad_citation" or (selection_mode == "citation_repair" and not is_repair)
        ):
            citation_text = first_chapter
        content = json.dumps({"cards": [{
            "front": f"What is the focus of section {len(passages)}?",
            "back": citation,
            "citations": [{
                "source_type": "media",
                "source_id": str(media_id),
                "citation_text": citation_text,
                "locator": {"chunk_index": 99, "timestamp_seconds": 120},
            }],
        }]})
        if selection_mode == "bad_repair" and final_chapter in excerpt:
            content = '{"cards":[]}' if is_repair else "malformed response"
        if selection_mode == "nonstudy_section" and final_chapter not in excerpt:
            content = '{"cards":[]}'
        cap_reason = "MAX_TOKENS" if selection_mode == "output_limit_native" else "length"
        return {"choices": [{"finish_reason": cap_reason if is_capped else "stop", "message": {"content": content}}]}

    monkeypatch.setattr(service_mod, "perform_chat_api_call_async", generate)
    if selection_mode == "budget":
        monkeypatch.setattr(service_mod, "MAX_STUDY_PACK_PASSES", 1)
    if selection_mode == "output_limit_budget":
        monkeypatch.setattr(service_mod, "MAX_STUDY_PACK_PASSES", 3)
    try:
        if selection_mode == "stale" or selection_mode.startswith("empty"):
            match = "Review its sections" if selection_mode == "stale" else "Select at least one"
            with pytest.raises(ValueError, match=match):
                await service.create_from_request(request)
            assert passages == []  # nosec B101
            assert not db.get_deck_by_name("Plant study guide")  # nosec B101
            return
        if selection_mode == "budget":
            with pytest.raises(ValueError, match="too many generation passes"):
                await service.create_from_request(request)
            assert not passages  # nosec B101
            return
        if selection_mode == "output_limit_budget":
            with pytest.raises(ValueError, match="too many generation passes"):
                await service.create_from_request(request)
            assert len(capped_passages) == 1  # nosec B101
            assert not db.get_deck_by_name("Plant study guide")  # nosec B101
            return
        if selection_mode == "bad_citation":
            with pytest.raises(ValueError, match="citation_text must match source evidence"):
                await service.create_from_request(request)
            assert len(passages) == 3 and len(repair_prompts) == 1  # nosec B101
            assert not db.get_deck_by_name("Plant study guide")  # nosec B101
            assert db.get_study_pack(1) is None  # nosec B101
            return
        if selection_mode == "bad_repair":
            with pytest.raises(ValueError, match="repair returned no cards"):
                await service.create_from_request(request)
            assert not db.get_deck_by_name("Plant study guide")  # nosec B101
            return
        result = await service.create_from_request(request)
    finally:
        media.close_connection()

    expected_text = final_chapter if selection_mode == "last" else book
    assert "".join("".join(passage.split()) for passage in passages) == "".join(expected_text.split())  # nosec B101
    assert max(map(len, passages)) <= 12_000  # nosec B101
    citation = db.list_flashcard_citations(result.card_uuids[-1])[0]
    assert citation["citation_text"] == final_chapter  # nosec B101
    saved_locator = json.loads(citation["locator"])
    assert saved_locator["study_pack_chapter"] == {  # nosec B101
        "section_index": 1,
        "section_revision": revision,
        "title": final_chapter,
        "source_title": "A study guide to plants",
    }
    assert "chunk_index" not in saved_locator and "timestamp_seconds" not in saved_locator  # nosec B101

    if selection_mode == "citation_repair":
        (system_prompt, user_prompt), = repair_prompts
        assert "citation_text must match source evidence" in user_prompt  # nosec B101
        assert "verbatim" in system_prompt  # nosec B101

    if selection_mode in {"all", "output_limit", "output_limit_native"}:
        first_chapter_cards = 2
        if selection_mode.startswith("output_limit"):
            assert len(capped_passages) == 1 and not repair_prompts  # nosec B101
            assert max(map(len, passages)) <= 6_100  # nosec B101
            first_chapter_cards = 3
        assert len(passages) == first_chapter_cards + 1  # nosec B101
        first_set_id = f"{media_id}:0:{revision}"
        all_cards = browse_deck_cards(db, result.deck_id, chapter_id=None, limit=20, offset=0)
        assert all_cards["all_cards_count"] == all_cards["total"] == len(passages)  # nosec B101
        assert [(chapter["title"], chapter["card_count"]) for chapter in all_cards["chapters"]] == [  # nosec B101
            (first_chapter, first_chapter_cards),
            (final_chapter, 1),
        ]
        assert [card["uuid"] for card in all_cards["items"]] == result.card_uuids  # nosec B101
        first_set = browse_deck_cards(db, result.deck_id, chapter_id=first_set_id, limit=20, offset=0)
        assert [card["uuid"] for card in first_set["items"]] == result.card_uuids[:first_chapter_cards]  # nosec B101
        assert all(card["deck_id"] == result.deck_id for card in all_cards["items"])  # nosec B101

    pack = db.get_study_pack(result.pack_id)
    snapshot = pack["source_bundle_json"]
    if isinstance(snapshot, str):
        snapshot = json.loads(snapshot)
    assert len(snapshot["items"][0]["evidence_text"]) <= 12_000  # nosec B101
    (selection,) = extract_study_pack_source_items(snapshot)
    assert "excerpt_text" not in selection  # nosec B101
    assert selection["locator"]["section_indices"] == indices  # nosec B101
    assert selection["locator"]["section_revision"] == revision  # nosec B101
