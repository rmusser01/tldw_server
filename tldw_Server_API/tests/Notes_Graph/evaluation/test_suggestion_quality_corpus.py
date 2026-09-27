from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import (
    content_fingerprint,
    estimate_tokens,
    reconstruct_evidence,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_generation import (
    MAX_ESTIMATED_INPUT_TOKENS,
    MAX_NEW_TAG_SUGGESTIONS,
    MAX_OUTPUT_TOKENS,
    MAX_RELATIONSHIP_SUGGESTIONS,
    MAX_TAG_CATALOG,
    MAX_TAG_SUGGESTIONS,
    SuggestionGenerationError,
    build_generation_request,
    parse_and_validate_generation,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_retrieval import SuggestionRetriever

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 28, 14, 0, tzinfo=timezone.utc)
DATASET_ID = "quality-evaluation"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "suggestion_grounding_cases.json"
CASE_KEYS = {
    "id",
    "domain",
    "match_kind",
    "source",
    "candidates",
    "memberships",
    "existing_tags",
    "response",
    "expected_targets",
    "expected_tags",
    "expected_top30",
    "cutoff_expected_targets",
    "largest",
}
SOURCE_KEYS = {"id", "title", "content", "repeat"}
CANDIDATE_KEYS = {"id", "title", "content", "repeat", "role", "expected"}
MEMBERSHIP_KEYS = {"tags", "sources"}
TAG_MEMBERSHIP_KEYS = {"tag", "note_ids"}
SOURCE_MEMBERSHIP_KEYS = {"source", "external_ref", "note_ids"}
RESPONSE_KEYS = {"relationships", "tags"}
RELATIONSHIP_KEYS = {
    "target_id",
    "rationale",
    "source_evidence",
    "target_evidence",
}
TAG_KEYS = {"existing_tag", "new_tag", "rationale", "source_evidence"}
PROBE_KEYS = {"id", "case_id", "mutation", "expected_error"}


def _load_corpus() -> dict[str, object]:
    corpus = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert set(corpus) == {"schema_version", "cases", "validation_probes"}
    assert corpus["schema_version"] == 1
    cases = corpus["cases"]
    probes = corpus["validation_probes"]
    assert isinstance(cases, list) and cases, "quality corpus must contain cases"
    assert isinstance(probes, list) and probes, "quality corpus must contain validation probes"
    case_ids: set[str] = set()
    note_ids: set[str] = set()
    for case in cases:
        assert set(case) == CASE_KEYS
        assert case["id"] not in case_ids
        case_ids.add(case["id"])
        assert case["domain"] in {"medical", "technical", "research", "general"}
        assert case["match_kind"] in {"direct", "weak"}
        assert isinstance(case["largest"], bool)
        assert set(case["source"]) == SOURCE_KEYS
        assert isinstance(case["source"]["repeat"], int) and case["source"]["repeat"] >= 1
        assert set(case["memberships"]) == MEMBERSHIP_KEYS
        assert set(case["response"]) == RESPONSE_KEYS
        assert isinstance(case["existing_tags"], list)
        assert isinstance(case["expected_targets"], list)
        assert isinstance(case["expected_tags"], list)
        for candidate in case["candidates"]:
            assert set(candidate) == CANDIDATE_KEYS
            assert candidate["role"] in {
                "same_owner",
                "cross_owner",
                "already_linked",
                "rejected",
            }
            assert isinstance(candidate["repeat"], int) and candidate["repeat"] >= 1
            assert isinstance(candidate["expected"], bool)
        all_ids = [case["source"]["id"]] + [item["id"] for item in case["candidates"]]
        assert not note_ids.intersection(all_ids), "note IDs must be globally unique"
        note_ids.update(all_ids)
        assert set(case["expected_targets"]) == {
            item["id"] for item in case["candidates"] if item["expected"]
        }
        assert isinstance(case["expected_top30"], list)
        assert isinstance(case["cutoff_expected_targets"], list)
        assert set(case["cutoff_expected_targets"]) <= set(case["expected_targets"])
        assert all(note_id in all_ids for note_id in case["expected_top30"])
        for membership in case["memberships"]["tags"]:
            assert set(membership) == TAG_MEMBERSHIP_KEYS
            assert len(membership["note_ids"]) >= 2
            assert case["source"]["id"] in membership["note_ids"]
            assert all(note_id in all_ids for note_id in membership["note_ids"])
        for membership in case["memberships"]["sources"]:
            assert set(membership) == SOURCE_MEMBERSHIP_KEYS
            assert len(membership["note_ids"]) >= 2
            assert case["source"]["id"] in membership["note_ids"]
            assert all(note_id in all_ids for note_id in membership["note_ids"])
        for relationship in case["response"]["relationships"]:
            assert set(relationship) == RELATIONSHIP_KEYS
        for tag in case["response"]["tags"]:
            assert set(tag) == TAG_KEYS
            assert (tag["existing_tag"] is None) != (tag["new_tag"] is None)
    expected_matrix = {
        (domain, match_kind)
        for domain in ("medical", "technical", "research", "general")
        for match_kind in ("direct", "weak")
    }
    actual_matrix = {(case["domain"], case["match_kind"]) for case in cases}
    assert actual_matrix == expected_matrix, f"domain/match coverage {len(actual_matrix)}/8"
    assert sum(bool(case["largest"]) for case in cases) == 1
    for probe in probes:
        assert set(probe) == PROBE_KEYS
        assert probe["case_id"] in case_ids
        assert probe["mutation"] in {
            "unknown_candidate",
            "cross_owner",
            "already_linked",
            "rejected",
            "invalid_evidence",
        }
        assert probe["expected_error"] == "notes_graph_suggestion_unknown_reference"
    assert {probe["mutation"] for probe in probes} == {
        "unknown_candidate",
        "cross_owner",
        "already_linked",
        "rejected",
        "invalid_evidence",
    }
    return corpus


def _expanded(item: dict[str, object]) -> str:
    return (str(item["content"]) + " ") * int(item["repeat"])


def _authorize(db: CharactersRAGDB) -> None:
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (db.client_id, DATASET_ID),
        )


def _fingerprint(db: CharactersRAGDB, note_id: str) -> str:
    note = db.get_note_by_id(note_id, include_deleted=True)
    assert note is not None
    return content_fingerprint(note["title"], note["content"])


def _reject_pair(db: CharactersRAGDB, source_id: str, target_id: str, suffix: str) -> None:
    store = db.note_graph_suggestion_store
    admitted = store.admit_run(
        dataset_id=DATASET_ID,
        source_note_id=source_id,
        source_fingerprint=_fingerprint(db, source_id),
        provider="recorded",
        model="quality",
        capability_revision="quality-v1",
        prompt_contract_version="notes-graph-suggestions-v1",
        idempotency_key=f"reject-run-{suffix}",
        now=NOW,
    )
    queued = store.bind_admitted_run(
        dataset_id=DATASET_ID,
        run_id=admitted.run.id,
        expected_state="admitting",
        expected_revision=admitted.run.revision,
        job_id=f"reject-job-{suffix}",
        completion_token=f"reject-placeholder-{suffix}",
        replay_envelope={"run_id": admitted.run.id, "state": "queued"},
        now=NOW,
    )
    running = store.start_run(
        dataset_id=DATASET_ID,
        run_id=queued.id,
        expected_state="queued",
        expected_revision=queued.revision,
        expected_job_id=queued.job_id,
        acquired_completion_token=f"reject-lease-{suffix}",
        now=NOW,
    )
    suggestion_id = f"rejected-{suffix}"
    publishing = store.stage_suggestions(
        dataset_id=DATASET_ID,
        run_id=running.id,
        expected_state="running",
        expected_revision=running.revision,
        expected_job_id=running.job_id,
        expected_completion_token=running.expected_completion_token,
        result_digest=f"sha256:{'d' * 64}",
        candidates=(
            {
                "id": suggestion_id,
                "kind": "related_note",
                "target_note_id": target_id,
                "target_fingerprint": _fingerprint(db, target_id),
                "match_strength": "strong",
                "rationale": "Recorded rejection fixture",
                "evidence": (),
            },
        ),
        invalid_item_count=0,
        now=NOW,
    )
    store.activate_staged_run(
        dataset_id=DATASET_ID,
        run_id=publishing.id,
        expected_state="publishing",
        expected_revision=publishing.revision,
        observed_job_id=publishing.job_id,
        observed_completion_token=publishing.expected_completion_token,
        observed_result_digest=publishing.result_digest,
        now=NOW,
    )
    store.reject_suggestion(
        dataset_id=DATASET_ID,
        suggestion_id=suggestion_id,
        expected_revision=1,
        expected_source_fingerprint=_fingerprint(db, source_id),
        expected_target_fingerprint=_fingerprint(db, target_id),
        idempotency_key=f"reject-decision-{suffix}",
        now=NOW,
    )


def _seed_case(tmp_path: Path, case: dict[str, object]):
    db_path = tmp_path / f"{case['id']}.db"
    db = CharactersRAGDB(str(db_path), client_id="quality-owner")
    _authorize(db)
    source = case["source"]
    conversation_by_note: dict[str, str] = {}
    for index, membership in enumerate(case["memberships"]["sources"]):
        conversation_id = db.add_conversation(
            {
                "title": f"Quality source {case['id']} {index}",
                "source": membership["source"],
                "external_ref": membership["external_ref"],
            }
        )
        assert conversation_id is not None
        for note_id in membership["note_ids"]:
            assert note_id not in conversation_by_note
            conversation_by_note[note_id] = conversation_id
    db.add_note(
        source["title"],
        _expanded(source),
        note_id=source["id"],
        conversation_id=conversation_by_note.get(source["id"]),
    )
    cross_owner = CharactersRAGDB(str(db_path), client_id="other-owner")
    try:
        for candidate in case["candidates"]:
            owner = cross_owner if candidate["role"] == "cross_owner" else db
            owner.add_note(
                candidate["title"],
                _expanded(candidate),
                note_id=candidate["id"],
                conversation_id=conversation_by_note.get(candidate["id"]),
            )
    finally:
        cross_owner.close_all_connections()
    keyword_ids = {tag: db.add_keyword(tag) for tag in case["existing_tags"]}
    for membership in case["memberships"]["tags"]:
        keyword_id = keyword_ids.get(membership["tag"])
        if keyword_id is None:
            keyword_id = db.add_keyword(membership["tag"])
            keyword_ids[membership["tag"]] = keyword_id
        assert keyword_id is not None
        for note_id in membership["note_ids"]:
            assert db.link_note_to_keyword(note_id, keyword_id)
    for index, candidate in enumerate(case["candidates"]):
        if candidate["role"] == "already_linked":
            db.notes_link_store.upsert(
                edge_id=f"14000000-0000-4000-8000-{index:012d}",
                payload={
                    "source_note_id": source["id"],
                    "target_note_id": candidate["id"],
                    "type": "manual",
                    "directed": False,
                    "weight": 1.0,
                    "label": None,
                    "properties": {},
                    "created_at": NOW.isoformat(),
                    "last_modified": NOW.isoformat(),
                    "created_by": "quality-fixture",
                },
                expected_version=None,
            )
        elif candidate["role"] == "rejected":
            _reject_pair(db, source["id"], candidate["id"], f"{case['id']}-{index}")
    retrieval = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id=DATASET_ID,
        source_note_id=source["id"],
    )
    prepared = build_generation_request(
        retrieval=retrieval,
        source_title=source["title"],
        source_content=_expanded(source),
    )
    return db, retrieval, prepared


def _recorded_payload(case: dict[str, object], prepared) -> dict[str, object]:
    existing_by_display = {item.display_tag: item.tag_id for item in prepared.existing_tags}
    relationships = [
        {
            "target_note_id": item["target_id"],
            "rationale": item["rationale"],
            "source_evidence_ids": [prepared.source_evidence_ids[item["source_evidence"]]],
            "target_evidence_ids": [
                prepared.candidate_evidence_ids[item["target_id"]][item["target_evidence"]]
            ],
        }
        for item in case["response"]["relationships"]
    ]
    tags = [
        {
            "existing_tag_id": (
                existing_by_display[item["existing_tag"]]
                if item["existing_tag"] is not None
                else None
            ),
            "new_tag": item["new_tag"],
            "rationale": item["rationale"],
            "source_evidence_ids": [prepared.source_evidence_ids[item["source_evidence"]]],
        }
        for item in case["response"]["tags"]
    ]
    return {"relationships": relationships, "tags": tags}


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip()).casefold()


def _assert_expected_retrieval(case_id: str, retrieved_ids: list[str], expected_ids: list[str]) -> None:
    matched = [note_id for note_id in expected_ids if note_id in retrieved_ids]
    assert len(matched) == len(expected_ids), (
        f"{case_id} expected-target retrieval {len(matched)}/{len(expected_ids)}; "
        f"missing={sorted(set(expected_ids) - set(matched))}"
    )


def _assert_unique_relationships(case_id: str, target_ids: list[str], expected_count: int) -> None:
    assert len(target_ids) == expected_count, (
        f"{case_id} accepted relationship count {len(target_ids)}/{expected_count}"
    )
    assert len(target_ids) == len(set(target_ids)), (
        f"{case_id} duplicate accepted relationship targets: {target_ids}"
    )


def test_offline_grounding_corpus_meets_quality_and_budget_contracts(tmp_path) -> None:
    corpus = _load_corpus()
    expected_total = retrieved_total = 0
    evidence_total = evidence_valid = 0
    expected_tags = accepted_tags = 0
    largest_seen = False
    prepared_by_case = {}
    case_by_id = {case["id"]: case for case in corpus["cases"]}

    for case in corpus["cases"]:
        db, retrieval, prepared = _seed_case(tmp_path, case)
        prepared_by_case[case["id"]] = prepared
        try:
            retrieved_order = [item.note_id for item in retrieval.candidates]
            retrieved_ids = set(retrieved_order)
            expected = set(case["expected_targets"])
            expected_total += len(expected)
            retrieved_total += len(expected & retrieved_ids)
            forbidden = {
                item["id"]
                for item in case["candidates"]
                if item["role"] in {"cross_owner", "already_linked", "rejected"}
            }
            assert not forbidden & retrieved_ids, (
                f"{case['id']} invalid retrieval output: {sorted(forbidden & retrieved_ids)}"
            )
            membership_targets = {
                note_id
                for membership_kind in ("tags", "sources")
                for membership in case["memberships"][membership_kind]
                for note_id in membership["note_ids"]
                if note_id != case["source"]["id"]
            }
            _assert_expected_retrieval(case["id"], retrieved_order, sorted(membership_targets))
            if case["largest"]:
                assert retrieved_order == case["expected_top30"]

            payload = _recorded_payload(case, prepared)
            raw = json.dumps(payload, ensure_ascii=False)
            result = parse_and_validate_generation(raw, prepared=prepared)
            result_targets = [item.target_note_id for item in result.relationships]
            _assert_unique_relationships(case["id"], result_targets, len(case["expected_targets"]))
            assert set(result_targets) <= retrieved_ids
            assert set(result_targets) == set(case["expected_targets"])

            valid_source = {item.reference for item in prepared.source_evidence}
            valid_targets = {
                note_id: {item.reference for item in evidence}
                for note_id, evidence in prepared.candidate_evidence.items()
            }
            for relationship in result.relationships:
                for reference in relationship.source_evidence:
                    evidence_total += 1
                    evidence_valid += int(
                        reference in valid_source
                        and reconstruct_evidence(
                            reference,
                            title=case["source"]["title"],
                            content=_expanded(case["source"]),
                        )
                        is not None
                    )
                candidate = next(
                    item for item in case["candidates"] if item["id"] == relationship.target_note_id
                )
                for reference in relationship.target_evidence:
                    evidence_total += 1
                    evidence_valid += int(
                        reference in valid_targets[relationship.target_note_id]
                        and reconstruct_evidence(
                            reference,
                            title=candidate["title"],
                            content=_expanded(candidate),
                        )
                        is not None
                    )
            for tag in result.tags:
                for reference in tag.source_evidence:
                    evidence_total += 1
                    evidence_valid += int(reference in valid_source)
            normalized_tags = [tag.normalized_tag for tag in result.tags]
            assert len(normalized_tags) == len(set(normalized_tags)), (
                f"{case['id']} duplicate normalized tags survived"
            )
            assert all(tag == _normalized(tag) for tag in normalized_tags)
            expected_case_tags = set(case["expected_tags"])
            expected_tags += len(expected_case_tags)
            accepted_tags += len(expected_case_tags & set(normalized_tags))

            if case["largest"]:
                largest_seen = True
                assert len(case["candidates"]) > 30
                assert len(retrieval.candidates) == 30
                _assert_expected_retrieval(
                    case["id"],
                    retrieved_order,
                    case["cutoff_expected_targets"],
                )
                assert len(retrieval.tag_catalog) == MAX_TAG_CATALOG == 100
                assert len(prepared.candidate_ids) == 30
                assert prepared.estimated_input_tokens <= MAX_ESTIMATED_INPUT_TOKENS
                assert estimate_tokens(raw) <= MAX_OUTPUT_TOKENS
                assert len(result.relationships) == MAX_RELATIONSHIP_SUGGESTIONS == 5
                assert len(result.tags) == MAX_TAG_SUGGESTIONS == 5
                assert sum(tag.is_new for tag in result.tags) == MAX_NEW_TAG_SUGGESTIONS == 2
                assert len(prepared.source_evidence) == 4
                assert all(len(evidence) == 2 for evidence in prepared.candidate_evidence.values())

                cutoff_mutation = [
                    note_id
                    for note_id in retrieved_order
                    if note_id != case["cutoff_expected_targets"][-1]
                ]
                with pytest.raises(AssertionError, match=r"expected-target retrieval 2/3"):
                    _assert_expected_retrieval(
                        case["id"],
                        cutoff_mutation,
                        case["cutoff_expected_targets"],
                    )
            with pytest.raises(AssertionError, match="duplicate accepted relationship targets"):
                _assert_unique_relationships(
                    case["id"],
                    [*result_targets, result_targets[0]],
                    len(result_targets) + 1,
                )
        finally:
            db.close_all_connections()

    assert retrieved_total / expected_total >= 0.9, (
        f"top-30 expected-target recall {retrieved_total}/{expected_total}"
    )
    assert evidence_valid == evidence_total, (
        f"evidence-reference validity {evidence_valid}/{evidence_total}"
    )
    assert accepted_tags == expected_tags, (
        f"tag normalization acceptance {accepted_tags}/{expected_tags}"
    )
    assert largest_seen

    invalid_accepted = 0
    for probe in corpus["validation_probes"]:
        case = case_by_id[probe["case_id"]]
        prepared = prepared_by_case[probe["case_id"]]
        payload = _recorded_payload(case, prepared)
        relationship = payload["relationships"][0]
        if probe["mutation"] == "invalid_evidence":
            relationship["source_evidence_ids"] = ["unknown-evidence"]
        elif probe["mutation"] == "unknown_candidate":
            relationship["target_note_id"] = "19999999-0000-4000-8000-000000000999"
        else:
            relationship["target_note_id"] = next(
                item["id"]
                for item in case["candidates"]
                if item["role"] == probe["mutation"]
            )
        try:
            parse_and_validate_generation(json.dumps(payload), prepared=prepared)
        except SuggestionGenerationError as exc:
            assert exc.code == probe["expected_error"]
        else:
            invalid_accepted += 1
    assert invalid_accepted == 0, (
        f"cross-owner/unknown/already-linked/rejected/invalid-evidence output "
        f"{invalid_accepted}/{len(corpus['validation_probes'])}"
    )
