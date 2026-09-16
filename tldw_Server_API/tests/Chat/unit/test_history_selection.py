"""Pure H1 history selection contract tests."""

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import (
    CompareHistorySelectionV1,
    HistorySelectionEnvelopeV1,
)
from tldw_Server_API.app.core.Chat.history_selection import (
    HistorySelectionError,
    canonical_comparison_json,
    canonical_selection_json,
    comparison_digest,
    resolve_comparison_projection,
    resolve_history_selection,
    resolve_legacy_projection,
    resolve_parent_path,
    selection_digest,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "history_selection_v1.json"


def test_selected_branch_and_before_root_are_stable() -> None:
    rows = [
        {"id": "u1", "revision": "1", "parent_id": None, "role": "user", "settled": True},
        {"id": "a1", "revision": "1", "parent_id": "u1", "role": "assistant", "settled": True},
        {"id": "a2", "revision": "1", "parent_id": "u1", "role": "assistant", "settled": True},
        {"id": "u2", "revision": "1", "parent_id": "a2", "role": "user", "settled": True},
    ]
    assert [row["id"] for row in resolve_parent_path(rows, {"kind": "after_message", "message_id": "a1"})] == ["u1", "a1"]
    assert resolve_parent_path(rows, {"kind": "before_message", "message_id": "u1"}) == []
    assert resolve_parent_path(rows, {"kind": "empty"}) == []


@pytest.mark.parametrize("rows,code", [
    ([{"id": "a", "parent_id": None}, {"id": "a", "parent_id": None}], "duplicate_message_id"),
    ([{"id": "a", "parent_id": "missing"}], "missing_parent"),
    ([{"id": "a", "parent_id": "b"}, {"id": "b", "parent_id": "a"}], "cyclic_ancestry"),
])
def test_invalid_ancestry_rejects(rows: list[dict], code: str) -> None:
    with pytest.raises(HistorySelectionError) as error:
        resolve_parent_path(rows, {"kind": "after_message", "message_id": "a"})
    assert error.value.code == code


@given(st.integers(min_value=1, max_value=80), st.integers(min_value=0, max_value=79))
def test_valid_paths_are_unique_and_before_is_proper_prefix(length: int, candidate: int) -> None:
    rows = [{"id": str(index), "parent_id": str(index - 1) if index else None} for index in range(length)]
    index = candidate % length
    original = [row.copy() for row in rows]
    after = resolve_parent_path(rows, {"kind": "after_message", "message_id": str(index)})
    before = resolve_parent_path(rows, {"kind": "before_message", "message_id": str(index)})
    assert len({row["id"] for row in after}) == len(after)
    assert before == after[:-1]
    assert rows == original


@given(st.integers(min_value=1, max_value=80))
def test_generated_cycle_and_orphan_reject(length: int) -> None:
    rows = [{"id": str(index), "parent_id": str(index - 1) if index else None} for index in range(length)]
    rows[0]["parent_id"] = str(length - 1)
    with pytest.raises(HistorySelectionError, match="cyclic_ancestry"):
        resolve_parent_path(rows, {"kind": "after_message", "message_id": str(length - 1)})
    rows[0]["parent_id"] = "absent"
    with pytest.raises(HistorySelectionError, match="missing_parent"):
        resolve_parent_path(rows, {"kind": "after_message", "message_id": str(length - 1)})


def test_shared_canonical_vectors() -> None:
    fixture = json.loads(FIXTURE.read_text())
    for vector in fixture["selection_vectors"]:
        assert canonical_selection_json(vector["selection"]) == vector["canonical_json"]
        assert selection_digest(vector["selection"]) == vector["selection_digest"]
    comparison = fixture["comparison_vector"]
    assert canonical_comparison_json(comparison["selection"]) == comparison["canonical_json"]
    assert comparison_digest(comparison["selection"]) == comparison["selection_digest"]
    assert [row["id"] for row in resolve_comparison_projection(comparison["source_rows"], "A", "uA")] == comparison["expected_path_ids"]


def test_native_resolution_returns_stale_without_writable_selection() -> None:
    snapshot = {
        "version": 1, "owner_key": "local:p", "conversation_id": "c",
        "fences": {"conversation": "1", "history": "1", "settings": "1"},
        "nodes": [{"id": "u1", "revision": "1", "parent_id": None, "role": "user", "settled": True}],
        "source_digest": "source", "interpretation_status": {"kind": "parent_graph_v1"},
        "storage_context_digest": "storage",
    }
    view = {
        "view_session_id": "v", "owner_key": "local:p", "conversation_id": "c",
        "interpretation": {"kind": "parent_graph_v1"}, "cursor": {"kind": "after_message", "message_id": "gone"},
        "selection_revision": 3,
    }
    assert resolve_history_selection(snapshot, view, "send", "request") == {"status": "stale_selection", "code": "missing_cursor"}


def test_cross_conversation_parent_rejects() -> None:
    rows = [
        {"id": "u1", "parent_id": None, "conversation_id": "c1"},
        {"id": "a1", "parent_id": "u1", "conversation_id": "c2"},
    ]
    with pytest.raises(HistorySelectionError, match="cross_conversation_parent"):
        resolve_parent_path(rows, {"kind": "after_message", "message_id": "a1"})


def test_orphan_outside_chosen_branch_rejects() -> None:
    rows = [
        {"id": "root", "parent_id": None},
        {"id": "selected", "parent_id": "root"},
        {"id": "orphan", "parent_id": "gone"},
    ]
    with pytest.raises(HistorySelectionError, match="missing_parent"):
        resolve_parent_path(rows, {"kind": "after_message", "message_id": "selected"})


def test_legacy_projection_rejects_cross_conversation_member() -> None:
    rows = [
        {"id": "a", "conversation_id": "c1"},
        {"id": "b", "conversation_id": "c2"},
    ]
    with pytest.raises(HistorySelectionError, match="cross_conversation_parent"):
        resolve_legacy_projection(rows, ["a", "b"], {"kind": "after_message", "message_id": "b"})


def test_strict_envelope_rejects_unknown_fields_and_malformed_cursor() -> None:
    selection = json.loads(FIXTURE.read_text())["selection_vectors"][0]["selection"]
    assert HistorySelectionEnvelopeV1.model_validate({"selection": selection}).selection.selection_digest == ""
    with pytest.raises(ValidationError):
        HistorySelectionEnvelopeV1.model_validate({"selection": {**selection, "unexpected": True}})
    with pytest.raises(ValidationError):
        HistorySelectionEnvelopeV1.model_validate({"selection": {**selection, "cursor": {"kind": "empty", "message_id": "forged"}}})


def test_comparison_selection_accepts_tagged_vector_and_rejects_extra_fields() -> None:
    comparison = json.loads(FIXTURE.read_text())["comparison_vector"]["selection"]
    assert CompareHistorySelectionV1.model_validate(comparison).model_id == "A"
    with pytest.raises(ValidationError):
        CompareHistorySelectionV1.model_validate({**comparison, "interpretation": {"kind": "parent_graph_v1"}})
