"""Pure H1 history selection contract tests."""

import json
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import (
    CapturedHistoryV1,
    CompareHistorySelectionV1,
    HistorySelectionEnvelopeV1,
)
from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import (
    HistorySelectionSnapshotV1 as WireHistorySelectionSnapshotV1,
)
from tldw_Server_API.app.core.Chat.history_selection import (
    HistoryFencesV1,
    HistoryMessageRevisionV1,
    HistorySelectionError,
    HistorySelectionSnapshotV1,
    HistorySelectionV1,
    bind_selected_history_content,
    canonical_comparison_json,
    canonical_selection_json,
    comparison_digest,
    resolve_comparison_projection,
    resolve_history_selection,
    resolve_legacy_projection,
    resolve_parent_path,
    selection_digest,
    selection_to_wire,
    snapshot_to_wire,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "history_selection_v1.json"


def test_history_selection_error_survives_generator_transaction_boundary() -> None:
    @contextmanager
    def transaction_boundary():
        yield

    with pytest.raises(HistorySelectionError) as caught:
        with transaction_boundary():
            raise HistorySelectionError("stale_selection")
    assert caught.value.code == "stale_selection"
    assert str(caught.value) == "stale_selection"


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


def test_capture_binds_text_and_multiple_images_to_manifest() -> None:
    selection = json.loads(FIXTURE.read_text())["selection_vectors"][0]["selection"]
    rows = [
        {"id": "u1", "revision": "1", "parent_id": None, "role": "user", "settled": True},
        {"id": "a1", "revision": "2", "parent_id": "u1", "role": "assistant", "settled": True},
    ]
    content = [
        {"id": "u1", "revision": "1", "message": "look", "images": ["image://1", "image://2"]},
        {"id": "a1", "revision": "2", "message": "seen", "images": []},
    ]
    bound = bind_selected_history_content(rows, content)
    assert [item["id"] for item in bound] == ["u1", "a1"]
    content[0]["images"].append("image://later")
    content[0]["message"] = "later"
    assert bound[0]["message"] == "look"
    assert bound[0]["images"] == ("image://1", "image://2")
    content[0]["images"].pop()
    content[0]["message"] = "look"
    capture = {
        "status": "captured",
        "snapshot": {"version": 1, "owner_key": "local:p", "conversation_id": "c", "fences": selection["fences"],
                     "nodes": rows, "source_digest": "source", "interpretation_status": {"kind": "parent_graph_v1"},
                     "storage_context_digest": "storage"},
        "rows": rows,
        "selected_content": content,
        "view": {"view_session_id": "v", "owner_key": "local:p", "conversation_id": "c",
                 "interpretation": {"kind": "parent_graph_v1"},
                 "cursor": {"kind": "after_message", "message_id": "a1"}, "selection_revision": 1},
        "purpose": "send", "storage_context_digest": "storage",
    }
    validated = CapturedHistoryV1.model_validate(capture)
    assert validated.selected_content[0].images == ("image://1", "image://2")
    content[0]["images"].append("image://late")
    assert validated.selected_content[0].images == ("image://1", "image://2")
    content[0]["images"].pop()
    assert CapturedHistoryV1.model_validate_json(validated.model_dump_json()) == validated
    for drift in [list(reversed(content)), [{**content[0], "revision": "changed"}, content[1]], content[:1]]:
        with pytest.raises((ValidationError, HistorySelectionError)):
            CapturedHistoryV1.model_validate({**capture, "selected_content": drift})
    with pytest.raises(ValidationError):
        CapturedHistoryV1.model_validate({**capture, "rows": [rows[0], rows[0]], "selected_content": [content[0], content[0]]})


def test_core_values_copy_and_freeze_nested_inputs() -> None:
    interpretation = {"kind": "parent_graph_v1"}
    cursor = {"kind": "after_message", "message_id": "a1"}
    metadata = {"revision": "m1", "children": [{"id": "img1"}]}
    node = {"id": "a1", "revision": "1", "parent_id": None, "metadata": metadata}
    status = {"kind": "legacy_linear_v1", "projection_id": "p", "ordered_path_ids": ["a1"]}
    fences = HistoryFencesV1("1", "1", "1")
    selection = HistorySelectionV1(1, "owner", "c", interpretation, cursor, 1, "send", (HistoryMessageRevisionV1("a1", "1"),), fences, "storage", "request", "digest")
    snapshot = HistorySelectionSnapshotV1(1, "owner", "c", fences, (node,), "source", status, "storage")
    interpretation["kind"] = "legacy_linear_v1"
    cursor["message_id"] = "a2"
    metadata["children"][0]["id"] = "img2"
    status["ordered_path_ids"].append("a2")
    assert selection.interpretation["kind"] == "parent_graph_v1"
    assert selection.cursor["message_id"] == "a1"
    assert snapshot.nodes[0]["metadata"]["children"][0]["id"] == "img1"
    assert snapshot.interpretation_status["ordered_path_ids"] == ("a1",)
    with pytest.raises((TypeError, FrozenInstanceError)):
        selection.interpretation["kind"] = "changed"
    with pytest.raises((TypeError, FrozenInstanceError)):
        snapshot.nodes[0]["metadata"]["children"][0]["id"] = "changed"
    assert HistorySelectionEnvelopeV1.model_validate({"selection": selection_to_wire(selection)}).selection.owner_key == "owner"
    wire_snapshot = snapshot_to_wire(snapshot)
    assert wire_snapshot["nodes"][0]["metadata"]["children"][0]["id"] == "img1"


def test_wire_snapshot_and_comparison_rows_are_strict_and_deeply_immutable() -> None:
    fixture = json.loads(FIXTURE.read_text())
    rows = fixture["comparison_vector"]["source_rows"]
    snapshot = WireHistorySelectionSnapshotV1.model_validate({
        "version": 1, "owner_key": "local:p", "conversation_id": "compare-c",
        "fences": {"conversation": "1", "history": "1", "settings": "1"},
        "nodes": rows, "source_digest": "source", "interpretation_status": {"kind": "parent_graph_v1"},
        "storage_context_digest": "storage",
    })
    assert snapshot.nodes[0].comparison.cluster_id == "r1"
    core_snapshot = HistorySelectionSnapshotV1(
        1, "local:p", "compare-c", HistoryFencesV1("1", "1", "1"),
        tuple(rows), "source", {"kind": "parent_graph_v1"}, "storage",
    )
    assert WireHistorySelectionSnapshotV1.model_validate(snapshot_to_wire(core_snapshot)) == snapshot
    assert WireHistorySelectionSnapshotV1.model_validate_json(snapshot.model_dump_json()) == snapshot
    rows[0]["comparison"]["cluster_id"] = "changed"
    assert snapshot.nodes[0].comparison.cluster_id == "r1"
    with pytest.raises((TypeError, AttributeError, FrozenInstanceError, ValidationError)):
        snapshot.nodes[0].comparison.cluster_id = "changed"
    with pytest.raises((TypeError, AttributeError, FrozenInstanceError, ValidationError)):
        snapshot.nodes[0] = snapshot.nodes[1]


def test_legacy_descendants_keep_their_protected_base_and_null_branch():
    nodes = [
        {"id": "a", "parent_id": None}, {"id": "b", "parent_id": None},
        {"id": "shared", "parent_id": None},
        {"id": "x", "parent_id": "shared", "legacy_projection_id": "first"},
        {"id": "y", "parent_id": "shared", "legacy_projection_id": "second"},
        {"id": "fresh", "parent_id": None, "legacy_projection_id": "first"},
    ]
    def path(base, target, projection):
        return [row["id"] for row in resolve_legacy_projection(nodes, base,
            {"kind": "after_message", "message_id": target}, projection_id=projection)]
    assert path(["a", "shared"], "x", "first") == ["a", "shared", "x"]
    assert path(["b", "shared"], "y", "second") == ["b", "shared", "y"]
    assert path(["a", "shared"], "fresh", "first") == ["fresh"]
    assert path(["a", "shared"], "shared", "first") == ["a", "shared"]
    with pytest.raises(HistorySelectionError, match="interpretation_mismatch"):
        path(["a", "shared"], "y", "first")
