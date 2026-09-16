"""Pure, owner-neutral H1 history selection values and canonical encoding.

Owners validate storage context separately. A digest is provenance, not authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class HistorySelectionError(ValueError):
    """A stable code for malformed or unavailable ancestry."""

    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True)
class HistoryMessageRevisionV1:
    """Immutable accepted source member identity."""

    id: str
    revision: str


@dataclass(frozen=True)
class HistoryFencesV1:
    """Independent conversation, history and settings revisions."""

    conversation: str
    history: str
    settings: str


@dataclass(frozen=True)
class HistorySelectionV1:
    """Finalized selection submitted to an owner for admission."""

    version: int
    owner_key: str
    conversation_id: str
    interpretation: Mapping[str, str]
    cursor: Mapping[str, str]
    selection_revision: int
    purpose: str
    messages: tuple[HistoryMessageRevisionV1, ...]
    fences: HistoryFencesV1
    storage_context_digest: str
    request_context_digest: str
    selection_digest: str

    def __post_init__(self) -> None:
        """Detach caller-owned nested values before a selection can be admitted."""

        object.__setattr__(self, "interpretation", _freeze_json(self.interpretation))
        object.__setattr__(self, "cursor", _freeze_json(self.cursor))
        object.__setattr__(self, "messages", tuple(self.messages))


@dataclass(frozen=True)
class HistorySelectionSnapshotV1:
    """A complete owner-captured lightweight source manifest."""

    version: int
    owner_key: str
    conversation_id: str
    fences: HistoryFencesV1
    nodes: tuple[Mapping[str, Any], ...]
    source_digest: str
    interpretation_status: Mapping[str, Any]
    storage_context_digest: str

    def __post_init__(self) -> None:
        """Freeze the complete manifest and selected interpretation evidence."""

        object.__setattr__(self, "nodes", tuple(_freeze_json(row) for row in self.nodes))
        object.__setattr__(self, "interpretation_status", _freeze_json(self.interpretation_status))


@dataclass(frozen=True)
class HistoryAdmissionV1:
    """Owner-issued accepted input and immutable selected membership."""

    version: int
    owner_key: str
    conversation_id: str
    selection_digest: str
    messages: tuple[HistoryMessageRevisionV1, ...]
    input_message_id: str
    input_message_revision: str
    originating_selection_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))


def _freeze_json(value: Any) -> Any:
    """Copy H1's JSON-shaped captured fields into standard immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(member) for key, member in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(member) for member in value)
    return value


def _wire_json(value: Any) -> Any:
    """Convert captured JSON values to plain wire containers without deepcopy."""

    if isinstance(value, Mapping):
        return {key: _wire_json(member) for key, member in value.items()}
    if isinstance(value, tuple):
        return [_wire_json(member) for member in value]
    return value


def selection_to_wire(selection: HistorySelectionV1) -> dict[str, Any]:
    """Serialize a frozen core selection for strict Pydantic validation."""

    return {
        "version": selection.version,
        "owner_key": selection.owner_key,
        "conversation_id": selection.conversation_id,
        "interpretation": _wire_json(selection.interpretation),
        "cursor": _wire_json(selection.cursor),
        "selection_revision": selection.selection_revision,
        "purpose": selection.purpose,
        "messages": [{"id": member.id, "revision": member.revision} for member in selection.messages],
        "fences": {"conversation": selection.fences.conversation, "history": selection.fences.history, "settings": selection.fences.settings},
        "storage_context_digest": selection.storage_context_digest,
        "request_context_digest": selection.request_context_digest,
        "selection_digest": selection.selection_digest,
    }


def snapshot_to_wire(snapshot: HistorySelectionSnapshotV1) -> dict[str, Any]:
    """Serialize frozen manifest fields as ordinary JSON-shaped wire values."""

    return {
        "version": snapshot.version,
        "owner_key": snapshot.owner_key,
        "conversation_id": snapshot.conversation_id,
        "fences": {"conversation": snapshot.fences.conversation, "history": snapshot.fences.history, "settings": snapshot.fences.settings},
        "nodes": _wire_json(snapshot.nodes),
        "source_digest": snapshot.source_digest,
        "interpretation_status": _wire_json(snapshot.interpretation_status),
        "storage_context_digest": snapshot.storage_context_digest,
    }


def bind_selected_history_content(
    rows: Sequence[Mapping[str, Any]], content: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    """Reject content loaded at another source revision or in another order."""

    if len(rows) != len(content) or len({row["id"] for row in rows}) != len(rows) or any(
        row["id"] != item["id"] or row["revision"] != item["revision"]
        for row, item in zip(rows, content)
    ):
        raise HistorySelectionError("selected_content_mismatch")
    return tuple(
        _freeze_json({"id": item["id"], "revision": item["revision"], "message": item["message"], "images": item["images"]})
        for item in content
    )


def resolve_parent_path(
    nodes: Sequence[Mapping[str, Any]], cursor: Mapping[str, str]
) -> list[Mapping[str, Any]]:
    """Follow explicit parent IDs without sorting, content checks or mutation."""

    by_id: dict[str, Mapping[str, Any]] = {}
    conversation_id: str | None = None
    for row in nodes:
        row_id = row["id"]
        if not row_id or row_id in by_id:
            raise HistorySelectionError("duplicate_message_id")
        row_conversation = row.get("conversation_id")
        if row_conversation:
            if conversation_id and conversation_id != row_conversation:
                raise HistorySelectionError("cross_conversation_parent")
            conversation_id = row_conversation
        by_id[row_id] = row

    resolved: set[str] = set()
    for row in nodes:
        walking: set[str] = set()
        current: Mapping[str, Any] | None = row
        while current is not None and current["id"] not in resolved:
            if current["id"] in walking:
                raise HistorySelectionError("cyclic_ancestry")
            walking.add(current["id"])
            parent_id = current.get("parent_id")
            if parent_id is None:
                break
            parent = by_id.get(parent_id)
            if parent is None:
                raise HistorySelectionError("missing_parent")
            if current.get("conversation_id") and parent.get("conversation_id") != current["conversation_id"]:
                raise HistorySelectionError("cross_conversation_parent")
            current = parent
        resolved.update(walking)

    if cursor["kind"] not in ("empty", "before_message", "after_message"):
        raise HistorySelectionError("invalid_cursor")
    if cursor["kind"] == "empty":
        return []
    target = by_id.get(cursor["message_id"])
    if target is None:
        raise HistorySelectionError("missing_cursor")
    reversed_path: list[Mapping[str, Any]] = []
    visited: set[str] = set()
    current: Mapping[str, Any] = target
    while True:
        if current["id"] in visited:
            raise HistorySelectionError("cyclic_ancestry")
        visited.add(current["id"])
        reversed_path.append(current)
        parent_id = current.get("parent_id")
        if parent_id is None:
            break
        parent = by_id.get(parent_id)
        if parent is None:
            raise HistorySelectionError("missing_parent")
        if current.get("conversation_id") and parent.get("conversation_id") != current["conversation_id"]:
            raise HistorySelectionError("cross_conversation_parent")
        current = parent
    reversed_path.reverse()
    return reversed_path[:-1] if cursor["kind"] == "before_message" else reversed_path


def resolve_legacy_projection(
    nodes: Sequence[Mapping[str, Any]], ordered_path_ids: Sequence[str], cursor: Mapping[str, str]
) -> list[Mapping[str, Any]]:
    """Read reviewed membership without rewriting source rows or parent edges."""

    by_id = {row["id"]: row for row in nodes}
    if len(by_id) != len(nodes):
        raise HistorySelectionError("duplicate_message_id")
    conversations = {row.get("conversation_id") for row in nodes if row.get("conversation_id")}
    if len(conversations) > 1:
        raise HistorySelectionError("cross_conversation_parent")
    if len(set(ordered_path_ids)) != len(ordered_path_ids):
        raise HistorySelectionError("duplicate_projection_member")
    try:
        path = [by_id[row_id] for row_id in ordered_path_ids]
    except KeyError as exc:
        raise HistorySelectionError("missing_projection_member") from exc
    if cursor["kind"] not in ("empty", "before_message", "after_message"):
        raise HistorySelectionError("invalid_cursor")
    if cursor["kind"] == "empty":
        return []
    try:
        at = ordered_path_ids.index(cursor["message_id"])
    except ValueError as exc:
        raise HistorySelectionError("missing_cursor") from exc
    return path[: at + (cursor["kind"] == "after_message")]


def canonical_selection_tuple(selection: Mapping[str, Any]) -> list[Any]:
    """Return the exact nine-member H1 normal selection tuple."""

    interpretation = selection["interpretation"]
    cursor = selection["cursor"]
    return [
        1,
        selection["owner_key"],
        selection["conversation_id"],
        [interpretation["kind"], interpretation.get("projection_id")],
        [cursor["kind"], cursor.get("message_id")],
        selection["purpose"],
        [[member["id"], member["revision"]] for member in selection["messages"]],
        selection["storage_context_digest"],
        selection["request_context_digest"],
    ]


def canonical_selection_json(selection: Mapping[str, Any]) -> str:
    """Encode compact JSON with literal Unicode and no normalization."""

    return json.dumps(canonical_selection_tuple(selection), ensure_ascii=False, separators=(",", ":"))


def selection_digest(selection: Mapping[str, Any]) -> str:
    """Hash the canonical normal tuple as UTF-8 SHA-256."""

    return hashlib.sha256(canonical_selection_json(selection).encode("utf-8")).hexdigest()


def canonical_comparison_tuple(selection: Mapping[str, Any]) -> list[Any]:
    """Comparison tuple is tagged and independent of normal parent ancestry."""

    cursor = selection["cursor"]
    return [
        1, "comparison", selection["owner_key"], selection["conversation_id"],
        selection["model_id"], selection["cluster_id"],
        [cursor["kind"], cursor.get("message_id")],
        [[member["id"], member["revision"]] for member in selection["messages"]],
        selection["storage_context_digest"], selection["request_context_digest"],
    ]


def canonical_comparison_json(selection: Mapping[str, Any]) -> str:
    """Encode a comparison selection with the same Unicode rules."""

    return json.dumps(canonical_comparison_tuple(selection), ensure_ascii=False, separators=(",", ":"))


def comparison_digest(selection: Mapping[str, Any]) -> str:
    """Hash the tagged comparison tuple as UTF-8 SHA-256."""

    return hashlib.sha256(canonical_comparison_json(selection).encode("utf-8")).hexdigest()


def resolve_comparison_projection(
    nodes: Sequence[Mapping[str, Any]], model_id: str, boundary_id: str
) -> list[Mapping[str, Any]]:
    """Retain common rounds and one model in source order, ignoring cross-model edges."""

    seen: set[str] = set()
    conversation_id: str | None = None
    boundary_index: int | None = None
    for index, row in enumerate(nodes):
        row_id = row["id"]
        if not row_id or row_id in seen:
            raise HistorySelectionError("duplicate_message_id")
        seen.add(row_id)
        row_conversation = row.get("conversation_id")
        if row_conversation:
            if conversation_id and conversation_id != row_conversation:
                raise HistorySelectionError("cross_conversation_parent")
            conversation_id = row_conversation
        if row_id == boundary_id:
            comparison = row.get("comparison") or {}
            if not comparison or not (comparison.get("common") or comparison.get("model_id") == model_id):
                raise HistorySelectionError("invalid_comparison_boundary")
            boundary_index = index
    if boundary_index is None:
        raise HistorySelectionError("invalid_comparison_boundary")
    return [
        row for row in nodes[: boundary_index + 1]
        if row.get("comparison", {}).get("common") or row.get("comparison", {}).get("model_id") == model_id
    ]


def resolve_history_selection(
    snapshot: Mapping[str, Any], view: Mapping[str, Any], purpose: str, request_context_digest: str
) -> dict[str, Any]:
    """Resolve a coherent owner snapshot to a selection or structured non-ready state."""

    if snapshot["version"] != 1:
        return {"status": "unsupported_history_capability", "code": "unsupported_version"}
    if snapshot["owner_key"] != view["owner_key"] or snapshot["conversation_id"] != view["conversation_id"]:
        return {"status": "invalid_history", "code": "owner_conversation_mismatch"}
    status = snapshot["interpretation_status"]
    if status["kind"] == "legacy_review_required":
        return {"status": "legacy_review_required", "code": "legacy_review_required"}
    interpretation = view["interpretation"]
    if status["kind"] != interpretation["kind"] or (
        status["kind"] == "legacy_linear_v1" and status["projection_id"] != interpretation["projection_id"]
    ):
        return {"status": "stale_selection", "code": "interpretation_mismatch"}
    try:
        nodes = snapshot["nodes"]
        if any(row.get("conversation_id") and row["conversation_id"] != snapshot["conversation_id"] for row in nodes):
            raise HistorySelectionError("cross_conversation_parent")
        rows = (
            resolve_parent_path(nodes, view["cursor"])
            if status["kind"] == "parent_graph_v1"
            else resolve_legacy_projection(nodes, status["ordered_path_ids"], view["cursor"])
        )
        if any(not row["settled"] for row in rows):
            raise HistorySelectionError("unsettled_message")
        selection = {
            "version": 1,
            "owner_key": snapshot["owner_key"],
            "conversation_id": snapshot["conversation_id"],
            "interpretation": interpretation,
            "cursor": view["cursor"],
            "selection_revision": view["selection_revision"],
            "purpose": purpose,
            "messages": [{"id": row["id"], "revision": row["revision"]} for row in rows],
            "fences": snapshot["fences"],
            "storage_context_digest": snapshot["storage_context_digest"],
            "request_context_digest": request_context_digest,
        }
        selection["selection_digest"] = selection_digest(selection)
        return {"status": "ready", "selection": selection, "rows": rows}
    except HistorySelectionError as exc:
        return {"status": "stale_selection" if exc.code == "missing_cursor" else "invalid_history", "code": exc.code}
