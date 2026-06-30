"""Shared bounded dialogue-tree expansion engine for persona robustness flows."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Any, Callable


@dataclass(frozen=True)
class DialogueTreeBudget:
    max_depth: int = 1
    max_branching: int = 2
    max_candidates: int = 16
    max_provider_calls: int = 16

    def __post_init__(self) -> None:
        for field_name in (
            "max_depth",
            "max_branching",
            "max_candidates",
            "max_provider_calls",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")


@dataclass(frozen=True)
class TreeCandidate:
    action_type: str
    text: str = ""
    tool_plan: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DialogueTreeNode:
    node_id: str
    parent_node_id: str | None
    depth: int
    candidate: TreeCandidate | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DialogueTreeResult:
    nodes: tuple[DialogueTreeNode, ...]
    children_by_parent: Mapping[str, tuple[str, ...]]
    max_depth_seen: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(
            self,
            "children_by_parent",
            MappingProxyType(
                {
                    str(parent): tuple(children)
                    for parent, children in dict(self.children_by_parent).items()
                }
            ),
        )


class DialogueTreeEngine:
    def __init__(
        self,
        budget: DialogueTreeBudget,
        generators: list[Callable[[DialogueTreeNode], list[TreeCandidate]]],
    ) -> None:
        self.budget = budget
        self.generators = generators

    def expand(self, root_payload: dict[str, Any] | None = None) -> DialogueTreeResult:
        root = DialogueTreeNode(
            node_id="root",
            parent_node_id=None,
            depth=0,
            candidate=None,
            payload=dict(root_payload or {}),
        )
        nodes: list[DialogueTreeNode] = [root]
        children_by_parent: dict[str, list[str]] = defaultdict(list)
        frontier: deque[DialogueTreeNode] = deque([root])
        selected_candidates = 0
        provider_calls = 0
        max_depth_seen = 0

        while frontier:
            node = frontier.popleft()
            max_depth_seen = max(max_depth_seen, node.depth)
            if node.depth >= self.budget.max_depth:
                continue
            if self.budget.max_branching == 0:
                continue
            if selected_candidates >= self.budget.max_candidates:
                continue
            if provider_calls >= self.budget.max_provider_calls:
                continue

            generated_candidates, calls_used = self._collect_candidates(
                node=node,
                remaining_provider_calls=self.budget.max_provider_calls - provider_calls,
            )
            provider_calls += calls_used
            remaining_capacity = self.budget.max_candidates - selected_candidates
            selected = generated_candidates[: min(self.budget.max_branching, remaining_capacity)]

            for candidate in selected:
                child_position = len(children_by_parent[node.node_id]) + 1
                child_node = DialogueTreeNode(
                    node_id=f"{node.node_id}.{child_position}",
                    parent_node_id=node.node_id,
                    depth=node.depth + 1,
                    candidate=candidate,
                )
                children_by_parent[node.node_id].append(child_node.node_id)
                nodes.append(child_node)
                frontier.append(child_node)
                selected_candidates += 1
                max_depth_seen = max(max_depth_seen, child_node.depth)

                if selected_candidates >= self.budget.max_candidates:
                    break

        return DialogueTreeResult(
            nodes=tuple(nodes),
            children_by_parent=MappingProxyType(
                {parent: tuple(children) for parent, children in children_by_parent.items()}
            ),
            max_depth_seen=max_depth_seen,
        )

    def _collect_candidates(
        self,
        *,
        node: DialogueTreeNode,
        remaining_provider_calls: int,
    ) -> tuple[list[TreeCandidate], int]:
        candidates: list[TreeCandidate] = []
        calls_used = 0
        for generator in self.generators:
            if calls_used >= remaining_provider_calls:
                break
            generated = generator(node) or []
            calls_used += 1
            candidates.extend(generated)

        return sorted(candidates, key=self._candidate_sort_key), calls_used

    @staticmethod
    def _candidate_sort_key(candidate: TreeCandidate) -> tuple[str, str, str, str]:
        metadata = _stable_payload_digest(candidate.metadata)
        tool_plan = _stable_payload_digest(candidate.tool_plan)
        return (candidate.text, candidate.action_type, metadata, tool_plan)


def _stable_payload_digest(value: Any) -> str:
    normalized = _normalize_for_sort(value)
    payload = repr(normalized).encode("utf-8", errors="replace")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def _normalize_for_sort(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _normalize_for_sort(sub_value))
            for key, sub_value in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, list):
        return tuple(_normalize_for_sort(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_normalize_for_sort(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_normalize_for_sort(item) for item in value), key=repr))
    return value


__all__ = [
    "DialogueTreeBudget",
    "DialogueTreeEngine",
    "DialogueTreeNode",
    "DialogueTreeResult",
    "TreeCandidate",
]
