import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from tldw_Server_API.app.core.MCP_unified.modules.implementations.kanban_module import KanbanModule
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.DB_Management.Kanban_DB import ConflictError, NotFoundError, InputError


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeKanbanDB:
    def __init__(self) -> None:
        self._boards: Dict[int, Dict[str, Any]] = {}
        self._lists: Dict[int, Dict[str, Any]] = {}
        self._cards: Dict[int, Dict[str, Any]] = {}
        self._labels: Dict[int, Dict[str, Any]] = {}
        self._card_labels: Dict[int, set[int]] = {}
        self._comments: Dict[int, Dict[str, Any]] = {}
        self._checklists: Dict[int, Dict[str, Any]] = {}
        self._checklist_items: Dict[int, Dict[str, Any]] = {}
        self._board_counter = 0
        self._list_counter = 0
        self._card_counter = 0
        self._label_counter = 0
        self._comment_counter = 0
        self._checklist_counter = 0
        self._checklist_item_counter = 0
        self._workflow_policy_counter = 0
        self._workflow_policies: Dict[int, Dict[str, Any]] = {}
        self._workflow_states: Dict[int, Dict[str, Any]] = {}
        self._workflow_events: Dict[int, List[Dict[str, Any]]] = {}

    def create_board(
        self,
        name: str,
        client_id: str,
        description: Optional[str] = None,
        activity_retention_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._board_counter += 1
        board = {
            "id": self._board_counter,
            "name": name,
            "client_id": client_id,
            "description": description,
            "activity_retention_days": activity_retention_days,
            "metadata": metadata,
            "archived": 0,
            "deleted": 0,
        }
        self._boards[self._board_counter] = board
        return dict(board)

    def list_boards(
        self,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        boards = [
            b for b in self._boards.values()
            if (include_archived or not b.get("archived"))
            and (include_deleted or not b.get("deleted"))
        ]
        total = len(boards)
        return boards[offset: offset + limit], total

    def get_board(self, board_id: int, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
        board = self._boards.get(board_id)
        if not board:
            return None
        if board.get("deleted") and not include_deleted:
            return None
        return dict(board)

    def create_list(
        self,
        board_id: int,
        name: str,
        client_id: str,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        if board_id not in self._boards:
            raise NotFoundError("Board not found", entity="board", entity_id=board_id)
        self._list_counter += 1
        lst = {
            "id": self._list_counter,
            "board_id": board_id,
            "name": name,
            "client_id": client_id,
            "position": position or 0,
            "archived": 0,
            "deleted": 0,
        }
        self._lists[self._list_counter] = lst
        return dict(lst)

    def list_lists(
        self,
        board_id: int,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        if board_id not in self._boards:
            raise NotFoundError("Board not found", entity="board", entity_id=board_id)
        return [
            dict(lst)
            for lst in self._lists.values()
            if lst["board_id"] == board_id
            and (include_archived or not lst.get("archived"))
            and (include_deleted or not lst.get("deleted"))
        ]

    def create_card(
        self,
        list_id: int,
        title: str,
        client_id: str,
        description: Optional[str] = None,
        position: Optional[int] = None,
        due_date: Optional[str] = None,
        start_date: Optional[str] = None,
        priority: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if list_id not in self._lists:
            raise NotFoundError("List not found", entity="list", entity_id=list_id)
        self._card_counter += 1
        card = {
            "id": self._card_counter,
            "list_id": list_id,
            "board_id": self._lists[list_id]["board_id"],
            "title": title,
            "description": description,
            "client_id": client_id,
            "position": position or 0,
            "due_date": due_date,
            "start_date": start_date,
            "priority": priority,
            "metadata": metadata,
            "archived": 0,
            "deleted": 0,
        }
        self._cards[self._card_counter] = card
        return dict(card)

    def list_cards(
        self,
        list_id: int,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        if list_id not in self._lists:
            raise NotFoundError("List not found", entity="list", entity_id=list_id)
        return [
            dict(card)
            for card in self._cards.values()
            if card["list_id"] == list_id
            and (include_archived or not card.get("archived"))
            and (include_deleted or not card.get("deleted"))
        ]

    def move_card(
        self,
        card_id: int,
        target_list_id: int,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        card = self._cards.get(card_id)
        if not card:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        if target_list_id not in self._lists:
            raise NotFoundError("List not found", entity="list", entity_id=target_list_id)
        if self._lists[target_list_id]["board_id"] != card["board_id"]:
            raise InputError("Cannot move card to a list in a different board")
        card["list_id"] = target_list_id
        if position is not None:
            card["position"] = position
        return dict(card)

    def search_cards(
        self,
        query: str,
        board_id: Optional[int] = None,
        label_ids: Optional[List[int]] = None,
        priority: Optional[str] = None,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        needle = query.lower()
        results = []
        for card in self._cards.values():
            if board_id and card["board_id"] != board_id:
                continue
            if priority and card.get("priority") != priority:
                continue
            if not include_archived and card.get("archived"):
                continue
            hay = f"{card.get('title', '')} {card.get('description', '')}".lower()
            if needle in hay:
                results.append(dict(card))
        total = len(results)
        return results[offset: offset + limit], total

    def create_label(self, board_id: int, name: str, color: Optional[str] = None) -> Dict[str, Any]:
        if board_id not in self._boards:
            raise NotFoundError("Board not found", entity="board", entity_id=board_id)
        self._label_counter += 1
        label = {
            "id": self._label_counter,
            "board_id": board_id,
            "name": name,
            "color": color,
        }
        self._labels[self._label_counter] = label
        return dict(label)

    def list_labels(self, board_id: int) -> List[Dict[str, Any]]:
        if board_id not in self._boards:
            raise NotFoundError("Board not found", entity="board", entity_id=board_id)
        return [dict(lbl) for lbl in self._labels.values() if lbl["board_id"] == board_id]

    def update_label(
        self,
        label_id: int,
        name: Optional[str] = None,
        color: Optional[str] = None,
    ) -> Dict[str, Any]:
        label = self._labels.get(label_id)
        if not label:
            raise NotFoundError("Label not found", entity="label", entity_id=label_id)
        if name is not None:
            label["name"] = name
        if color is not None:
            label["color"] = color
        return dict(label)

    def delete_label(self, label_id: int) -> bool:
        if label_id not in self._labels:
            return False
        self._labels.pop(label_id, None)
        for card_id in list(self._card_labels.keys()):
            self._card_labels[card_id].discard(label_id)
        return True

    def assign_label_to_card(self, card_id: int, label_id: int) -> bool:
        card = self._cards.get(card_id)
        label = self._labels.get(label_id)
        if not card:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        if not label:
            raise NotFoundError("Label not found", entity="label", entity_id=label_id)
        if label["board_id"] != card["board_id"]:
            raise InputError("Label does not belong to the card's board")
        self._card_labels.setdefault(card_id, set()).add(label_id)
        return True

    def remove_label_from_card(self, card_id: int, label_id: int) -> bool:
        if card_id not in self._cards:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        if label_id not in self._labels:
            raise NotFoundError("Label not found", entity="label", entity_id=label_id)
        labels = self._card_labels.setdefault(card_id, set())
        if label_id in labels:
            labels.discard(label_id)
            return True
        return False

    def get_card_labels(self, card_id: int) -> List[Dict[str, Any]]:
        if card_id not in self._cards:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        label_ids = self._card_labels.get(card_id, set())
        return [dict(self._labels[lid]) for lid in label_ids if lid in self._labels]

    def create_comment(self, card_id: int, content: str) -> Dict[str, Any]:
        if card_id not in self._cards:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        self._comment_counter += 1
        comment = {
            "id": self._comment_counter,
            "card_id": card_id,
            "content": content,
            "deleted": 0,
        }
        self._comments[self._comment_counter] = comment
        return dict(comment)

    def list_comments(
        self,
        card_id: int,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if card_id not in self._cards:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        comments = [
            dict(comment)
            for comment in self._comments.values()
            if comment["card_id"] == card_id
            and (include_deleted or not comment.get("deleted"))
        ]
        total = len(comments)
        return comments[offset: offset + limit], total

    def update_comment(self, comment_id: int, content: str) -> Dict[str, Any]:
        comment = self._comments.get(comment_id)
        if not comment:
            raise NotFoundError("Comment not found", entity="comment", entity_id=comment_id)
        comment["content"] = content
        return dict(comment)

    def delete_comment(self, comment_id: int, hard_delete: bool = False) -> bool:
        comment = self._comments.get(comment_id)
        if not comment:
            return False
        if hard_delete:
            self._comments.pop(comment_id, None)
            return True
        comment["deleted"] = 1
        return True

    def create_checklist(
        self,
        card_id: int,
        name: str,
        position: Optional[int] = None,
    ) -> Dict[str, Any]:
        if card_id not in self._cards:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        self._checklist_counter += 1
        checklist = {
            "id": self._checklist_counter,
            "card_id": card_id,
            "name": name,
            "position": position or 0,
        }
        self._checklists[self._checklist_counter] = checklist
        return dict(checklist)

    def list_checklists(self, card_id: int) -> List[Dict[str, Any]]:
        if card_id not in self._cards:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        return [
            dict(ch)
            for ch in self._checklists.values()
            if ch["card_id"] == card_id
        ]

    def update_checklist(self, checklist_id: int, name: Optional[str] = None) -> Dict[str, Any]:
        checklist = self._checklists.get(checklist_id)
        if not checklist:
            raise NotFoundError("Checklist not found", entity="checklist", entity_id=checklist_id)
        if name is not None:
            checklist["name"] = name
        return dict(checklist)

    def delete_checklist(self, checklist_id: int) -> bool:
        if checklist_id not in self._checklists:
            return False
        self._checklists.pop(checklist_id, None)
        for item_id in list(self._checklist_items.keys()):
            if self._checklist_items[item_id]["checklist_id"] == checklist_id:
                self._checklist_items.pop(item_id, None)
        return True

    def create_checklist_item(
        self,
        checklist_id: int,
        name: str,
        position: Optional[int] = None,
        checked: bool = False,
    ) -> Dict[str, Any]:
        if checklist_id not in self._checklists:
            raise NotFoundError("Checklist not found", entity="checklist", entity_id=checklist_id)
        self._checklist_item_counter += 1
        item = {
            "id": self._checklist_item_counter,
            "checklist_id": checklist_id,
            "name": name,
            "position": position or 0,
            "checked": bool(checked),
        }
        self._checklist_items[self._checklist_item_counter] = item
        return dict(item)

    def list_checklist_items(self, checklist_id: int) -> List[Dict[str, Any]]:
        if checklist_id not in self._checklists:
            raise NotFoundError("Checklist not found", entity="checklist", entity_id=checklist_id)
        return [
            dict(item)
            for item in self._checklist_items.values()
            if item["checklist_id"] == checklist_id
        ]

    def update_checklist_item(
        self,
        item_id: int,
        name: Optional[str] = None,
        checked: Optional[bool] = None,
    ) -> Dict[str, Any]:
        item = self._checklist_items.get(item_id)
        if not item:
            raise NotFoundError("Checklist item not found", entity="checklist_item", entity_id=item_id)
        if name is not None:
            item["name"] = name
        if checked is not None:
            item["checked"] = bool(checked)
        return dict(item)

    def delete_checklist_item(self, item_id: int) -> bool:
        if item_id not in self._checklist_items:
            return False
        self._checklist_items.pop(item_id, None)
        return True

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _ensure_workflow_policy_for_board(self, board_id: int) -> Dict[str, Any]:
        policy = self._workflow_policies.get(board_id)
        if policy:
            return policy
        return self.upsert_workflow_policy(board_id=board_id)

    def _ensure_workflow_state_for_card(self, card_id: int) -> Dict[str, Any]:
        state = self._workflow_states.get(card_id)
        if state:
            return state
        card = self._cards.get(card_id)
        if not card:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        policy = self._ensure_workflow_policy_for_board(card["board_id"])
        default_status = policy["statuses"][0]["status_key"]
        now = self._now()
        created = {
            "card_id": card_id,
            "policy_id": policy["id"],
            "workflow_status_key": default_status,
            "lease_owner": None,
            "lease_expires_at": None,
            "approval_state": "none",
            "pending_transition_id": None,
            "retry_counters": None,
            "last_transition_at": None,
            "last_actor": None,
            "version": 1,
            "created_at": now,
            "updated_at": now,
        }
        self._workflow_states[card_id] = created
        return dict(created)

    def _append_workflow_event(
        self,
        card_id: int,
        event_type: str,
        from_status_key: Optional[str],
        to_status_key: Optional[str],
        actor: str,
        idempotency_key: str,
        correlation_id: Optional[str],
        reason: Optional[str],
        before_snapshot: Optional[Dict[str, Any]],
        after_snapshot: Optional[Dict[str, Any]],
    ) -> None:
        events = self._workflow_events.setdefault(card_id, [])
        event_id = len(events) + 1
        events.append(
            {
                "id": event_id,
                "card_id": card_id,
                "event_type": event_type,
                "from_status_key": from_status_key,
                "to_status_key": to_status_key,
                "actor": actor,
                "reason": reason,
                "idempotency_key": idempotency_key,
                "correlation_id": correlation_id,
                "before_snapshot": before_snapshot,
                "after_snapshot": after_snapshot,
                "created_at": self._now(),
            }
        )

    def upsert_workflow_policy(
        self,
        *,
        board_id: int,
        statuses: Optional[List[Dict[str, Any]]] = None,
        transitions: Optional[List[Dict[str, Any]]] = None,
        is_paused: bool = False,
        is_draining: bool = False,
        default_lease_ttl_sec: int = 900,
        strict_projection: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if board_id not in self._boards:
            raise NotFoundError("Board not found", entity="board", entity_id=board_id)
        existing = self._workflow_policies.get(board_id)
        if existing:
            policy = dict(existing)
            policy["version"] = int(policy.get("version", 1)) + 1
        else:
            self._workflow_policy_counter += 1
            now = self._now()
            policy = {
                "id": self._workflow_policy_counter,
                "board_id": board_id,
                "version": 1,
                "created_at": now,
            }
        policy["statuses"] = statuses if statuses is not None else policy.get("statuses") or [
            {"status_key": "todo", "display_name": "To Do", "sort_order": 0, "is_active": True, "is_terminal": False}
        ]
        policy["transitions"] = transitions if transitions is not None else policy.get("transitions") or []
        policy["is_paused"] = bool(is_paused)
        policy["is_draining"] = bool(is_draining)
        policy["default_lease_ttl_sec"] = int(default_lease_ttl_sec)
        policy["strict_projection"] = bool(strict_projection)
        policy["metadata"] = policy.get("metadata") if metadata is None and existing else metadata
        policy["updated_at"] = self._now()
        self._workflow_policies[board_id] = policy
        return dict(policy)

    def get_workflow_policy(self, board_id: int) -> Optional[Dict[str, Any]]:
        policy = self._workflow_policies.get(board_id)
        return dict(policy) if policy else None

    def update_workflow_policy_flags(
        self,
        *,
        board_id: int,
        is_paused: Optional[bool] = None,
        is_draining: Optional[bool] = None,
    ) -> Dict[str, Any]:
        policy = self._ensure_workflow_policy_for_board(board_id)
        policy["version"] = int(policy.get("version", 1)) + 1
        if is_paused is not None:
            policy["is_paused"] = bool(is_paused)
        if is_draining is not None:
            policy["is_draining"] = bool(is_draining)
        policy["updated_at"] = self._now()
        self._workflow_policies[board_id] = policy
        return dict(policy)

    def list_workflow_statuses(self, board_id: int) -> List[Dict[str, Any]]:
        policy = self._ensure_workflow_policy_for_board(board_id)
        return [dict(status) for status in policy.get("statuses", [])]

    def list_workflow_transitions(self, board_id: int) -> List[Dict[str, Any]]:
        policy = self._ensure_workflow_policy_for_board(board_id)
        return [dict(transition) for transition in policy.get("transitions", [])]

    def get_card_workflow_state(self, card_id: int) -> Dict[str, Any]:
        return dict(self._ensure_workflow_state_for_card(card_id))

    def patch_card_workflow_state(
        self,
        *,
        card_id: int,
        workflow_status_key: Optional[str],
        expected_version: int,
        lease_owner: Optional[str],
        idempotency_key: str,
        correlation_id: Optional[str] = None,
        last_actor: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self._ensure_workflow_state_for_card(card_id)
        if int(expected_version) != int(state["version"]):
            raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)
        before = dict(state)
        if workflow_status_key is not None:
            state["workflow_status_key"] = workflow_status_key
        state["lease_owner"] = lease_owner
        state["last_actor"] = last_actor
        state["last_transition_at"] = self._now()
        state["version"] = int(state["version"]) + 1
        state["updated_at"] = self._now()
        self._workflow_states[card_id] = state
        self._append_workflow_event(
            card_id=card_id,
            event_type="state_patched",
            from_status_key=before["workflow_status_key"],
            to_status_key=state["workflow_status_key"],
            actor=last_actor or "tester",
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            reason=None,
            before_snapshot=before,
            after_snapshot=dict(state),
        )
        return dict(state)

    def claim_card_workflow(
        self,
        *,
        card_id: int,
        owner: str,
        lease_ttl_sec: Optional[int] = None,
        idempotency_key: str,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self._ensure_workflow_state_for_card(card_id)
        before = dict(state)
        ttl = int(lease_ttl_sec or 900)
        state["lease_owner"] = owner
        state["lease_expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).strftime("%Y-%m-%d %H:%M:%S")
        state["last_actor"] = owner
        state["version"] = int(state["version"]) + 1
        state["updated_at"] = self._now()
        self._workflow_states[card_id] = state
        self._append_workflow_event(
            card_id=card_id,
            event_type="workflow_claimed",
            from_status_key=before["workflow_status_key"],
            to_status_key=state["workflow_status_key"],
            actor=owner,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            reason="claim_acquired",
            before_snapshot=before,
            after_snapshot=dict(state),
        )
        return dict(state)

    def release_card_workflow(
        self,
        *,
        card_id: int,
        owner: str,
        idempotency_key: str,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self._ensure_workflow_state_for_card(card_id)
        if state.get("lease_owner") and state.get("lease_owner") != owner:
            raise ConflictError("lease_mismatch", entity="card_workflow_state", entity_id=card_id)
        before = dict(state)
        state["lease_owner"] = None
        state["lease_expires_at"] = None
        state["last_actor"] = owner
        state["version"] = int(state["version"]) + 1
        state["updated_at"] = self._now()
        self._workflow_states[card_id] = state
        self._append_workflow_event(
            card_id=card_id,
            event_type="workflow_released",
            from_status_key=before["workflow_status_key"],
            to_status_key=state["workflow_status_key"],
            actor=owner,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            reason="claim_released",
            before_snapshot=before,
            after_snapshot=dict(state),
        )
        return dict(state)

    def transition_card_workflow(
        self,
        *,
        card_id: int,
        to_status_key: str,
        actor: str,
        expected_version: int,
        idempotency_key: str,
        correlation_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self._ensure_workflow_state_for_card(card_id)
        if int(expected_version) != int(state["version"]):
            raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)
        card = self._cards.get(card_id)
        if not card:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        policy = self._ensure_workflow_policy_for_board(card["board_id"])
        if bool(policy.get("is_paused", False)):
            raise ConflictError("policy_paused", entity="workflow_policy", entity_id=policy["id"])
        transitions = policy.get("transitions", [])
        edge = next(
            (
                t for t in transitions
                if t.get("from_status_key") == state["workflow_status_key"]
                and t.get("to_status_key") == to_status_key
                and bool(t.get("is_active", True))
            ),
            None,
        )
        if edge is None:
            raise ConflictError("transition_not_allowed", entity="card_workflow_state", entity_id=card_id)
        if bool(edge.get("requires_claim", True)):
            if state.get("lease_owner") != actor:
                raise ConflictError("lease_required", entity="card_workflow_state", entity_id=card_id)
        before = dict(state)
        if bool(edge.get("requires_approval", False)):
            state["approval_state"] = "awaiting_approval"
            state["pending_transition_id"] = int(edge.get("id", 1))
            state["last_actor"] = actor
            state["version"] = int(state["version"]) + 1
            state["updated_at"] = self._now()
            self._workflow_states[card_id] = state
            self._append_workflow_event(
                card_id=card_id,
                event_type="workflow_approval_requested",
                from_status_key=before["workflow_status_key"],
                to_status_key=to_status_key,
                actor=actor,
                idempotency_key=idempotency_key,
                correlation_id=correlation_id,
                reason=reason,
                before_snapshot=before,
                after_snapshot=dict(state),
            )
            return dict(state)
        state["workflow_status_key"] = to_status_key
        state["approval_state"] = "none"
        state["pending_transition_id"] = None
        state["last_transition_at"] = self._now()
        state["last_actor"] = actor
        state["version"] = int(state["version"]) + 1
        state["updated_at"] = self._now()
        self._workflow_states[card_id] = state
        self._append_workflow_event(
            card_id=card_id,
            event_type="workflow_transitioned",
            from_status_key=before["workflow_status_key"],
            to_status_key=state["workflow_status_key"],
            actor=actor,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            reason=reason,
            before_snapshot=before,
            after_snapshot=dict(state),
        )
        return dict(state)

    def decide_card_workflow_approval(
        self,
        *,
        card_id: int,
        reviewer: str,
        decision: str,
        expected_version: int,
        idempotency_key: str,
        correlation_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self._ensure_workflow_state_for_card(card_id)
        if int(expected_version) != int(state["version"]):
            raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)
        if state.get("approval_state") != "awaiting_approval":
            raise ConflictError("approval_required", entity="card_workflow_state", entity_id=card_id)
        card = self._cards.get(card_id)
        if not card:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        policy = self._ensure_workflow_policy_for_board(card["board_id"])
        transition = next(
            (
                t for t in policy.get("transitions", [])
                if t.get("from_status_key") == state["workflow_status_key"]
                and bool(t.get("requires_approval", False))
            ),
            None,
        )
        if transition is None:
            raise ConflictError("transition_not_allowed", entity="card_workflow_state", entity_id=card_id)
        before = dict(state)
        if decision == "approved":
            state["workflow_status_key"] = transition.get("approve_to_status_key") or transition.get("to_status_key")
            state["approval_state"] = "approved"
        else:
            state["workflow_status_key"] = transition.get("reject_to_status_key") or state["workflow_status_key"]
            state["approval_state"] = "rejected"
        state["pending_transition_id"] = None
        state["last_transition_at"] = self._now()
        state["last_actor"] = reviewer
        state["version"] = int(state["version"]) + 1
        state["updated_at"] = self._now()
        self._workflow_states[card_id] = state
        self._append_workflow_event(
            card_id=card_id,
            event_type="workflow_approval_decided",
            from_status_key=before["workflow_status_key"],
            to_status_key=state["workflow_status_key"],
            actor=reviewer,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            reason=reason or decision,
            before_snapshot=before,
            after_snapshot=dict(state),
        )
        return dict(state)

    def list_card_workflow_events(self, *, card_id: int, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        if card_id not in self._cards:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)
        events = list(self._workflow_events.get(card_id, []))
        events.sort(key=lambda row: row["id"], reverse=True)
        return events[offset: offset + limit]

    def list_stale_workflow_claims(self, *, board_id: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        now = self._now()
        stale: List[Dict[str, Any]] = []
        for card_id, state in self._workflow_states.items():
            lease_owner = state.get("lease_owner")
            lease_expires_at = state.get("lease_expires_at")
            if not lease_owner or not lease_expires_at or str(lease_expires_at) > now:
                continue
            card = self._cards.get(card_id)
            if not card:
                continue
            if board_id is not None and card.get("board_id") != board_id:
                continue
            stale.append(
                {
                    "card_id": card_id,
                    "board_id": card["board_id"],
                    "list_id": card["list_id"],
                    "title": card["title"],
                    "workflow_status_key": state["workflow_status_key"],
                    "lease_owner": lease_owner,
                    "lease_expires_at": lease_expires_at,
                    "version": state["version"],
                    "updated_at": state["updated_at"],
                }
            )
        return stale[:limit]

    def force_reassign_workflow_claim(
        self,
        *,
        card_id: int,
        new_owner: str,
        idempotency_key: str,
        correlation_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = self._ensure_workflow_state_for_card(card_id)
        before = dict(state)
        state["lease_owner"] = new_owner
        state["lease_expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=900)).strftime("%Y-%m-%d %H:%M:%S")
        state["last_actor"] = new_owner
        state["version"] = int(state["version"]) + 1
        state["updated_at"] = self._now()
        self._workflow_states[card_id] = state
        self._append_workflow_event(
            card_id=card_id,
            event_type="workflow_claim_reassigned",
            from_status_key=before["workflow_status_key"],
            to_status_key=state["workflow_status_key"],
            actor=new_owner,
            idempotency_key=idempotency_key,
            correlation_id=correlation_id,
            reason=reason,
            before_snapshot=before,
            after_snapshot=dict(state),
        )
        return dict(state)


@pytest.mark.asyncio
async def test_kanban_module_basic_flow(tmp_path):
    mod = KanbanModule(ModuleConfig(name="kanban"))
    fake_db = FakeKanbanDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[assignment]

    db_path = tmp_path / "kanban.db"
    ctx = SimpleNamespace(user_id="1", db_paths={"kanban": db_path})

    empty = await mod.execute_tool("kanban.boards.list", {}, context=ctx)
    _ensure(empty["total"] == 0, f"Unexpected board list payload: {empty!r}")

    created_board = await mod.execute_tool(
        "kanban.boards.create",
        {"name": "Work Board"},
        context=ctx,
    )
    board_id = created_board["board"]["id"]

    fetched = await mod.execute_tool(
        "kanban.boards.get",
        {"board_id": board_id},
        context=ctx,
    )
    _ensure(fetched["board"]["name"] == "Work Board", f"Unexpected board payload: {fetched!r}")

    created_list = await mod.execute_tool(
        "kanban.lists.create",
        {"board_id": board_id, "name": "Todo"},
        context=ctx,
    )
    list_id = created_list["list"]["id"]

    lists = await mod.execute_tool(
        "kanban.lists.list",
        {"board_id": board_id},
        context=ctx,
    )
    _ensure(len(lists["lists"]) == 1, f"Unexpected list payload: {lists!r}")

    created_card = await mod.execute_tool(
        "kanban.cards.create",
        {"list_id": list_id, "title": "Ship MCP"},
        context=ctx,
    )
    card_id = created_card["card"]["id"]
    _ensure(created_card["card"]["title"] == "Ship MCP", f"Unexpected card payload: {created_card!r}")

    cards = await mod.execute_tool(
        "kanban.cards.list",
        {"list_id": list_id},
        context=ctx,
    )
    _ensure(len(cards["cards"]) == 1, f"Unexpected cards payload: {cards!r}")

    created_label = await mod.execute_tool(
        "kanban.labels.create",
        {"board_id": board_id, "name": "Urgent", "color": "red"},
        context=ctx,
    )
    label_id = created_label["label"]["id"]
    assigned = await mod.execute_tool(
        "kanban.labels.assign",
        {"card_id": card_id, "label_id": label_id},
        context=ctx,
    )
    _ensure(any(lbl["id"] == label_id for lbl in assigned["labels"]), f"Unexpected assigned labels: {assigned!r}")

    comments = await mod.execute_tool(
        "kanban.comments.create",
        {"card_id": card_id, "content": "Looks good"},
        context=ctx,
    )
    comment_id = comments["comment"]["id"]
    comment_list = await mod.execute_tool(
        "kanban.comments.list",
        {"card_id": card_id},
        context=ctx,
    )
    _ensure(comment_list["total"] == 1, f"Unexpected comment list payload: {comment_list!r}")

    checklist = await mod.execute_tool(
        "kanban.checklists.create",
        {"card_id": card_id, "name": "Launch"},
        context=ctx,
    )
    checklist_id = checklist["checklist"]["id"]
    item = await mod.execute_tool(
        "kanban.checklists.items.create",
        {"checklist_id": checklist_id, "name": "Announce"},
        context=ctx,
    )
    item_id = item["item"]["id"]
    updated_item = await mod.execute_tool(
        "kanban.checklists.items.update",
        {"item_id": item_id, "checked": True},
        context=ctx,
    )
    _ensure(updated_item["item"]["checked"] is True, f"Unexpected checklist item payload: {updated_item!r}")

    moved_list = await mod.execute_tool(
        "kanban.lists.create",
        {"board_id": board_id, "name": "Done"},
        context=ctx,
    )
    moved_list_id = moved_list["list"]["id"]
    moved = await mod.execute_tool(
        "kanban.cards.move",
        {"card_id": card_id, "target_list_id": moved_list_id},
        context=ctx,
    )
    _ensure(moved["card"]["list_id"] == moved_list_id, f"Unexpected moved card payload: {moved!r}")

    searched = await mod.execute_tool(
        "kanban.cards.search",
        {"query": "mcp"},
        context=ctx,
    )
    _ensure(searched["total"] == 1, f"Unexpected search payload: {searched!r}")


@pytest.mark.asyncio
async def test_kanban_workflow_transition_tool_requires_expected_version_and_idempotency(tmp_path):
    mod = KanbanModule(ModuleConfig(name="kanban"))
    fake_db = FakeKanbanDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[assignment]

    ctx = SimpleNamespace(user_id="1", db_paths={"kanban": tmp_path / "kanban.db"})

    with pytest.raises(ValueError, match="expected_version"):
        await mod.execute_tool(
            "kanban.workflow.task.transition",
            {
                "card_id": 1,
                "to_status_key": "impl",
                "actor": "builder",
                "correlation_id": "corr-missing-version",
            },
            context=ctx,
        )

    with pytest.raises(ValueError, match="idempotency_key"):
        await mod.execute_tool(
            "kanban.workflow.task.transition",
            {
                "card_id": 1,
                "to_status_key": "impl",
                "actor": "builder",
                "expected_version": 1,
                "correlation_id": "corr-missing-idem",
            },
            context=ctx,
        )


@pytest.mark.asyncio
async def test_kanban_workflow_control_tools_require_admin(tmp_path):
    mod = KanbanModule(ModuleConfig(name="kanban"))
    fake_db = FakeKanbanDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[assignment]

    non_admin_ctx = SimpleNamespace(user_id="1", db_paths={"kanban": tmp_path / "kanban.db"}, metadata={"roles": []})

    board = await mod.execute_tool("kanban.boards.create", {"name": "Workflow Board"}, context=non_admin_ctx)
    board_id = board["board"]["id"]
    created_list = await mod.execute_tool(
        "kanban.lists.create",
        {"board_id": board_id, "name": "Todo"},
        context=non_admin_ctx,
    )
    card = await mod.execute_tool(
        "kanban.cards.create",
        {"list_id": created_list["list"]["id"], "title": "Workflow Card"},
        context=non_admin_ctx,
    )
    card_id = card["card"]["id"]

    with pytest.raises(ValueError, match="forbidden"):
        await mod.execute_tool("kanban.workflow.control.pause", {"board_id": board_id}, context=non_admin_ctx)

    with pytest.raises(ValueError, match="forbidden"):
        await mod.execute_tool(
            "kanban.workflow.recovery.force_reassign",
            {
                "card_id": card_id,
                "new_owner": "builder",
                "idempotency_key": "mcp-force-reassign",
                "correlation_id": "corr-force-reassign",
                "reason": "stale lease",
            },
            context=non_admin_ctx,
        )


@pytest.mark.asyncio
async def test_kanban_workflow_tool_roundtrip(tmp_path):
    mod = KanbanModule(ModuleConfig(name="kanban"))
    fake_db = FakeKanbanDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[assignment]

    admin_ctx = SimpleNamespace(
        user_id="1",
        db_paths={"kanban": tmp_path / "kanban.db"},
        metadata={"roles": ["admin"]},
    )

    board = await mod.execute_tool("kanban.boards.create", {"name": "WF Board"}, context=admin_ctx)
    board_id = board["board"]["id"]
    lst = await mod.execute_tool("kanban.lists.create", {"board_id": board_id, "name": "Todo"}, context=admin_ctx)
    card = await mod.execute_tool(
        "kanban.cards.create",
        {"list_id": lst["list"]["id"], "title": "WF Card"},
        context=admin_ctx,
    )
    card_id = card["card"]["id"]

    policy = await mod.execute_tool(
        "kanban.workflow.policy.upsert",
        {
            "board_id": board_id,
            "statuses": [
                {"status_key": "todo", "display_name": "To Do", "sort_order": 0},
                {"status_key": "impl", "display_name": "Implement", "sort_order": 1},
            ],
            "transitions": [
                {
                    "id": 1,
                    "from_status_key": "todo",
                    "to_status_key": "impl",
                    "requires_claim": False,
                    "requires_approval": False,
                    "is_active": True,
                }
            ],
        },
        context=admin_ctx,
    )
    _ensure(policy["policy"]["board_id"] == board_id, f"Unexpected workflow policy payload: {policy!r}")

    state = await mod.execute_tool("kanban.workflow.task.state.get", {"card_id": card_id}, context=admin_ctx)
    transitioned = await mod.execute_tool(
        "kanban.workflow.task.transition",
        {
            "card_id": card_id,
            "to_status_key": "impl",
            "actor": "builder",
            "expected_version": state["state"]["version"],
            "idempotency_key": "mcp-transition-1",
            "correlation_id": "corr-mcp-transition-1",
            "reason": "begin impl",
        },
        context=admin_ctx,
    )
    _ensure(transitioned["state"]["workflow_status_key"] == "impl", f"Unexpected transition payload: {transitioned!r}")

    events = await mod.execute_tool(
        "kanban.workflow.task.events.list",
        {"card_id": card_id, "limit": 10, "offset": 0},
        context=admin_ctx,
    )
    _ensure(events["events"][0]["correlation_id"] == "corr-mcp-transition-1", f"Unexpected workflow events payload: {events!r}")


@pytest.mark.asyncio
async def test_kanban_workflow_policy_upsert_omits_metadata_when_not_supplied(tmp_path):
    mod = KanbanModule(ModuleConfig(name="kanban"))

    class SpyDB:
        def __init__(self) -> None:
            self.last_kwargs: Dict[str, Any] | None = None

        def upsert_workflow_policy(self, **kwargs: Any) -> Dict[str, Any]:
            self.last_kwargs = dict(kwargs)
            return {
                "id": 1,
                "board_id": int(kwargs["board_id"]),
                "version": 1,
                "is_paused": bool(kwargs.get("is_paused", False)),
                "is_draining": bool(kwargs.get("is_draining", False)),
                "default_lease_ttl_sec": int(kwargs.get("default_lease_ttl_sec", 900)),
                "strict_projection": bool(kwargs.get("strict_projection", True)),
                "metadata": {"persisted": True},
                "statuses": [],
                "transitions": [],
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            }

    spy_db = SpyDB()
    mod._open_db = lambda ctx: spy_db  # type: ignore[assignment]
    ctx = SimpleNamespace(user_id="1", db_paths={"kanban": tmp_path / "kanban.db"}, metadata={"roles": ["admin"]})

    out = await mod.execute_tool(
        "kanban.workflow.policy.upsert",
        {
            "board_id": 42,
            "default_lease_ttl_sec": 1800,
        },
        context=ctx,
    )

    _ensure(out["policy"]["board_id"] == 42, f"Unexpected policy payload: {out!r}")
    _ensure(spy_db.last_kwargs is not None, "workflow policy upsert kwargs were not captured")
    _ensure("metadata" not in spy_db.last_kwargs, f"Unexpected metadata passthrough: {spy_db.last_kwargs!r}")


@pytest.mark.asyncio
async def test_kanban_workflow_policy_upsert_parses_boolean_like_inputs(tmp_path):
    mod = KanbanModule(ModuleConfig(name="kanban"))

    class SpyDB:
        def __init__(self) -> None:
            self.last_kwargs: Dict[str, Any] | None = None

        def upsert_workflow_policy(self, **kwargs: Any) -> Dict[str, Any]:
            self.last_kwargs = dict(kwargs)
            return {
                "id": 1,
                "board_id": int(kwargs["board_id"]),
                "version": 1,
                "is_paused": kwargs["is_paused"],
                "is_draining": kwargs["is_draining"],
                "default_lease_ttl_sec": int(kwargs.get("default_lease_ttl_sec", 900)),
                "strict_projection": kwargs["strict_projection"],
                "metadata": None,
                "statuses": [],
                "transitions": [],
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            }

    spy_db = SpyDB()
    mod._open_db = lambda ctx: spy_db  # type: ignore[assignment]
    ctx = SimpleNamespace(user_id="1", db_paths={"kanban": tmp_path / "kanban.db"}, metadata={"roles": ["admin"]})

    out = await mod.execute_tool(
        "kanban.workflow.policy.upsert",
        {
            "board_id": 42,
            "is_paused": "false",
            "is_draining": "yes",
            "strict_projection": "0",
            "default_lease_ttl_sec": 1800,
        },
        context=ctx,
    )

    _ensure(out["policy"]["is_paused"] is False, f"Unexpected parsed policy payload: {out!r}")
    _ensure(out["policy"]["is_draining"] is True, f"Unexpected parsed policy payload: {out!r}")
    _ensure(out["policy"]["strict_projection"] is False, f"Unexpected parsed policy payload: {out!r}")
    _ensure(spy_db.last_kwargs is not None, "workflow policy kwargs were not captured")
    _ensure(spy_db.last_kwargs["is_paused"] is False, f"Unexpected parsed kwargs: {spy_db.last_kwargs!r}")
    _ensure(spy_db.last_kwargs["is_draining"] is True, f"Unexpected parsed kwargs: {spy_db.last_kwargs!r}")
    _ensure(spy_db.last_kwargs["strict_projection"] is False, f"Unexpected parsed kwargs: {spy_db.last_kwargs!r}")


@pytest.mark.asyncio
async def test_kanban_workflow_policy_upsert_rejects_invalid_boolean_like_inputs(tmp_path):
    mod = KanbanModule(ModuleConfig(name="kanban"))
    fake_db = FakeKanbanDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[assignment]
    ctx = SimpleNamespace(user_id="1", db_paths={"kanban": tmp_path / "kanban.db"}, metadata={"roles": ["admin"]})

    with pytest.raises(ValueError, match="strict_projection must be a boolean-like value"):
        await mod.execute_tool(
            "kanban.workflow.policy.upsert",
            {
                "board_id": 42,
                "strict_projection": "maybe",
            },
            context=ctx,
        )


@pytest.mark.asyncio
async def test_kanban_workflow_policy_schema_allows_boolean_like_inputs():
    mod = KanbanModule(ModuleConfig(name="kanban"))

    tools = await mod.get_tools()
    tool = next(tool for tool in tools if tool["name"] == "kanban.workflow.policy.upsert")
    properties = tool["inputSchema"]["properties"]

    for field_name in ("is_paused", "is_draining", "strict_projection"):
        variants = properties[field_name]["oneOf"]
        variant_types = {variant["type"] for variant in variants}
        _ensure(
            variant_types == {"boolean", "integer", "string"},
            f"Unexpected schema variants for {field_name}: {variants!r}",
        )
        string_variant = next(variant for variant in variants if variant["type"] == "string")
        _ensure(
            {"0", "1", "false", "n", "no", "off", "on", "true", "y", "yes"}.issubset(set(string_variant["enum"])),
            f"Unexpected boolean-like token allowlist for {field_name}: {string_variant!r}",
        )
        _ensure(
            {"TRUE", "False", "YES", "NO", "ON", "OFF"}.issubset(set(string_variant["enum"])),
            f"Schema should include case variants accepted by runtime parsing for {field_name}: {string_variant!r}",
        )
