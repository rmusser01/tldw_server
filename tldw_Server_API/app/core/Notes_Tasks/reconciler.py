"""Reconcile parsed note checklist items with durable task records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists
from tldw_Server_API.app.core.Notes_Tasks.models import (
    ParsedChecklistItem,
    ReconciliationResult,
    TaskActor,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.chacha.task_store import TaskConnection


@dataclass(frozen=True)
class _ProjectedTask:
    task: dict[str, Any]
    projection: dict[str, Any]


class NotesTaskReconciler:
    """Apply parsed markdown checklist projections to task storage."""

    def reconcile_note(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        note_version: int,
        content: str,
        actor: TaskActor,
    ) -> ReconciliationResult:
        parsed = parse_note_checklists(note_id=note_id, note_version=note_version, content=content)

        with db.transaction() as conn:
            self._require_current_note(
                db=db,
                conn=conn,
                note_id=note_id,
                note_version=note_version,
                content=content,
            )
            live_tasks = self._load_live_projected_tasks(db=db, conn=conn, note_id=note_id)
            parsed_hash_counts = Counter(item.locator.normalized_text_hash for item in parsed.items)
            live_hash_counts = Counter(
                projected.projection["normalized_text_hash"]
                for projected in live_tasks
            )

            matched_live_indexes: set[int] = set()
            matched_task_ids: list[str] = []
            created_task_ids: list[str] = []
            unlinked_task_ids: list[str] = []
            created_count = 0
            updated_count = 0
            ambiguous_count = 0
            placeholder_warning_count = 0

            for item in parsed.items:
                if not item.text.strip():
                    placeholder_warning_count += 1
                    continue

                match_index = self._find_match(
                    item=item,
                    live_tasks=live_tasks,
                    matched_live_indexes=matched_live_indexes,
                    parsed_hash_counts=parsed_hash_counts,
                    live_hash_counts=live_hash_counts,
                )

                if match_index is None:
                    if self._is_ambiguous_hash(item, parsed_hash_counts, live_hash_counts):
                        ambiguous_count += 1
                    task = db.create_task(
                        note_id=note_id,
                        text=item.text,
                        status=self._status_for_item(item),
                        metadata=item.metadata,
                        actor_type=actor.actor_type,
                        actor_id=actor.actor_id,
                        conn=conn,
                    )
                    db.set_task_projection(
                        task_id=task["id"],
                        note_id=note_id,
                        note_version=note_version,
                        line_number=item.locator.line_number,
                        start_offset=item.locator.start_offset,
                        end_offset=item.locator.end_offset,
                        normalized_text_hash=item.locator.normalized_text_hash,
                        occurrence_index=item.locator.occurrence_index,
                        block_fingerprint=item.locator.block_fingerprint,
                        raw_line=item.raw_line,
                        has_child_content=item.has_child_content,
                        conn=conn,
                    )
                    created_count += 1
                    created_task_ids.append(task["id"])
                    continue

                matched_live_indexes.add(match_index)
                projected = live_tasks[match_index]
                task = projected.task
                changed_task = self._task_record_differs(task, item)
                if changed_task:
                    task = db.update_task_record(
                        task_id=task["id"],
                        expected_version=int(task["version"]),
                        text=item.text,
                        status=self._status_for_item(item),
                        metadata=item.metadata,
                        actor_type=actor.actor_type,
                        actor_id=actor.actor_id,
                        conn=conn,
                    )
                    updated_count += 1
                db.set_task_projection(
                    task_id=task["id"],
                    note_id=note_id,
                    note_version=note_version,
                    line_number=item.locator.line_number,
                    start_offset=item.locator.start_offset,
                    end_offset=item.locator.end_offset,
                    normalized_text_hash=item.locator.normalized_text_hash,
                    occurrence_index=item.locator.occurrence_index,
                    block_fingerprint=item.locator.block_fingerprint,
                    raw_line=item.raw_line,
                    has_child_content=item.has_child_content,
                    conn=conn,
                )
                matched_task_ids.append(task["id"])

            for index, projected in enumerate(live_tasks):
                if index in matched_live_indexes:
                    continue
                task = projected.task
                unlinked = db.mark_task_unlinked(
                    task_id=task["id"],
                    expected_version=int(task["version"]),
                    actor_type=actor.actor_type,
                    actor_id=actor.actor_id,
                    conn=conn,
                )
                unlinked_task_ids.append(unlinked["id"])

            parser_warning_count = sum(len(item.warnings) for item in parsed.items)
            warning_count = parser_warning_count + ambiguous_count + placeholder_warning_count
            db.set_reconciliation_state(
                note_id=note_id,
                note_version=note_version,
                status="clean" if warning_count == 0 else "warnings",
                item_count=len(parsed.items),
                warning_count=warning_count,
                conn=conn,
            )

        return ReconciliationResult(
            note_id=note_id,
            note_version=note_version,
            parsed_count=len(parsed.items),
            created_count=created_count,
            updated_count=updated_count,
            unlinked_count=len(unlinked_task_ids),
            ambiguous_count=ambiguous_count,
            matched_task_ids=matched_task_ids,
            created_task_ids=created_task_ids,
            unlinked_task_ids=unlinked_task_ids,
            warning_count=warning_count,
        )

    @staticmethod
    def _status_for_item(item: ParsedChecklistItem) -> str:
        return "done" if item.checked else "open"

    @staticmethod
    def _task_record_differs(task: dict[str, Any], item: ParsedChecklistItem) -> bool:
        return (
            task["text"] != item.text
            or task["status"] != NotesTaskReconciler._status_for_item(item)
            or (task.get("metadata_json") or {}) != item.metadata
        )

    @staticmethod
    def _require_current_note(
        *,
        db: CharactersRAGDB,
        conn: TaskConnection,
        note_id: str,
        note_version: int,
        content: str,
    ) -> None:
        note = db.task_store.get_note_reconciliation_snapshot(
            note_id=note_id,
            conn=conn,
        )
        if note is None or bool(note["deleted"]):
            raise ConflictError("Note not found for task reconciliation.", entity="notes", entity_id=note_id)
        if int(note["version"]) != int(note_version):
            raise ConflictError(
                (
                    f"Note version mismatch for task reconciliation. "
                    f"Expected {note_version}, found {note['version']}."
                ),
                entity="notes",
                entity_id=note_id,
            )
        if note["content"] != content:
            raise ConflictError(
                "Note content changed before task reconciliation.",
                entity="notes",
                entity_id=note_id,
            )

    @staticmethod
    def _load_live_projected_tasks(
        *,
        db: CharactersRAGDB,
        conn: TaskConnection,
        note_id: str,
    ) -> list[_ProjectedTask]:
        projected_pairs = db.task_store.list_live_projected_tasks(
            note_id=note_id,
            conn=conn,
        )
        return [
            _ProjectedTask(task=pair["task"], projection=pair["projection"])
            for pair in projected_pairs
        ]

    def _find_match(
        self,
        *,
        item: ParsedChecklistItem,
        live_tasks: list[_ProjectedTask],
        matched_live_indexes: set[int],
        parsed_hash_counts: Counter[str],
        live_hash_counts: Counter[str],
    ) -> int | None:
        locator_matches = [
            index
            for index, projected in enumerate(live_tasks)
            if index not in matched_live_indexes
            and self._projection_matches_locator(projected.projection, item)
        ]
        if len(locator_matches) == 1:
            return locator_matches[0]

        if self._is_ambiguous_hash(item, parsed_hash_counts, live_hash_counts):
            return None

        unique_hash_matches = [
            index
            for index, projected in enumerate(live_tasks)
            if index not in matched_live_indexes
            and projected.projection["normalized_text_hash"] == item.locator.normalized_text_hash
            and projected.projection["occurrence_index"] == item.locator.occurrence_index
            and projected.projection["block_fingerprint"] == item.locator.block_fingerprint
        ]
        if len(unique_hash_matches) == 1:
            return unique_hash_matches[0]
        return None

    @staticmethod
    def _is_ambiguous_hash(
        item: ParsedChecklistItem,
        parsed_hash_counts: Counter[str],
        live_hash_counts: Counter[str],
    ) -> bool:
        text_hash = item.locator.normalized_text_hash
        return parsed_hash_counts[text_hash] > 1 or live_hash_counts[text_hash] > 1

    @staticmethod
    def _projection_matches_locator(projection: dict[str, Any], item: ParsedChecklistItem) -> bool:
        locator = item.locator
        return (
            projection["line_number"] == locator.line_number
            and projection["start_offset"] == locator.start_offset
            and projection["normalized_text_hash"] == locator.normalized_text_hash
            and projection["occurrence_index"] == locator.occurrence_index
        )
