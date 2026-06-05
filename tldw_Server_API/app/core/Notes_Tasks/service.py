"""Service entry point for note-backed task reconciliation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tldw_Server_API.app.core.Notes_Tasks.models import ReconciliationResult, TaskActor
from tldw_Server_API.app.core.Notes_Tasks.reconciler import NotesTaskReconciler

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class NotesTaskService:
    """Coordinate task-backed checklist reconciliation for saved notes."""

    def __init__(self, reconciler: NotesTaskReconciler | None = None) -> None:
        self._reconciler = reconciler or NotesTaskReconciler()

    def reconcile_note(
        self,
        *,
        db: CharactersRAGDB,
        note_id: str,
        note_version: int,
        content: str,
        actor: TaskActor,
    ) -> ReconciliationResult:
        return self._reconciler.reconcile_note(
            db=db,
            note_id=note_id,
            note_version=note_version,
            content=content,
            actor=actor,
        )
