"""Atomically publish generated Study entries and their Workspace receipt."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.chacha.research_study_store import ResearchStudyStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Research_Workspace.study_output_models import (
    ResearchFlashcardPublication,
    ResearchStudyGeneration,
    ResearchStudyOutput,
    ResearchStudyOutputsPage,
    ResearchStudyPublication,
)

_MAX_GENERATION_BYTES = 32 * 1024
_MAX_REQUEST_BYTES = 288 * 1024


class ResearchStudyPublicationError(ValueError):
    """Safe, content-free rejection from the Study publication contract."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _project(row: dict[str, Any]) -> ResearchStudyOutput:
    return ResearchStudyOutput(
        workspace_id=row["workspace_id"],
        owner_id=row["owner_id"],
        version=row["version"],
        name=row["name"],
        created_at=row["created_at"],
        generation=ResearchStudyGeneration.model_validate_json(row["generation_json"]),
    )


class ResearchStudyPublisher:
    """Publish and resolve native Study owners within a captured Workspace."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db
        self.store = ResearchStudyStore(db)

    def _workspace(self, workspace_id: str, *, conn: Any = None) -> dict:
        workspace = self.db.workspace_store.get_workspace(workspace_id, conn=conn, for_update=conn is not None)
        if workspace is None:
            raise ResearchStudyPublicationError(404, "Workspace not found")
        return workspace

    def get(self, workspace_id: str, kind: str, owner_id: int) -> ResearchStudyOutput:
        """Resolve only the registered visible native owner."""
        row = self.store.get(workspace_id, kind, owner_id)
        if row is None:
            raise ResearchStudyPublicationError(404, "Study output not found")
        return _project(row)

    def page(self, workspace_id: str, *, limit: int, offset: int) -> ResearchStudyOutputsPage:
        """Project a page of generated native owners for an existing Workspace."""
        self._workspace(workspace_id)
        rows, total = self.store.page(workspace_id, limit=limit, offset=offset)
        return ResearchStudyOutputsPage(items=[_project(row) for row in rows], limit=limit, offset=offset, total=total)

    def save(self, workspace_id: str, request: ResearchStudyPublication) -> ResearchStudyOutput:
        """Commit all native entries and the operation receipt in one transaction."""
        generation = request.model_dump(include=set(ResearchStudyGeneration.model_fields), exclude_none=True)
        generation_json = ResearchStudyGeneration.model_validate(generation).model_dump_json(exclude_none=True)
        request_json = request.model_dump_json()
        if (
            len(generation_json.encode("utf-8")) > _MAX_GENERATION_BYTES
            or len(request_json.encode("utf-8")) > _MAX_REQUEST_BYTES
        ):
            raise ResearchStudyPublicationError(413, "Study output exceeds the allowed size")
        fingerprint = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        kind = request.output_kind
        with self.db.transaction() as conn:
            workspace = self._workspace(workspace_id, conn=conn)
            existing = self.store.find_operation(conn, workspace_id, request.operation_id)
            if existing:
                if existing["request_fingerprint"] != fingerprint:
                    raise ResearchStudyPublicationError(409, "Study operation already has a different request")
                owner_id = existing["owner_id"]
            else:
                if workspace.get("archived") or workspace.get("native_chat_admission_closed"):
                    raise ResearchStudyPublicationError(409, "Workspace is closed to new Study outputs")
                if not self.store.sources_current(conn, workspace_id, request.source_snapshot):
                    raise ResearchStudyPublicationError(409, "Selected workspace sources changed")
                if request.supersedes_owner_id and not self.store.get(
                    workspace_id, kind, int(request.supersedes_owner_id), conn=conn
                ):
                    raise ResearchStudyPublicationError(409, "Previous Study output is no longer in this workspace")
                owner_id = self._write_study_owner(workspace_id, request)
                self.store.add_receipt(
                    conn,
                    workspace_id,
                    operation_id=request.operation_id,
                    fingerprint=fingerprint,
                    kind=kind,
                    owner_id=owner_id,
                    generation_json=generation_json,
                )
            row = self.store.get(workspace_id, kind, owner_id, conn=conn)
            if row is None:
                raise ResearchStudyPublicationError(409, "Saved Study owner is no longer in this workspace")
            return _project(row)

    def _write_study_owner(self, workspace_id: str, request: ResearchStudyPublication) -> int:
        sources = {source.source_id: int(source.catalog_item_id) for source in request.source_snapshot}
        if isinstance(request, ResearchFlashcardPublication):
            native = self.db.flashcard_store
            name = request.title or "Research flashcards"
            if native.get_deck_by_name(name, include_deleted=True):
                suffix = hashlib.sha256(f"{workspace_id}/{request.operation_id}".encode()).hexdigest()[:12]
                name = f"{name[:105]} ({suffix})"
            owner_id = native.add_deck(name, workspace_id=workspace_id, restore_deleted=False)
            for card in request.items:
                native.add_flashcard(
                    {
                        "deck_id": owner_id,
                        "front": card.front,
                        "back": card.back,
                        "extra": json.dumps(
                            {
                                "source_citations": [
                                    {"source_id": source_id, "media_id": sources[source_id]}
                                    for source_id in card.citations
                                ]
                            }
                        ),
                    }
                )
            return owner_id

        owner_id = self.db.create_quiz(request.title or "Research quiz", workspace_id=workspace_id)
        for order, question in enumerate(request.items):
            self.db.create_question(
                owner_id,
                question_type="multiple_choice",
                question_text=question.question,
                correct_answer=question.correct_index,
                options=list(question.options),
                explanation=question.explanation,
                order_index=order,
                source_citations=[
                    {"source_type": "media", "source_id": str(sources[source_id]), "media_id": sources[source_id]}
                    for source_id in question.citations
                ],
            )
        return owner_id
