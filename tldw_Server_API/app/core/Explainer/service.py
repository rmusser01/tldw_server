"""Service layer for Explainer workspace CRUD operations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tldw_Server_API.app.core.DB_Management.Explainer_DB import InputError
from tldw_Server_API.app.core.Explainer.models import (
    ExplainerGrounding,
    ExplainerNode,
    ExplainerNodeStatus,
    ExplainerOutputIntent,
    ExplainerSelectedSource,
    ExplainerSession,
    ExplainerSessionSummary,
)
from tldw_Server_API.app.core.Explainer.jobs import (
    ExplainerJobAccepted,
    ExplainerTerminalJobError,
    enqueue_explainer_node_expansion_job,
    is_explainer_generation_configured,
)
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository


class ExplainerValidationError(ValueError):
    """Raised when an Explainer API request violates workspace rules."""


class ExplainerNotFoundError(LookupError):
    """Raised when an Explainer resource is not visible to the requesting user."""


class ExplainerService:
    """Application service for ownership-aware Explainer CRUD behavior."""

    def __init__(self, repo: ExplainerRepository, job_manager: Any | None = None) -> None:
        self.repo = repo
        self.job_manager = job_manager

    def create_session(
        self,
        *,
        owner_user_id: str,
        title: str,
        mode: str,
        output_intent: str,
        grounding: str,
        depth_preset: str,
        selected_sources: list[dict[str, Any]],
        root_prompt: str,
    ) -> ExplainerSession:
        self._validate_grounding(grounding=grounding, selected_sources=selected_sources)
        return self.repo.create_session(
            owner_user_id=owner_user_id,
            title=title,
            mode=mode,
            output_intent=output_intent,
            grounding=grounding,
            depth_preset=depth_preset,
            selected_sources=selected_sources,
            root_prompt=root_prompt,
        )

    def list_sessions(self, *, owner_user_id: str) -> list[ExplainerSession]:
        return self.repo.list_sessions(owner_user_id=owner_user_id)

    def list_session_summaries(
        self,
        *,
        owner_user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ExplainerSessionSummary], int]:
        return self.repo.list_session_summaries(
            owner_user_id=owner_user_id,
            limit=limit,
            offset=offset,
        )

    def get_session(self, session_id: str, *, owner_user_id: str) -> ExplainerSession:
        session = self.repo.get_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            raise ExplainerNotFoundError("Explainer session not found")
        return session

    def update_session(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        title: str | None = None,
        output_intent: str | None = None,
        grounding: str | None = None,
        depth_preset: str | None = None,
        selected_sources: list[dict[str, Any]] | None = None,
    ) -> ExplainerSession:
        existing = self.get_session(session_id, owner_user_id=owner_user_id)
        effective_grounding = grounding or existing.grounding
        effective_sources = (
            selected_sources
            if selected_sources is not None
            else [asdict(source) for source in existing.selected_sources]
        )
        self._validate_grounding(
            grounding=effective_grounding,
            selected_sources=effective_sources,
        )
        session = self.repo.update_session(
            session_id,
            owner_user_id=owner_user_id,
            title=title,
            output_intent=output_intent,
            grounding=grounding,
            depth_preset=depth_preset,
            selected_sources=selected_sources,
        )
        if session is None:
            raise ExplainerNotFoundError("Explainer session not found")
        return session

    def archive_session(self, session_id: str, *, owner_user_id: str) -> ExplainerSession:
        session = self.repo.archive_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            raise ExplainerNotFoundError("Explainer session not found")
        return session

    def create_node(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        title: str,
        parent_id: str | None = None,
        body: str | None = None,
        kind: str,
        intent: str,
        status: str,
        evidence_state: str,
        outside_knowledge_used: bool,
        citations: list[dict[str, Any]] | None = None,
    ) -> ExplainerNode:
        node = self.repo.create_node(
            session_id,
            owner_user_id=owner_user_id,
            title=title,
            parent_id=parent_id,
            body=body,
            kind=kind,
            intent=intent,
            status=status,
            evidence_state=evidence_state,
            outside_knowledge_used=outside_knowledge_used,
            citations=citations,
        )
        if node is None:
            raise ExplainerNotFoundError("Explainer session not found")
        return node

    def update_node(
        self,
        session_id: str,
        node_id: str,
        *,
        owner_user_id: str,
        updates: dict[str, Any],
    ) -> ExplainerNode:
        node = self.repo.update_node(
            session_id,
            node_id,
            owner_user_id=owner_user_id,
            **updates,
        )
        if node is None:
            raise ExplainerNotFoundError("Explainer node not found")
        return node

    def delete_node(self, session_id: str, node_id: str, *, owner_user_id: str) -> dict[str, str]:
        deleted = self.repo.delete_node(session_id, node_id, owner_user_id=owner_user_id)
        if not deleted:
            raise ExplainerNotFoundError("Explainer node not found")
        return {"status": "deleted", "id": node_id}

    def enqueue_node_expansion(
        self,
        *,
        session_id: str,
        node_id: str,
        owner_user_id: str,
        intent: str | None = None,
    ) -> ExplainerJobAccepted:
        if self.job_manager is None:
            raise ExplainerValidationError("Explainer Jobs backend unavailable")
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        node = session.nodes.get(node_id)
        if node is None:
            raise ExplainerNotFoundError("Explainer node not found")
        effective_intent = intent or node.intent or session.output_intent
        self._validate_intent(effective_intent)
        if session.grounding != ExplainerGrounding.SOURCE_ONLY.value and not is_explainer_generation_configured():
            raise ExplainerValidationError("Explainer generation is not configured")
        try:
            accepted = enqueue_explainer_node_expansion_job(
                jm=self.job_manager,
                session=session,
                node=node,
                owner_user_id=owner_user_id,
                intent=effective_intent,
            )
        except ExplainerTerminalJobError as exc:
            raise ExplainerValidationError(str(exc)) from exc
        queued = self.repo.update_node(
            session_id,
            node_id,
            owner_user_id=owner_user_id,
            status=ExplainerNodeStatus.QUEUED.value,
        )
        if queued is None:
            raise ExplainerNotFoundError("Explainer node not found")
        return accepted

    def answer_question(
        self,
        session_id: str,
        node_id: str,
        *,
        owner_user_id: str,
        selected_option_id: str | None = None,
        selected_custom_answer: str | None = None,
    ) -> ExplainerNode:
        if not (selected_option_id or selected_custom_answer):
            raise ExplainerValidationError("selected answer is required")
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if node_id not in session.nodes:
            raise ExplainerNotFoundError("Explainer node not found")
        node = self.repo.update_node(
            session_id,
            node_id,
            owner_user_id=owner_user_id,
            selected_option_id=selected_option_id,
            selected_custom_answer=selected_custom_answer,
            status=ExplainerNodeStatus.COMPLETE.value,
        )
        if node is None:
            raise ExplainerNotFoundError("Explainer node not found")
        return node

    @staticmethod
    def _validate_grounding(
        *,
        grounding: str,
        selected_sources: list[dict[str, Any] | ExplainerSelectedSource],
    ) -> None:
        if grounding == ExplainerGrounding.SOURCE_ONLY.value and not selected_sources:
            raise ExplainerValidationError("source_only grounding requires at least one selected source")

    @staticmethod
    def _validate_intent(intent: str) -> None:
        allowed = {item.value for item in ExplainerOutputIntent}
        if intent not in allowed:
            raise ExplainerValidationError(f"intent must be one of {sorted(allowed)}")


def map_explainer_service_error(exc: Exception) -> tuple[int, str]:
    """Return HTTP status/detail for Explainer service/repository exceptions."""
    if isinstance(exc, ExplainerNotFoundError):
        return 404, str(exc)
    if isinstance(exc, (ExplainerValidationError, InputError)):
        return 422, str(exc)
    return 500, "Explainer service unavailable"
