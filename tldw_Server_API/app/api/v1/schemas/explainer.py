"""Pydantic schemas for the Explainer workspace API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tldw_Server_API.app.core.Explainer.models import (
    ExplainerCitation,
    ExplainerDepthPreset,
    ExplainerEvidenceState,
    ExplainerGrounding,
    ExplainerMode,
    ExplainerNode,
    ExplainerNodeKind,
    ExplainerNodeStatus,
    ExplainerOutputIntent,
    ExplainerSelectedSource,
    ExplainerSession,
)


class _ExplainerSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)


class ExplainerSelectedSourceRequest(_ExplainerSchema):
    source_id: str = Field(alias="sourceId")
    source_type: str = Field(alias="sourceType")
    title: str
    snapshot_version: str | None = Field(default=None, alias="snapshotVersion")
    metadata: dict[str, Any] | None = None


class ExplainerSelectedSourceResponse(ExplainerSelectedSourceRequest):
    added_at: str = Field(alias="addedAt")

    @classmethod
    def from_domain(cls, source: ExplainerSelectedSource) -> "ExplainerSelectedSourceResponse":
        return cls(
            source_id=source.source_id,
            source_type=source.source_type,
            title=source.title,
            added_at=source.added_at,
            snapshot_version=source.snapshot_version,
            metadata=source.metadata,
        )


class ExplainerCitationResponse(_ExplainerSchema):
    id: str
    source_id: str = Field(alias="sourceId")
    source_type: str = Field(alias="sourceType")
    title: str
    excerpt: str
    location_label: str | None = Field(default=None, alias="locationLabel")
    start_offset: int | None = Field(default=None, alias="startOffset")
    end_offset: int | None = Field(default=None, alias="endOffset")
    url: str | None = None
    snapshot_hash: str | None = Field(default=None, alias="snapshotHash")

    @classmethod
    def from_domain(cls, citation: ExplainerCitation) -> "ExplainerCitationResponse":
        return cls(
            id=citation.id,
            source_id=citation.source_id,
            source_type=citation.source_type,
            title=citation.title,
            excerpt=citation.excerpt,
            location_label=citation.location_label,
            start_offset=citation.start_offset,
            end_offset=citation.end_offset,
            url=citation.url,
            snapshot_hash=citation.snapshot_hash,
        )


class ExplainerCitationRequest(_ExplainerSchema):
    source_id: str = Field(alias="sourceId")
    source_type: str = Field(alias="sourceType")
    title: str
    excerpt: str
    location_label: str | None = Field(default=None, alias="locationLabel")
    start_offset: int | None = Field(default=None, alias="startOffset")
    end_offset: int | None = Field(default=None, alias="endOffset")
    url: str | None = None
    snapshot_hash: str | None = Field(default=None, alias="snapshotHash")


class ExplainerNodeResponse(_ExplainerSchema):
    id: str
    session_id: str = Field(alias="sessionId")
    parent_id: str | None = Field(default=None, alias="parentId")
    ordinal: int
    title: str
    body: str | None = None
    kind: str
    intent: str
    status: str
    evidence_state: str = Field(alias="evidenceState")
    outside_knowledge_used: bool = Field(alias="outsideKnowledgeUsed")
    citations: list[ExplainerCitationResponse] = Field(default_factory=list)
    question_options: list[dict[str, Any]] | None = Field(default=None, alias="questionOptions")
    selected_option_id: str | None = Field(default=None, alias="selectedOptionId")
    selected_custom_answer: str | None = Field(default=None, alias="selectedCustomAnswer")
    generation_metadata: dict[str, Any] | None = Field(default=None, alias="generationMetadata")
    child_node_ids: list[str] = Field(default_factory=list, alias="childNodeIds")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, node: ExplainerNode) -> "ExplainerNodeResponse":
        return cls(
            id=node.id,
            session_id=node.session_id,
            parent_id=node.parent_id,
            ordinal=node.ordinal,
            title=node.title,
            body=node.body,
            kind=node.kind,
            intent=node.intent,
            status=node.status,
            evidence_state=node.evidence_state,
            outside_knowledge_used=node.outside_knowledge_used,
            citations=[ExplainerCitationResponse.from_domain(citation) for citation in node.citations],
            question_options=node.question_options,
            selected_option_id=node.selected_option_id,
            selected_custom_answer=node.selected_custom_answer,
            generation_metadata=node.generation_metadata,
            child_node_ids=node.child_node_ids,
            created_at=node.created_at,
            updated_at=node.updated_at,
        )


class ExplainerSessionCreateRequest(_ExplainerSchema):
    title: str
    mode: ExplainerMode
    output_intent: ExplainerOutputIntent = Field(alias="outputIntent")
    grounding: ExplainerGrounding
    depth_preset: ExplainerDepthPreset = Field(alias="depthPreset")
    selected_sources: list[ExplainerSelectedSourceRequest] = Field(default_factory=list, alias="selectedSources")
    root_prompt: str = Field(alias="rootPrompt")


class ExplainerSessionPatchRequest(_ExplainerSchema):
    title: str | None = None
    output_intent: ExplainerOutputIntent | None = Field(default=None, alias="outputIntent")
    grounding: ExplainerGrounding | None = None
    depth_preset: ExplainerDepthPreset | None = Field(default=None, alias="depthPreset")
    selected_sources: list[ExplainerSelectedSourceRequest] | None = Field(default=None, alias="selectedSources")


class ExplainerNodeCreateRequest(_ExplainerSchema):
    parent_id: str | None = Field(default=None, alias="parentId")
    title: str
    body: str | None = None
    kind: ExplainerNodeKind = ExplainerNodeKind.EXPLANATION
    intent: ExplainerOutputIntent = ExplainerOutputIntent.EXPLAIN
    status: ExplainerNodeStatus = ExplainerNodeStatus.IDLE
    evidence_state: ExplainerEvidenceState = Field(default=ExplainerEvidenceState.UNCITED, alias="evidenceState")
    outside_knowledge_used: bool = Field(default=False, alias="outsideKnowledgeUsed")
    citations: list[ExplainerCitationRequest] = Field(default_factory=list)


class ExplainerNodePatchRequest(_ExplainerSchema):
    title: str | None = None
    body: str | None = None
    status: ExplainerNodeStatus | None = None
    evidence_state: ExplainerEvidenceState | None = Field(default=None, alias="evidenceState")
    outside_knowledge_used: bool | None = Field(default=None, alias="outsideKnowledgeUsed")
    selected_option_id: str | None = Field(default=None, alias="selectedOptionId")
    selected_custom_answer: str | None = Field(default=None, alias="selectedCustomAnswer")
    question_options: list[dict[str, Any]] | None = Field(default=None, alias="questionOptions")
    generation_metadata: dict[str, Any] | None = Field(default=None, alias="generationMetadata")
    citations: list[ExplainerCitationRequest] | None = None


class ExplainerSessionResponse(_ExplainerSchema):
    id: str
    owner_user_id: str = Field(alias="ownerUserId")
    title: str
    mode: str
    status: str
    output_intent: str = Field(alias="outputIntent")
    grounding: str
    depth_preset: str = Field(alias="depthPreset")
    selected_sources: list[ExplainerSelectedSourceResponse] = Field(alias="selectedSources")
    root_node_ids: list[str] = Field(alias="rootNodeIds")
    nodes: dict[str, ExplainerNodeResponse]
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")
    archived_at: str | None = Field(default=None, alias="archivedAt")

    @classmethod
    def from_domain(cls, session: ExplainerSession) -> "ExplainerSessionResponse":
        return cls(
            id=session.id,
            owner_user_id=session.owner_user_id,
            title=session.title,
            mode=session.mode,
            status=session.status,
            output_intent=session.output_intent,
            grounding=session.grounding,
            depth_preset=session.depth_preset,
            selected_sources=[
                ExplainerSelectedSourceResponse.from_domain(source)
                for source in session.selected_sources
            ],
            root_node_ids=session.root_node_ids,
            nodes={
                node_id: ExplainerNodeResponse.from_domain(node)
                for node_id, node in session.nodes.items()
            },
            created_at=session.created_at,
            updated_at=session.updated_at,
            archived_at=session.archived_at,
        )


class ExplainerSessionListResponse(_ExplainerSchema):
    items: list[ExplainerSessionResponse]
    total: int


class ExplainerDeleteNodeResponse(_ExplainerSchema):
    id: str
    status: str
