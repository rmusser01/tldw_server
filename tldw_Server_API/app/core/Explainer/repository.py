"""Ownership-aware repository for Explainer workspace persistence."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import defaultdict
from typing import Any

from tldw_Server_API.app.core.DB_Management.Explainer_DB import (
    ExplainerDatabase,
    InputError,
)
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
    ExplainerSessionStatus,
)


class ExplainerRepository:
    """CRUD repository that enforces user ownership on session access."""

    def __init__(self, db: ExplainerDatabase) -> None:
        self.db = db

    def create_session(
        self,
        *,
        owner_user_id: str,
        title: str,
        mode: str,
        output_intent: str,
        grounding: str,
        depth_preset: str,
        selected_sources: list[ExplainerSelectedSource | dict[str, Any]],
        root_prompt: str,
    ) -> ExplainerSession:
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        title = _require_text(title, "title")
        root_prompt = _require_text(root_prompt, "root_prompt")
        _require_enum(mode, ExplainerMode, "mode")
        _require_enum(output_intent, ExplainerOutputIntent, "output_intent")
        _require_enum(grounding, ExplainerGrounding, "grounding")
        _require_enum(depth_preset, ExplainerDepthPreset, "depth_preset")

        now = self.db.utcnow_iso()
        session_id = _new_id("exp_sess")
        root_node_id = _new_id("exp_node")
        normalized_sources = [
            _normalize_source(source, fallback_added_at=now)
            for source in selected_sources
        ]

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO explainer_sessions (
                    id, owner_user_id, title, mode, status, output_intent, grounding,
                    depth_preset, created_at, updated_at, archived_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    session_id,
                    owner_user_id,
                    title,
                    mode,
                    ExplainerSessionStatus.ACTIVE.value,
                    output_intent,
                    grounding,
                    depth_preset,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO explainer_nodes (
                    id, session_id, parent_id, ordinal, title, body, kind, intent,
                    status, evidence_state, outside_knowledge_used, question_options_json,
                    selected_option_id, selected_custom_answer, generation_metadata_json,
                    created_at, updated_at, deleted_at
                )
                VALUES (?, ?, NULL, 0, ?, NULL, ?, ?, ?, ?, 0, NULL, NULL, NULL, NULL, ?, ?, NULL)
                """,
                (
                    root_node_id,
                    session_id,
                    root_prompt,
                    ExplainerNodeKind.QUESTION.value,
                    output_intent,
                    ExplainerNodeStatus.IDLE.value,
                    ExplainerEvidenceState.UNCITED.value,
                    now,
                    now,
                ),
            )
            self._replace_sources(
                conn,
                session_id=session_id,
                owner_user_id=owner_user_id,
                sources=normalized_sources,
                now=now,
            )

        loaded = self.get_session(session_id, owner_user_id=owner_user_id)
        if loaded is None:
            raise RuntimeError("created Explainer session could not be reloaded")
        return loaded

    def list_sessions(
        self,
        *,
        owner_user_id: str,
        include_archived: bool = False,
    ) -> list[ExplainerSession]:
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        conn = self.db.get_connection()
        if include_archived:
            rows = conn.execute(
                """
                SELECT id
                FROM explainer_sessions
                WHERE owner_user_id = ?
                ORDER BY updated_at DESC, created_at DESC
                """,
                (owner_user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id
                FROM explainer_sessions
                WHERE owner_user_id = ? AND archived_at IS NULL
                ORDER BY updated_at DESC, created_at DESC
                """,
                (owner_user_id,),
            ).fetchall()
        sessions: list[ExplainerSession] = []
        for row in rows:
            session = self.get_session(row["id"], owner_user_id=owner_user_id, include_archived=include_archived)
            if session is not None:
                sessions.append(session)
        return sessions

    def get_session(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        include_archived: bool = False,
    ) -> ExplainerSession | None:
        session_id = _require_text(session_id, "session_id")
        owner_user_id = _require_text(owner_user_id, "owner_user_id")
        conn = self.db.get_connection()
        if include_archived:
            session_row = conn.execute(
                """
                SELECT *
                FROM explainer_sessions
                WHERE id = ? AND owner_user_id = ?
                """,
                (session_id, owner_user_id),
            ).fetchone()
        else:
            session_row = conn.execute(
                """
                SELECT *
                FROM explainer_sessions
                WHERE id = ? AND owner_user_id = ? AND archived_at IS NULL
                """,
                (session_id, owner_user_id),
            ).fetchone()
        if session_row is None:
            return None

        node_rows = conn.execute(
            """
            SELECT *
            FROM explainer_nodes
            WHERE session_id = ? AND deleted_at IS NULL
            ORDER BY parent_id IS NOT NULL, parent_id, ordinal, created_at
            """,
            (session_id,),
        ).fetchall()
        source_rows = conn.execute(
            """
            SELECT *
            FROM explainer_selected_sources
            WHERE session_id = ? AND owner_user_id = ? AND deleted_at IS NULL
            ORDER BY ordinal, added_at
            """,
            (session_id, owner_user_id),
        ).fetchall()
        citation_rows = conn.execute(
            """
            SELECT *
            FROM explainer_citations
            WHERE session_id = ? AND owner_user_id = ? AND deleted_at IS NULL
            ORDER BY ordinal, created_at
            """,
            (session_id, owner_user_id),
        ).fetchall()
        return self._build_session(session_row, node_rows, source_rows, citation_rows)

    def update_session(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        title: str | None = None,
        output_intent: str | None = None,
        grounding: str | None = None,
        depth_preset: str | None = None,
        selected_sources: list[ExplainerSelectedSource | dict[str, Any]] | None = None,
    ) -> ExplainerSession | None:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            return None

        if output_intent is not None:
            _require_enum(output_intent, ExplainerOutputIntent, "output_intent")
        if grounding is not None:
            _require_enum(grounding, ExplainerGrounding, "grounding")
        if depth_preset is not None:
            _require_enum(depth_preset, ExplainerDepthPreset, "depth_preset")

        next_title = _require_text(title, "title") if title is not None else session.title
        next_output_intent = output_intent or session.output_intent
        next_grounding = grounding or session.grounding
        next_depth_preset = depth_preset or session.depth_preset

        now = self.db.utcnow_iso()
        with self.db.transaction() as conn:
            if any(
                value is not None
                for value in (title, output_intent, grounding, depth_preset)
            ):
                conn.execute(
                    """
                    UPDATE explainer_sessions
                    SET title = ?, output_intent = ?, grounding = ?,
                        depth_preset = ?, updated_at = ?
                    WHERE id = ? AND owner_user_id = ? AND archived_at IS NULL
                    """,
                    (
                        next_title,
                        next_output_intent,
                        next_grounding,
                        next_depth_preset,
                        now,
                        session_id,
                        owner_user_id,
                    ),
                )
            if selected_sources is not None:
                normalized_sources = [
                    _normalize_source(source, fallback_added_at=now)
                    for source in selected_sources
                ]
                self._replace_sources(
                    conn,
                    session_id=session_id,
                    owner_user_id=owner_user_id,
                    sources=normalized_sources,
                    now=now,
                )
                conn.execute(
                    """
                    UPDATE explainer_sessions
                    SET updated_at = ?
                    WHERE id = ? AND owner_user_id = ? AND archived_at IS NULL
                    """,
                    (now, session_id, owner_user_id),
                )

        return self.get_session(session_id, owner_user_id=owner_user_id)

    def archive_session(
        self,
        session_id: str,
        *,
        owner_user_id: str,
    ) -> ExplainerSession | None:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            return None
        now = self.db.utcnow_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE explainer_sessions
                SET status = ?, archived_at = ?, updated_at = ?
                WHERE id = ? AND owner_user_id = ? AND archived_at IS NULL
                """,
                (ExplainerSessionStatus.ARCHIVED.value, now, now, session_id, owner_user_id),
            )
        return self.get_session(session_id, owner_user_id=owner_user_id, include_archived=True)

    def create_node(
        self,
        session_id: str,
        *,
        owner_user_id: str,
        title: str,
        parent_id: str | None = None,
        body: str | None = None,
        kind: str = ExplainerNodeKind.EXPLANATION.value,
        intent: str = ExplainerOutputIntent.EXPLAIN.value,
        status: str = ExplainerNodeStatus.IDLE.value,
        evidence_state: str = ExplainerEvidenceState.UNCITED.value,
        outside_knowledge_used: bool = False,
    ) -> ExplainerNode | None:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None:
            return None
        if parent_id is not None and parent_id not in session.nodes:
            raise InputError("parent_id does not belong to session")
        _require_enum(kind, ExplainerNodeKind, "kind")
        _require_enum(intent, ExplainerOutputIntent, "intent")
        _require_enum(status, ExplainerNodeStatus, "status")
        _require_enum(evidence_state, ExplainerEvidenceState, "evidence_state")
        now = self.db.utcnow_iso()
        node_id = _new_id("exp_node")
        with self.db.transaction() as conn:
            ordinal = self._next_child_ordinal(conn, session_id=session_id, parent_id=parent_id)
            conn.execute(
                """
                INSERT INTO explainer_nodes (
                    id, session_id, parent_id, ordinal, title, body, kind, intent,
                    status, evidence_state, outside_knowledge_used, question_options_json,
                    selected_option_id, selected_custom_answer, generation_metadata_json,
                    created_at, updated_at, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, NULL)
                """,
                (
                    node_id,
                    session_id,
                    parent_id,
                    ordinal,
                    _require_text(title, "title"),
                    body,
                    kind,
                    intent,
                    status,
                    evidence_state,
                    1 if outside_knowledge_used else 0,
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE explainer_sessions
                SET updated_at = ?
                WHERE id = ? AND owner_user_id = ?
                """,
                (now, session_id, owner_user_id),
            )
        loaded = self.get_session(session_id, owner_user_id=owner_user_id)
        return None if loaded is None else loaded.nodes[node_id]

    def update_node(
        self,
        session_id: str,
        node_id: str,
        *,
        owner_user_id: str,
        title: str | None = None,
        body: str | None = None,
        status: str | None = None,
        evidence_state: str | None = None,
        outside_knowledge_used: bool | None = None,
        selected_option_id: str | None = None,
        selected_custom_answer: str | None = None,
        question_options: list[dict[str, Any]] | None = None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> ExplainerNode | None:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None or node_id not in session.nodes:
            return None
        existing_node = session.nodes[node_id]

        if status is not None:
            _require_enum(status, ExplainerNodeStatus, "status")
        if evidence_state is not None:
            _require_enum(evidence_state, ExplainerEvidenceState, "evidence_state")
        if not any(
            value is not None
            for value in (
                title,
                body,
                status,
                evidence_state,
                outside_knowledge_used,
                selected_option_id,
                selected_custom_answer,
                question_options,
                generation_metadata,
            )
        ):
            return existing_node

        now = self.db.utcnow_iso()
        next_outside_knowledge_used = (
            outside_knowledge_used
            if outside_knowledge_used is not None
            else existing_node.outside_knowledge_used
        )
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE explainer_nodes
                SET title = ?, body = ?, status = ?, evidence_state = ?,
                    outside_knowledge_used = ?, selected_option_id = ?,
                    selected_custom_answer = ?, question_options_json = ?,
                    generation_metadata_json = ?, updated_at = ?
                WHERE id = ? AND session_id = ? AND deleted_at IS NULL
                """,
                (
                    _require_text(title, "title") if title is not None else existing_node.title,
                    body if body is not None else existing_node.body,
                    status or existing_node.status,
                    evidence_state or existing_node.evidence_state,
                    1 if next_outside_knowledge_used else 0,
                    selected_option_id
                    if selected_option_id is not None
                    else existing_node.selected_option_id,
                    selected_custom_answer
                    if selected_custom_answer is not None
                    else existing_node.selected_custom_answer,
                    _json_dumps(question_options)
                    if question_options is not None
                    else _json_dumps(existing_node.question_options),
                    _json_dumps(generation_metadata)
                    if generation_metadata is not None
                    else _json_dumps(existing_node.generation_metadata),
                    now,
                    node_id,
                    session_id,
                ),
            )
            conn.execute(
                """
                UPDATE explainer_sessions
                SET updated_at = ?
                WHERE id = ? AND owner_user_id = ?
                """,
                (now, session_id, owner_user_id),
            )
        loaded = self.get_session(session_id, owner_user_id=owner_user_id)
        return None if loaded is None else loaded.nodes.get(node_id)

    def delete_node(
        self,
        session_id: str,
        node_id: str,
        *,
        owner_user_id: str,
    ) -> bool:
        session = self.get_session(session_id, owner_user_id=owner_user_id)
        if session is None or node_id not in session.nodes:
            return False
        now = self.db.utcnow_iso()
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE explainer_nodes
                SET deleted_at = ?, updated_at = ?
                WHERE id = ? AND session_id = ? AND deleted_at IS NULL
                """,
                (now, now, node_id, session_id),
            )
            conn.execute(
                """
                UPDATE explainer_sessions
                SET updated_at = ?
                WHERE id = ? AND owner_user_id = ?
                """,
                (now, session_id, owner_user_id),
            )
        return True

    def _build_session(
        self,
        session_row: sqlite3.Row,
        node_rows: list[sqlite3.Row],
        source_rows: list[sqlite3.Row],
        citation_rows: list[sqlite3.Row],
    ) -> ExplainerSession:
        citations_by_node: dict[str, list[ExplainerCitation]] = defaultdict(list)
        for row in citation_rows:
            citations_by_node[row["node_id"]].append(
                ExplainerCitation(
                    id=row["id"],
                    source_id=row["source_id"],
                    source_type=row["source_type"],
                    title=row["title"],
                    excerpt=row["excerpt"],
                    location_label=row["location_label"],
                    start_offset=row["start_offset"],
                    end_offset=row["end_offset"],
                    url=row["url"],
                    snapshot_hash=row["snapshot_hash"],
                )
            )

        child_ids_by_parent: dict[str | None, list[str]] = defaultdict(list)
        nodes: dict[str, ExplainerNode] = {}
        for row in node_rows:
            child_ids_by_parent[row["parent_id"]].append(row["id"])
            nodes[row["id"]] = ExplainerNode(
                id=row["id"],
                session_id=row["session_id"],
                parent_id=row["parent_id"],
                ordinal=row["ordinal"],
                title=row["title"],
                body=row["body"],
                kind=row["kind"],
                intent=row["intent"],
                status=row["status"],
                evidence_state=row["evidence_state"],
                outside_knowledge_used=bool(row["outside_knowledge_used"]),
                citations=citations_by_node.get(row["id"], []),
                question_options=_json_loads(row["question_options_json"]),
                selected_option_id=row["selected_option_id"],
                selected_custom_answer=row["selected_custom_answer"],
                generation_metadata=_json_loads(row["generation_metadata_json"]),
                child_node_ids=[],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                deleted_at=row["deleted_at"],
            )
        for parent_id, child_ids in child_ids_by_parent.items():
            if parent_id is not None and parent_id in nodes:
                nodes[parent_id].child_node_ids = child_ids

        selected_sources = [
            ExplainerSelectedSource(
                source_id=row["source_id"],
                source_type=row["source_type"],
                title=row["title"],
                added_at=row["added_at"],
                snapshot_version=row["snapshot_version"],
                metadata=_json_loads(row["metadata_json"]),
            )
            for row in source_rows
        ]

        return ExplainerSession(
            id=session_row["id"],
            owner_user_id=session_row["owner_user_id"],
            title=session_row["title"],
            mode=session_row["mode"],
            status=session_row["status"],
            output_intent=session_row["output_intent"],
            grounding=session_row["grounding"],
            depth_preset=session_row["depth_preset"],
            selected_sources=selected_sources,
            root_node_ids=child_ids_by_parent.get(None, []),
            nodes=nodes,
            created_at=session_row["created_at"],
            updated_at=session_row["updated_at"],
            archived_at=session_row["archived_at"],
        )

    def _replace_sources(
        self,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        owner_user_id: str,
        sources: list[ExplainerSelectedSource],
        now: str,
    ) -> None:
        conn.execute(
            """
            UPDATE explainer_selected_sources
            SET deleted_at = ?
            WHERE session_id = ? AND owner_user_id = ? AND deleted_at IS NULL
            """,
            (now, session_id, owner_user_id),
        )
        for ordinal, source in enumerate(sources):
            conn.execute(
                """
                INSERT INTO explainer_selected_sources (
                    id, session_id, owner_user_id, ordinal, source_id, source_type, title,
                    added_at, snapshot_version, metadata_json, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    _new_id("exp_src"),
                    session_id,
                    owner_user_id,
                    ordinal,
                    source.source_id,
                    source.source_type,
                    source.title,
                    source.added_at,
                    source.snapshot_version,
                    _json_dumps(source.metadata),
                ),
            )

    @staticmethod
    def _next_child_ordinal(
        conn: sqlite3.Connection,
        *,
        session_id: str,
        parent_id: str | None,
    ) -> int:
        if parent_id is None:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(ordinal), -1) AS max_ordinal
                FROM explainer_nodes
                WHERE session_id = ? AND parent_id IS NULL AND deleted_at IS NULL
                """,
                (session_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(ordinal), -1) AS max_ordinal
                FROM explainer_nodes
                WHERE session_id = ? AND parent_id = ? AND deleted_at IS NULL
                """,
                (session_id, parent_id),
            ).fetchone()
        return int(row["max_ordinal"]) + 1


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _require_text(value: str, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise InputError(f"{field_name} is required")
    return text


def _require_enum(value: str, enum_type: type, field_name: str) -> None:
    allowed = {item.value for item in enum_type}
    if value not in allowed:
        raise InputError(f"{field_name} must be one of {sorted(allowed)}")


def _normalize_source(
    source: ExplainerSelectedSource | dict[str, Any],
    *,
    fallback_added_at: str,
) -> ExplainerSelectedSource:
    if isinstance(source, ExplainerSelectedSource):
        return source
    return ExplainerSelectedSource(
        source_id=_require_text(str(source.get("source_id", "")), "source_id"),
        source_type=_require_text(str(source.get("source_type", "")), "source_type"),
        title=_require_text(str(source.get("title", "")), "source title"),
        added_at=str(source.get("added_at") or fallback_added_at),
        snapshot_version=source.get("snapshot_version"),
        metadata=source.get("metadata"),
    )


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)
