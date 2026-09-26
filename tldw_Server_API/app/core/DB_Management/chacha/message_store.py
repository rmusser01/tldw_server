from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Chat.history_selection import (
    HistoryFencesV1,
    HistorySelectionError,
    HistorySelectionSnapshotV1,
    resolve_legacy_projection,
    resolve_parent_path,
    snapshot_to_wire,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    BackendType,
    CharactersRAGDBError,
    ConflictError,
    FTSQueryTranslator,
    InputError,
    logger,
)
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


MAX_CHAT_ATTACHMENT_READ_BYTES = 32 * 1024 * 1024

class MessageStore:
    """Focused persistence seam for message CRUD operations."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db
        self._last_message_order_timestamp: str | None = None
        self._message_order_lock = threading.Lock()

    @staticmethod
    def _history_digest(value: Any) -> str:
        """Hash finite canonical JSON for native storage provenance."""
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _history_owner_key(owner_client_id: str, owner_key: str | None) -> str:
        """Use an adapter namespace externally; the fallback is internal to this DB."""
        if not isinstance(owner_client_id, str) or not owner_client_id.strip():
            raise NotFoundError("Conversation not found.")
        if owner_key is not None and (not isinstance(owner_key, str) or not owner_key.strip()):
            raise HistorySelectionError("invalid_owner_key")
        return owner_key if owner_key is not None else f"native-client:{owner_client_id}"

    def _lock_history_owner(self, conn: Any, conversation_id: str, owner_client_id: str) -> None:
        """Lock only the conversation fence, never existing messages or metadata."""
        query = (
            "SELECT id FROM conversations WHERE id = ? AND client_id = ? AND deleted = FALSE FOR UPDATE"
            if self._db.backend_type == BackendType.POSTGRESQL
            else "SELECT id FROM conversations WHERE id = ? AND client_id = ? AND deleted = FALSE"
        )
        row = conn.execute(
            query,
            (conversation_id, owner_client_id),
        ).fetchone()
        if row is None:
            raise NotFoundError("Conversation not found.")

    def _read_history_snapshot(
        self,
        conversation_id: str,
        *,
        owner_client_id: str,
        owner_key: str,
        projection_id: str | None,
        conn: Any,
        selected_ids: tuple[str, ...] = (),
        include_message_versions: bool = False,
    ) -> tuple[HistorySelectionSnapshotV1, tuple[dict[str, Any], ...]]:
        """One statement captures all fences, rows, metadata, assets and an optional base.

        Hash binary values in the engine, returning bytes only for requested content.
        Window gating enforces a 32 MiB selected image budget before bytes cross the
        database boundary. No display loader, per-message query or row cap is used.
        """
        postgres = self._db.backend_type == BackendType.POSTGRESQL
        image_data_sql = "CASE WHEN mi.message_id IS NOT NULL THEN mi.image_data ELSE m.image_data END"
        if postgres:
            requested_sql = "SELECT jsonb_array_elements_text(CAST(? AS JSONB)) AS id"

            def text_hash(field: str) -> str:
                return f"encode(sha256(convert_to(COALESCE({field}, ''), 'UTF8')), 'hex')"

            binary_hash = f"encode(sha256({image_data_sql}), 'hex')"
            byte_length = "octet_length"
        else:

            requested_sql = "SELECT value AS id FROM json_each(?)"

            def text_hash(field: str) -> str:
                return f"h1_sha256(COALESCE({field}, ''))"

            binary_hash = f"h1_sha256({image_data_sql})"
            byte_length = "length"
        # Keep per-conversation payloads out of the per-message window sort. Even
        # CASE around a joined payload can materialize it once per message first.
        # Scalar reads in the first-row CASE preserve one-statement coherence.
        # All interpolated expressions are fixed backend SQL, never caller identifiers.
        query = f"""
            WITH requested AS ({requested_sql})
            SELECT c.version AS conversation_version, c.history_version,
                   cs.settings_version, cs.conversation_id AS settings_row_id,
                   bs.conversation_id AS behavior_row_id,
                   bs.status AS behavior_status, bs.schema_version AS behavior_schema_version,
                   bs.digest AS behavior_digest, bs.size_bytes AS behavior_size_bytes,
                   CASE WHEN ROW_NUMBER() OVER (ORDER BY m.timestamp, m.last_modified, m.id, mi.position) = 1
                        THEN {text_hash('(SELECT behavior.canonical_json FROM conversation_behavior_snapshots behavior WHERE behavior.conversation_id = c.id)')}
                        END AS behavior_content_hash,
                   CASE WHEN ROW_NUMBER() OVER (ORDER BY m.timestamp, m.last_modified, m.id, mi.position) = 1
                        THEN (SELECT settings.settings_json FROM conversation_settings settings
                              WHERE settings.conversation_id = c.id) END AS settings_json,
                   CASE WHEN ROW_NUMBER() OVER (ORDER BY m.timestamp, m.last_modified, m.id, mi.position) = 1
                        THEN (SELECT projection.confirmation_json FROM conversation_history_projections projection
                              WHERE projection.conversation_id = c.id AND projection.client_id = c.client_id
                                AND projection.owner_key = ? AND projection.projection_id = ?) END AS projection_json,
                   c.character_id, c.assistant_kind, c.assistant_id, c.persona_memory_mode,
                   c.scope_type, c.workspace_id,
                   m.id, m.parent_message_id, m.sender, m.version AS message_version,
                   m.history_admission_json, SUBSTR(COALESCE(m.content, ''), 1, 200) AS preview,
                   {text_hash('m.content')} AS content_hash,
                   mm.message_id AS metadata_id,
                   {text_hash('mm.tool_calls_json')} AS tool_calls_hash,
                   {text_hash('mm.extra_json')} AS extra_hash,
                   CAST(mm.last_modified AS TEXT) AS metadata_modified,
                   mi.message_id AS ordered_image_message_id, mi.position,
                   CASE WHEN mi.message_id IS NOT NULL THEN mi.image_mime_type ELSE m.image_mime_type END AS image_mime,
                   {binary_hash} AS image_hash,
                   CASE WHEN requested.id IS NOT NULL THEN m.content END AS selected_text,
                   CASE WHEN requested.id IS NOT NULL THEN mm.tool_calls_json END AS selected_tools,
                   CASE WHEN requested.id IS NOT NULL THEN mm.extra_json END AS selected_extra,
                   SUM(CASE WHEN requested.id IS NOT NULL
                       THEN COALESCE({byte_length}({image_data_sql}), 0) ELSE 0 END)
                       OVER () AS selected_image_bytes,
                   CASE WHEN requested.id IS NOT NULL AND
                       SUM(CASE WHEN requested.id IS NOT NULL
                           THEN COALESCE({byte_length}({image_data_sql}), 0) ELSE 0 END)
                           OVER () <= 33554432
                       THEN {image_data_sql} END AS selected_image
            FROM conversations c
            LEFT JOIN conversation_settings cs ON cs.conversation_id = c.id
            LEFT JOIN conversation_behavior_snapshots bs ON bs.conversation_id = c.id
            LEFT JOIN messages m ON m.conversation_id = c.id AND m.deleted = FALSE
            LEFT JOIN message_metadata mm ON mm.message_id = m.id
            LEFT JOIN message_images mi ON mi.message_id = m.id
            LEFT JOIN requested ON requested.id = m.id
            WHERE c.id = ? AND c.client_id = ? AND c.deleted = FALSE
            ORDER BY m.timestamp, m.last_modified, m.id, mi.position
        """  # nosec B608 - fixed SQL fragments; all input values are bound
        result = conn.execute(
            query, (json.dumps(selected_ids), owner_key, projection_id, conversation_id, owner_client_id)
        )
        records = result.fetchall()
        if not records:
            raise NotFoundError("Conversation not found.")
        header = dict(records[0])
        if int(header["selected_image_bytes"] or 0) > 33554432:
            raise HistorySelectionError("selected_content_too_large")
        nodes: dict[str, dict[str, Any]] = {}
        contents: dict[str, dict[str, Any]] = {}
        provenance: dict[str, Any] = {}
        selected_set = set(selected_ids)
        for raw in records:
            record = dict(raw)
            mid = record["id"]
            if mid is None:
                continue
            if mid not in nodes:
                try:
                    authority = json.loads(record["history_admission_json"] or "null")
                except (TypeError, ValueError):
                    authority = None
                provenance[mid] = authority
                node = {
                    "id": mid,
                    "conversation_id": conversation_id,
                    "parent_id": record["parent_message_id"],
                    "role": record["sender"],
                    "preview": record["preview"],
                    "settled": not isinstance(authority, dict) or authority.get("settled", True) is True,
                    "metadata": [],
                    "assets": [],
                    "revision": str(record["message_version"]),
                }
                if isinstance(authority, dict) and authority.get("version") == 1:
                    interpretation = authority.get("interpretation", {})
                    if interpretation.get("kind") == "legacy_linear_v1":
                        node["legacy_projection_id"] = interpretation.get("projection_id")
                if record["metadata_id"] is not None:
                    node["metadata"].append(
                        {
                            "id": mid,
                            "kind": "message_metadata",
                            "revision": self._history_digest(
                                [record["tool_calls_hash"], record["extra_hash"], record["metadata_modified"]]
                            ),
                        }
                    )
                nodes[mid] = node
                if mid in selected_set:
                    contents[mid] = {
                        "id": mid,
                        "message": record["selected_text"] or "",
                        "images": [],
                        "tool_calls": json.loads(record["selected_tools"] or "null"),
                        "extra_metadata": json.loads(record["selected_extra"] or "null"),
                    }
                    if include_message_versions:
                        # Composition needs the physical version to distinguish an
                        # unedited placeholder; never expose it in capture or hashes.
                        contents[mid]["_message_version"] = record["message_version"]
                node["_source"] = [record["message_version"], record["content_hash"], record["history_admission_json"]]
            if (record["ordered_image_message_id"] is not None
                    or record["image_hash"] is not None or record["image_mime"] is not None):
                position = record["position"] if record["position"] is not None else "primary"
                nodes[mid]["assets"].append(
                    {
                        "id": f"{mid}:{position}",
                        "kind": "image",
                        "revision": self._history_digest([record["image_hash"], record["image_mime"]]),
                    }
                )
                if mid in selected_set:
                    image_bytes = record["selected_image"]
                    if record["ordered_image_message_id"] is not None and (
                        not isinstance(record["position"], int)
                        or record["position"] != len(contents[mid]["images"])
                    ):
                        raise HistorySelectionError("selected_attachment_unavailable")
                    if (not isinstance(image_bytes, (bytes, bytearray, memoryview))
                            or not image_bytes or not record["image_mime"]):
                        raise HistorySelectionError("selected_attachment_unavailable")
                    contents[mid]["images"].append(
                        "data:"
                        + record["image_mime"]
                        + ";base64,"
                        + base64.b64encode(bytes(image_bytes)).decode("ascii")
                    )
        for node in nodes.values():
            source = node.pop("_source")
            node["revision"] = self._history_digest([source, node])
            if node["id"] in contents:
                contents[node["id"]]["revision"] = node["revision"]
                contents[node["id"]]["images"] = tuple(contents[node["id"]]["images"])
        manifest = tuple(nodes.values())
        status: dict[str, Any] = {"kind": "legacy_review_required"}
        if projection_id is not None:
            if header["projection_json"] is None:
                raise HistorySelectionError("missing_projection")
            accepted = json.loads(header["projection_json"])
            reviewed = {row["id"]: row["revision"] for row in accepted["source_members"]}
            path = accepted["ordered_path_ids"]
            if any(mid not in nodes or nodes[mid]["revision"] != reviewed[mid] for mid in path):
                raise HistorySelectionError("stale_projection")
            status = {"kind": "legacy_linear_v1", "projection_id": projection_id, "ordered_path_ids": path}
        else:
            try:
                resolve_parent_path(manifest, {"kind": "empty"})
                unversioned = [
                    node
                    for node in manifest
                    if not (
                        isinstance(provenance[node["id"]], dict)
                        and provenance[node["id"]].get("version") == 1
                        and provenance[node["id"]].get("interpretation", {}).get("kind") == "parent_graph_v1"
                    )
                ]
                unversioned_ids = {node["id"] for node in unversioned}
                roots = [node for node in unversioned if node["parent_id"] is None]
                parents = [node["parent_id"] for node in unversioned if node["parent_id"] is not None]
                # Unprotected legacy data must itself be a unique complete chain.
                legacy_chain = not unversioned or (
                    len(roots) == 1
                    and len(parents) == len(set(parents))
                    and all(parent in unversioned_ids for parent in parents)
                )
                if legacy_chain and not any(node.get("legacy_projection_id") for node in manifest):
                    status = {"kind": "parent_graph_v1"}
            except HistorySelectionError:
                status = {"kind": "legacy_review_required"}
        fences = HistoryFencesV1(
            str(header["conversation_version"]), str(header["history_version"]), str(header["settings_version"] or 0)
        )
        context = {
            key: header[key]
            for key in (
                "settings_json",
                "settings_row_id",
                "behavior_row_id",
                "behavior_status",
                "behavior_schema_version",
                "behavior_digest",
                "behavior_size_bytes",
                "behavior_content_hash",
                "character_id",
                "assistant_kind",
                "assistant_id",
                "persona_memory_mode",
                "scope_type",
                "workspace_id",
            )
        }
        # Only the explicitly empty plain policy is representable by H1's legacy copier.
        # Row presence is digest-bound: unreadable required state is never a default.
        plain_settings = header["settings_row_id"] is None
        if not plain_settings:
            try:
                plain_settings = json.loads(header["settings_json"]) == {}
            except (TypeError, ValueError):
                plain_settings = False
        context_digest = self._history_digest(context)
        native_fork_context = {
            "policy": "plain_v1",
            "storage_context_digest": context_digest,
            "supported": plain_settings
            and header["behavior_row_id"] is None
            and all(header[key] is None for key in (
                "character_id", "assistant_kind", "assistant_id", "persona_memory_mode"
            )),
        }
        snapshot = HistorySelectionSnapshotV1(
            version=1,
            owner_key=owner_key,
            conversation_id=conversation_id,
            fences=fences,
            nodes=manifest,
            source_digest=self._history_digest(manifest),
            interpretation_status=status,
            storage_context_digest=context_digest,
            native_fork_context=native_fork_context,
        )
        if any(mid not in contents for mid in selected_ids):
            raise HistorySelectionError("selected_content_mismatch")
        return snapshot, tuple(contents[mid] for mid in selected_ids)

    def get_conversation_history_snapshot(
        self,
        conversation_id: str,
        *,
        owner_client_id: str,
        owner_key: str | None = None,
        projection_id: str | None = None,
        conn: Any | None = None,
        lock_for_update: bool = False,
    ) -> HistorySelectionSnapshotV1:
        """Capture a complete owned manifest, optionally using a caller's admission fence.

        `owner_client_id` is authenticated identity, independently checked against the
        conversation. Browser adapters must supply their verified server/account
        `owner_key`; the default key is only suitable inside this database.
        An explicit projection ID selects the view's immutable accepted legacy base.
        """
        namespace = self._history_owner_key(owner_client_id, owner_key)
        transaction = nullcontext(conn) if conn is not None else self._db.transaction()
        with transaction as active:
            if lock_for_update:
                self._lock_history_owner(active, conversation_id, owner_client_id)
            snapshot, _ = self._read_history_snapshot(
                conversation_id,
                owner_client_id=owner_client_id,
                owner_key=namespace,
                projection_id=projection_id,
                conn=active,
            )
            return snapshot

    def get_conversation_history_selected_content(
        self,
        conversation_id: str,
        message_ids: Sequence[str],
        *,
        snapshot: HistorySelectionSnapshotV1,
        owner_client_id: str,
        owner_key: str | None = None,
        conn: Any | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Read all selected text/images/tools/extra metadata bound to a captured source.

        Raises on any source or fence drift; callers must retry capture, never compose
        mixed revisions. Returned dictionaries are detached from database state.
        """
        namespace = self._history_owner_key(owner_client_id, owner_key)
        ids = tuple(message_ids)
        if (
            snapshot.owner_key != namespace
            or snapshot.conversation_id != conversation_id
            or len(set(ids)) != len(ids)
            or any(not isinstance(mid, str) for mid in ids)
        ):
            raise HistorySelectionError("selected_content_mismatch")
        projection_id = snapshot.interpretation_status.get("projection_id")
        transaction = nullcontext(conn) if conn is not None else self._db.transaction()
        with transaction as active:
            fresh, content = self._read_history_snapshot(
                conversation_id,
                owner_client_id=owner_client_id,
                owner_key=namespace,
                projection_id=projection_id,
                conn=active,
                selected_ids=ids,
            )
            if (
                snapshot.source_digest != fresh.source_digest
                or snapshot.fences != fresh.fences
                or snapshot.storage_context_digest != fresh.storage_context_digest
            ):
                raise HistorySelectionError("stale_source")
            return content

    def confirm_legacy_history_projection(
        self,
        confirmation: Mapping[str, Any],
        *,
        owner_client_id: str,
        owner_key: str | None = None,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """Authorize, replay or CAS-insert an immutable LegacyHistoryProjectionV1 wire value.

        A matching authorized replay precedes fresh source validation, including after
        appends or source edits. Reusing its ID for different confirmation bytes fails.
        Caller-owned transactions are never committed here.
        """
        namespace = self._history_owner_key(owner_client_id, owner_key)
        body = dict(confirmation)
        required = {
            "version",
            "projection_id",
            "owner_key",
            "conversation_id",
            "source_digest",
            "fences",
            "source_members",
            "ordered_path_ids",
            "cursor",
            "selection_revision",
        }
        if (
            set(body) != required
            or type(body["version"]) is not int
            or body["version"] != 1
            or body["owner_key"] != namespace
            or not isinstance(body["projection_id"], str)
            or not body["projection_id"]
            or type(body["selection_revision"]) is not int
            or body["selection_revision"] < 0
        ):
            raise HistorySelectionError("invalid_projection")
        try:
            canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise HistorySelectionError("invalid_projection") from exc
        body = json.loads(canonical)
        cursor = body["cursor"]
        path = body["ordered_path_ids"]
        if (
            not isinstance(body["conversation_id"], str)
            or not body["conversation_id"]
            or not isinstance(path, list)
            or any(not isinstance(mid, str) or not mid for mid in path)
            or not isinstance(cursor, dict)
            or cursor.get("kind") not in {"empty", "before_message", "after_message"}
            or set(cursor) != ({"kind"} if cursor.get("kind") == "empty" else {"kind", "message_id"})
            or (cursor.get("kind") != "empty" and not isinstance(cursor.get("message_id"), str))
        ):
            raise HistorySelectionError("invalid_projection")
        cid = body["conversation_id"]
        transaction = nullcontext(conn) if conn is not None else self._db.transaction()
        with transaction as active:
            self._lock_history_owner(active, cid, owner_client_id)
            existing = active.execute(
                "SELECT confirmation_json, projection_digest, created_at FROM conversation_history_projections "
                "WHERE client_id = ? AND owner_key = ? AND conversation_id = ? AND projection_id = ?",
                (owner_client_id, namespace, cid, body["projection_id"]),
            ).fetchone()
            if existing is not None:
                record = dict(existing)
                if record["confirmation_json"] != canonical:
                    raise HistorySelectionError("projection_id_conflict")
                return {
                    **json.loads(canonical),
                    "projection_digest": record["projection_digest"],
                    "created_at": record["created_at"],
                }
            fresh, _ = self._read_history_snapshot(
                cid, owner_client_id=owner_client_id, owner_key=namespace, projection_id=None, conn=active
            )
            wire = snapshot_to_wire(fresh)
            members = [{"id": row["id"], "revision": row["revision"]} for row in wire["nodes"]]
            if (
                body["source_digest"] != fresh.source_digest
                or body["fences"] != wire["fences"]
                or body["source_members"] != members
            ):
                raise HistorySelectionError("stale_source")
            try:
                resolve_legacy_projection(wire["nodes"], body["ordered_path_ids"], body["cursor"])
            except (HistorySelectionError, KeyError, TypeError) as exc:
                raise HistorySelectionError("invalid_projection") from exc
            digest = self._history_digest(body)
            created_at = self._db._get_current_utc_timestamp_iso()
            active.execute(
                "INSERT INTO conversation_history_projections (projection_id, conversation_id, client_id, owner_key, "
                "interpretation_version, source_digest, history_fence, source_members_json, ordered_path_ids_json, "
                "confirmation_json, projection_digest, created_at) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
                (
                    body["projection_id"],
                    cid,
                    owner_client_id,
                    namespace,
                    body["source_digest"],
                    fresh.fences.history,
                    json.dumps(members),
                    json.dumps(body["ordered_path_ids"]),
                    canonical,
                    digest,
                    created_at,
                ),
            )
            return {**json.loads(canonical), "projection_digest": digest, "created_at": created_at}

    def validate_history_selection(
        self, conversation_id: str, selection: Mapping[str, Any], *,
        owner_client_id: str, owner_key: str, conn: Any,
    ) -> tuple[HistorySelectionSnapshotV1, tuple[dict[str, Any], ...]]:
        """Re-resolve retained membership and context under the caller's owner fence."""
        from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import HistorySelectionV1
        from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection

        body = HistorySelectionV1.model_validate(selection).model_dump(mode="json")
        if body["owner_key"] != owner_key or body["conversation_id"] != conversation_id:
            raise HistorySelectionError("owner_conversation_mismatch")
        fresh, content = self._read_history_snapshot(
            conversation_id, owner_client_id=owner_client_id, owner_key=owner_key,
            projection_id=body["interpretation"].get("projection_id"), conn=conn,
            selected_ids=tuple(row["id"] for row in body["messages"]),
            include_message_versions=True,
        )
        result = resolve_history_selection(snapshot_to_wire(fresh), body, body["purpose"], body["request_context_digest"])
        if result["status"] != "ready":
            raise HistorySelectionError(result["code"])
        # Fence changes alone are not evidence of a changed retained path or context.
        if result["selection"]["selection_digest"] != body["selection_digest"] or result["selection"]["messages"] != body["messages"]:
            raise HistorySelectionError("stale_selection")
        return fresh, content

    def _write_history_authority(self, conn: Any, message_id: str, authority: Mapping[str, Any]) -> None:
        """Write owner-only provenance; public CRUD and sync never call this seam."""
        conn.execute(
            "UPDATE messages SET history_admission_json = ? WHERE id = ?",
            (json.dumps(authority, sort_keys=True, ensure_ascii=False, separators=(",", ":")), message_id),
        )

    def _history_message_state(
        self,
        conversation_id: str,
        message_id: str,
        *,
        owner_client_id: str,
        owner_key: str,
        conn: Any,
    ) -> str:
        """Hash the accepted row's state without revalidating its old source projection.

        The immutable interpretation tag is part of row state, but resolving its
        historical base is admission-only. Later source edits are unrelated to
        the accepted input's content, parent, role, metadata and assets.
        """
        fresh, content = self._read_history_snapshot(
            conversation_id,
            owner_client_id=owner_client_id,
            owner_key=owner_key,
            projection_id=None,
            conn=conn,
            selected_ids=(message_id,),
        )
        node = next((row for row in snapshot_to_wire(fresh)["nodes"] if row["id"] == message_id), None)
        if node is None or not content:
            raise HistorySelectionError("stale_parent")
        node.pop("revision")
        item = {key: value for key, value in content[0].items() if key != "revision"}
        return self._history_digest([node, item])

    @staticmethod
    def _history_intent_digest(message: Mapping[str, Any]) -> str:
        """Stable retry identity includes ordered image bytes and explicit parent presence."""

        def encode(value: Any) -> Any:
            if isinstance(value, (bytes, bytearray, memoryview)):
                return {"bytes": base64.b64encode(bytes(value)).decode("ascii")}
            if isinstance(value, Mapping):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item) for item in value]
            return value

        return MessageStore._history_digest(encode(message))

    def append_selected_history_input(
        self,
        conversation_id: str,
        selection: Mapping[str, Any],
        message: Mapping[str, Any],
        *,
        owner_client_id: str,
        owner_key: str,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """Atomically validate selected history and append one accepted current input."""
        body, data = dict(selection), dict(message)
        if (
            body.get("owner_key") != owner_key
            or body.get("conversation_id") != conversation_id
            or body.get("purpose") != "send"
        ):
            raise HistorySelectionError("owner_conversation_mismatch")
        if not data.get("id") or data.get("sender") not in {"user", "tool", "system"}:
            raise HistorySelectionError("invalid_input")
        if data.get("conversation_id", conversation_id) != conversation_id:
            raise HistorySelectionError("owner_conversation_mismatch")
        intent = self._history_intent_digest(data)
        with nullcontext(conn) if conn is not None else self._db.transaction() as active:
            self._lock_history_owner(active, conversation_id, owner_client_id)
            existing = active.execute("SELECT history_admission_json FROM messages WHERE id = ?", (data["id"],)).fetchone()
            if existing is not None:
                authority = json.loads(existing["history_admission_json"] or "null")
                if (
                    not isinstance(authority, dict)
                    or authority.get("intent_digest") != intent
                    or authority.get("selection") != body
                ):
                    raise HistorySelectionError("message_id_conflict")
                self._validate_history_parent(
                    conversation_id,
                    authority["admission"],
                    owner_client_id=owner_client_id,
                    owner_key=owner_key,
                    conn=active,
                )
                return authority["admission"]
            fresh, _ = self.validate_history_selection(
                conversation_id, body, owner_client_id=owner_client_id, owner_key=owner_key, conn=active
            )
            parent = body["messages"][-1]["id"] if body["messages"] else None
            if "parent_message_id" in data and data["parent_message_id"] != parent:
                raise HistorySelectionError("parent_mismatch")
            data.update(conversation_id=conversation_id, parent_message_id=parent, client_id=owner_client_id)
            mid = self.add_message(data, conn=active)
            if data.get("tool_calls") is not None or data.get("extra_metadata") is not None:
                self._add_message_metadata_with_conn(mid, data.get("tool_calls"), data.get("extra_metadata"), active)
            scope = dict(
                active.execute(
                    "SELECT scope_type, workspace_id FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
            )
            admission = {
                "version": 1,
                "owner_key": owner_key,
                "conversation_id": conversation_id,
                "input_message_id": mid,
                "input_message_revision": "1",
                "selection_digest": body["selection_digest"],
                "messages": body["messages"],
                "originating_selection_revision": body["selection_revision"],
            }
            authority = {
                "version": 1,
                "interpretation": body["interpretation"],
                "settled": True,
                "admission": admission,
                "selection": body,
                "intent_digest": intent,
                "storage_context_digest": fresh.storage_context_digest,
                "scope": scope,
            }
            self._write_history_authority(active, mid, authority)
            authority["input_state_digest"] = self._history_message_state(
                conversation_id, mid, owner_client_id=owner_client_id, owner_key=owner_key, conn=active
            )
            self._write_history_authority(active, mid, authority)
            return admission

    def append_selected_history_inputs(
        self,
        conversation_id: str,
        selection: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        *,
        owner_client_id: str,
        owner_key: str,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """Accept a server-owned current-input chain once, under one owner transaction."""
        body = dict(selection)
        if not messages or any(message.get("sender") not in {"user", "tool"} for message in messages):
            raise HistorySelectionError("invalid_input")
        if (
            body.get("owner_key") != owner_key
            or body.get("conversation_id") != conversation_id
            or body.get("purpose") != "send"
        ):
            raise HistorySelectionError("owner_conversation_mismatch")
        with nullcontext(conn) if conn is not None else self._db.transaction() as active:
            self._lock_history_owner(active, conversation_id, owner_client_id)
            consumed = active.execute(
                "SELECT history_admission_json FROM messages "
                "WHERE conversation_id = ? AND history_admission_json IS NOT NULL",
                (conversation_id,),
            ).fetchall()
            for row in consumed:
                authority = json.loads(row["history_admission_json"])
                if (
                    authority.get("server_completion") is True
                    and authority.get("admission", {}).get("owner_key") == owner_key
                    and authority.get("admission", {}).get("selection_digest") == body.get("selection_digest")
                ):
                    raise HistorySelectionError("selection_already_consumed")
            self.validate_history_selection(
                conversation_id, body, owner_client_id=owner_client_id, owner_key=owner_key, conn=active
            )
            scope = dict(
                active.execute(
                    "SELECT scope_type, workspace_id FROM conversations WHERE id = ?", (conversation_id,)
                ).fetchone()
            )
            parent = body["messages"][-1]["id"] if body["messages"] else None
            chain = []
            for message in messages:
                data = dict(message)
                if "parent_message_id" in data and data["parent_message_id"] != parent:
                    raise HistorySelectionError("parent_mismatch")
                data.update(
                    id=self._db._generate_uuid(),
                    conversation_id=conversation_id,
                    parent_message_id=parent,
                    client_id=owner_client_id,
                )
                mid = self.add_message(data, conn=active)
                if data.get("tool_calls") is not None or data.get("extra_metadata") is not None:
                    self._add_message_metadata_with_conn(mid, data.get("tool_calls"), data.get("extra_metadata"), active)
                self._write_history_authority(
                    active, mid, {"version": 1, "interpretation": body["interpretation"], "settled": True}
                )
                chain.append(
                    {
                        "id": mid,
                        "version": "1",
                        "state": self._history_message_state(
                            conversation_id, mid, owner_client_id=owner_client_id, owner_key=owner_key, conn=active
                        ),
                    }
                )
                parent = mid
            admission = {
                "version": 1,
                "owner_key": owner_key,
                "conversation_id": conversation_id,
                "input_message_id": parent,
                "input_message_revision": "1",
                "selection_digest": body["selection_digest"],
                "messages": body["messages"],
                "originating_selection_revision": body["selection_revision"],
            }
            authority = {
                "version": 1,
                "interpretation": body["interpretation"],
                "settled": True,
                "server_completion": True,
                "admission": admission,
                "selection": body,
                "scope": scope,
                "input_state_digest": chain[-1]["state"],
                "input_chain": chain,
            }
            for item in chain:
                self._write_history_authority(active, item["id"], authority)
            return admission

    def _validate_history_parent(
        self,
        conversation_id: str,
        reference: Mapping[str, Any],
        *,
        owner_client_id: str,
        owner_key: str,
        conn: Any,
    ) -> dict[str, Any]:
        """Check owner-issued acceptance and live input state without checking current branch."""
        if reference.get("owner_key") != owner_key or reference.get("conversation_id") != conversation_id:
            raise HistorySelectionError("owner_conversation_mismatch")
        row = conn.execute(
            "SELECT version, history_admission_json FROM messages "
            "WHERE id = ? AND conversation_id = ? AND deleted = FALSE",
            (reference["input_message_id"], conversation_id),
        ).fetchone()
        if row is None or str(row["version"]) != reference["input_message_revision"]:
            raise HistorySelectionError("stale_parent")
        authority = json.loads(row["history_admission_json"] or "null")
        if not isinstance(authority, dict) or any(
            authority.get("admission", {}).get(key) != value for key, value in reference.items()
        ):
            raise HistorySelectionError("invalid_admission")
        state = self._history_message_state(
            conversation_id, reference["input_message_id"], owner_client_id=owner_client_id, owner_key=owner_key, conn=conn
        )
        if state != authority["input_state_digest"]:
            raise HistorySelectionError("stale_parent")
        for item in authority.get("input_chain", []):
            version = conn.execute(
                "SELECT version FROM messages WHERE id = ? AND conversation_id = ? AND deleted = FALSE",
                (item["id"], conversation_id),
            ).fetchone()
            if version is None or str(version["version"]) != item["version"]:
                raise HistorySelectionError("stale_parent")
            state = self._history_message_state(
                conversation_id, item["id"], owner_client_id=owner_client_id, owner_key=owner_key, conn=conn
            )
            if state != item["state"]:
                raise HistorySelectionError("stale_parent")
        scope = dict(
            conn.execute("SELECT scope_type, workspace_id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        )
        if scope != authority["scope"]:
            raise HistorySelectionError("stale_scope")
        return authority

    def settle_history_admission(
        self,
        conversation_id: str,
        reference: Mapping[str, Any],
        message: Mapping[str, Any],
        *,
        owner_client_id: str,
        owner_key: str,
        conn: Any | None = None,
    ) -> str:
        """Settle an assistant/tool result against its immutable accepted input."""
        from tldw_Server_API.app.api.v1.schemas.history_selection_schemas import HistoryAdmissionReferenceV1

        binding = HistoryAdmissionReferenceV1.model_validate(reference).model_dump(mode="json")
        data = dict(message)
        if not data.get("id") or data.get("sender") not in {"assistant", "tool"}:
            raise HistorySelectionError("invalid_settlement")
        if data.get("conversation_id", conversation_id) != conversation_id:
            raise HistorySelectionError("owner_conversation_mismatch")
        if "parent_message_id" in data and data["parent_message_id"] != binding["input_message_id"]:
            raise HistorySelectionError("parent_mismatch")
        intent = self._history_intent_digest(data)
        with nullcontext(conn) if conn is not None else self._db.transaction() as active:
            self._lock_history_owner(active, conversation_id, owner_client_id)
            authority = self._validate_history_parent(
                conversation_id, binding, owner_client_id=owner_client_id, owner_key=owner_key, conn=active
            )
            existing = active.execute(
                "SELECT history_admission_json, deleted, version FROM messages WHERE id = ?", (data["id"],)
            ).fetchone()
            if existing is not None:
                saved = json.loads(existing["history_admission_json"] or "null")
                if (
                    existing["deleted"]
                    or existing["version"] != 1
                    or not isinstance(saved, dict)
                    or saved.get("settlement") != binding
                    or saved.get("intent_digest") != intent
                ):
                    raise HistorySelectionError("message_id_conflict")
                state = self._history_message_state(
                    conversation_id, data["id"], owner_client_id=owner_client_id, owner_key=owner_key, conn=active
                )
                if state != saved.get("result_state_digest"):
                    raise HistorySelectionError("message_id_conflict")
                return data["id"]
            data.update(
                conversation_id=conversation_id, parent_message_id=binding["input_message_id"], client_id=owner_client_id
            )
            mid = self.add_message(data, conn=active)
            if data.get("tool_calls") is not None or data.get("extra_metadata") is not None:
                self._add_message_metadata_with_conn(mid, data.get("tool_calls"), data.get("extra_metadata"), active)
            result_authority = {
                "version": 1,
                "interpretation": authority["interpretation"],
                "settled": True,
                "settlement": binding,
                "intent_digest": intent,
            }
            # Install the stable interpretation before hashing the row. Legacy
            # nodes derive their projection tag from this protected provenance.
            self._write_history_authority(active, mid, result_authority)
            result_authority["result_state_digest"] = self._history_message_state(
                conversation_id,
                mid,
                owner_client_id=owner_client_id,
                owner_key=owner_key,
                conn=active,
            )
            self._write_history_authority(active, mid, result_authority)
            return mid

    def _next_message_order_timestamp(self) -> str:
        """Return a millisecond ISO timestamp that is monotonic for message inserts."""
        with self._message_order_lock:
            now = self._db._get_current_utc_timestamp_iso()
            last = self._last_message_order_timestamp
            if last is not None and now <= last:
                try:
                    bumped = datetime.fromisoformat(last.replace("Z", "+00:00")) + timedelta(milliseconds=1)
                    now = bumped.isoformat(timespec="milliseconds").replace("+00:00", "Z")
                except ValueError:
                    logger.warning("Could not parse last message timestamp {}; using current timestamp.", last)
            self._last_message_order_timestamp = now
            return now

    # ------------------------------------------------------------------
    # Message creation
    # ------------------------------------------------------------------

    @staticmethod
    def _row_value(row: Any, key: str, index: int = 0) -> Any:
        if isinstance(row, dict):
            return row.get(key)
        mapping = getattr(row, "_mapping", None)
        if mapping is not None:
            return mapping.get(key)
        try:
            return row[key]
        except (IndexError, KeyError, TypeError):
            return row[index]

    def _advance_history_version(self, conn: Any, conversation_id: str) -> None:
        """Advance the resume history fence within the message mutation transaction."""
        cursor = conn.execute(
            "UPDATE conversations "
            "SET history_version = history_version + 1, last_modified = ? "
            "WHERE id = ? AND deleted = FALSE",
            (self._db._get_current_utc_timestamp_iso(), conversation_id),
        )
        if cursor.rowcount != 1:
            raise InputError(  # noqa: TRY003
                f"Cannot mutate message history: Conversation ID '{conversation_id}' not found or deleted."
            )

    def lock_message_for_edit(self, message_id: str, *, conn: Any) -> None:
        """Lock a live PostgreSQL message before its conversation is locked."""
        if self._db.backend_type != BackendType.POSTGRESQL:
            return
        row = conn.execute(
            "SELECT id FROM messages "
            "WHERE id = ? AND deleted = FALSE "
            "FOR UPDATE",
            (message_id,),
        ).fetchone()
        if row is None:
            raise ConflictError(
                f"Message ID {message_id} is no longer available for editing.",
                entity="messages",
                entity_id=message_id,
            )

    def lock_message_metadata_for_edit(self, message_id: str, *, conn: Any) -> None:
        """Ensure and lock message metadata before the conversation row."""
        if self._db.backend_type != BackendType.POSTGRESQL:
            return
        conn.execute(
            "INSERT INTO message_metadata(message_id, last_modified) "
            "VALUES (?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(message_id) DO NOTHING",
            (message_id,),
        )
        row = conn.execute(
            "SELECT message_id FROM message_metadata "
            "WHERE message_id = ? FOR UPDATE",
            (message_id,),
        ).fetchone()
        if row is None:
            raise ConflictError(
                f"Message metadata for {message_id} is no longer available.",
                entity="message_metadata",
                entity_id=message_id,
            )

    def add_message(
        self,
        msg_data: dict[str, Any],
        *,
        conn: Any | None = None,
    ) -> str | None:
        """
        Adds a new message to a conversation, optionally with image data.

        `id` (UUID string) is auto-generated if not provided in `msg_data`.
        Requires 'conversation_id', 'sender'. Message must have 'content' (text) or image attachments.
        `client_id` defaults to DB instance's `client_id`. `version` is set to 1.
        `timestamp` defaults to current UTC time if not provided; `last_modified` is set to current UTC time.

        Verifies that the parent conversation (given by `conversation_id`) exists and is not deleted.
        FTS updates (`messages_fts` for content) and `sync_log` entries are handled by SQL triggers.

        Args:
            msg_data: Dictionary with message data.
                      Required: 'conversation_id', 'sender'. At least one of 'content' or images.
                      Optional: 'id', 'parent_message_id', 'content' (str),
                                'image_data' (bytes), 'image_mime_type' (str, required if image_data present),
                                'images' (iterable of {'data','mime'}), 'timestamp', 'ranking', 'client_id'.

        Returns:
            The string UUID of the newly added message.

        Raises:
            InputError: If required fields are missing, if both 'content' and attachments are absent,
                        or if the parent conversation is not found or is deleted.
            ConflictError: If a message with the provided 'id' (if any) already exists.
            CharactersRAGDBError: For other database errors (e.g., FK violation for conversation_id).
        """
        images_payload_raw = msg_data.pop('images', None)
        normalized_images: list[tuple[bytes, str]] = []
        if images_payload_raw:
            for entry in images_payload_raw:
                img_bytes: bytes | None = None
                img_mime: str | None = None
                if isinstance(entry, dict):
                    img_bytes = entry.get("data") or entry.get("image_data")
                    img_mime = entry.get("mime") or entry.get("image_mime_type")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    img_bytes, img_mime = entry[0], entry[1]
                if img_bytes is None or img_mime is None:
                    continue
                if isinstance(img_bytes, memoryview):
                    img_bytes = img_bytes.tobytes()
                normalized_images.append((img_bytes, str(img_mime)))

        # Enforce maximum image sizes (single and multi-image) using settings override
        try:
            from tldw_Server_API.app.core.config import settings  # noqa: E402
            _max_img_bytes = int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            _max_img_bytes = 5 * 1024 * 1024  # 5MB default

        # Validate primary image size if present
        primary_img = msg_data.get('image_data')
        if isinstance(primary_img, memoryview):
            primary_img = primary_img.tobytes()
        if isinstance(primary_img, (bytes, bytearray)) and len(primary_img) > _max_img_bytes:
            raise InputError(  # noqa: TRY003
                f"Primary image attachment exceeds maximum size of {_max_img_bytes} bytes"
            )

        # Validate any additional images provided via 'images'
        if normalized_images:
            for b, _m in normalized_images:
                if b is None:
                    continue
                if isinstance(b, memoryview):
                    b = b.tobytes()
                if isinstance(b, (bytes, bytearray)) and len(b) > _max_img_bytes:
                    raise InputError(  # noqa: TRY003
                        f"Message image attachment exceeds maximum size of {_max_img_bytes} bytes"
                    )

        msg_id = msg_data.get('id') or self._db._generate_uuid()

        required_fields = ['conversation_id', 'sender']
        for field in required_fields:
            if field not in msg_data:
                raise InputError(f"Required field '{field}' is missing for message.")  # noqa: TRY003
        if not msg_data.get('content') and not msg_data.get('image_data') and not normalized_images:
            raise InputError("Message must have text content or image data.")  # noqa: TRY003
        if msg_data.get('image_data') and not msg_data.get('image_mime_type'):
            raise InputError("image_mime_type is required if image_data is provided.")  # noqa: TRY003

        if normalized_images and not msg_data.get('image_data'):
            first_bytes, first_mime = normalized_images[0]
            msg_data['image_data'] = first_bytes
            msg_data['image_mime_type'] = first_mime

        client_id = msg_data.get('client_id') or self._db.client_id
        if not client_id:
            raise InputError("Client ID is required for message.")  # noqa: TRY003

        now = self._next_message_order_timestamp()
        timestamp = msg_data.get('timestamp') or now

        query = """
                INSERT INTO messages (id, conversation_id, parent_message_id, sender, content,
                                      image_data, image_mime_type,
                                      timestamp, ranking, last_modified, client_id, version, deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
        if self._db.backend_type == BackendType.POSTGRESQL:
            params = (
                msg_id, msg_data['conversation_id'], msg_data.get('parent_message_id'),
                msg_data['sender'], msg_data.get('content', ''),
                msg_data.get('image_data'), msg_data.get('image_mime_type'),
                timestamp, msg_data.get('ranking'), now, client_id, 1, False
            )
        else:
            params = (
                msg_id, msg_data['conversation_id'], msg_data.get('parent_message_id'),
                msg_data['sender'], msg_data.get('content', ''),
                msg_data.get('image_data'), msg_data.get('image_mime_type'),
                timestamp, msg_data.get('ranking'), now, client_id, 1, 0
            )
        try:
            transaction = nullcontext(conn) if conn is not None else self._db.transaction()
            with transaction as transaction_conn:
                conv_cursor = transaction_conn.execute(
                    "SELECT 1 FROM conversations WHERE id = ? AND deleted = FALSE",
                    (msg_data['conversation_id'],),
                )
                if not conv_cursor.fetchone():
                    raise InputError(  # noqa: TRY003, TRY301
                        f"Cannot add message: Conversation ID '{msg_data['conversation_id']}' not found or deleted."
                    )
                transaction_conn.execute(query, params)
                if normalized_images:
                    self._insert_message_images(msg_id, normalized_images, conn=transaction_conn)
                self._advance_history_version(transaction_conn, msg_data['conversation_id'])
            logger.info(
                'Added message ID: {} to conversation {} (Images stored: {}).',
                msg_id,
                msg_data['conversation_id'],
                len(normalized_images) if normalized_images else ("Yes" if msg_data.get('image_data') else "No"),
            )
            return msg_id  # noqa: TRY300
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: messages.id" in str(e):
                raise ConflictError(  # noqa: TRY003
                    f"Message with ID '{msg_id}' already exists.",
                    entity="messages",
                    entity_id=msg_id,
                ) from e
            raise CharactersRAGDBError(f"Database integrity error adding message: {e}") from e  # noqa: TRY003
        except BackendDatabaseError as e:
            error_text = str(e).lower()
            if "duplicate key" in error_text or "unique constraint" in error_text:
                raise ConflictError(  # noqa: TRY003
                    f"Message with ID '{msg_id}' already exists.",
                    entity="messages",
                    entity_id=msg_id,
                ) from e
            raise CharactersRAGDBError(f"Database error adding message: {e}") from e  # noqa: TRY003
        except InputError:
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error adding message: {e}")
            raise

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def _insert_message_images(
        self,
        message_id: str,
        images: list[tuple[bytes, str]],
        *,
        conn: Any | None = None,
    ) -> None:
        """Insert or replace message images for the given message."""
        if not images:
            return
        params: list[tuple[str, int, bytes, str]] = []
        for idx, (img_bytes, img_mime) in enumerate(images):
            if img_bytes is None or img_mime is None:
                continue
            if isinstance(img_bytes, memoryview):
                img_bytes = img_bytes.tobytes()
            params.append((message_id, idx, img_bytes, img_mime))
        if not params:
            return
        query = (
            "INSERT INTO message_images (message_id, position, image_data, image_mime_type) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(message_id, position) DO UPDATE SET "
            "image_data=excluded.image_data, image_mime_type=excluded.image_mime_type, "
            "created_at=CURRENT_TIMESTAMP"
        )
        if conn is not None:
            conn.executemany(query, params)
        else:
            self._db.execute_many(query, params, commit=False)

    def append_message_image(
        self,
        message_id: str,
        image_bytes: bytes,
        mime_type: str,
        *,
        commit: bool = True,
        conn: Any | None = None,
    ) -> int:
        """Append one image to a message after the current maximum image position."""
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise InputError("image_bytes must be bytes-like.")  # noqa: TRY003
        if not mime_type:
            raise InputError("mime_type is required for message images.")  # noqa: TRY003

        try:
            from tldw_Server_API.app.core.config import settings  # noqa: E402

            max_image_bytes = int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            max_image_bytes = 5 * 1024 * 1024
        if len(image_bytes) > max_image_bytes:
            raise InputError(  # noqa: TRY003
                f"Message image attachment exceeds maximum size of {max_image_bytes} bytes"
            )

        def _append_once(transaction_conn: Any, *, use_db_executor: bool = False) -> int:
            def _execute(query: str, params: tuple[Any, ...]) -> Any:
                if use_db_executor:
                    return self._db.execute_query(query, params)
                return transaction_conn.execute(query, params)

            message_cursor = _execute(
                "SELECT conversation_id FROM messages WHERE id = ? AND deleted = FALSE",
                (message_id,),
            )
            message_row = message_cursor.fetchone()
            if message_row is None:
                raise InputError(  # noqa: TRY003
                    f"Cannot append image: Message ID '{message_id}' not found or deleted."
                )
            conversation_id = str(self._row_value(message_row, "conversation_id"))

            cursor = _execute(
                "SELECT COALESCE(MAX(position), -1) + 1 AS next_position "
                "FROM message_images WHERE message_id = ?",
                (message_id,),
            )
            row = cursor.fetchone()
            position = int(self._row_value(row, "next_position") if row is not None else 0)
            _execute(
                """
                INSERT INTO message_images (message_id, position, image_data, image_mime_type)
                VALUES (?, ?, ?, ?)
                """,
                (message_id, position, bytes(image_bytes), str(mime_type)),
            )
            self._advance_history_version(transaction_conn, conversation_id)
            return position

        def _append_with_retries(existing_conn: Any | None = None) -> int:
            last_error: Exception | None = None
            for _ in range(5):
                try:
                    if existing_conn is not None:
                        return _append_once(existing_conn, use_db_executor=True)
                    with self._db.transaction() as owned_conn:
                        return _append_once(owned_conn, use_db_executor=True)
                except sqlite3.IntegrityError as exc:
                    last_error = exc
                    continue
            raise ConflictError(  # noqa: TRY003
                f"Concurrent append conflict for message image positions on message_id={message_id}",
            ) from last_error

        if conn is not None:
            return _append_once(conn)
        if not commit:
            return _append_with_retries(self._db.get_connection())
        return _append_with_retries()

    def get_message_images(self, message_id: str, *, strict: bool = False) -> list[dict[str, Any]]:
        """Fetch ordered images; strict reads propagate errors and reject gaps."""
        try:
            cursor = self._db.execute_query(
                "SELECT message_id, position, image_data, image_mime_type FROM message_images "
                "WHERE message_id = ? ORDER BY position ASC",
                (message_id,),
            )
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            images: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
                img_bytes = record.get("image_data")
                if isinstance(img_bytes, memoryview):
                    record["image_data"] = img_bytes.tobytes()
                if strict and (record["position"] != len(images) or not record.get("image_data")):
                    raise CharactersRAGDBError("Saved chat attachment positions or data are incomplete")
                images.append(record)
            return images  # noqa: TRY300
        except CharactersRAGDBError as e:
            if strict:
                raise
            logger.error(f"Failed to fetch images for message {message_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Message retrieval
    # ------------------------------------------------------------------

    def get_source_message_projection(
        self,
        conversation_id: str,
        *,
        max_chars: int,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Return one bounded, statement-consistent source-message snapshot."""
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise InputError("conversation_id cannot be empty.")  # noqa: TRY003
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
            raise InputError("max_chars must be a positive integer.")  # noqa: TRY003
        if owner_user_id is not None and (not isinstance(owner_user_id, str) or not owner_user_id.strip()):
            raise InputError("owner_user_id must be a non-empty string.")  # noqa: TRY003

        is_postgres = self._db.backend_type == BackendType.POSTGRESQL
        if is_postgres and owner_user_id is None:
            raise InputError("owner_user_id is required for PostgreSQL source projections.")  # noqa: TRY003
        owner_clause = " AND c.client_id = ?" if is_postgres and owner_user_id else ""
        owner_params = [owner_user_id.strip()] if owner_clause else []
        false_literal = "FALSE" if is_postgres else "0"
        true_literal = "TRUE" if is_postgres else "1"
        invalid_expression = (
            "FALSE"
            if is_postgres
            else "(INSTR(COALESCE(m.sender, ''), CHAR(0)) > 0 "
            "OR INSTR(m.content, CHAR(0)) > 0)"
        )
        # Every eligible formatted message costs at least four characters, and
        # every message after the first also costs a one-character separator.
        # One row beyond the maximum possible fit is sufficient to prove overflow.
        row_limit = (max_chars + 2) // 5 + 1
        char_budget = max_chars + 1
        failure_type = "UnknownDatabaseError"

        try:
            cursor = self._db.execute_query(
                f"""
                    WITH settings AS (
                        SELECT CAST(? AS INTEGER) AS char_budget
                    ),
                    live_conversation AS (
                        SELECT c.id
                        FROM conversations c
                        WHERE c.id = ?
                          AND c.deleted = {false_literal}
                          {owner_clause}
                        LIMIT 1
                    ),
                    eligible AS (
                        SELECT
                            m.id,
                            m.timestamp,
                            m.last_modified,
                            COALESCE(NULLIF(m.sender, ''), 'unknown') || ': ' || m.content
                                AS formatted_text,
                            {invalid_expression} AS source_invalid
                        FROM messages m
                        JOIN live_conversation c ON c.id = m.conversation_id
                        WHERE m.deleted = {false_literal}
                          AND m.content IS NOT NULL
                          AND m.content != ''
                        ORDER BY m.timestamp ASC, m.last_modified ASC, m.id ASC
                        LIMIT ?
                    ),
                    numbered AS (
                        SELECT
                            id,
                            formatted_text,
                            source_invalid,
                            ROW_NUMBER() OVER (
                                ORDER BY timestamp ASC, last_modified ASC, id ASC
                            ) AS ordinal,
                            LENGTH(formatted_text) AS source_length
                        FROM eligible
                    ),
                    costed AS (
                        SELECT
                            ordinal,
                            formatted_text,
                            source_invalid,
                            source_length,
                            CASE WHEN ordinal = 1 THEN 0 ELSE 1 END AS separator_chars,
                            COALESCE(
                                SUM(
                                    source_length + CASE WHEN ordinal = 1 THEN 0 ELSE 1 END
                                ) OVER (
                                    ORDER BY ordinal ASC
                                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                                ),
                                0
                            ) AS prior_chars
                        FROM numbered
                    ),
                    allocated AS (
                        SELECT
                            costed.ordinal,
                            costed.formatted_text,
                            costed.source_invalid,
                            costed.source_length,
                            CASE
                                WHEN settings.char_budget
                                     - costed.prior_chars
                                     - costed.separator_chars <= 0
                                THEN NULL
                                WHEN costed.source_length <= settings.char_budget
                                     - costed.prior_chars
                                     - costed.separator_chars
                                THEN costed.source_length
                                ELSE settings.char_budget
                                     - costed.prior_chars
                                     - costed.separator_chars
                            END AS char_cap
                        FROM costed
                        CROSS JOIN settings
                    ),
                    stats AS (
                        SELECT
                            COALESCE(
                                MAX(CASE WHEN source_invalid THEN 1 ELSE 0 END),
                                0
                            ) AS source_invalid,
                            COALESCE(
                                MAX(
                                    CASE
                                        WHEN char_cap IS NULL OR source_length > char_cap
                                        THEN 1
                                        ELSE 0
                                    END
                                ),
                                0
                            ) AS source_truncated
                        FROM allocated
                    ),
                    projected AS (
                        SELECT
                            ordinal,
                            SUBSTR(
                                formatted_text,
                                1,
                                CAST(char_cap AS INTEGER)
                            ) AS source_text
                        FROM allocated
                        WHERE char_cap IS NOT NULL AND char_cap > 0
                    )
                    SELECT
                        projected.ordinal,
                        projected.source_text,
                        {true_literal} AS conversation_exists,
                        stats.source_invalid,
                        stats.source_truncated
                    FROM projected
                    CROSS JOIN stats
                    UNION ALL
                    SELECT
                        NULL AS ordinal,
                        NULL AS source_text,
                        {true_literal} AS conversation_exists,
                        stats.source_invalid,
                        stats.source_truncated
                    FROM live_conversation
                    CROSS JOIN stats
                    WHERE NOT EXISTS (SELECT 1 FROM projected)
                    ORDER BY ordinal ASC
                """,  # nosec B608 - interpolated fragments are fixed by backend type.
                (
                    char_budget,
                    conversation_id,
                    *owner_params,
                    row_limit,
                ),
                log_params=False,
                log_errors=False,
            )
            records = [dict(row) for row in cursor.fetchall()]
            if not records:
                return {
                    "rows": [],
                    "conversation_exists": False,
                    "invalid": False,
                    "truncated": False,
                }

            first = records[0]
            invalid = first.get("source_invalid")
            truncated = first.get("source_truncated")
            if not isinstance(invalid, (bool, int)) or invalid not in (0, 1):
                raise CharactersRAGDBError("Invalid source-message validation marker.")
            if not isinstance(truncated, (bool, int)) or truncated not in (0, 1):
                raise CharactersRAGDBError("Invalid source-message truncation marker.")

            projected_rows: list[dict[str, str]] = []
            for record in records:
                if record.get("source_invalid") != invalid or record.get("source_truncated") != truncated:
                    raise CharactersRAGDBError("Inconsistent source-message projection markers.")
                source_text = record.get("source_text")
                if source_text is None:
                    continue
                if not isinstance(source_text, str) or not source_text:
                    raise CharactersRAGDBError("Invalid bounded source-message projection.")
                projected_rows.append({"source_text": source_text})

            return {
                "rows": projected_rows,
                "conversation_exists": True,
                "invalid": bool(invalid),
                "truncated": bool(truncated),
            }
        except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
            failure_type = type(exc).__name__

        logger.error(
            "Database error fetching bounded source messages ({})",
            failure_type,
        )
        raise CharactersRAGDBError("Source-message projection failed.")

    def get_message_conversation_id(self, message_id: str) -> str | None:
        """Return the conversation_id for a message if it exists and is not deleted."""
        query = "SELECT conversation_id FROM messages WHERE id = ? AND deleted = FALSE"
        try:
            cursor = self._db.execute_query(query, (message_id,))
            row = cursor.fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                return row.get("conversation_id")
            return row[0] if row else None
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching conversation_id for message {message_id}: {e}")
            raise

    def get_message_by_id(self, message_id: str, include_deleted: bool = False, *, strict_images: bool = False) -> dict[str, Any] | None:
        """
        Retrieves a specific message by its UUID.

        Only non-deleted messages are returned. Includes all fields, such as
        `image_data` (BLOB) and `image_mime_type` if present.

        Args:
            message_id: The string UUID of the message.
            include_deleted: Include soft-deleted messages and conversations.
            strict_images: Propagate attachment read failures and reject missing
                attachment positions or bytes instead of returning partial data.

        Returns:
            A dictionary with message data if found and not deleted, else None.

        Raises:
            CharactersRAGDBError: For database errors.
        """
        deleted_clause = "" if include_deleted else "AND m.deleted = FALSE AND c.deleted = FALSE"
        query = (
            "SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content, "
            "m.image_data, m.image_mime_type, m.timestamp, m.ranking, m.last_modified, "
            "m.version, m.client_id, m.deleted "
            "FROM messages m "
            "JOIN conversations c ON c.id = m.conversation_id "
            f"WHERE m.id = ? {deleted_clause}"  # nosec B608
        )
        try:
            cursor = self._db.execute_query(query, (message_id,))
            row = cursor.fetchone()
            if not row:
                return None
            if isinstance(row, dict):
                record = dict(row)
            else:
                columns = [col[0] for col in cursor.description] if cursor.description else []
                record = {columns[idx]: row[idx] for idx in range(len(columns))}
            img_blob = record.get("image_data")
            if isinstance(img_blob, memoryview):
                record["image_data"] = img_blob.tobytes()
            record["images"] = self.get_message_images(message_id, **({"strict": True} if strict_images else {}))
            return record  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching message ID {message_id}: {e}")
            raise

    def append_message_from_sync(
        self,
        *,
        stable_message_id: str,
        conversation_id: str,
        sender: str,
        content: str | None,
        timestamp: str | None,
        sync_client_id: str,
        object_revision: int,
        payload_hash: str,
        parent_message_id: str | None = None,
        ranking: int | None = None,
        projection_message_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a chat message from Sync v2 with stable-ID dedupe and divergence preservation."""

        normalized_stable_id = str(stable_message_id).strip()
        if not normalized_stable_id:
            raise InputError("stable_message_id cannot be empty.")  # noqa: TRY003
        if object_revision < 1:
            raise InputError("object_revision must be greater than zero.")  # noqa: TRY003

        projection_id = projection_message_id or normalized_stable_id
        forced_conflict = projection_message_id is not None
        existing_versions = self.get_messages_by_sync_stable_id(normalized_stable_id, include_deleted=True)
        projection_id_blocked = False
        for version in existing_versions:
            sync_meta = ((version.get("metadata") or {}).get("extra") or {}).get("sync_v2") or {}
            if sync_meta.get("payload_hash") == payload_hash:
                return {
                    "message_id": version["id"],
                    "stable_message_id": normalized_stable_id,
                    "created": False,
                    "idempotent": True,
                    "conflict": bool(sync_meta.get("projection_conflict")),
                }
        for version in existing_versions:
            if version["id"] == projection_id:
                sync_meta = ((version.get("metadata") or {}).get("extra") or {}).get("sync_v2", {})
                sync_payload_hash = sync_meta.get("payload_hash")
                if sync_payload_hash and sync_payload_hash != payload_hash:
                    projection_id_blocked = True
                    continue
                if (
                    not sync_payload_hash
                    and not self._sync_projection_matches(
                        version,
                        conversation_id=conversation_id,
                        parent_message_id=parent_message_id,
                        sender=sender,
                        content=content,
                        timestamp=timestamp,
                        ranking=ranking,
                        sync_client_id=sync_client_id,
                    )
                ):
                    projection_id_blocked = True
                    continue
                projection_conflict = forced_conflict or bool(
                    sync_meta.get("projection_conflict")
                )
                self._set_sync_v2_message_metadata_or_raise(
                    message_id=projection_id,
                    stable_message_id=normalized_stable_id,
                    payload_hash=payload_hash,
                    object_revision=object_revision,
                    projection_conflict=projection_conflict,
                )
                return {
                    "message_id": projection_id,
                    "stable_message_id": normalized_stable_id,
                    "created": False,
                    "idempotent": True,
                    "conflict": projection_conflict,
                }

        is_conflict = forced_conflict or bool(existing_versions)
        if is_conflict and (projection_message_id is None or projection_id_blocked):
            projection_id = self._available_sync_conflict_projection_id(
                normalized_stable_id,
                object_revision,
                existing_versions,
            )

        with self._db.transaction() as conn:
            message_id = self.add_message(
                {
                    "id": projection_id,
                    "conversation_id": conversation_id,
                    "parent_message_id": parent_message_id,
                    "sender": sender,
                    "content": content or "",
                    "timestamp": timestamp,
                    "ranking": ranking,
                    "client_id": sync_client_id,
                },
                conn=conn,
            )
            if message_id is None:
                raise CharactersRAGDBError("Failed to append Sync v2 message projection.")  # noqa: TRY003
            if object_revision != 1:
                conn.execute(
                    "UPDATE messages SET version = ?, client_id = ? WHERE id = ?",
                    (object_revision, sync_client_id, message_id),
                )
            self._set_sync_v2_message_metadata_or_raise(
                message_id=message_id,
                stable_message_id=normalized_stable_id,
                payload_hash=payload_hash,
                object_revision=object_revision,
                projection_conflict=is_conflict,
                conn=conn,
            )
        return {
            "message_id": message_id,
            "stable_message_id": normalized_stable_id,
            "created": True,
            "idempotent": False,
            "conflict": is_conflict,
        }

    def tombstone_message_from_sync(
        self,
        *,
        stable_message_id: str,
        sync_client_id: str,
        object_revision: int,
        object_hash: str,
    ) -> bool:
        """Soft-delete all projections for a stable message from an accepted Sync v2 tombstone."""

        normalized_stable_id = str(stable_message_id).strip()
        if not normalized_stable_id:
            raise InputError("stable_message_id cannot be empty.")  # noqa: TRY003
        if object_revision < 1:
            raise InputError("object_revision must be greater than zero.")  # noqa: TRY003

        existing_versions = self.get_messages_by_sync_stable_id(normalized_stable_id, include_deleted=True)
        matched_versions = [
            version
            for version in existing_versions
            if (((version.get("metadata") or {}).get("extra") or {}).get("sync_v2") or {}).get("payload_hash")
            == object_hash
        ]
        if not matched_versions:
            matched_versions = [
                version
                for version in existing_versions
                if version["id"] == normalized_stable_id
                and not (((version.get("metadata") or {}).get("extra") or {}).get("sync_v2") or {}).get(
                    "payload_hash"
                )
            ]
        if not matched_versions:
            raise ConflictError(  # noqa: TRY003
                "Message projection matching Sync v2 tombstone base hash was not found.",
                entity="messages",
                entity_id=normalized_stable_id,
            )

        matched_ids = {str(version["id"]) for version in matched_versions}
        now = self._db._get_current_utc_timestamp_iso()
        affected_conversation_ids = {
            str(version["conversation_id"])
            for version in existing_versions
            if not bool(version["deleted"])
        }
        with self._db.transaction() as conn:
            for version in existing_versions:
                if bool(version["deleted"]):
                    continue
                conn.execute(
                    """
                    UPDATE messages
                       SET deleted = ?,
                           last_modified = ?,
                           version = ?,
                           client_id = ?
                     WHERE id = ?
                    """,
                    (True, now, object_revision, sync_client_id, version["id"]),
                )
            for version in existing_versions:
                sync_meta = dict(((version.get("metadata") or {}).get("extra") or {}).get("sync_v2") or {})
                sync_meta.setdefault("stable_message_id", normalized_stable_id)
                sync_meta.setdefault("payload_hash", object_hash if str(version["id"]) in matched_ids else "")
                sync_meta.update(
                    {
                        "object_revision": object_revision,
                        "tombstoned": True,
                    }
                )
                persisted = self.set_message_metadata_extra(
                    version["id"],
                    {"sync_v2": sync_meta},
                    merge=True,
                    conn=conn,
                    _advance_history=False,
                )
                if not persisted:
                    raise CharactersRAGDBError(  # noqa: TRY003
                        f"Failed to persist Sync v2 tombstone metadata for message {version['id']}."
                    )
            for affected_conversation_id in sorted(affected_conversation_ids):
                self._advance_history_version(conn, affected_conversation_id)
        return True

    def get_messages_by_sync_stable_id(
        self,
        stable_message_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch message projections associated with a Sync v2 stable message ID."""

        self._db._ensure_message_metadata_table()
        deleted_clause = "" if include_deleted else "AND m.deleted = FALSE AND c.deleted = FALSE"
        query = (
            "SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content, "
            "m.image_data, m.image_mime_type, m.timestamp, m.ranking, m.last_modified, "
            "m.version, m.client_id, m.deleted, mm.tool_calls_json, mm.extra_json "
            "FROM messages m "
            "LEFT JOIN message_metadata mm ON mm.message_id = m.id "
            "JOIN conversations c ON c.id = m.conversation_id "
            f"WHERE 1 = 1 {deleted_clause} "  # nosec B608
            "ORDER BY m.timestamp ASC, m.last_modified ASC, m.id ASC"
        )
        cursor = self._db.execute_query(query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        results: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
            extra = json.loads(record.pop("extra_json") or "{}")
            if not isinstance(extra, dict):
                extra = {}
            tool_calls = json.loads(record.pop("tool_calls_json") or "null")
            metadata = {"tool_calls": tool_calls, "extra": extra}
            sync_meta = extra.get("sync_v2") if isinstance(extra, dict) else None
            fallback_match = record["id"] == stable_message_id or str(record["id"]).startswith(
                f"{stable_message_id}__sync_conflict__"
            )
            metadata_match = isinstance(sync_meta, dict) and sync_meta.get("stable_message_id") == stable_message_id
            if not metadata_match and not fallback_match:
                continue
            if not isinstance(sync_meta, dict):
                extra["sync_v2"] = {"stable_message_id": stable_message_id}
            img_blob = record.get("image_data")
            if isinstance(img_blob, memoryview):
                record["image_data"] = img_blob.tobytes()
            record["images"] = self.get_message_images(record["id"])
            record["metadata"] = metadata
            results.append(record)
        return results

    def _set_sync_v2_message_metadata_or_raise(
        self,
        *,
        message_id: str,
        stable_message_id: str,
        payload_hash: str,
        object_revision: int,
        projection_conflict: bool,
        conn: Any | None = None,
    ) -> None:
        persisted = self.set_message_metadata_extra(
            message_id,
            {
                "sync_v2": {
                    "stable_message_id": stable_message_id,
                    "payload_hash": payload_hash,
                    "object_revision": object_revision,
                    "projection_conflict": projection_conflict,
                }
            },
            merge=True,
            conn=conn,
            _advance_history=False,
        )
        if not persisted:
            raise CharactersRAGDBError(  # noqa: TRY003
                f"Failed to persist Sync v2 metadata for message {message_id}."
            )

    @staticmethod
    def _sync_projection_matches(
        version: dict[str, Any],
        *,
        conversation_id: str,
        parent_message_id: str | None,
        sender: str,
        content: str | None,
        timestamp: str | None,
        ranking: int | None,
        sync_client_id: str,
    ) -> bool:
        """Return whether a metadata-less row matches the incoming Sync v2 projection."""

        if version.get("conversation_id") != conversation_id:
            return False
        if version.get("parent_message_id") != parent_message_id:
            return False
        if version.get("sender") != sender:
            return False
        if version.get("content") != (content or ""):
            return False
        if timestamp is not None and version.get("timestamp") != timestamp:
            return False
        if version.get("ranking") != ranking:
            return False
        return version.get("client_id") == sync_client_id

    @staticmethod
    def _available_sync_conflict_projection_id(
        stable_message_id: str,
        object_revision: int,
        existing_versions: list[dict[str, Any]],
    ) -> str:
        existing_ids = {str(version["id"]) for version in existing_versions}
        base_projection_id = f"{stable_message_id}__sync_conflict__{object_revision}"
        projection_id = base_projection_id
        suffix = 2
        while projection_id in existing_ids:
            projection_id = f"{base_projection_id}_{suffix}"
            suffix += 1
        return projection_id

    # ------------------------------------------------------------------
    # Message listing / querying
    # ------------------------------------------------------------------

    def get_messages_for_conversation(self, conversation_id: str, limit: int = 100, offset: int = 0,
                                      order_by_timestamp: str = "ASC", include_deleted: bool = False, *,
                                      strict_images: bool = False,
                                      image_byte_limit: int = MAX_CHAT_ATTACHMENT_READ_BYTES) -> list[dict[str, Any]]:
        """
        Lists messages for a specific conversation.
        Returns non-deleted messages, ordered by `timestamp` according to `order_by_timestamp`.
        Crucially, it also ensures the parent conversation is not soft-deleted.
        """
        if order_by_timestamp.upper() not in ["ASC", "DESC"]:
            raise InputError("order_by_timestamp must be 'ASC' or 'DESC'.")  # noqa: TRY003
        order_direction = order_by_timestamp.upper()

        # The new query joins with conversations to check its 'deleted' status.
        delete_clause = "" if include_deleted else "AND m.deleted = FALSE"

        query = """
            SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content,
                   m.image_data, m.image_mime_type, m.timestamp, m.ranking,
                   m.last_modified, m.version, m.client_id, m.deleted
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.conversation_id = ?
              {delete_clause}
              AND c.deleted = FALSE
            ORDER BY m.timestamp {order_direction}, m.last_modified {order_direction}, m.id {order_direction}
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        try:
            if strict_images:
                # A single statement gives SQLite and PostgreSQL one snapshot.
                # Over-budget pages expose their size but no attachment blobs.
                strict_query = """
                    WITH page AS (
                        SELECT m.id, m.conversation_id, m.parent_message_id, m.sender, m.content,
                               m.timestamp, m.ranking, m.last_modified, m.version, m.client_id, m.deleted,
                               LENGTH(m.image_data) AS primary_bytes
                        FROM messages m JOIN conversations c ON m.conversation_id = c.id
                        WHERE m.conversation_id = ? {delete_clause} AND c.deleted = FALSE
                        ORDER BY m.timestamp {order_direction}, m.last_modified {order_direction}, m.id {order_direction}
                        LIMIT ? OFFSET ?
                    ), sizes AS (
                        SELECT COALESCE(SUM(CASE WHEN EXISTS (
                            SELECT 1 FROM message_images mi WHERE mi.message_id = page.id
                        ) THEN (SELECT COALESCE(SUM(LENGTH(mi.image_data)), 0)
                                FROM message_images mi WHERE mi.message_id = page.id)
                        ELSE COALESCE(page.primary_bytes, 0) END), 0) AS image_bytes FROM page
                    ), bounds AS (SELECT ? AS max_bytes)
                    SELECT page.*,
                           CASE WHEN sizes.image_bytes <= bounds.max_bytes AND mi.position IS NULL
                                THEN m.image_data ELSE NULL END AS image_data,
                           m.image_mime_type,
                           sizes.image_bytes AS page_image_bytes,
                           mi.position AS image_position,
                           CASE WHEN sizes.image_bytes <= bounds.max_bytes
                                THEN mi.image_data ELSE NULL END AS ordered_image_data,
                           mi.image_mime_type AS ordered_image_mime_type
                    FROM page JOIN messages m ON m.id = page.id CROSS JOIN sizes CROSS JOIN bounds
                    LEFT JOIN message_images mi ON mi.message_id = page.id AND sizes.image_bytes <= bounds.max_bytes
                    ORDER BY page.timestamp {order_direction}, page.last_modified {order_direction},
                             page.id {order_direction}, mi.position ASC
                """.format_map(locals())  # nosec B608
                cursor = self._db.execute_query(strict_query, (conversation_id, limit, offset, image_byte_limit))
                columns = [column[0] for column in cursor.description] if cursor.description else []
                by_id: dict[str, dict[str, Any]] = {}
                for row in cursor.fetchall():
                    record = dict(row) if isinstance(row, dict) else dict(zip(columns, row))
                    if record.pop("page_image_bytes") > image_byte_limit:
                        raise InputError("Chat attachment page exceeds the image read limit")
                    position = record.pop("image_position")
                    image_data = record.pop("ordered_image_data")
                    image_mime = record.pop("ordered_image_mime_type")
                    record.pop("primary_bytes")
                    if isinstance(record.get("image_data"), memoryview):
                        record["image_data"] = record["image_data"].tobytes()
                    message = by_id.setdefault(record["id"], {**record, "images": []})
                    if position is not None:
                        if position != len(message["images"]):
                            raise CharactersRAGDBError("Saved chat attachment positions are incomplete or invalid")
                        if isinstance(image_data, memoryview):
                            image_data = image_data.tobytes()
                        message["images"].append({"message_id": record["id"], "position": position,
                                                  "image_data": image_data, "image_mime_type": image_mime})
                        if position == 0:
                            message["image_data"] = image_data
                            message["image_mime_type"] = image_mime
                return list(by_id.values())
            cursor = self._db.execute_query(query, (conversation_id, limit, offset))
            raw_rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in raw_rows:
                record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
                image_blob = record.get("image_data")
                if isinstance(image_blob, memoryview):
                    record["image_data"] = image_blob.tobytes()
                record["images"] = self.get_message_images(record["id"])
                results.append(record)
            return results  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error(f"Database error fetching messages for conversation ID {conversation_id}: {e}")
            raise

    def count_root_messages_for_conversation(self, conversation_id: str) -> int:
        """Count root (parentless) messages for a conversation."""
        query = (
            "SELECT COUNT(1) FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND m.parent_message_id IS NULL "
            "AND m.deleted = FALSE AND c.deleted = FALSE"
        )
        try:
            cursor = self._db.execute_query(query, (conversation_id,))
            row = cursor.fetchone()
            if row is None:
                return 0
            try:
                return int(row[0])
            except _CHACHA_NONCRITICAL_EXCEPTIONS:
                return int(row.get("COUNT(1)") or row.get("count") or 0)
        except CharactersRAGDBError as e:
            logger.error("Database error counting root messages for conversation {}: {}", conversation_id, e)
            raise

    def get_root_messages_for_conversation(
        self,
        conversation_id: str,
        *,
        limit: int,
        offset: int,
        order_by_timestamp: str = "ASC",
    ) -> list[dict[str, Any]]:
        """Fetch root (parentless) messages with minimal columns for tree building."""
        if order_by_timestamp.upper() not in ["ASC", "DESC"]:
            raise InputError("order_by_timestamp must be 'ASC' or 'DESC'.")  # noqa: TRY003
        order_direction = order_by_timestamp.upper()
        query = """
            SELECT m.id, m.parent_message_id, m.sender, m.content, m.timestamp
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.conversation_id = ?
              AND m.parent_message_id IS NULL
              AND m.deleted = FALSE
              AND c.deleted = FALSE
            ORDER BY m.timestamp {order_direction}, m.last_modified {order_direction}, m.id {order_direction}
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        try:
            cursor = self._db.execute_query(query, (conversation_id, limit, offset))
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
                results.append(record)
            return results  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error("Database error fetching root messages for conversation {}: {}", conversation_id, e)
            raise

    def get_messages_for_conversation_by_parent_ids(
        self,
        conversation_id: str,
        parent_ids: list[str],
        *,
        order_by_timestamp: str = "ASC",
    ) -> list[dict[str, Any]]:
        """Fetch child messages for the given parent IDs with minimal columns."""
        if not parent_ids:
            return []
        if order_by_timestamp.upper() not in ["ASC", "DESC"]:
            raise InputError("order_by_timestamp must be 'ASC' or 'DESC'.")  # noqa: TRY003
        order_direction = order_by_timestamp.upper()
        placeholders = ",".join(["?"] * len(parent_ids))
        query = """
            SELECT m.id, m.parent_message_id, m.sender, m.content, m.timestamp
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.conversation_id = ?
              AND m.parent_message_id IN ({placeholders})
              AND m.deleted = FALSE
              AND c.deleted = FALSE
            ORDER BY m.timestamp {order_direction}, m.last_modified {order_direction}, m.id {order_direction}
        """.format_map(locals())  # nosec B608
        params = [conversation_id, *parent_ids]
        try:
            cursor = self._db.execute_query(query, tuple(params))
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description] if cursor.description else []
            results: list[dict[str, Any]] = []
            for row in rows:
                record = dict(row) if isinstance(row, dict) else {columns[idx]: row[idx] for idx in range(len(columns))}
                results.append(record)
            return results  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error(
                'Database error fetching child messages for conversation {}: {}',
                conversation_id,
                e,
            )
            raise

    def has_system_message_for_conversation(
        self,
        conversation_id: str,
        include_deleted: bool = False,
    ) -> bool:
        """Check whether a conversation has at least one system message."""
        if include_deleted:
            query = """
                SELECT 1
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.conversation_id = ?
                  AND lower(m.sender) = 'system'
                  AND c.deleted = ?
                LIMIT 1
            """
            params = (conversation_id, False)
        else:
            query = """
                SELECT 1
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.conversation_id = ?
                  AND lower(m.sender) = 'system'
                  AND m.deleted = ?
                  AND c.deleted = ?
                LIMIT 1
            """
            params = (conversation_id, False, False)
        try:
            cursor = self._db.execute_query(query, params)
            return cursor.fetchone() is not None
        except CharactersRAGDBError as e:
            logger.error(
                'Database error checking system messages for conversation ID {}: {}',
                conversation_id,
                e,
            )
            raise

    # ------------------------------------------------------------------
    # Message update
    # ------------------------------------------------------------------

    def update_message(
        self,
        message_id: str,
        update_data: dict[str, Any],
        expected_version: int,
        *,
        conn: Any | None = None,
    ) -> bool | None:
        """
        Updates an existing message using optimistic locking.

        Succeeds if `expected_version` matches the current database version.
        `version` is incremented, `last_modified` updated, and `client_id` set.
        Updatable fields from `update_data`: 'content', 'ranking', 'parent_message_id'.
        Image data can also be updated: 'image_data' and 'image_mime_type'.
        If 'image_data' is set to `None` in `update_data`, both 'image_data' and
        'image_mime_type' columns will be set to NULL in the database.
        Other fields in `update_data` are ignored. `update_data` must not be empty.

        FTS updates (`messages_fts` for content changes) and `sync_log` entries
        are handled by SQL triggers.

        Args:
            message_id: The UUID of the message to update.
            update_data: Dictionary with fields to update. Must not be empty.
                         If 'image_data' is updated, 'image_mime_type' should also be
                         provided, unless 'image_data' is set to None.
            expected_version: The client's expected version of the record.

        Returns:
            True if the update was successful.

        Raises:
            InputError: If `update_data` is empty.
            ConflictError: If the message is not found, is soft-deleted, or if `expected_version`
                           does not match the current database version.
            CharactersRAGDBError: For database integrity errors (e.g., invalid `parent_message_id`)
                                  or other database issues.
        """
        if not update_data:
            raise InputError("No data provided for message update.")  # noqa: TRY003

        now = self._db._get_current_utc_timestamp_iso()
        fields_to_update_sql = []
        params_for_set_clause = []

        allowed_to_update = ['content', 'ranking', 'parent_message_id', 'image_data', 'image_mime_type']

        # Special handling for clearing image
        if 'image_data' in update_data and update_data['image_data'] is None:
            fields_to_update_sql.append("image_data = NULL")
            fields_to_update_sql.append("image_mime_type = NULL")
            # Remove these keys from update_data to avoid processing them again
            # in the loop if they were explicitly set to None
            # This isn't strictly necessary with current loop logic but good for clarity
            update_data.pop('image_data', None)
            update_data.pop('image_mime_type', None)

        for key, value in update_data.items():
            if key in allowed_to_update:
                fields_to_update_sql.append(f"{key} = ?")
                params_for_set_clause.append(value)
            elif key not in ['id', 'conversation_id', 'sender', 'timestamp', 'last_modified', 'version', 'client_id', 'deleted']:
                logger.warning(
                    f"Attempted to update immutable or unknown field '{key}' in message ID {message_id}, skipping.")

        if not fields_to_update_sql:  # If only image was cleared, this list might be empty now if no other fields
            logger.info(f"No updatable content fields provided for message ID {message_id}, but metadata will be updated if version matches.")
            # Proceed to metadata update; SQL query will be constructed accordingly

        next_version_val = expected_version + 1

        current_fields_to_update_sql = list(fields_to_update_sql)
        current_params_for_set_clause = list(params_for_set_clause)

        current_fields_to_update_sql.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        current_params_for_set_clause.extend([now, next_version_val, self._db.client_id])

        where_values = [message_id, expected_version]
        final_params_for_execute = tuple(current_params_for_set_clause + where_values)

        query = f"UPDATE messages SET {', '.join(current_fields_to_update_sql)} WHERE id = ? AND version = ? AND deleted = FALSE"  # nosec B608

        try:
            transaction = nullcontext(conn) if conn is not None else self._db.transaction()
            with transaction as transaction_conn:
                current_db_version = self._db._get_current_db_version(
                    transaction_conn,
                    "messages",
                    "id",
                    message_id,
                )

                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Message ID {message_id} update failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="messages", entity_id=message_id
                    )

                conversation_row = transaction_conn.execute(
                    "SELECT conversation_id FROM messages WHERE id = ?",
                    (message_id,),
                ).fetchone()
                cursor = transaction_conn.execute(query, final_params_for_execute)

                if cursor.rowcount == 0:
                    check_again_cursor = transaction_conn.execute(
                        "SELECT version, deleted FROM messages WHERE id = ?",
                        (message_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    msg = f"Update for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = f"Message ID {message_id} disappeared."
                    elif final_state['deleted']:
                        msg = f"Message ID {message_id} was soft-deleted concurrently."
                    elif final_state['version'] != expected_version:
                        msg = f"Message ID {message_id} version changed to {final_state['version']} concurrently."
                    raise ConflictError(msg, entity="messages", entity_id=message_id)  # noqa: TRY301

                self._advance_history_version(
                    transaction_conn,
                    str(self._row_value(conversation_row, "conversation_id")),
                )
                logger.info(
                    f"Updated message ID {message_id} from version {expected_version} to version {next_version_val}. Fields updated: {fields_to_update_sql if fields_to_update_sql else 'None'}")
                return True
        except sqlite3.IntegrityError as e:
            logger.error(f"SQLite integrity error updating message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise CharactersRAGDBError(f"Database integrity error updating message: {e}") from e  # noqa: TRY003
        except ConflictError:
            raise
        except InputError:  # Should not be raised from here directly, but for completeness
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error updating message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Soft delete
    # ------------------------------------------------------------------

    def soft_delete_message(
        self,
        message_id: str,
        expected_version: int,
        *,
        conn: Any | None = None,
    ) -> bool | None:
        """
        Soft-deletes a message using optimistic locking.

        Sets `deleted` to 1, updates `last_modified`, increments `version`, and sets `client_id`.
        Succeeds if `expected_version` matches the current DB version and the record is active.
        If already soft-deleted, returns True (idempotent).

        FTS updates (removal from `messages_fts`) and `sync_log` entries are handled by SQL triggers.

        Args:
            message_id: The UUID of the message to soft-delete.
            expected_version: The client's expected version of the record.

        Returns:
            True if the soft-delete was successful or if the message was already soft-deleted.

        Raises:
            ConflictError: If not found (and not already deleted), or if active with a version mismatch.
            CharactersRAGDBError: For other database errors.
        """
        now = self._db._get_current_utc_timestamp_iso()
        next_version_val = expected_version + 1

        query = "UPDATE messages SET deleted = TRUE, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = FALSE"
        params = (now, next_version_val, self._db.client_id, message_id, expected_version)

        try:
            transaction = nullcontext(conn) if conn is not None else self._db.transaction()
            with transaction as transaction_conn:
                try:
                    current_db_version = self._db._get_current_db_version(
                        transaction_conn,
                        "messages",
                        "id",
                        message_id,
                    )
                except ConflictError:
                    check_status_cursor = transaction_conn.execute(
                        "SELECT deleted, version FROM messages WHERE id = ?",
                        (message_id,),
                    )
                    record_status = check_status_cursor.fetchone()
                    if record_status and record_status['deleted']:
                        logger.info(f"Message ID {message_id} already soft-deleted. Success (idempotent).")
                        return True
                    raise  # Re-raise if not found or other conflict

                if current_db_version != expected_version:
                    raise ConflictError(  # noqa: TRY003, TRY301
                        f"Soft delete for Message ID {message_id} failed: version mismatch (db has {current_db_version}, client expected {expected_version}).",
                        entity="messages", entity_id=message_id
                    )

                conversation_row = transaction_conn.execute(
                    "SELECT conversation_id FROM messages WHERE id = ?",
                    (message_id,),
                ).fetchone()
                cursor = transaction_conn.execute(query, params)

                if cursor.rowcount == 0:
                    check_again_cursor = transaction_conn.execute(
                        "SELECT version, deleted FROM messages WHERE id = ?",
                        (message_id,),
                    )
                    final_state = check_again_cursor.fetchone()
                    msg = f"Soft delete for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    if not final_state:
                        msg = f"Message ID {message_id} disappeared."
                    elif final_state['deleted']:
                        logger.info(f"Message ID {message_id} was soft-deleted concurrently. Success.")
                        return True
                    elif final_state['version'] != expected_version:
                        msg = f"Message ID {message_id} version changed to {final_state['version']} concurrently."
                    else:
                        msg = f"Soft delete for message ID {message_id} (expected v{expected_version}) affected 0 rows."
                    raise ConflictError(msg, entity="messages", entity_id=message_id)  # noqa: TRY301

                self._advance_history_version(
                    transaction_conn,
                    str(self._row_value(conversation_row, "conversation_id")),
                )
                logger.info(
                    f"Soft-deleted message ID {message_id} (was v{expected_version}), new version {next_version_val}.")
                return True
        except ConflictError:
            raise
        except CharactersRAGDBError as e:
            logger.error(f"Database error soft-deleting message ID {message_id} (expected v{expected_version}): {e}",
                         exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def search_messages_by_content(
        self,
        content_query: str,
        conversation_id: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Searches messages by content using FTS.

        Matches against the 'content' field in `messages_fts`.
        Optionally filters by `conversation_id`. Returns non-deleted messages,
        ordered by relevance (rank).

        Args:
            content_query: The search term for content. Supports FTS query syntax.
            conversation_id: Optional conversation UUID to filter results.
            limit: Maximum number of results. Defaults to 10.
            offset: Number of matching rows to skip. Defaults to 0.

        Returns:
            A list of matching message dictionaries. Can be empty.

        Raises:
            CharactersRAGDBError: For database search errors.
        """
        if self._db.backend_type == BackendType.POSTGRESQL:
            tsquery = FTSQueryTranslator.normalize_query(content_query, 'postgresql')
            if not tsquery:
                logger.debug("Message content query normalized to empty tsquery for input '{}'", content_query)
                return []

            base_query = [
                "SELECT m.*, c.title AS conversation_title, ts_rank(m.messages_fts_tsv, to_tsquery('english', ?)) AS rank",
                "FROM messages m",
                "JOIN conversations c ON c.id = m.conversation_id",
                "WHERE m.deleted = FALSE",
                "AND c.deleted = FALSE AND c.client_id = ?",
                "AND m.messages_fts_tsv @@ to_tsquery('english', ?)",
            ]
            params_list: list[Any] = [tsquery, self._db.client_id, tsquery]

            if conversation_id:
                base_query.append("AND m.conversation_id = ?")
                params_list.append(conversation_id)

            base_query.append("ORDER BY rank DESC, m.last_modified DESC")
            base_query.append("LIMIT ? OFFSET ?")
            params_list.extend([limit, offset])

            try:
                cursor = self._db.execute_query("\n".join(base_query), tuple(params_list))
                return [dict(row) for row in cursor.fetchall()]
            except CharactersRAGDBError as exc:
                logger.error("PostgreSQL FTS search failed for messages term '{}': {}", content_query, exc)
                raise

        safe_literal = content_query.replace('"', '""')
        safe_search_term = f'"{safe_literal}"' if '"' in content_query else safe_literal
        if not safe_search_term.strip():
            return []
        base_query = """
                     SELECT m.*, c.title AS conversation_title
                     FROM messages_fts, messages m
                     JOIN conversations c ON c.id = m.conversation_id
                     WHERE messages_fts.rowid = m.rowid \
                       AND messages_fts MATCH ? \
                       AND m.deleted = FALSE \
                       AND c.deleted = FALSE \
                     """
        params_list = [safe_search_term]
        if conversation_id:
            base_query += " AND m.conversation_id = ?"
            params_list.append(conversation_id)

        base_query += " ORDER BY bm25(messages_fts) ASC, m.last_modified DESC LIMIT ? OFFSET ?"
        params_list.extend([limit, offset])

        try:
            try:
                # Retain valid FTS syntax; normalize only prose parse failures.
                rows = self._db.get_connection().execute(base_query, tuple(params_list)).fetchall()
            except sqlite3.OperationalError as exc:
                if not any(marker in str(exc).lower() for marker in ("fts5: syntax error", "no such column:")):
                    raise
                normalized = FTSQueryTranslator.normalize_query(content_query, "sqlite")
                if normalized == safe_search_term:
                    raise
                params_list[0] = normalized
                rows = self._db.get_connection().execute(base_query, tuple(params_list)).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Message search failed: {exc}") from exc  # noqa: TRY003
        except CharactersRAGDBError as e:
            logger.error("Error searching messages for content '{}': {}", safe_search_term, e)
            raise

    # ------------------------------------------------------------------
    # Message metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_json_value(value: Any) -> Any:
        """Decode stored JSON text while accepting backend-decoded JSON values."""
        return json.loads(value) if isinstance(value, str) else value

    def _add_message_metadata_with_conn(
        self,
        message_id: str,
        tool_calls: Any | None,
        extra: Any | None,
        conn: Any,
        *,
        advance_history: bool = True,
    ) -> bool:
        cursor = conn.execute(
            "SELECT m.conversation_id, mm.message_id AS metadata_message_id, "
            "mm.tool_calls_json, mm.extra_json "
            "FROM messages m LEFT JOIN message_metadata mm ON mm.message_id = m.id "
            "WHERE m.id = ?",
            (message_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return False

        metadata_message_id = self._row_value(row, "metadata_message_id", 1)
        if metadata_message_id is not None:
            stored_tool_calls = self._metadata_json_value(self._row_value(row, "tool_calls_json", 2))
            stored_extra = self._metadata_json_value(self._row_value(row, "extra_json", 3))
            if stored_tool_calls == tool_calls and stored_extra == extra:
                return True

        conn.execute(
            "INSERT INTO message_metadata(message_id, tool_calls_json, extra_json, last_modified) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(message_id) DO UPDATE SET tool_calls_json=excluded.tool_calls_json, "
            "extra_json=excluded.extra_json, last_modified=CURRENT_TIMESTAMP",
            (
                message_id,
                json.dumps(tool_calls) if tool_calls is not None else None,
                json.dumps(extra) if extra is not None else None,
            ),
        )
        if advance_history:
            conversation_id = str(self._row_value(row, "conversation_id"))
            self._advance_history_version(conn, conversation_id)
        return True

    def add_message_metadata(
        self,
        message_id: str,
        tool_calls: Any | None = None,
        extra: Any | None = None,
        *,
        conn: Any | None = None,
    ) -> bool:
        """Upsert per-message metadata such as tool calls.

        Stores JSON-serialized metadata in an auxiliary table `message_metadata`.
        The table is created on-demand if missing.
        """
        try:
            self._db._ensure_message_metadata_table()
            transaction = nullcontext(conn) if conn is not None else self._db.transaction()
            with transaction as transaction_conn:
                return self._add_message_metadata_with_conn(
                    message_id,
                    tool_calls,
                    extra,
                    transaction_conn,
                )
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            if conn is not None:
                raise
            logger.warning(f"add_message_metadata failed for message {message_id}: {e}")
            return False

    def get_message_metadata(
        self,
        message_id: str,
        *,
        conn: Any | None = None,
    ) -> dict[str, Any] | None:
        """Fetch metadata for a message if present."""
        try:
            self._db._ensure_message_metadata_table()
            if conn is not None:
                cursor = conn.execute(
                    "SELECT tool_calls_json, extra_json, last_modified "
                    "FROM message_metadata WHERE message_id = ?",
                    (message_id,),
                )
            else:
                cursor = self._db.execute_query(
                    "SELECT tool_calls_json, extra_json, last_modified "
                    "FROM message_metadata WHERE message_id = ?",
                    (message_id,),
                )
            row = cursor.fetchone()
            if not row:
                return None
            tc = self._row_value(row, "tool_calls_json")
            ex = self._row_value(row, "extra_json", 1)
            lm = self._row_value(row, "last_modified", 2)
            return {
                "tool_calls": self._metadata_json_value(tc) if tc is not None else None,
                "extra": self._metadata_json_value(ex) if ex is not None else None,
                "last_modified": lm,
            }
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            return None

    def get_message_metadata_map(self, message_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch metadata for multiple messages in a single query."""
        if not message_ids:
            return {}

        try:
            self._db._ensure_message_metadata_table()
            if self._db.backend_type == BackendType.SQLITE:
                placeholders = ",".join(["?"] * len(message_ids))
                query = (
                    "SELECT message_id, tool_calls_json, extra_json, last_modified "
                    f"FROM message_metadata WHERE message_id IN ({placeholders})"  # nosec B608
                )
                cursor = self._db.execute_query(query, tuple(message_ids))
                rows = cursor.fetchall()
            else:
                placeholders = ",".join(["%s"] * len(message_ids))
                query = (
                    "SELECT message_id, tool_calls_json, extra_json, last_modified "
                    f"FROM message_metadata WHERE message_id IN ({placeholders})"  # nosec B608
                )
                result = self._db.backend.execute(query, tuple(message_ids))
                rows = result.fetchall()

            metadata_by_message_id: dict[str, dict[str, Any]] = {}
            for row in rows:
                try:
                    message_id = str(row["message_id"])
                    tc = row["tool_calls_json"]
                    ex = row["extra_json"]
                    lm = row["last_modified"]
                except _CHACHA_NONCRITICAL_EXCEPTIONS:
                    message_id = str(row[0])
                    tc = row[1]
                    ex = row[2]
                    lm = row[3]
                metadata_by_message_id[message_id] = {
                    "tool_calls": json.loads(tc) if tc else None,
                    "extra": json.loads(ex) if ex else None,
                    "last_modified": lm,
                }
            return metadata_by_message_id
        except _CHACHA_NONCRITICAL_EXCEPTIONS:
            return {}

    def _get_message_metadata_for_merge(
        self,
        message_id: str,
        conn: Any,
    ) -> dict[str, Any] | None:
        """Read metadata under the row lock used by PostgreSQL merge writers."""
        if self._db.backend_type == BackendType.POSTGRESQL:
            conn.execute(
                "INSERT INTO message_metadata(message_id, last_modified) "
                "SELECT id, CURRENT_TIMESTAMP FROM messages WHERE id = ? "
                "ON CONFLICT(message_id) DO NOTHING",
                (message_id,),
            )
            query = (
                "SELECT tool_calls_json, extra_json, last_modified "
                "FROM message_metadata WHERE message_id = ? FOR UPDATE"
            )
        else:
            query = (
                "SELECT tool_calls_json, extra_json, last_modified "
                "FROM message_metadata WHERE message_id = ?"
            )

        row = conn.execute(query, (message_id,)).fetchone()
        if row is None:
            # SQLite merge writers retain the existing create-on-first-write
            # behavior; the downstream message join still rejects missing IDs.
            return {} if self._db.backend_type == BackendType.SQLITE else None
        tool_calls = self._row_value(row, "tool_calls_json")
        extra = self._row_value(row, "extra_json", 1)
        return {
            "tool_calls": self._metadata_json_value(tool_calls) if tool_calls is not None else None,
            "extra": self._metadata_json_value(extra) if extra is not None else None,
            "last_modified": self._row_value(row, "last_modified", 2),
        }

    def set_message_metadata_extra(
        self,
        message_id: str,
        extra: dict[str, Any],
        merge: bool = True,
        *,
        conn: Any | None = None,
        _advance_history: bool = True,
    ) -> bool:
        """Set or merge structured extra metadata for a message.

        Expected shape for `extra`:
          {
            "tool_results": { "<tool_call_id>": <any-json-serializable> },
            ... other namespaced keys ...,
            "version": 1
          }

        If merge=True and existing extra exists, perform a shallow merge; nested maps like
        tool_results are merged key-wise.
        """
        try:
            self._db._ensure_message_metadata_table()
            transaction = nullcontext(conn) if conn is not None else self._db.transaction()
            with transaction as transaction_conn:
                current = self._get_message_metadata_for_merge(message_id, transaction_conn)
                if current is None:
                    return False
                current_extra = current.get('extra') or {}
                if merge and isinstance(current_extra, dict) and isinstance(extra, dict):
                    merged = dict(current_extra)
                    # Merge tool_results specially
                    tr_existing = merged.get('tool_results') if isinstance(merged.get('tool_results'), dict) else {}
                    tr_incoming = extra.get('tool_results') if isinstance(extra.get('tool_results'), dict) else {}
                    if tr_existing or tr_incoming:
                        merged['tool_results'] = {**tr_existing, **tr_incoming}
                    # Merge top-level keys (favor incoming)
                    for k, v in extra.items():
                        if k == 'tool_results':
                            continue
                        merged[k] = v
                    new_extra = merged
                else:
                    new_extra = extra
                return self._add_message_metadata_with_conn(
                    message_id,
                    current.get('tool_calls'),
                    new_extra,
                    transaction_conn,
                    advance_history=_advance_history,
                )
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            if conn is not None:
                raise
            logger.warning(f"set_message_metadata_extra failed for {message_id}: {e}")
            return False

    def set_message_rag_context(
        self,
        message_id: str,
        rag_context: dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Store RAG context (citations, retrieved documents, search settings) with a message.

        This persists RAG search results and citations in message_metadata.extra_json
        under the 'rag_context' key for later retrieval and export.

        Args:
            message_id: The message ID to attach RAG context to
            rag_context: Dict containing:
                - search_query: The original search query
                - search_mode: Search mode used (fts/vector/hybrid)
                - settings_snapshot: Key RAG settings used
                - retrieved_documents: List of retrieved docs with scores/excerpts
                - generated_answer: AI-generated answer (if any)
                - citations: Citation metadata
                - claims_verified: Verification results (if any)
                - timestamp: ISO timestamp
                - feedback_id: Analytics ID
            merge: If True, merge with existing extra data; if False, replace entire extra

        Returns:
            bool: True if successful, False otherwise
        """
        return self.set_message_metadata_extra(
            message_id,
            {"rag_context": rag_context},
            merge=merge,
        )

    def get_message_rag_context(self, message_id: str) -> dict[str, Any] | None:
        """
        Retrieve RAG context stored with a message.

        Returns the rag_context dict from message_metadata.extra_json,
        or None if no RAG context is stored.
        """
        try:
            metadata = self.get_message_metadata(message_id)
            if not metadata:
                return None
            extra = metadata.get('extra')
            if not isinstance(extra, dict):
                return None
            return extra.get('rag_context')
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"get_message_rag_context failed for {message_id}: {e}")
            return None

    def get_messages_with_rag_context(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        include_rag_context: bool = True
    ) -> list[dict[str, Any]]:
        """
        Retrieve messages for a conversation with optional RAG context attached.

        This is optimized for the Knowledge QA page to load conversation history
        with full citation data.

        Args:
            conversation_id: The conversation to fetch messages from
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            include_rag_context: If True, attach rag_context to each message

        Returns:
            List of message dicts, each optionally including 'rag_context' key
        """
        try:
            messages = self.get_messages_for_conversation(
                conversation_id,
                limit=limit,
                offset=offset
            )

            if not include_rag_context:
                return messages

            # Attach RAG context to each message
            for msg in messages:
                msg_id = msg.get('id')
                if msg_id:
                    rag_context = self.get_message_rag_context(msg_id)
                    if rag_context:
                        msg['rag_context'] = rag_context

            return messages  # noqa: TRY300
        except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"get_messages_with_rag_context failed for conversation {conversation_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Count / query helpers
    # ------------------------------------------------------------------

    def count_messages_for_conversation(self, conversation_id: str, include_deleted: bool = False) -> int:
        """
        Count messages for a conversation, ensuring the parent conversation is active.

        Args:
            conversation_id: Conversation UUID
            include_deleted: If True, include soft-deleted messages

        Returns:
            Integer count of messages.

        Raises:
            CharactersRAGDBError on database failure.
        """
        base_query = (
            "SELECT COUNT(1) FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND c.deleted = FALSE"
        )
        params = [conversation_id]
        if not include_deleted:
            base_query += " AND m.deleted = FALSE"
        try:
            cursor = self._db.execute_query(base_query, tuple(params))
            row = cursor.fetchone()
            # row may be tuple or dict depending on connection row factory
            if row is None:
                return 0
            try:
                return int(row[0])
            except _CHACHA_NONCRITICAL_EXCEPTIONS:
                return int(row.get("COUNT(1)") or row.get("count") or 0)
        except CharactersRAGDBError as e:
            logger.error(f"Database error counting messages for conversation {conversation_id}: {e}")
            raise

    def count_messages_for_conversations(
        self,
        conversation_ids: list[str],
        include_deleted: bool = False,
    ) -> dict[str, int]:
        """
        Count messages for multiple conversations in a single query.

        Args:
            conversation_ids: List of conversation UUIDs.
            include_deleted: If True, include soft-deleted messages.

        Returns:
            Mapping of conversation_id -> message count.
        """
        if not conversation_ids:
            return {}
        placeholders = ",".join(["?"] * len(conversation_ids))
        base_query = (
            f"SELECT m.conversation_id, COUNT(1) as cnt "  # nosec B608
            f"FROM messages m "
            f"JOIN conversations c ON m.conversation_id = c.id "
            f"WHERE m.conversation_id IN ({placeholders}) AND c.deleted = FALSE"
        )
        if not include_deleted:
            base_query += " AND m.deleted = FALSE"
        base_query += " GROUP BY m.conversation_id"
        try:
            cursor = self._db.execute_query(base_query, tuple(conversation_ids))
            rows = cursor.fetchall()
            result: dict[str, int] = dict.fromkeys(conversation_ids, 0)
            for row in rows:
                if isinstance(row, dict):
                    conv_id = row.get("conversation_id")
                    cnt = row.get("cnt") or row.get("COUNT(1)") or 0
                else:
                    conv_id = row[0]
                    cnt = row[1]
                if conv_id is not None:
                    result[str(conv_id)] = int(cnt or 0)
            return result  # noqa: TRY300
        except CharactersRAGDBError as e:
            logger.error("Database error counting messages for conversations: {}", e)
            raise

    def get_latest_message_for_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Fetch the most recent non-deleted message for a conversation."""
        query = (
            "SELECT m.id, m.timestamp, m.content, m.sender "
            "FROM messages m JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND m.deleted = FALSE AND c.deleted = FALSE "
            "ORDER BY m.timestamp DESC, m.last_modified DESC, m.id DESC LIMIT 1"
        )
        try:
            cursor = self._db.execute_query(query, (conversation_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row) if isinstance(row, dict) else {
                "id": row[0],
                "timestamp": row[1],
                "content": row[2],
                "sender": row[3],
            }
        except CharactersRAGDBError as exc:
            logger.error("Database error fetching latest message for conversation {}: {}", conversation_id, exc)
            raise

    def count_messages_since(
        self,
        conversation_id: str,
        since_message_id: str | None,
    ) -> int:
        """Count messages after the given message_id within a conversation."""
        if not since_message_id:
            return self.count_messages_for_conversation(conversation_id)

        try:
            since_message = self.get_message_by_id(since_message_id)
        except CharactersRAGDBError:
            return self.count_messages_for_conversation(conversation_id)

        if not since_message:
            return self.count_messages_for_conversation(conversation_id)

        since_timestamp = since_message.get("timestamp")
        if not since_timestamp:
            return self.count_messages_for_conversation(conversation_id)

        query = (
            "SELECT COUNT(1) FROM messages m "
            "JOIN conversations c ON m.conversation_id = c.id "
            "WHERE m.conversation_id = ? AND m.deleted = FALSE AND c.deleted = FALSE "
            "AND m.timestamp > ?"
        )
        try:
            cursor = self._db.execute_query(query, (conversation_id, since_timestamp))
            row = cursor.fetchone()
            if row is None:
                return 0
            try:
                return int(row[0])
            except _CHACHA_NONCRITICAL_EXCEPTIONS:
                return int(row.get("COUNT(1)") or row.get("count") or 0)
        except CharactersRAGDBError as exc:
            logger.error("Database error counting messages after {}: {}", since_message_id, exc)
            raise
