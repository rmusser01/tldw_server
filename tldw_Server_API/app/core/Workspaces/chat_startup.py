"""Atomic, permanent owner-bound acceptance and replay of strict Workspace startup."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.api.v1.schemas.workspace_chat_startup_schemas import (
    WorkspaceChatStartupRequest,
    startup_request_fingerprint,
)
from tldw_Server_API.app.core.DB_Management.backends.base import UniqueConstraintError
from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_store import (
    WorkspaceStartupError,
    WorkspaceStartupResult,
    startup_binding_digest,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona.conversation_admission import PersonaAdmissionError, require_current_persona
from tldw_Server_API.app.core.Workspaces.assistant_defaults import (
    WorkspaceDefaultUnavailable,
    insert_resolved_workspace_conversation,
    resolve_workspace_assistant_startup,
)


class _ReceiptRetry(Exception):
    """Exit owned work before changing from fresh acceptance to winner replay."""

    def __init__(self, error: Exception | None = None) -> None:
        """Retain an insertion conflict only for propagation if no winner exists."""
        self.error = error
        super().__init__()


def map_workspace_default_unavailable(error: WorkspaceDefaultUnavailable) -> WorkspaceStartupError:
    """Translate only the typed resolver failure, not HTTP/storage errors generally."""
    code = (
        "persona_feature_disabled" if error.reason == "persona_feature_disabled" else "workspace_assistant_unavailable"
    )
    return WorkspaceStartupError(code, error.status_code, error.reason)


def _require_workspace(workspace: Mapping[str, Any] | None, *, fresh: bool) -> None:
    """Enforce current access and lifecycle before disclosing any receipt target."""
    if workspace is None or workspace.get("deleted"):
        raise WorkspaceStartupError("workspace_not_found", 404)
    if workspace.get("native_chat_admission_closed") or workspace.get("system_operation_state") is not None:
        raise WorkspaceStartupError("workspace_chat_admission_closed", 409)
    if fresh and workspace.get("archived"):
        raise WorkspaceStartupError("workspace_archived", 409)


def _replay(
    db: CharactersRAGDB,
    *,
    owner_id: str,
    key_digest: str,
    fingerprint: str,
    receipt: Mapping[str, Any],
    conn: Any,
) -> WorkspaceStartupResult:
    """Lock Workspace, original Persona, conversation, then receipt without FK inversion."""
    store = db.workspace_chat_startups
    workspace_id = receipt["workspace_id"]
    _require_workspace(store.lock_receipt_workspace(workspace_id, conn=conn), fresh=False)
    current = store.get_receipt(owner_id, key_digest, conn=conn)
    if current is None or current["workspace_id"] != workspace_id:
        raise WorkspaceStartupError("workspace_chat_startup_changed", 409)
    if current["request_fingerprint"] != fingerprint:
        raise WorkspaceStartupError("idempotency_key_conflict", 409)
    conversation_id = current["conversation_id"]
    if conversation_id is None:
        raise WorkspaceStartupError("workspace_chat_deleted", 410)
    # This read chooses the Persona lock only; no authority is granted until the
    # locked conversation and final receipt have been validated below.
    hint = db.get_conversation_by_id(conversation_id, include_deleted=True)
    admission_error = None
    if hint is not None and not hint.get("deleted"):
        try:
            require_current_persona(db, owner_id=owner_id, conversation=hint, conn=conn)
        except PersonaAdmissionError as error:
            admission_error = error
    current = store.get_receipt(owner_id, key_digest, conn=conn)
    if current is None:
        raise WorkspaceStartupError("workspace_chat_startup_changed", 409)
    conversation = store.lock_conversation(conversation_id, conn=conn)
    current = store.get_receipt(owner_id, key_digest, conn=conn, for_update=True)
    if current is None or current["workspace_id"] != workspace_id:
        raise WorkspaceStartupError("workspace_chat_startup_changed", 409)
    if current["request_fingerprint"] != fingerprint:
        raise WorkspaceStartupError("idempotency_key_conflict", 409)
    if current["conversation_id"] is None or conversation is None or conversation.get("deleted"):
        raise WorkspaceStartupError("workspace_chat_deleted", 410)
    if (
        current["conversation_id"] != conversation_id
        or current["invalidated_at"] is not None
        or startup_binding_digest(db, conversation) != current["binding_digest"]
    ):
        raise WorkspaceStartupError("workspace_chat_startup_changed", 409)
    if admission_error is not None:
        raise WorkspaceStartupError(admission_error.code, admission_error.status_code, admission_error.reason) from None
    if hint is None or hint.get("deleted") or startup_binding_digest(db, hint) != current["binding_digest"]:
        # A different Persona may not be locked after acquiring the conversation.
        raise WorkspaceStartupError("workspace_chat_startup_changed", 409)
    return WorkspaceStartupResult(conversation=conversation, replayed=True)


def _attempt_startup(
    db: CharactersRAGDB,
    *,
    owner_id: str,
    request: WorkspaceChatStartupRequest,
    key_digest: str,
    fingerprint: str,
    receipt_limit: int,
    chat_limit: int | None,
    title_timestamp: str,
    retry: _ReceiptRetry | None,
) -> WorkspaceStartupResult:
    """Hold the result until the owned transaction has committed successfully."""
    store = db.workspace_chat_startups
    store.require_outermost()
    with db.transaction() as conn:
        store.lock_owner(owner_id, conn=conn)
        receipt = store.get_receipt(owner_id, key_digest, conn=conn)
        if receipt is not None:
            result = _replay(
                db, owner_id=owner_id, key_digest=key_digest, fingerprint=fingerprint, receipt=receipt, conn=conn
            )
        else:
            if retry is not None:
                if retry.error is not None:
                    raise retry.error
                raise WorkspaceStartupError("workspace_chat_startup_changed", 409)
            workspace = store.lock_receipt_workspace(request.workspace_id, conn=conn, for_create=True)
            if store.get_receipt(owner_id, key_digest, conn=conn) is not None:
                raise _ReceiptRetry()
            _require_workspace(workspace, fresh=True)
            if (
                request.workspace_assistant_selection == "inherit"
                and workspace["version"] != request.workspace_assistant_default_version
            ):
                raise WorkspaceStartupError("workspace_assistant_version_conflict", 409)
            adapted = request.model_dump(exclude_unset=True)
            adapted.pop("workspace_assistant_selection")
            adapted.pop("workspace_assistant_default_version", None)
            if request.workspace_assistant_selection == "none":
                adapted.update(assistant_kind=None, assistant_id=None, character_id=None)
            try:
                resolved = resolve_workspace_assistant_startup(
                    db,
                    user_id=owner_id,
                    request=ChatSessionCreate.model_validate(adapted),
                    conn=conn,
                )
            except WorkspaceDefaultUnavailable as error:
                if store.get_receipt(owner_id, key_digest, conn=conn) is not None:
                    raise _ReceiptRetry() from None
                raise map_workspace_default_unavailable(error) from None
            if store.get_receipt(owner_id, key_digest, conn=conn) is not None:
                raise _ReceiptRetry()
            try:
                require_current_persona(db, owner_id=owner_id, conversation=resolved.request.model_dump(), conn=conn)
            except PersonaAdmissionError as error:
                if store.get_receipt(owner_id, key_digest, conn=conn) is not None:
                    raise _ReceiptRetry() from None
                raise WorkspaceStartupError(error.code, error.status_code, error.reason) from None
            if store.get_receipt(owner_id, key_digest, conn=conn) is not None:
                raise _ReceiptRetry()
            if store.count_receipts(owner_id, conn=conn) >= receipt_limit:
                raise WorkspaceStartupError("workspace_chat_receipt_capacity_exceeded", 409)
            if (
                chat_limit is not None
                and store.count_live_chats(owner_id, request.workspace_id, conn=conn) >= chat_limit
            ):
                raise WorkspaceStartupError("workspace_chat_quota_exceeded", 429)
            conversation_id = str(uuid4())
            payload = dict(
                adapted,
                id=conversation_id,
                root_id=conversation_id,
                client_id=owner_id,
                parent_conversation_id=None,
                forked_from_message_id=None,
            )
            insert_resolved_workspace_conversation(
                db,
                resolved=resolved,
                conversation_data=payload,
                title_timestamp=title_timestamp,
                conn=conn,
            )
            conversation = store.lock_conversation(conversation_id, conn=conn)
            if conversation is None:
                raise WorkspaceStartupError("workspace_chat_startup_changed", 409)
            try:
                store.insert_receipt(
                    owner_id,
                    key_digest,
                    fingerprint,
                    startup_binding_digest(db, conversation),
                    request.workspace_id,
                    conversation_id,
                    conn=conn,
                )
            except (sqlite3.IntegrityError, UniqueConstraintError) as error:
                if isinstance(error, sqlite3.IntegrityError) and getattr(error, "sqlite_errorname", None) not in {
                    "SQLITE_CONSTRAINT_UNIQUE",
                    "SQLITE_CONSTRAINT_PRIMARYKEY",
                }:
                    raise
                raise _ReceiptRetry(error) from None
            result = WorkspaceStartupResult(conversation=conversation, replayed=False)
    return result


def start_workspace_chat(
    db: CharactersRAGDB,
    *,
    owner_id: str,
    request: WorkspaceChatStartupRequest,
    idempotency_key: str,
    receipt_limit: int,
    chat_limit: int | None,
    title_timestamp: str,
) -> WorkspaceStartupResult:
    """Accept once or replay under current authority in at most two owned transactions."""
    if owner_id != db.owner_user_id:
        raise WorkspaceStartupError("workspace_chat_startup_owner_mismatch", 404)
    if (
        not isinstance(idempotency_key, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", idempotency_key) is None
    ):
        raise WorkspaceStartupError("invalid_idempotency_key", 422)
    if (
        type(receipt_limit) is not int
        or receipt_limit < 1
        or (chat_limit is not None and (type(chat_limit) is not int or chat_limit < 0))
    ):
        raise WorkspaceStartupError("workspace_chat_startup_configuration_invalid", 503)
    fingerprint = startup_request_fingerprint(request)
    key_digest = hashlib.sha256(idempotency_key.encode("ascii")).hexdigest()
    retry = None
    for attempt in range(2):
        try:
            return _attempt_startup(
                db,
                owner_id=owner_id,
                request=request,
                key_digest=key_digest,
                fingerprint=fingerprint,
                receipt_limit=receipt_limit,
                chat_limit=chat_limit,
                title_timestamp=title_timestamp,
                retry=retry,
            )
        except _ReceiptRetry as error:
            if attempt:
                raise WorkspaceStartupError("workspace_chat_startup_changed", 409) from None
            retry = error
    raise AssertionError("Unreachable startup retry state")
