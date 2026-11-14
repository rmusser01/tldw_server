"""Write an imported chat and its whole message graph in one transaction (D7 P8).

The store composes the ordinary conversation and message stores on a single
connection. An imported chat therefore has the same rows, search index entries
and change-log entries as a chat written message by message, and either all of
it is committed or none of it is.

Each imported message carries owner-only provenance (``history_admission_json``)
that marks it as part of an explicit parent graph and records the fingerprint
of the import request. That fingerprint is how a repeated import is recognized;
see ``core.Chat.conversation_import``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Chat.conversation_import import (
    import_fingerprint_from_authority,
    import_message_authority,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, UniqueConstraintError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
    logger,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

# Only these conversation fields are taken from the caller. Owner, scope,
# version and deleted state are fixed by the store.
_CONVERSATION_FIELDS = (
    "id",
    "title",
    "state",
    "character_id",
    "assistant_kind",
    "assistant_id",
    "persona_memory_mode",
    "root_id",
    "parent_conversation_id",
    "forked_from_message_id",
)

# The provenance is stored as compact JSON, so an import marker always contains this text.
_IMPORT_MARKER_LIKE = '%"import":{%'


def _is_unique_violation(error: BaseException) -> bool:
    """PostgreSQL reports a taken primary key as a backend error, sometimes wrapped."""
    return isinstance(error, UniqueConstraintError) or isinstance(error.__cause__, UniqueConstraintError)


class ChatImportStore:
    """Atomic writes and replay lookups for imported chats."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _require_own_transaction(self) -> None:
        """Refuse to run inside a transaction that someone else opened.

        Joined to an open transaction, the import would be committed or rolled
        back by whoever opened it, and "all or nothing" would no longer be this
        method's to promise. SQLite reports an open transaction on the thread's
        connection. PostgreSQL connections may hold an open read, which the
        import's own transaction settles, so there only a managed transaction
        that is already in progress counts.
        """
        state = self._db._connection_state()
        if self._db.backend_type == BackendType.SQLITE:
            joined = bool(getattr(getattr(state, "conn", None), "in_transaction", False))
        else:
            joined = bool(getattr(state, "tx_depth", 0))
        if joined:
            raise CharactersRAGDBError("A chat import needs its own transaction.")  # noqa: TRY003

    def get_import_state(self, conversation_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Return the conversation with this id and the import fingerprint stored with its messages.

        Args:
            conversation_id: The conversation id, live or in trash.

        Returns:
            ``(row, fingerprint)``. ``row`` is None when the caller can see no
            conversation with that id. ``fingerprint`` is None for a chat that
            was not created by an import.
        """
        row = self._db.get_conversation_by_id(conversation_id, include_deleted=True)
        if row is None:
            return None, None
        cursor = self._db.execute_query(
            "SELECT history_admission_json FROM messages "
            "WHERE conversation_id = ? AND history_admission_json LIKE ?",
            (conversation_id, _IMPORT_MARKER_LIKE),
            read_only=True,
        )
        # Every imported message carries the same fingerprint, so the first valid marker decides.
        while (candidate := cursor.fetchone()) is not None:
            fingerprint = import_fingerprint_from_authority(candidate["history_admission_json"])
            if fingerprint is not None:
                return dict(row), fingerprint
        return dict(row), None

    def import_conversation(
        self,
        conversation: Mapping[str, Any],
        messages: Sequence[Mapping[str, Any]],
        *,
        owner_client_id: str,
        request_fingerprint: str,
    ) -> str:
        """Insert a conversation and all of its messages, or nothing.

        Args:
            conversation: ``id``, ``title``, ``created_at`` and
                ``last_modified`` (UTC ISO strings), plus optional ``state``,
                assistant binding and fork lineage. Any other key is ignored.
            messages: Messages with every parent before its children. Each has
                ``id``, ``parent_message_id``, ``sender``, ``content``,
                ``timestamp``, ``images`` (``{"data", "mime"}``) and
                ``extra_metadata``. Their timestamps are stored as given.
            owner_client_id: The owner of every row written. Nothing in the
                payload can change it.
            request_fingerprint: SHA-256 hex digest of the import request.

        Returns:
            The conversation id.

        Raises:
            InputError: The arguments are malformed. Nothing is written.
            ConflictError: The conversation id (``entity="conversations"``) or a
                message id (``entity="messages"``) is already taken. Nothing is
                written.
            CharactersRAGDBError: A transaction is already open on this
                connection, or any other database failure. Nothing is written.
        """
        owner = str(owner_client_id or "").strip()
        if not owner:
            raise InputError("An imported chat needs an owner.")  # noqa: TRY003
        try:
            authority = import_message_authority(request_fingerprint)
        except ValueError as error:
            raise InputError(str(error)) from error  # noqa: TRY003
        conversation_id = conversation.get("id")
        created_at = conversation.get("created_at")
        last_modified = conversation.get("last_modified")
        if not all(isinstance(value, str) and value for value in (conversation_id, created_at, last_modified)):
            raise InputError("An imported chat needs an id, created_at and last_modified.")  # noqa: TRY003
        if not messages:
            raise InputError("An imported chat needs at least one message.")  # noqa: TRY003
        known: set[str] = set()
        for message in messages:
            parent = message.get("parent_message_id")
            if not message.get("id") or message["id"] in known or (parent is not None and parent not in known):
                raise InputError("Imported messages need unique ids, with every parent before its children.")  # noqa: TRY003
            known.add(message["id"])

        conv_data = {field: conversation.get(field) for field in _CONVERSATION_FIELDS}
        conv_data.update(client_id=owner, scope_type="global", workspace_id=None)

        self._require_own_transaction()
        with self._db.transaction() as conn:
            if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is not None:
                raise ConflictError(  # noqa: TRY003
                    f"Conversation with ID '{conversation_id}' already exists.",
                    entity="conversations",
                    entity_id=conversation_id,
                )
            try:
                self._db.conversation_store.add_conversation(conv_data, conn=conn)
            except UniqueConstraintError as error:
                # A row the caller cannot see (another owner's, under PostgreSQL RLS) holds the id.
                raise ConflictError(  # noqa: TRY003
                    f"Conversation with ID '{conversation_id}' already exists.",
                    entity="conversations",
                    entity_id=conversation_id,
                ) from error

            for message in messages:
                message_id = message["id"]
                try:
                    self._db.message_store.add_message(
                        {
                            "id": message_id,
                            "conversation_id": conversation_id,
                            "parent_message_id": message.get("parent_message_id"),
                            "sender": message["sender"],
                            "content": message.get("content") or "",
                            "timestamp": message["timestamp"],
                            "images": list(message.get("images") or ()),
                            "client_id": owner,
                        },
                        conn=conn,
                    )
                except ConflictError:
                    raise
                except (CharactersRAGDBError, UniqueConstraintError) as error:
                    if _is_unique_violation(error):
                        raise ConflictError(  # noqa: TRY003
                            f"Message with ID '{message_id}' already exists.",
                            entity="messages",
                            entity_id=message_id,
                        ) from error
                    raise
                extra = message.get("extra_metadata")
                if extra:
                    self._db.message_store._add_message_metadata_with_conn(
                        message_id, None, dict(extra), conn, advance_history=False
                    )
                self._db.message_store._write_history_authority(conn, message_id, authority)

            # Writing the messages stamped the conversation with the current time; put its own dates back.
            conn.execute(
                "UPDATE conversations SET created_at = ?, last_modified = ? WHERE id = ?",
                (created_at, last_modified, conversation_id),
            )
        logger.info("Imported conversation {} with {} messages.", conversation_id, len(messages))
        return conversation_id
