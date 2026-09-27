# Notes_Library.py
# Description: This module provides a service layer for managing notes and note keywords
#
# Imports
import re
import sqlite3  # For exception handling in _get_db
import threading
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

#
# Third-Party Imports
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)
from tldw_Server_API.app.core.Notes.organization_capture import (
    active_coordinator,
    capture_note_tombstone,
    capture_note_upsert,
    capture_plan,
    stable_note_id,
)

#
#######################################################################################################################
#
# Functions:

logger = logger


class NotesInteropService:
    """
    A service layer for interacting with the notes-specific functionalities
    of CharactersRAGDB, designed for per-user database instances.
    This service manages CharactersRAGDB instances for each user and exposes
    methods for note and related keyword operations.
    """

    def __init__(self, base_db_directory: Union[str, Path], api_client_id: str):
        """
        Initializes the NotesInteropService.

        Args:
            base_db_directory: The base directory where user-specific SQLite database
                               files will be stored (e.g., "data/user_dbs").
                               Each user's DB will be in a sub-file like "user_{user_id}.sqlite".
            api_client_id: A client ID string representing this API application.
                           This ID is passed to CharactersRAGDB instances.
        """
        # Keep path as provided to avoid /var vs /private/var mismatch in tests
        self.base_db_directory = Path(base_db_directory)
        self.api_client_id = api_client_id
        self._db_instances: dict[str, CharactersRAGDB] = {}
        # Backward-compatible lock naming expected by tests
        self._lock = threading.Lock()
        self._db_lock = self._lock  # alias used internally
        self._resolved_base_db_directory: Optional[Path] = None

        try:
            self.base_db_directory.mkdir(parents=True, exist_ok=True)
            self._resolved_base_db_directory = self.base_db_directory.resolve()
            logger.info(f"NotesInteropService initialized. Base DB directory: {self.base_db_directory}")
        except OSError as e:
            logger.exception(f"Failed to create base DB directory {self.base_db_directory}: {e}")
            # This is a critical failure for the service's operation.
            raise CharactersRAGDBError(f"Failed to create base DB directory {self.base_db_directory}: {e}") from e

    def _get_db(self, user_id: Union[str, int]) -> CharactersRAGDB:
        """
        Retrieves or creates a CharactersRAGDB instance for a given user_id.
        Instances are cached for efficiency. This method is thread-safe.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            A CharactersRAGDB instance for the specified user.

        Raises:
            ValueError: If user_id is empty.
            CharactersRAGDBError: If the database initialization fails.
        """
        # Accept numeric IDs (common in AuthNZ) by normalizing to string
        if isinstance(user_id, int):
            user_id = str(user_id)
        elif not isinstance(user_id, str) or not user_id.strip():
            # Keep error message for backward-compatibility with existing tests
            raise ValueError("user_id must be a non-empty string.")
        else:
            user_id = user_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", user_id):
            raise ValueError("user_id contains unsupported characters; only letters, numbers, underscores, and hyphens are allowed.")
        db_directory_resolved = self._resolved_base_db_directory
        if db_directory_resolved is None:
            raise CharactersRAGDBError("NotesInteropService has no resolved base directory; initialization may have failed.")
        # Fast path: check if instance already exists without lock
        if user_id in self._db_instances:
            return self._db_instances[user_id]

        # Slow path: acquire lock and double-check
        with self._db_lock:
            if user_id not in self._db_instances:
                user_dir = self.base_db_directory / user_id
                try:
                    user_dir.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    logger.exception(
                        "Failed to create notes directory '{}' for user_id '{}': {}",
                        user_dir,
                        user_id,
                        exc,
                    )
                    raise CharactersRAGDBError(
                        f"Failed to prepare user directory for notes at {user_dir}: {exc}"
                    ) from exc
                candidate_path = (user_dir / "ChaChaNotes.db").resolve()
                try:
                    candidate_path.relative_to(db_directory_resolved)
                except ValueError as exc:
                    logger.exception(
                        "Resolved database path '{}' escapes base directory '{}' for user_id '{}'.",
                        candidate_path,
                        db_directory_resolved,
                        user_id,
                    )
                    raise CharactersRAGDBError("Resolved database path escapes configured base directory.") from exc
                db_path = candidate_path
                logger.info(f"Creating or loading DB for user_id '{user_id}' at path: {db_path}")
                try:
                    db_instance = CharactersRAGDB(db_path=db_path, client_id=self.api_client_id)
                    # If the factory returns the same object for multiple users (e.g., in tests),
                    # wrap subsequent users with lightweight proxies to ensure unique identities.
                    same_underlying = any(existing is db_instance for existing in self._db_instances.values())
                    self._db_instances[user_id] = (_DBProxy(db_instance) if same_underlying else db_instance)
                    logger.info(f"Successfully initialized DB for user_id '{user_id}'.")
                except (CharactersRAGDBError, SchemaError, sqlite3.Error) as e:
                    logger.error(f"Failed to initialize DB for user_id '{user_id}' at {db_path}: {e}", exc_info=True)
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error initializing DB for user_id '{user_id}' at {db_path}: {e}",
                                 exc_info=True)
                    raise CharactersRAGDBError(f"Unexpected error initializing DB for user {user_id}: {e}") from e

            return self._db_instances[user_id]

    # --- Note Methods (Test-friendly wrappers) ---

    def create_note(self, *, title: str, content: str, user_id: str) -> Any:
        """Create a note. Uses mock-style `create_note` if available, else DB `add_note`."""
        db = self._get_db(user_id)
        if hasattr(db, "create_note"):
            return db.create_note(title=title, content=content, user_id=user_id)
        return self.add_note(user_id=user_id, title=title, content=content)

    def add_note(self, user_id: str, title: str, content: str, note_id: Optional[str] = None) -> str:
        """
        Adds a new note for the specified user.

        Args:
            user_id: The ID of the user.
            title: The title of the note.
            content: The content of the note.
            note_id: Optional UUID for the note. If None, one will be generated.

        Returns:
            The ID of the newly created note.

        Raises:
            InputError, ConflictError, CharactersRAGDBError: From ChaChaDB.
            ValueError: If user_id is invalid.
        """
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            key = coordinator.request_fingerprint(
                "notes-interop.note.create",
                {"note_id": note_id, "title": title, "content": content},
            )
            captured_id = note_id or stable_note_id("notes-interop", key)
            capture_note_upsert(
                coordinator,
                note_id=captured_id,
                title=title,
                content=content,
                source="notes-interop",
                key=key,
            )
            return captured_id
        # ChaChaDB's add_note returns `str | None` in its type hint.
        # Current implementation of ChaChaDB.add_note returns `str` on success or raises.
        created_note_id = db.add_note(title=title, content=content, note_id=note_id)
        if created_note_id is None:
            # This path suggests an issue if add_note could return None without raising.
            logger.error(f"add_note for user {user_id} returned None unexpectedly for title '{title}'.")
            raise CharactersRAGDBError("Failed to create note, received None ID unexpectedly.")
        return created_note_id

    def get_note(self, *, note_id: str, user_id: str) -> Optional[dict[str, Any]]:
        """Get a note by id. Uses mock-style `get_note` if available, else DB `get_note_by_id`."""
        db = self._get_db(user_id)
        if hasattr(db, "get_note"):
            return db.get_note(note_id)
        return db.get_note_by_id(note_id=note_id)

    def get_note_by_id(self, user_id: str, note_id: str) -> Optional[dict[str, Any]]:
        """Retrieves a specific note by its ID for the given user."""
        db = self._get_db(user_id)
        return db.get_note_by_id(note_id=note_id)

    def list_notes(self, user_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Lists notes for the given user."""
        db = self._get_db(user_id)
        return db.list_notes(limit=limit, offset=offset)

    def update_note(self, *args, **kwargs) -> Any:
        """
        Update a note.

        Supports both calling styles used across the codebase and tests:
        - Positional: update_note(user_id, note_id, update_data, expected_version)
        - Keyword-only: update_note(user_id=..., note_id=..., title=..., content=..., expected_version=..., update_data=...)
        """
        def _coerce_expected_version(value: Any) -> int:
            if value is None:
                raise ValueError("expected_version is required for note updates.")
            try:
                coerced = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("expected_version must be an integer for note updates.") from exc
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("expected_version must be a whole number for note updates.")
            if coerced < 0:
                raise ValueError("expected_version must be non-negative for note updates.")
            return coerced

        # Positional compatibility path used by unit tests
        if args and len(args) >= 4:
            user_id, note_id, update_data, expected_version = args[:4]
            # `update_data` is expected to be a dict; rely on ChaChaNotes_DB to validate keys.
            db = self._get_db(user_id)
            coerced_version = _coerce_expected_version(expected_version)
            coordinator = active_coordinator(db, user_id=user_id)
            if coordinator is not None:
                current = db.get_note_by_id(note_id)
                if current is None:
                    return False
                capture_note_upsert(
                    coordinator,
                    note_id=note_id,
                    title=str(update_data.get("title", current.get("title") or "")),
                    content=str(update_data.get("content", current.get("content") or "")),
                    conversation_id=current.get("conversation_id"),
                    message_id=current.get("message_id"),
                    expected_version=coerced_version,
                    source="notes-interop",
                )
                return True
            return db.update_note(note_id=note_id, update_data=update_data, expected_version=coerced_version)

        # Keyword/modern path
        user_id: str = kwargs.get("user_id")
        if not user_id:
            raise ValueError("user_id is required")
        note_id: str = kwargs.get("note_id")
        if not note_id:
            raise ValueError("note_id is required")
        expected_version = kwargs.get("expected_version")
        title: Optional[str] = kwargs.get("title")
        content: Optional[str] = kwargs.get("content")
        update_data: Optional[dict[str, Any]] = kwargs.get("update_data")

        db = self._get_db(user_id)
        data: dict[str, Any] = update_data.copy() if update_data else {}
        if title is not None:
            data["title"] = title
        if content is not None:
            data["content"] = content
        coerced_version = _coerce_expected_version(expected_version)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            current = db.get_note_by_id(note_id)
            if current is None:
                return False
            capture_note_upsert(
                coordinator,
                note_id=note_id,
                title=str(data.get("title", current.get("title") or "")),
                content=str(data.get("content", current.get("content") or "")),
                conversation_id=current.get("conversation_id"),
                message_id=current.get("message_id"),
                expected_version=coerced_version,
                source="notes-interop",
            )
            return True
        return db.update_note(note_id=note_id, update_data=data, expected_version=coerced_version)

    def delete_note(self, *, note_id: str, user_id: str) -> Any:
        """Delete a note through active Sync, or use the legacy inactive path."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            current = db.get_note_by_id(note_id=note_id, include_deleted=True)
            if current is None:
                return False
            expected_version = int(current["version"])
            if bool(current.get("deleted")):
                expected_version -= 1
            capture_note_tombstone(
                coordinator,
                note_id=note_id,
                expected_version=expected_version,
                source="notes-interop",
            )
            return True
        if hasattr(db, "delete_note"):
            return db.delete_note(note_id)
        raise CharactersRAGDBError("delete_note is not supported by this database")

    def soft_delete_note(self, user_id: str, note_id: str, expected_version: int) -> bool:
        """Soft-deletes a note for the given user with optimistic locking."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            capture_note_tombstone(
                coordinator,
                note_id=note_id,
                expected_version=expected_version,
                source="notes-interop",
            )
            return True
        return db.soft_delete_note(note_id=note_id, expected_version=expected_version)

    def search_notes(self, *args, **kwargs) -> list[dict[str, Any]]:
        """
        Search notes.

        Supports both calling styles:
        - Positional: search_notes(user_id, term, limit=..., offset=...)
        - Keyword: search_notes(user_id=..., query=... or search_term=..., limit=..., offset=...)
        """
        if args and len(args) >= 2:
            # Positional style: search_notes(user_id, term, [limit], [offset])
            user_id = args[0]
            term = args[1]
            # If provided positionally, honor those; otherwise fall back to kwargs/defaults
            limit = args[2] if len(args) >= 3 and args[2] is not None else kwargs.get("limit", 10)
            offset = args[3] if len(args) >= 4 and args[3] is not None else kwargs.get("offset", 0)
            # Validate search term early to avoid DB-level failures
            if term is None or (isinstance(term, str) and term.strip() == ""):
                raise ValueError("search term must be a non-empty string")
            db = self._get_db(user_id)
            try:
                return db.search_notes(search_term=term, limit=int(limit), offset=int(offset))
            except TypeError:
                return db.search_notes(query=term, limit=int(limit), offset=int(offset))
        else:
            user_id = kwargs.get("user_id")
            term = kwargs.get("query") if kwargs.get("query") is not None else kwargs.get("search_term")
            limit = kwargs.get("limit", 10)
            offset = kwargs.get("offset", 0)
            # Validate search term early
            if term is None or (isinstance(term, str) and term.strip() == ""):
                raise ValueError("search term must be a non-empty string")
            db = self._get_db(user_id)
            if "query" in kwargs:
                try:
                    return db.search_notes(query=term, limit=int(limit), offset=int(offset))
                except TypeError:
                    return db.search_notes(search_term=term, limit=int(limit), offset=int(offset))
            else:
                try:
                    return db.search_notes(search_term=term, limit=int(limit), offset=int(offset))
                except TypeError:
                    return db.search_notes(query=term, limit=int(limit), offset=int(offset))

    # --- Note-Keyword Linking Methods ---

    def link_note_keyword(self, *, note_id: str, keyword_id: int, user_id: str) -> Any:
        """Link note to keyword. Uses mock-style `link_note_keyword` if available."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            return self._set_note_keyword_link(
                coordinator,
                note_id=note_id,
                keyword_id=keyword_id,
                present=True,
            )
        if hasattr(db, "link_note_keyword"):
            return db.link_note_keyword(note_id, keyword_id)
        return db.link_note_to_keyword(note_id=note_id, keyword_id=keyword_id)

    def link_note_to_keyword(self, user_id: str, note_id: str, keyword_id: int) -> bool:
        """Links a note to a keyword for the given user."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            return self._set_note_keyword_link(
                coordinator,
                note_id=note_id,
                keyword_id=keyword_id,
                present=True,
            )
        return db.link_note_to_keyword(note_id=note_id, keyword_id=keyword_id)

    def unlink_note_keyword(self, *, note_id: str, keyword_id: int, user_id: str) -> Any:
        """Unlink note from keyword. Uses mock-style `unlink_note_keyword` if available."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            return self._set_note_keyword_link(
                coordinator,
                note_id=note_id,
                keyword_id=keyword_id,
                present=False,
            )
        if hasattr(db, "unlink_note_keyword"):
            return db.unlink_note_keyword(note_id, keyword_id)
        return db.unlink_note_from_keyword(note_id=note_id, keyword_id=keyword_id)

    def unlink_note_from_keyword(self, user_id: str, note_id: str, keyword_id: int) -> bool:
        """Unlinks a note from a keyword for the given user."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            return self._set_note_keyword_link(
                coordinator,
                note_id=note_id,
                keyword_id=keyword_id,
                present=False,
            )
        return db.unlink_note_from_keyword(note_id=note_id, keyword_id=keyword_id)

    def get_keywords_for_note(self, user_id: str, note_id: str) -> list[dict[str, Any]]:
        """Retrieves all keywords linked to a specific note for the given user."""
        db = self._get_db(user_id)
        return db.get_keywords_for_note(note_id=note_id)

    def get_notes_for_keyword(self, user_id: str, keyword_id: int, limit: int = 50, offset: int = 0) -> list[
        dict[str, Any]]:
        """Retrieves all notes linked to a specific keyword for the given user."""
        db = self._get_db(user_id)
        return db.get_notes_for_keyword(keyword_id=keyword_id, limit=limit, offset=offset)

    # --- Keyword Methods (as they relate to notes functionality) ---

    def create_keyword(self, *, keyword: str, user_id: str) -> Optional[int]:
        """Create a keyword. Uses mock-style `create_keyword` if available, else DB `add_keyword`."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            return self._capture_keyword_create(coordinator, keyword)
        if hasattr(db, "create_keyword"):
            return db.create_keyword(keyword=keyword, user_id=user_id)
        return db.add_keyword(keyword_text=keyword)

    def add_keyword(self, user_id: str, keyword_text: str) -> Optional[int]:
        """Adds a new keyword for the specified user. Returns keyword ID."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            return self._capture_keyword_create(coordinator, keyword_text)
        return db.add_keyword(keyword_text=keyword_text)

    def get_keyword(self, *, keyword_id: int, user_id: str) -> Optional[dict[str, Any]]:
        """Get keyword by id. Uses mock-style `get_keyword` if available."""
        db = self._get_db(user_id)
        if hasattr(db, "get_keyword"):
            return db.get_keyword(keyword_id)
        return db.get_keyword_by_id(keyword_id=keyword_id)

    def get_keyword_by_id(self, user_id: str, keyword_id: int) -> Optional[dict[str, Any]]:
        """Retrieves a keyword by its ID for the given user."""
        db = self._get_db(user_id)
        return db.get_keyword_by_id(keyword_id=keyword_id)

    def get_keyword_by_text(self, user_id: str, keyword_text: str) -> Optional[dict[str, Any]]:
        """Retrieves a keyword by its text for the given user."""
        db = self._get_db(user_id)
        return db.get_keyword_by_text(keyword_text=keyword_text)

    def list_keywords(self, user_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Lists keywords for the given user."""
        db = self._get_db(user_id)
        return db.list_keywords(limit=limit, offset=offset)

    def delete_keyword(self, *, keyword_id: int, user_id: str) -> Any:
        """Delete a keyword for tests. Uses mock-style `delete_keyword` if available, else soft delete requires version."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            keyword = db.get_keyword_by_id(keyword_id=keyword_id)
            if keyword is None:
                raise CharactersRAGDBError("Keyword not found")
            return self._capture_keyword_delete(
                coordinator,
                keyword_id=keyword_id,
                expected_version=int(keyword["version"]),
            )
        if hasattr(db, "delete_keyword"):
            return db.delete_keyword(keyword_id)
        raise CharactersRAGDBError("delete_keyword without version not supported on real DB path")

    def soft_delete_keyword(self, user_id: str, keyword_id: int, expected_version: int) -> bool:
        """Soft-deletes a keyword for the given user with optimistic locking."""
        db = self._get_db(user_id)
        coordinator = active_coordinator(db, user_id=user_id)
        if coordinator is not None:
            return self._capture_keyword_delete(
                coordinator,
                keyword_id=keyword_id,
                expected_version=expected_version,
            )
        return db.soft_delete_keyword(keyword_id=keyword_id, expected_version=expected_version)

    @staticmethod
    def _capture_keyword_delete(coordinator, *, keyword_id: int, expected_version: int) -> bool:
        fields = {"keyword_id": keyword_id, "expected_version": expected_version}
        key = coordinator.request_fingerprint("notes-interop.keyword.delete", fields)
        replay = coordinator.replay_request_plan(
            source="notes-interop",
            idempotency_key=key,
            request_fingerprint=key,
            result_domain=None,
        )
        if replay is None:
            replay = coordinator.bind_request(
                coordinator.plan_resource_delete(
                    "notes.keyword",
                    keyword_id,
                    expected_version=expected_version,
                ),
                key,
            )
        capture_plan(coordinator, replay, source="notes-interop", key=key)
        return True

    @staticmethod
    def _capture_keyword_create(coordinator, keyword: str) -> int:
        key = coordinator.request_fingerprint(
            "notes-interop.keyword.create",
            {"keyword": str(keyword or "").strip()},
        )
        result = capture_plan(
            coordinator,
            coordinator.plan_keyword_create(keyword, idempotency_key=key),
            source="notes-interop",
            key=key,
        )
        if not isinstance(result, dict) or result.get("id") is None:
            raise CharactersRAGDBError("Captured keyword could not be reloaded")
        return int(result["id"])

    def _set_note_keyword_link(
        self,
        coordinator,
        *,
        note_id: str,
        keyword_id: int,
        present: bool,
    ) -> bool:
        keyword = coordinator.note_db.get_keyword_by_id(keyword_id)
        if keyword is None:
            raise CharactersRAGDBError("Keyword not found")
        fields = {
            "note_id": note_id,
            "keyword_sync_id": str(keyword["sync_id"]),
            "present": present,
        }
        key = coordinator.request_fingerprint("notes-interop.keyword.link", fields)
        plan = coordinator.plan_relationship(
            "notes.keyword_link",
            {
                "subject_type": "note",
                "subject_id": note_id,
                "keyword_sync_id": str(keyword["sync_id"]),
            },
            present,
            source="notes-interop",
            idempotency_key=key,
        )
        capture_plan(coordinator, plan, source="notes-interop", key=key)
        return True

    def search_keywords(
        self,
        *,
        user_id: str,
        query: Optional[str] = None,
        search_term: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search keywords. Supports both (query=...) and (search_term=...)."""
        term = query if query is not None else search_term
        db = self._get_db(user_id)
        if hasattr(db, "search_keywords"):
            try:
                return db.search_keywords(query=term, limit=limit)
            except TypeError:
                return db.search_keywords(search_term=term, limit=limit)
        return []

    # --- Resource Management ---

    def close_all_user_connections(self):
        """
        Closes all cached database connections.
        Useful for application shutdown.
        """
        with self._db_lock:
            logger.info(f"Closing all {len(self._db_instances)} cached user DB connections.")
            for user_id, db_instance in self._db_instances.items():
                try:
                    if hasattr(db_instance, "close_all_connections"):
                        db_instance.close_all_connections()
                    # Prefer mock-style close()
                    if hasattr(db_instance, "close"):
                        db_instance.close()
                    elif hasattr(db_instance, "close_connection"):
                        db_instance.close_connection()
                    logger.debug(f"Closed DB connection for user_id '{user_id}'.")
                except Exception as e:
                    logger.error(f"Error closing DB connection for user_id '{user_id}': {e}", exc_info=True)
            self._db_instances.clear()
        logger.info("All cached user DB connections have been processed for closure.")

    def close_user_connection(self, user_id: str):
        """
        Closes and removes a specific user's database connection from the cache.
        Useful if a user session ends and resources should be freed.
        """
        with self._db_lock:
            if user_id in self._db_instances:
                db_instance = self._db_instances.pop(user_id)  # Remove from cache
                try:
                    if hasattr(db_instance, "close_all_connections"):
                        db_instance.close_all_connections()
                    if hasattr(db_instance, "close"):
                        db_instance.close()
                    elif hasattr(db_instance, "close_connection"):
                        db_instance.close_connection()
                    logger.info(f"Closed and removed DB connection for user_id '{user_id}'.")
                except Exception as e:
                    logger.error(f"Error closing DB connection for user_id '{user_id}': {e}", exc_info=True)
                    # Even if close fails, it's removed from cache.
            else:
                logger.debug(f"No active DB connection found in cache for user_id '{user_id}' to close.")


class _DBProxy:
    """Deprecated: no longer used. Kept for import compatibility."""
    def __init__(self, inner: Any):
        self._inner = inner
    def __getattr__(self, item):
        return getattr(self._inner, item)


#
# # Example Usage (Conceptual - typically this would be in your API layer)
# if __name__ == "__main__":
#     # Basic logging setup for the example
#     logging.basicConfig(
#         level=logging.DEBUG,  # Set to INFO for less verbosity from ChaChaDB
#         format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s'
#     )
#
#     # Create a temporary directory for DBs for this example run
#     import tempfile
#     import shutil
#
#     temp_db_dir_obj = tempfile.TemporaryDirectory(prefix="chachadb_interop_test_")
#     temp_db_dir = temp_db_dir_obj.name
#     print(f"Temporary DB directory: {temp_db_dir}")
#
#     notes_service_instance = None  # Define for finally block
#
#     try:
#         # Initialize the service
#         notes_service_instance = NotesInteropService(
#             base_db_directory=temp_db_dir,
#             api_client_id="example_api_v1.0"
#         )
#
#         user1_id = "user123"
#         user2_id = "user456"
#
#         # --- User 1 operations ---
#         print(f"\n--- Operations for {user1_id} ---")
#         try:
#             note1_title = "User1's First Note"
#             note1_content = "This is the first note for user1."
#             u1_note1_id = notes_service_instance.add_note(user_id=user1_id, title=note1_title, content=note1_content)
#             print(f"Added note for {user1_id}: ID={u1_note1_id}, Title='{note1_title}'")
#
#             retrieved_note = notes_service_instance.get_note_by_id(user_id=user1_id, note_id=u1_note1_id)
#             print(f"Retrieved note for {user1_id}: {retrieved_note}")
#             assert retrieved_note is not None and retrieved_note['title'] == note1_title
#
#             kw1_text = "important"
#             u1_kw1_id = notes_service_instance.add_keyword(user_id=user1_id, keyword_text=kw1_text)
#             print(f"Added keyword for {user1_id}: ID={u1_kw1_id}, Text='{kw1_text}'")
#             assert u1_kw1_id is not None
#
#             if u1_note1_id and u1_kw1_id:
#                 link_success = notes_service_instance.link_note_to_keyword(user_id=user1_id, note_id=u1_note1_id,
#                                                                            keyword_id=u1_kw1_id)
#                 print(f"Link note to keyword success: {link_success}")
#                 assert link_success
#                 keywords_for_note = notes_service_instance.get_keywords_for_note(user_id=user1_id, note_id=u1_note1_id)
#                 print(f"Keywords for note {u1_note1_id}: {keywords_for_note}")
#                 assert len(keywords_for_note) == 1 and keywords_for_note[0]['id'] == u1_kw1_id
#
#             user1_notes = notes_service_instance.list_notes(user_id=user1_id)
#             print(f"Notes for {user1_id}: {user1_notes}")
#             assert len(user1_notes) == 1
#
#             update_success = notes_service_instance.update_note(
#                 user_id=user1_id, note_id=u1_note1_id,
#                 update_data={"content": "Updated content for user1's first note."}, expected_version=1
#             )
#             print(f"Update note success: {update_success}")
#             assert update_success
#             updated_note = notes_service_instance.get_note_by_id(user_id=user1_id, note_id=u1_note1_id)
#             print(f"Updated note: {updated_note}")
#             assert updated_note['content'] == "Updated content for user1's first note." and updated_note['version'] == 2
#
#             try:
#                 notes_service_instance.update_note(
#                     user_id=user1_id, note_id=u1_note1_id,
#                     update_data={"content": "Another update attempt."}, expected_version=1  # Wrong version
#                 )
#             except ConflictError as ce:
#                 print(f"Caught expected ConflictError for optimistic lock: {ce}")
#
#         except (InputError, ConflictError, CharactersRAGDBError, ValueError) as e:
#             print(f"Error during {user1_id} operations: {e}")
#
#         # --- User 2 operations (to show DB separation) ---
#         print(f"\n--- Operations for {user2_id} ---")
#         try:
#             note2_title = "User2's Special Note"
#             u2_note1_id = notes_service_instance.add_note(user_id=user2_id, title=note2_title,
#                                                           content="Content for user2.")
#             print(f"Added note for {user2_id}: ID={u2_note1_id}, Title='{note2_title}'")
#
#             user2_notes = notes_service_instance.list_notes(user_id=user2_id)
#             print(f"Notes for {user2_id}: {user2_notes}")
#             assert len(user2_notes) == 1 and user2_notes[0]['id'] == u2_note1_id
#
#             user1_notes_recheck = notes_service_instance.list_notes(user_id=user1_id)
#             print(f"Notes for {user1_id} (recheck): {user1_notes_recheck}")
#             assert len(user1_notes_recheck) == 1 and user1_notes_recheck[0]['id'] == u1_note1_id
#
#         except (InputError, ConflictError, CharactersRAGDBError, ValueError) as e:
#             print(f"Error during {user2_id} operations: {e}")
#
#         # --- Search example ---
#         print(f"\n--- Search for {user1_id} ---")
#         search_results_u1 = notes_service_instance.search_notes(user_id=user1_id, search_term="User1")
#         print(f"Search for 'User1' in {user1_id}'s notes: {search_results_u1}")
#         assert len(search_results_u1) > 0 and search_results_u1[0]['id'] == u1_note1_id
#
#         # --- Soft delete example ---
#         print(f"\n--- Soft delete for {user1_id} ---")
#         note_to_delete = notes_service_instance.get_note_by_id(user_id=user1_id, note_id=u1_note1_id)
#         if note_to_delete:
#             delete_success = notes_service_instance.soft_delete_note(user_id=user1_id, note_id=u1_note1_id,
#                                                                      expected_version=note_to_delete['version'])
#             print(f"Soft delete success: {delete_success}")
#             assert delete_success
#             deleted_note_check = notes_service_instance.get_note_by_id(user_id=user1_id, note_id=u1_note1_id)
#             print(f"Check after delete (should be None): {deleted_note_check}")
#             assert deleted_note_check is None
#             user1_notes_after_delete = notes_service_instance.list_notes(user_id=user1_id)
#             print(f"Notes for {user1_id} after delete: {user1_notes_after_delete}")
#             assert len(user1_notes_after_delete) == 0
#         else:
#             print(f"Note {u1_note1_id} not found for deletion, skipping delete test.")
#
#
#     finally:
#         if notes_service_instance:
#             notes_service_instance.close_all_user_connections()
#
#         temp_db_dir_obj.cleanup()  # Safely removes the temporary directory
#         print(f"\nCleaned up temporary DB directory: {temp_db_dir}")
#
#     print("\nExample script finished.")


#
# End of Notes_Library.py
#######################################################################################################################
