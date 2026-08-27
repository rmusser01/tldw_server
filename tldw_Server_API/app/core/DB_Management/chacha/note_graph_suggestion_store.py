"""Owner-bound SQL seam for Notes graph suggestion persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..ChaChaNotes_DB import BackendType

if TYPE_CHECKING:
    from ..ChaChaNotes_DB import CharactersRAGDB


class NoteGraphSuggestionStore:
    """Own the database boundary for future suggestion reads and transitions."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def owner_user_id(self) -> str:
        """Return the ChaChaNotes owner bound to this store instance."""
        return str(self._db.client_id)

    @property
    def is_postgres(self) -> bool:
        """Return whether this store is backed by PostgreSQL."""
        return self._db.backend_type == BackendType.POSTGRESQL
