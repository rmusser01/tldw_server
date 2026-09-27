"""Compatibility import for database-owned Personal Context key custody."""

from tldw_Server_API.app.core.DB_Management.Personal_Context_Key_Store import (
    ServerProfileKeyProvider,
)

__all__ = ["ServerProfileKeyProvider"]
