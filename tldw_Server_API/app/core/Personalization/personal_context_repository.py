"""Compatibility import for the database-owned Personal Context repository."""

from tldw_Server_API.app.core.DB_Management.Personal_Context_Repository import (
    PersonalContextRepository,
)

__all__ = ["PersonalContextRepository"]
