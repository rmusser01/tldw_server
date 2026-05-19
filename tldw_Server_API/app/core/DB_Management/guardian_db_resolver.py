"""
Guardian DB resolution helpers shared by core and API layers.
"""
from __future__ import annotations

from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


def coerce_guardian_storage_user_id(raw_user_id: object) -> int:
    """Normalize user IDs to the integer storage key used for guardian DB paths.

    Accepts ``int`` and non-empty ``str`` values that are convertible to ``int``.
    All other types (``None``, ``bool``, ``float``, empty strings, arbitrary
    objects) are rejected with a ``ValueError`` so that invalid or
    non-canonical user IDs never silently map to a real storage key.
    """
    # Reject None, booleans (bool is a subclass of int in Python), floats,
    # and any non-string/non-int type outright.
    if raw_user_id is None or isinstance(raw_user_id, bool):
        raise ValueError(
            f"Invalid guardian user ID: {raw_user_id!r} (type {type(raw_user_id).__name__})"
        )

    if isinstance(raw_user_id, int):
        return raw_user_id

    if isinstance(raw_user_id, str):
        stripped = raw_user_id.strip()
        if not stripped:
            raise ValueError("Invalid guardian user ID: empty or blank string")
        try:
            return int(stripped)
        except (ValueError, TypeError):
            raise ValueError(
                f"Invalid guardian user ID: string {raw_user_id!r} is not a valid integer"
            ) from None

    raise ValueError(
        f"Invalid guardian user ID: unsupported type {type(raw_user_id).__name__} ({raw_user_id!r})"
    )


def resolve_guardian_db_for_user_id(user_id: object) -> GuardianDB:
    """Return a GuardianDB instance bound to the provided user ID."""
    storage_user_id = coerce_guardian_storage_user_id(user_id)
    db_path = DatabasePaths.get_guardian_db_path(storage_user_id)
    return GuardianDB(str(db_path))
