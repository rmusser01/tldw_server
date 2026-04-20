"""
Unified database exception hierarchy for all DB modules.

All DB management modules should inherit from these base classes
instead of defining their own parallel hierarchies. This enables
consistent HTTP error mapping via map_db_error_to_http().

Migration strategy: Module-specific error classes (e.g., KanbanDBError,
CharactersRAGDBError) should be made subclasses of the appropriate
unified base. Existing code catching module-specific types will continue
to work since isinstance() respects inheritance.

Usage in endpoints::

    from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http

    try:
        result = db.some_operation(...)
    except (DatabaseError, InputError) as exc:
        raise map_db_error_to_http(exc) from exc
"""

from __future__ import annotations


class DatabaseError(Exception):
    """Base exception for all database-layer errors.

    HTTP mapping: 500 Internal Server Error
    """


class SchemaError(DatabaseError):
    """Database schema or migration issue.

    HTTP mapping: 500 Internal Server Error (with logging)
    """


class InputError(ValueError):
    """Invalid input to a database operation.

    Inherits from ValueError for backward compatibility with code
    that catches ValueError from DB modules.

    HTTP mapping: 400 Bad Request
    """


class ConflictError(DatabaseError):
    """Conflict detected (concurrent modification, unique constraint, etc.).

    HTTP mapping: 409 Conflict
    """


class NotFoundError(DatabaseError):
    """Requested resource does not exist.

    HTTP mapping: 404 Not Found
    """


class DataIntegrityError(DatabaseError):
    """Data integrity violation (FK constraint, check constraint, etc.).

    HTTP mapping: 422 Unprocessable Entity
    """


class MigrationError(DatabaseError):
    """Database migration failure.

    HTTP mapping: 500 Internal Server Error
    """


__all__ = [
    "ConflictError",
    "DataIntegrityError",
    "DatabaseError",
    "InputError",
    "MigrationError",
    "NotFoundError",
    "SchemaError",
]
