"""Tests for http_errors.map_db_error_to_http utility."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
    SchemaError,
)


class TestMapDbErrorToHttp:
    def test_input_error_maps_to_400(self):
        exc = InputError("bad input")
        result = map_db_error_to_http(exc)
        assert result.status_code == 400
        assert "bad input" in result.detail

    def test_conflict_error_maps_to_409(self):
        exc = ConflictError("conflict")
        result = map_db_error_to_http(exc)
        assert result.status_code == 409
        assert "conflict" in result.detail

    def test_schema_error_maps_to_500(self):
        exc = SchemaError("migration failed")
        result = map_db_error_to_http(exc)
        assert result.status_code == 500
        assert result.detail == "Database schema error"

    def test_database_error_maps_to_500_with_default_detail(self):
        exc = DatabaseError("connection lost")
        result = map_db_error_to_http(exc)
        assert result.status_code == 500
        assert result.detail == "Database error occurred"

    def test_database_error_uses_custom_default_detail(self):
        exc = DatabaseError("connection lost")
        result = map_db_error_to_http(exc, default_detail="Custom DB error")
        assert result.detail == "Custom DB error"

    def test_unknown_exception_maps_to_500(self):
        exc = RuntimeError("unexpected")
        result = map_db_error_to_http(exc)
        assert result.status_code == 500
        assert result.detail == "Internal server error"

    def test_input_error_with_empty_message(self):
        exc = InputError("")
        result = map_db_error_to_http(exc)
        assert result.status_code == 400
        assert result.detail == "Invalid input"

    def test_conflict_error_with_empty_message(self):
        exc = ConflictError("")
        result = map_db_error_to_http(exc)
        assert result.status_code == 409
        assert result.detail == "Conflict detected"
