"""Error-mapping characterization tests for writing endpoint helpers."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.endpoints import writing, writing_manuscripts
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_maps_input_error(module, entity_label):
    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(InputError("invalid writing payload"), entity_label)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "invalid writing payload"


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_maps_base_db_error(module, entity_label):
    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(CharactersRAGDBError("sqlite backend exploded"), entity_label)

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == f"Database error while processing {entity_label}"


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_preserves_not_found_conflict(module, entity_label):
    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(ConflictError("record not found"), entity_label)

    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert exc_info.value.detail == f"{entity_label} not found"


@pytest.mark.parametrize(
    ("module", "entity_label"),
    [
        (writing, "writing session"),
        (writing_manuscripts, "manuscript project"),
    ],
)
def test_writing_handle_db_errors_preserves_version_conflict(module, entity_label):
    with pytest.raises(HTTPException) as exc_info:
        module._handle_db_errors(ConflictError("version conflict"), entity_label)

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == "version conflict"
