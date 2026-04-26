from pathlib import Path
import threading

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
    SchemaError,
)


def _patch_chacha_init_failure(monkeypatch, tmp_path: Path, exc: Exception) -> None:
    deps.close_all_chacha_db_instances()
    deps.reset_chacha_shutdown_state()
    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_user_base_directory",
        lambda user_id: tmp_path / str(user_id),
    )

    def fail_create(*args, **kwargs):
        raise exc

    monkeypatch.setattr(deps, "_create_and_prepare_db", fail_create)


def _seed_existing_chacha_init_failure(monkeypatch, tmp_path: Path, user_id: int, exc: Exception) -> None:
    deps.close_all_chacha_db_instances()
    deps.reset_chacha_shutdown_state()
    user_dir = tmp_path / str(user_id)
    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_user_base_directory",
        lambda requested_user_id: tmp_path / str(requested_user_id),
    )
    init_event = threading.Event()
    init_event.set()
    with deps._chacha_db_lock:
        deps._chacha_db_init_events[str(user_dir)] = init_event
        deps._chacha_db_init_errors[str(user_dir)] = exc


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_maps_schema_error(monkeypatch, tmp_path):
    _patch_chacha_init_failure(monkeypatch, tmp_path, SchemaError("schema exploded"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(61, "61")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_maps_base_database_error(monkeypatch, tmp_path):
    _patch_chacha_init_failure(monkeypatch, tmp_path, CharactersRAGDBError("backend exploded"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(62, "62")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "ChaChaNotes DB unavailable"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_keeps_conflict_init_errors_as_500(monkeypatch, tmp_path):
    _patch_chacha_init_failure(monkeypatch, tmp_path, ConflictError("duplicate bootstrap state"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(63, "63")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "ChaChaNotes DB unavailable"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_keeps_input_init_errors_as_500(monkeypatch, tmp_path):
    _patch_chacha_init_failure(monkeypatch, tmp_path, InputError("invalid bootstrap state"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(67, "67")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "invalid bootstrap state"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_waiter_maps_schema_error(monkeypatch, tmp_path):
    _seed_existing_chacha_init_failure(monkeypatch, tmp_path, 64, SchemaError("schema exploded"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(64, "64")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_waiter_maps_base_database_error(monkeypatch, tmp_path):
    _seed_existing_chacha_init_failure(monkeypatch, tmp_path, 65, CharactersRAGDBError("backend exploded"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(65, "65")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "ChaChaNotes DB unavailable"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_waiter_keeps_conflict_init_errors_as_500(monkeypatch, tmp_path):
    _seed_existing_chacha_init_failure(monkeypatch, tmp_path, 66, ConflictError("duplicate bootstrap state"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(66, "66")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "ChaChaNotes DB unavailable"


@pytest.mark.asyncio
async def test_get_or_init_chacha_db_waiter_keeps_input_init_errors_as_500(monkeypatch, tmp_path):
    _seed_existing_chacha_init_failure(monkeypatch, tmp_path, 68, InputError("invalid bootstrap state"))

    try:
        with pytest.raises(HTTPException) as exc_info:
            await deps._get_or_init_db_instance(68, "68")
    finally:
        deps.close_all_chacha_db_instances()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "invalid bootstrap state"
