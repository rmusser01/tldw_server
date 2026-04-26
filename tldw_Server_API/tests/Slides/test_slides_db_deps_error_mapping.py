import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import Slides_DB_Deps as deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Slides.slides_db import SchemaError, SlidesDatabaseError


def _user() -> User:
    return User(id=42, username="slides-user")


def test_get_slides_db_maps_schema_error(monkeypatch):
    deps.cleanup_slides_db_cache()

    def fail_create(*args, **kwargs):
        raise SchemaError("schema exploded")

    monkeypatch.setattr(deps, "SlidesDatabase", fail_create)

    with pytest.raises(HTTPException) as exc_info:
        deps.get_slides_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Database schema error"


def test_get_slides_db_maps_base_database_error(monkeypatch):
    deps.cleanup_slides_db_cache()

    def fail_create(*args, **kwargs):
        raise SlidesDatabaseError("backend exploded")

    monkeypatch.setattr(deps, "SlidesDatabase", fail_create)

    with pytest.raises(HTTPException) as exc_info:
        deps.get_slides_db_for_user(current_user=_user())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Slides DB unavailable"
