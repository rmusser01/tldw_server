import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from importlib import import_module, reload

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.unit


@pytest.fixture()
def client_with_user(monkeypatch):
    async def override_user():
        return User(id=321, username="bulkuser", email=None, is_active=True)

    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    base_dir = Path.cwd() / "Databases" / "test_user_dbs"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    app = None
    try:
        mod = import_module("tldw_Server_API.app.main")
        mod = reload(mod)
        app = getattr(mod, "app")
        app.dependency_overrides[get_request_user] = override_user
        with TestClient(app) as client:
            yield client
    finally:
        if app is not None:
            app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def _insert_item(cdb: CollectionsDatabase, *, url: str, title: str, status: str, tags: list[str]) -> int:
    digest = hashlib.sha256(title.encode("utf-8")).hexdigest()
    row = cdb.upsert_content_item(
        origin="reading",
        origin_type="manual",
        origin_id=None,
        url=url,
        canonical_url=url,
        domain="example.com",
        title=title,
        summary="Summary",
        notes=None,
        content_hash=digest,
        word_count=3,
        published_at=datetime.utcnow().isoformat(),
        status=status,
        favorite=False,
        metadata={"seed": True},
        media_id=None,
        job_id=None,
        run_id=None,
        source_id=None,
        read_at=None,
        tags=tags,
    )
    return int(row.id)


def test_bulk_tags_and_status(client_with_user):
    client = client_with_user
    cdb = CollectionsDatabase.for_user(user_id=321)
    item_a = _insert_item(cdb, url="https://example.com/a", title="Alpha", status="saved", tags=["alpha"])
    item_b = _insert_item(cdb, url="https://example.com/b", title="Beta", status="saved", tags=["beta"])

    r = client.post(
        "/api/v1/items/bulk",
        json={"item_ids": [item_a, item_b], "action": "add_tags", "tags": ["gamma"]},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["succeeded"] == 2

    row_a = cdb.get_content_item(item_a)
    assert "gamma" in row_a.tags
    row_b = cdb.get_content_item(item_b)
    assert "gamma" in row_b.tags

    r = client.post(
        "/api/v1/items/bulk",
        json={"item_ids": [item_a], "action": "set_status", "status": "read"},
    )
    assert r.status_code == 200, r.text
    row_a = cdb.get_content_item(item_a)
    assert row_a.status == "read"
    assert row_a.read_at is not None


def test_bulk_tag_actions_match_by_identity_and_keep_display_casing(client_with_user):
    client = client_with_user
    cdb = CollectionsDatabase.for_user(user_id=321)
    item_id = _insert_item(cdb, url="https://example.com/casing", title="Casing", status="saved", tags=["Straße"])

    r = client.post(
        "/api/v1/items/bulk",
        json={"item_ids": [item_id], "action": "add_tags", "tags": ["STRASSE", "Machine Learning"]},
    )
    assert r.status_code == 200, r.text
    assert cdb.get_content_item(item_id).tags == ["Machine Learning", "Straße"]

    r = client.post(
        "/api/v1/items/bulk",
        json={"item_ids": [item_id], "action": "remove_tags", "tags": ["strasse"]},
    )
    assert r.status_code == 200, r.text
    assert cdb.get_content_item(item_id).tags == ["Machine Learning"]


def test_bulk_delete_soft_and_hard(client_with_user):
    client = client_with_user
    cdb = CollectionsDatabase.for_user(user_id=321)
    item_a = _insert_item(cdb, url="https://example.com/c", title="Gamma", status="saved", tags=["gamma"])
    item_b = _insert_item(cdb, url="https://example.com/d", title="Delta", status="saved", tags=["delta"])

    r = client.post(
        "/api/v1/items/bulk",
        json={"item_ids": [item_a], "action": "delete"},
    )
    assert r.status_code == 200, r.text
    row_a = cdb.get_content_item(item_a)
    assert row_a.status == "archived"

    r = client.post(
        "/api/v1/items/bulk",
        json={"item_ids": [item_b], "action": "delete", "hard": True},
    )
    assert r.status_code == 200, r.text
    with pytest.raises(KeyError):
        cdb.get_content_item(item_b)


def test_reading_bulk_alias(client_with_user):
    client = client_with_user
    cdb = CollectionsDatabase.for_user(user_id=321)
    item_id = _insert_item(cdb, url="https://example.com/alias", title="Alias", status="saved", tags=["alpha"])

    r = client.post(
        "/api/v1/reading/items/bulk",
        json={"item_ids": [item_id], "action": "set_status", "status": "read"},
    )
    assert r.status_code == 200, r.text
    row = cdb.get_content_item(item_id)
    assert row.status == "read"
    assert row.read_at is not None
