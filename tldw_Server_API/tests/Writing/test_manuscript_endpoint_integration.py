"""
Integration tests for Manuscript Management endpoints using a real ChaChaNotes DB.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.integration


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "manuscript_integration.db"
    db = CharactersRAGDB(str(db_path), client_id="integration_user")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db():
        return db

    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    monkeypatch.setenv("ROUTES_DISABLE", "media,audio")
    monkeypatch.setenv("SKIP_AUDIO_ROUTERS_IN_TESTS", "1")

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db

    with TestClient(fastapi_app) as c:
        yield c

    fastapi_app.dependency_overrides.clear()


PREFIX = "/api/v1/writing/manuscripts"


def test_full_manuscript_crud(client: TestClient):
    """Full lifecycle: create project -> part -> chapter -> scene -> structure -> update -> search -> delete."""

    # --- Create project ---
    resp = client.post(
        f"{PREFIX}/projects",
        json={"title": "My Novel", "author": "Test Author", "genre": "Fantasy"},
    )
    assert resp.status_code == 201, resp.text
    project = resp.json()
    project_id = project["id"]
    assert project["title"] == "My Novel"
    assert project["word_count"] == 0
    assert project["version"] == 1

    # --- Create part ---
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/parts",
        json={"title": "Part One", "sort_order": 1.0},
    )
    assert resp.status_code == 201, resp.text
    part = resp.json()
    part_id = part["id"]
    assert part["title"] == "Part One"

    # --- Create chapter under the part ---
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": "Chapter One", "part_id": part_id, "sort_order": 1.0},
    )
    assert resp.status_code == 201, resp.text
    chapter = resp.json()
    chapter_id = chapter["id"]
    assert chapter["title"] == "Chapter One"
    assert chapter["part_id"] == part_id

    # --- Create scene with content ---
    scene_text = "The dragon soared above the mountains casting long shadows across the valley below"
    resp = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={
            "title": "Opening Scene",
            "content": {"type": "doc", "content": [{"type": "paragraph", "text": scene_text}]},
            "content_plain": scene_text,
            "sort_order": 1.0,
        },
    )
    assert resp.status_code == 201, resp.text
    scene = resp.json()
    scene_id = scene["id"]
    assert scene["title"] == "Opening Scene"
    assert scene["word_count"] > 0

    # --- Get project structure ---
    resp = client.get(f"{PREFIX}/projects/{project_id}/structure")
    assert resp.status_code == 200, resp.text
    structure = resp.json()
    assert structure["project_id"] == project_id
    assert len(structure["parts"]) == 1
    assert len(structure["parts"][0]["chapters"]) == 1
    assert len(structure["parts"][0]["chapters"][0]["scenes"]) == 1

    # --- Update scene to change content ---
    new_text = "The ancient dragon descended gracefully landing before the trembling knight"
    resp = client.patch(
        f"{PREFIX}/scenes/{scene_id}",
        json={"content_plain": new_text},
        headers={"expected-version": str(scene["version"])},
    )
    assert resp.status_code == 200, resp.text
    updated_scene = resp.json()
    assert updated_scene["version"] == scene["version"] + 1
    # Word count should be recalculated
    expected_wc = len(new_text.split())
    assert updated_scene["word_count"] == expected_wc

    # --- Verify project word count propagated ---
    resp = client.get(f"{PREFIX}/projects/{project_id}")
    assert resp.status_code == 200, resp.text
    project_refreshed = resp.json()
    assert project_refreshed["word_count"] == expected_wc

    # --- Search scenes ---
    resp = client.get(
        f"{PREFIX}/projects/{project_id}/search",
        params={"q": "dragon"},
    )
    assert resp.status_code == 200, resp.text
    search = resp.json()
    assert search["query"] == "dragon"
    assert len(search["results"]) >= 1
    assert search["results"][0]["id"] == scene_id

    # --- Soft delete scene ---
    resp = client.delete(
        f"{PREFIX}/scenes/{scene_id}",
        headers={"expected-version": str(updated_scene["version"])},
    )
    assert resp.status_code == 204

    # --- Verify scene is gone ---
    resp = client.get(f"{PREFIX}/scenes/{scene_id}")
    assert resp.status_code == 404

    # --- Verify project word count dropped to 0 ---
    resp = client.get(f"{PREFIX}/projects/{project_id}")
    assert resp.status_code == 200
    assert resp.json()["word_count"] == 0


def test_deleted_project_hides_descendant_reads_and_search_results(client: TestClient):
    project = client.post(
        f"{PREFIX}/projects",
        json={"title": "Deleted Project Read Boundary"},
    ).json()
    project_id = project["id"]

    chapter = client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": "Chapter One"},
    ).json()
    chapter_id = chapter["id"]

    scene = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={
            "title": "Opening Scene",
            "content_plain": "A quiet dragon watched the valley from above",
        },
    ).json()
    scene_id = scene["id"]

    project_state = client.get(f"{PREFIX}/projects/{project_id}")
    assert project_state.status_code == 200, project_state.text

    delete_resp = client.delete(
        f"{PREFIX}/projects/{project_id}",
        headers={"expected-version": str(project_state.json()["version"])},
    )
    assert delete_resp.status_code == 204, delete_resp.text

    scene_resp = client.get(f"{PREFIX}/scenes/{scene_id}")
    assert scene_resp.status_code == 404, scene_resp.text

    list_resp = client.get(f"{PREFIX}/chapters/{chapter_id}/scenes")
    assert list_resp.status_code == 200, list_resp.text
    assert list_resp.json() == []

    search_resp = client.get(
        f"{PREFIX}/projects/{project_id}/search",
        params={"q": "dragon"},
    )
    assert search_resp.status_code == 200, search_resp.text
    assert search_resp.json()["results"] == []


def test_project_list_and_filter(client: TestClient):
    """Create multiple projects, list all, then filter by status."""

    # Create two draft projects
    resp1 = client.post(
        f"{PREFIX}/projects",
        json={"title": "Draft Novel A", "status": "draft"},
    )
    assert resp1.status_code == 201
    resp2 = client.post(
        f"{PREFIX}/projects",
        json={"title": "Completed Novel B", "status": "complete"},
    )
    assert resp2.status_code == 201

    # List all
    resp = client.get(f"{PREFIX}/projects")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 2
    titles = [p["title"] for p in data["projects"]]
    assert "Draft Novel A" in titles
    assert "Completed Novel B" in titles

    # Filter by status=draft
    resp = client.get(f"{PREFIX}/projects", params={"status": "draft"})
    assert resp.status_code == 200
    data = resp.json()
    statuses = {p["status"] for p in data["projects"]}
    assert statuses == {"draft"}
    assert any(p["title"] == "Draft Novel A" for p in data["projects"])

    # Filter by status=complete
    resp = client.get(f"{PREFIX}/projects", params={"status": "complete"})
    assert resp.status_code == 200
    data = resp.json()
    assert all(p["status"] == "complete" for p in data["projects"])
    assert any(p["title"] == "Completed Novel B" for p in data["projects"])


def test_project_list_includes_canonical_pagination(client: TestClient):
    client.post(f"{PREFIX}/projects", json={"title": "Pagination Novel A", "status": "draft"})
    client.post(f"{PREFIX}/projects", json={"title": "Pagination Novel B", "status": "complete"})

    page1_resp = client.get(f"{PREFIX}/projects", params={"limit": 1, "offset": 0})
    assert page1_resp.status_code == 200, page1_resp.text
    page1 = page1_resp.json()
    assert page1["total"] >= 2
    assert len(page1["projects"]) == 1
    assert page1["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": page1["total"],
        "has_more": True,
        "next_offset": 1,
    }
    assert page1["has_more"] is True
    assert page1["next_offset"] == 1

    page2_resp = client.get(f"{PREFIX}/projects", params={"limit": 1, "offset": 1})
    assert page2_resp.status_code == 200, page2_resp.text
    page2 = page2_resp.json()
    assert len(page2["projects"]) == 1
    assert page2["pagination"]["mode"] == "offset"
    assert page2["pagination"]["limit"] == 1
    assert page2["pagination"]["offset"] == 1
    assert page2["pagination"]["total"] == page2["total"]
    assert page2["has_more"] == page2["pagination"]["has_more"]
    assert page2["next_offset"] == page2["pagination"]["next_offset"]


def test_manuscript_version_history_and_trash_restore(client: TestClient):
    project_resp = client.post(f"{PREFIX}/projects", json={"title": "Versioned Novel"})
    assert project_resp.status_code == 201, project_resp.text
    project_id = project_resp.json()["id"]

    part_resp = client.post(f"{PREFIX}/projects/{project_id}/parts", json={"title": "Book One"})
    assert part_resp.status_code == 201, part_resp.text

    chapter_resp = client.post(f"{PREFIX}/projects/{project_id}/chapters", json={"title": "Chapter One"})
    assert chapter_resp.status_code == 201, chapter_resp.text
    chapter_id = chapter_resp.json()["id"]

    scene_resp = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={"title": "Opening", "content_plain": "First draft"},
    )
    assert scene_resp.status_code == 201, scene_resp.text
    scene = scene_resp.json()
    scene_id = scene["id"]

    version_resp = client.post(
        f"{PREFIX}/scene/{scene_id}/versions",
        json={"label": "First draft"},
    )
    assert version_resp.status_code == 201, version_resp.text
    version = version_resp.json()
    assert version["entity_type"] == "scene"
    assert version["entity_id"] == scene_id
    assert version["version_number"] == 1
    assert version["label"] == "First draft"
    assert version["payload"]["content_plain"] == "First draft"

    updated_resp = client.patch(
        f"{PREFIX}/scenes/{scene_id}",
        json={"content_plain": "Second draft"},
        headers={"expected-version": str(scene["version"])},
    )
    assert updated_resp.status_code == 200, updated_resp.text
    updated_scene = updated_resp.json()

    list_resp = client.get(f"{PREFIX}/scene/{scene_id}/versions")
    assert list_resp.status_code == 200, list_resp.text
    assert list_resp.json()["versions"][0]["version_number"] == 1

    get_resp = client.get(f"{PREFIX}/scene/{scene_id}/versions/1")
    assert get_resp.status_code == 200, get_resp.text
    assert get_resp.json()["payload"]["content_plain"] == "First draft"

    restore_version_resp = client.post(
        f"{PREFIX}/scene/{scene_id}/versions/1/restore",
        headers={"expected-version": str(updated_scene["version"])},
    )
    assert restore_version_resp.status_code == 200, restore_version_resp.text
    restored_scene = restore_version_resp.json()
    assert restored_scene["content_plain"] == "First draft"
    assert restored_scene["version"] == updated_scene["version"] + 1

    delete_resp = client.delete(
        f"{PREFIX}/scenes/{scene_id}",
        headers={"expected-version": str(restored_scene["version"])},
    )
    assert delete_resp.status_code == 204, delete_resp.text

    trash_resp = client.get(f"{PREFIX}/trash", params={"entity_type": "scene"})
    assert trash_resp.status_code == 200, trash_resp.text
    trash = trash_resp.json()
    assert trash["total"] == 1
    assert trash["items"][0]["id"] == scene_id
    assert trash["items"][0]["deleted"] is True

    restore_trash_resp = client.post(
        f"{PREFIX}/trash/scene/{scene_id}/restore",
        headers={"expected-version": str(trash["items"][0]["version"])},
    )
    assert restore_trash_resp.status_code == 200, restore_trash_resp.text
    restored_from_trash = restore_trash_resp.json()
    assert restored_from_trash["id"] == scene_id
    assert restored_from_trash["deleted"] is False
    assert restored_from_trash["version"] == trash["items"][0]["version"] + 1

    final_resp = client.get(f"{PREFIX}/scenes/{scene_id}")
    assert final_resp.status_code == 200, final_resp.text


def test_manuscript_restore_endpoints_reject_invalid_entity_type(client: TestClient):
    create_version_resp = client.post(f"{PREFIX}/invalid/entity-1/versions", json={})
    assert create_version_resp.status_code == 422

    list_versions_resp = client.get(f"{PREFIX}/invalid/entity-1/versions")
    assert list_versions_resp.status_code == 422

    get_version_resp = client.get(f"{PREFIX}/invalid/entity-1/versions/1")
    assert get_version_resp.status_code == 422

    restore_version_resp = client.post(f"{PREFIX}/invalid/entity-1/versions/1/restore")
    assert restore_version_resp.status_code == 422

    list_trash_resp = client.get(f"{PREFIX}/trash", params={"entity_type": "invalid"})
    assert list_trash_resp.status_code == 422

    restore_trash_resp = client.post(f"{PREFIX}/trash/invalid/entity-1/restore")
    assert restore_trash_resp.status_code == 422


def test_manuscript_restore_trash_rejects_deleted_parent_chapter(client: TestClient):
    project_resp = client.post(f"{PREFIX}/projects", json={"title": "Parent Guard Novel"})
    assert project_resp.status_code == 201, project_resp.text
    project_id = project_resp.json()["id"]

    chapter_resp = client.post(f"{PREFIX}/projects/{project_id}/chapters", json={"title": "Deleted Parent"})
    assert chapter_resp.status_code == 201, chapter_resp.text
    chapter = chapter_resp.json()

    scene_resp = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={"title": "Orphan Candidate", "content_plain": "Scene text"},
    )
    assert scene_resp.status_code == 201, scene_resp.text
    scene_id = scene_resp.json()["id"]

    current_chapter_resp = client.get(f"{PREFIX}/chapters/{chapter['id']}")
    assert current_chapter_resp.status_code == 200, current_chapter_resp.text
    current_chapter = current_chapter_resp.json()

    delete_chapter_resp = client.delete(
        f"{PREFIX}/chapters/{chapter['id']}",
        headers={"expected-version": str(current_chapter["version"])},
    )
    assert delete_chapter_resp.status_code == 204, delete_chapter_resp.text

    trash_resp = client.get(f"{PREFIX}/trash", params={"entity_type": "scene"})
    assert trash_resp.status_code == 200, trash_resp.text
    scene_trash = next(item for item in trash_resp.json()["items"] if item["id"] == scene_id)

    restore_trash_resp = client.post(
        f"{PREFIX}/trash/scene/{scene_id}/restore",
        headers={"expected-version": str(scene_trash["version"])},
    )

    assert restore_trash_resp.status_code == 409
    assert client.get(f"{PREFIX}/scenes/{scene_id}").status_code == 404


def test_project_settings_round_trip(client: TestClient):
    """Project settings survive create/get/list responses."""
    settings = {"theme": "dark", "editor": {"mode": "focus"}}
    resp = client.post(
        f"{PREFIX}/projects",
        json={"title": "Configured Novel", "settings": settings},
    )
    assert resp.status_code == 201, resp.text
    project = resp.json()
    project_id = project["id"]

    assert project["settings"] == settings

    resp = client.get(f"{PREFIX}/projects/{project_id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["settings"] == settings

    resp = client.get(f"{PREFIX}/projects")
    assert resp.status_code == 200, resp.text
    listed = next(item for item in resp.json()["projects"] if item["id"] == project_id)
    assert listed["settings"] == settings


def test_optimistic_locking(client: TestClient):
    """Verify that updating with a stale version returns 409."""

    # Create project
    resp = client.post(
        f"{PREFIX}/projects",
        json={"title": "Locking Test Project"},
    )
    assert resp.status_code == 201
    project = resp.json()
    project_id = project["id"]
    v1 = project["version"]

    # Update successfully
    resp = client.patch(
        f"{PREFIX}/projects/{project_id}",
        json={"title": "Updated Title"},
        headers={"expected-version": str(v1)},
    )
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["version"] == v1 + 1

    # Try again with the stale version -> should get 409
    resp = client.patch(
        f"{PREFIX}/projects/{project_id}",
        json={"title": "Stale Update"},
        headers={"expected-version": str(v1)},
    )
    assert resp.status_code == 409


def test_project_patch_null_clears_synopsis_and_settings(client: TestClient):
    create_resp = client.post(
        f"{PREFIX}/projects",
        json={"title": "Null Project", "synopsis": "Filled", "settings": {"theme": "dark"}},
    )
    assert create_resp.status_code == 201, create_resp.text
    project = create_resp.json()

    resp = client.patch(
        f"{PREFIX}/projects/{project['id']}",
        json={"synopsis": None, "settings": None},
        headers={"expected-version": str(project["version"])},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["synopsis"] is None
    assert resp.json()["settings"] == {}


def test_part_patch_null_clears_synopsis(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Part Nulls"}).json()
    part = client.post(
        f"{PREFIX}/projects/{project['id']}/parts",
        json={"title": "Part I", "synopsis": "Filled"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/parts/{part['id']}",
        json={"synopsis": None},
        headers={"expected-version": str(part['version'])},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["synopsis"] is None


def test_chapter_patch_null_clears_part_and_synopsis(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Chapter Nulls"}).json()
    part = client.post(f"{PREFIX}/projects/{project['id']}/parts", json={"title": "Part I"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1", "part_id": part["id"], "synopsis": "Filled"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/chapters/{chapter['id']}",
        json={"part_id": None, "synopsis": None},
        headers={"expected-version": str(chapter["version"])},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["part_id"] is None
    assert resp.json()["synopsis"] is None


def test_scene_patch_null_clears_synopsis(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Scene Nulls"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()
    scene = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={"title": "Scene 1", "synopsis": "Filled", "content_plain": "alpha beta"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={"synopsis": None},
        headers={"expected-version": str(scene['version'])},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["synopsis"] is None


def test_scene_patch_null_content_clears_content_json_only(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Scene Content Null"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()
    rich_content = {"type": "doc", "content": [{"type": "paragraph", "text": "alpha beta"}]}
    scene = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={
            "title": "Scene A",
            "content": rich_content,
            "content_plain": "alpha beta",
            "synopsis": "filled",
        },
    ).json()

    resp = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={"content": None},
        headers={"expected-version": str(scene["version"])},
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["content"] is None
    assert updated["content_json"] is None
    assert updated["content_plain"] == "alpha beta"
    assert updated["title"] == "Scene A"
    assert updated["word_count"] == 2


def test_scene_patch_null_content_plain_clears_plain_and_word_count(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Scene Plain Null"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()
    scene = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={
            "title": "Scene B",
            "content": {"type": "doc", "content": [{"type": "paragraph", "text": "one two three"}]},
            "content_plain": "one two three",
        },
    ).json()

    resp = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={"content_plain": None},
        headers={"expected-version": str(scene["version"])},
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["content_plain"] is None
    assert updated["content"] is None
    assert updated["content_json"] is None
    assert updated["word_count"] == 0


def test_scene_patch_with_content_and_content_plain_preserves_rich_content(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Scene Mixed Patch"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()
    scene = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={
            "title": "Scene C",
            "content": {"type": "doc", "content": [{"type": "paragraph", "text": "old"}]},
            "content_plain": "old text",
        },
    ).json()

    new_content = {"type": "doc", "content": [{"type": "paragraph", "text": "new rich body"}]}
    new_plain = "new plain body"
    resp = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={"content": new_content, "content_plain": new_plain},
        headers={"expected-version": str(scene["version"])},
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["content"] == new_content
    assert updated["content_json"] is not None
    assert updated["content_plain"] == new_plain
    assert updated["word_count"] == len(new_plain.split())


def test_scene_patch_with_content_and_content_plain_round_trips_on_get(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Scene Mixed Roundtrip"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()
    scene = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={
            "title": "Scene D",
            "content": {"type": "doc", "content": [{"type": "paragraph", "text": "old"}]},
            "content_plain": "old text",
        },
    ).json()

    new_content = {"type": "doc", "content": [{"type": "paragraph", "text": "new rich body"}]}
    new_plain = "new plain body"
    patch_resp = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={"content": new_content, "content_plain": new_plain},
        headers={"expected-version": str(scene["version"])},
    )
    assert patch_resp.status_code == 200, patch_resp.text

    get_resp = client.get(f"{PREFIX}/scenes/{scene['id']}")
    assert get_resp.status_code == 200, get_resp.text
    fetched = get_resp.json()
    assert fetched["content"] == new_content
    assert fetched["content_json"] is not None
    assert fetched["content_plain"] == new_plain


def test_create_project_rejects_whitespace_title(client: TestClient):
    resp = client.post(f"{PREFIX}/projects", json={"title": "   "})
    assert resp.status_code == 400, resp.text


def test_create_part_rejects_whitespace_title(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Whitespace Part Create"}).json()
    resp = client.post(
        f"{PREFIX}/projects/{project['id']}/parts",
        json={"title": "   "},
    )
    assert resp.status_code == 400, resp.text


def test_create_chapter_rejects_whitespace_title(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Whitespace Chapter"}).json()
    resp = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "   "},
    )
    assert resp.status_code == 400, resp.text


def test_chapter_patch_rejects_whitespace_title(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Whitespace Patch"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/chapters/{chapter['id']}",
        json={"title": "   "},
        headers={"expected-version": str(chapter["version"])},
    )

    assert resp.status_code == 400, resp.text


def test_create_scene_rejects_whitespace_title_when_provided(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Whitespace Scene Create"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()
    resp = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={"title": "   ", "content_plain": "alpha beta"},
    )
    assert resp.status_code == 400, resp.text


def test_part_patch_rejects_whitespace_title(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Whitespace Part"}).json()
    part = client.post(
        f"{PREFIX}/projects/{project['id']}/parts",
        json={"title": "Part I"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/parts/{part['id']}",
        json={"title": "   "},
        headers={"expected-version": str(part["version"])},
    )
    assert resp.status_code == 400, resp.text


def test_patch_requires_expected_version_header(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Missing Header"}).json()
    resp = client.patch(f"{PREFIX}/projects/{project['id']}", json={"synopsis": None})
    assert resp.status_code == 422, resp.text


def test_project_patch_rejects_empty_payload(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Empty Patch"}).json()
    resp = client.patch(
        f"{PREFIX}/projects/{project['id']}",
        json={},
        headers={"expected-version": str(project["version"])},
    )
    assert resp.status_code == 400, resp.text


def test_scene_patch_rejects_empty_payload(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Scene Empty Patch"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter 1"},
    ).json()
    scene = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={"title": "Scene 1", "content_plain": "alpha beta"},
    ).json()

    resp = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={},
        headers={"expected-version": str(scene["version"])},
    )
    assert resp.status_code == 400, resp.text


def test_reorder_rejects_stale_item_version(client: TestClient):
    project = client.post(f"{PREFIX}/projects", json={"title": "Reorder Conflict"}).json()
    chapter = client.post(
        f"{PREFIX}/projects/{project['id']}/chapters",
        json={"title": "Chapter R"},
    ).json()
    scene = client.post(
        f"{PREFIX}/chapters/{chapter['id']}/scenes",
        json={"title": "Scene 0", "sort_order": 0.0},
    ).json()

    patch_resp = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={"title": "Scene 0 Updated"},
        headers={"expected-version": str(scene["version"])},
    )
    assert patch_resp.status_code == 200, patch_resp.text

    resp = client.post(
        f"{PREFIX}/projects/{project['id']}/reorder",
        json={
            "entity_type": "scenes",
            "items": [{"id": scene["id"], "sort_order": 1.0, "version": scene["version"]}],
        },
    )
    assert resp.status_code == 409, resp.text


def test_reorder(client: TestClient):
    """Create scenes, reorder them, verify new sort order."""

    # Setup: project -> chapter -> 3 scenes
    resp = client.post(
        f"{PREFIX}/projects",
        json={"title": "Reorder Test"},
    )
    assert resp.status_code == 201
    project_id = resp.json()["id"]

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": "Chapter R"},
    )
    assert resp.status_code == 201
    chapter_id = resp.json()["id"]

    scene_ids = []
    for i in range(3):
        resp = client.post(
            f"{PREFIX}/chapters/{chapter_id}/scenes",
            json={"title": f"Scene {i}", "sort_order": float(i)},
        )
        assert resp.status_code == 201
        scene_ids.append(resp.json()["id"])

    # Reorder: reverse the scenes
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/reorder",
        json={
            "entity_type": "scenes",
            "items": [
                {"id": scene_ids[0], "sort_order": 3.0},
                {"id": scene_ids[1], "sort_order": 2.0},
                {"id": scene_ids[2], "sort_order": 1.0},
            ],
        },
    )
    assert resp.status_code == 204

    # Verify new order
    resp = client.get(f"{PREFIX}/chapters/{chapter_id}/scenes")
    assert resp.status_code == 200
    scenes = resp.json()
    assert len(scenes) == 3
    # Should be scene_ids[2], scene_ids[1], scene_ids[0] after ordering
    assert scenes[0]["id"] == scene_ids[2]
    assert scenes[1]["id"] == scene_ids[1]
    assert scenes[2]["id"] == scene_ids[0]


def test_reorder_rejects_explicit_null_version(client: TestClient) -> None:
    """Backward compatibility allows omitting version, but explicit null should still fail."""

    resp = client.post(
        f"{PREFIX}/projects",
        json={"title": "Reorder Null Version"},
    )
    assert resp.status_code == 201
    project_id = resp.json()["id"]

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": "Chapter R"},
    )
    assert resp.status_code == 201
    chapter_id = resp.json()["id"]

    resp = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={"title": "Scene 0", "sort_order": 0.0},
    )
    assert resp.status_code == 201
    scene_id = resp.json()["id"]

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/reorder",
        json={
            "entity_type": "scenes",
            "items": [{"id": scene_id, "sort_order": 1.0, "version": None}],
        },
    )
    assert resp.status_code == 422


def test_not_found(client: TestClient):
    """Fetching nonexistent entities returns 404."""

    resp = client.get(f"{PREFIX}/projects/nonexistent-uuid")
    assert resp.status_code == 404

    resp = client.get(f"{PREFIX}/parts/nonexistent-uuid")
    assert resp.status_code == 404

    resp = client.get(f"{PREFIX}/chapters/nonexistent-uuid")
    assert resp.status_code == 404

    resp = client.get(f"{PREFIX}/scenes/nonexistent-uuid")
    assert resp.status_code == 404
