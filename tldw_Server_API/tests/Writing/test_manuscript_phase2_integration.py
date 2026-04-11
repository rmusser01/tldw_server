"""
Integration tests for Manuscript Phase 2 endpoints (characters, world info, plot
tracking, citations, scene linking, research).
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.integration


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "manuscript_phase2_integration.db"
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


def _create_project(client: TestClient, title: str = "Test Project") -> dict:
    """Helper: create a project and return its JSON."""
    resp = client.post(f"{PREFIX}/projects", json={"title": title})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_chapter(client: TestClient, project_id: str, title: str = "Chapter 1") -> dict:
    """Helper: create a chapter and return its JSON."""
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": title},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _create_scene(client: TestClient, chapter_id: str, title: str = "Scene 1") -> dict:
    """Helper: create a scene and return its JSON."""
    resp = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={"title": title, "content_plain": "Some text here"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


# -----------------------------------------------------------------------
# Characters
# -----------------------------------------------------------------------


def test_character_crud(client: TestClient):
    """Create, list, get, update, and delete a character."""
    project = _create_project(client, "Character Test")
    project_id = project["id"]

    # Create character
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={
            "name": "Aldric",
            "role": "protagonist",
            "full_name": "Aldric of Thornwall",
            "age": "32",
            "motivation": "Avenge his family",
            "custom_fields": {"hair_color": "black"},
        },
    )
    assert resp.status_code == 201, resp.text
    char = resp.json()
    char_id = char["id"]
    assert char["name"] == "Aldric"
    assert char["role"] == "protagonist"
    assert char["full_name"] == "Aldric of Thornwall"
    assert char["custom_fields"]["hair_color"] == "black"
    assert char["version"] == 1

    # List characters
    resp = client.get(f"{PREFIX}/projects/{project_id}/characters")
    assert resp.status_code == 200, resp.text
    chars = resp.json()
    assert len(chars) == 1
    assert chars[0]["id"] == char_id

    # List with role filter
    resp = client.get(
        f"{PREFIX}/projects/{project_id}/characters",
        params={"role": "antagonist"},
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 0

    # Get by ID
    resp = client.get(f"{PREFIX}/characters/{char_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Aldric"

    # Update
    resp = client.patch(
        f"{PREFIX}/characters/{char_id}",
        json={"motivation": "Protect his kingdom", "custom_fields": {"hair_color": "silver"}},
        headers={"expected-version": "1"},
    )
    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["motivation"] == "Protect his kingdom"
    assert updated["custom_fields"]["hair_color"] == "silver"
    assert updated["version"] == 2

    # Delete
    resp = client.delete(
        f"{PREFIX}/characters/{char_id}",
        headers={"expected-version": "2"},
    )
    assert resp.status_code == 204

    # Verify deleted
    resp = client.get(f"{PREFIX}/characters/{char_id}")
    assert resp.status_code == 404


def test_character_create_rejects_whitespace_name(client: TestClient):
    project = _create_project(client, "Character Create Validation")
    project_id = project["id"]

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={"name": "   ", "role": "supporting"},
    )

    assert resp.status_code == 400, resp.text
    assert "name" in resp.json()["detail"].lower()


def test_character_patch_null_custom_fields_resets_to_empty_object(client: TestClient):
    project = _create_project(client, "Character Patch Validation")
    project_id = project["id"]

    create_resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={"name": "Aldric", "role": "protagonist", "custom_fields": {"hair_color": "black"}},
    )
    assert create_resp.status_code == 201, create_resp.text
    char_id = create_resp.json()["id"]

    resp = client.patch(
        f"{PREFIX}/characters/{char_id}",
        json={"custom_fields": None},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["custom_fields"] == {}
    assert updated["version"] == 2


def test_character_patch_rejects_empty_payload(client: TestClient):
    project = _create_project(client, "Character Empty Patch")
    project_id = project["id"]

    create_resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={"name": "Aldric", "role": "protagonist"},
    )
    assert create_resp.status_code == 201, create_resp.text
    char_id = create_resp.json()["id"]

    resp = client.patch(
        f"{PREFIX}/characters/{char_id}",
        json={},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 400, resp.text
    assert "no fields" in resp.json()["detail"].lower()


# -----------------------------------------------------------------------
# Character Relationships
# -----------------------------------------------------------------------


def test_character_relationships(client: TestClient):
    """Create two characters, link them, list, and delete the relationship."""
    project = _create_project(client, "Relationship Test")
    project_id = project["id"]

    # Create two characters
    resp1 = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={"name": "Alice", "role": "protagonist"},
    )
    assert resp1.status_code == 201
    alice_id = resp1.json()["id"]

    resp2 = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={"name": "Bob", "role": "supporting"},
    )
    assert resp2.status_code == 201
    bob_id = resp2.json()["id"]

    # Create relationship
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters/relationships",
        json={
            "from_character_id": alice_id,
            "to_character_id": bob_id,
            "relationship_type": "sibling",
            "description": "Twin siblings",
            "bidirectional": True,
        },
    )
    assert resp.status_code == 201, resp.text
    rel = resp.json()
    rel_id = rel["id"]
    assert rel["from_character_id"] == alice_id
    assert rel["to_character_id"] == bob_id
    assert rel["relationship_type"] == "sibling"

    # List relationships
    resp = client.get(f"{PREFIX}/projects/{project_id}/characters/relationships")
    assert resp.status_code == 200
    rels = resp.json()
    assert len(rels) == 1
    assert rels[0]["id"] == rel_id

    # Delete relationship
    resp = client.delete(
        f"{PREFIX}/characters/relationships/{rel_id}",
        headers={"expected-version": "1"},
    )
    assert resp.status_code == 204

    # Verify deleted
    resp = client.get(f"{PREFIX}/projects/{project_id}/characters/relationships")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


# -----------------------------------------------------------------------
# World Info
# -----------------------------------------------------------------------


def test_world_info_crud(client: TestClient):
    """Create, list (with kind filter), get, update, and delete world info."""
    project = _create_project(client, "World Info Test")
    project_id = project["id"]

    # Create location
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/world-info",
        json={
            "kind": "location",
            "name": "Thornwall Keep",
            "description": "An ancient fortress",
            "properties": {"population": 500},
            "tags": ["fortress", "ancient"],
        },
    )
    assert resp.status_code == 201, resp.text
    item = resp.json()
    item_id = item["id"]
    assert item["kind"] == "location"
    assert item["name"] == "Thornwall Keep"
    assert item["properties"]["population"] == 500
    assert "fortress" in item["tags"]
    assert item["version"] == 1

    # Create faction
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/world-info",
        json={"kind": "faction", "name": "The Order of Dawn"},
    )
    assert resp.status_code == 201

    # List all
    resp = client.get(f"{PREFIX}/projects/{project_id}/world-info")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    # List by kind=location
    resp = client.get(
        f"{PREFIX}/projects/{project_id}/world-info",
        params={"kind": "location"},
    )
    assert resp.status_code == 200
    filtered = resp.json()
    assert len(filtered) == 1
    assert filtered[0]["kind"] == "location"

    # Get by ID
    resp = client.get(f"{PREFIX}/world-info/{item_id}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Thornwall Keep"

    # Update
    resp = client.patch(
        f"{PREFIX}/world-info/{item_id}",
        json={"description": "A crumbling ancient fortress", "tags": ["fortress", "ruin"]},
        headers={"expected-version": "1"},
    )
    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["description"] == "A crumbling ancient fortress"
    assert "ruin" in updated["tags"]
    assert updated["version"] == 2

    # Delete
    resp = client.delete(
        f"{PREFIX}/world-info/{item_id}",
        headers={"expected-version": "2"},
    )
    assert resp.status_code == 204

    # Verify deleted
    resp = client.get(f"{PREFIX}/world-info/{item_id}")
    assert resp.status_code == 404


def test_world_info_create_rejects_whitespace_name(client: TestClient):
    project = _create_project(client, "World Info Create Validation")
    project_id = project["id"]

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/world-info",
        json={"kind": "location", "name": "   "},
    )

    assert resp.status_code == 400, resp.text
    assert "name" in resp.json()["detail"].lower()


def test_world_info_patch_null_parent_and_collections_reset(client: TestClient):
    project = _create_project(client, "World Info Patch Validation")
    project_id = project["id"]

    parent_resp = client.post(
        f"{PREFIX}/projects/{project_id}/world-info",
        json={"kind": "location", "name": "Parent", "properties": {"era": "ancient"}, "tags": ["root"]},
    )
    assert parent_resp.status_code == 201, parent_resp.text
    parent_id = parent_resp.json()["id"]

    create_resp = client.post(
        f"{PREFIX}/projects/{project_id}/world-info",
        json={
            "kind": "location",
            "name": "Child",
            "parent_id": parent_id,
            "properties": {"population": 500},
            "tags": ["keep"],
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    item_id = create_resp.json()["id"]

    resp = client.patch(
        f"{PREFIX}/world-info/{item_id}",
        json={"parent_id": None, "properties": None, "tags": None},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["parent_id"] is None
    assert updated["properties"] == {}
    assert updated["tags"] == []
    assert updated["version"] == 2


# -----------------------------------------------------------------------
# Plot Tracking
# -----------------------------------------------------------------------


def test_plot_tracking(client: TestClient):
    """Create plot line, add events, create plot holes, and list them."""
    project = _create_project(client, "Plot Test")
    project_id = project["id"]

    # Create plot line
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-lines",
        json={
            "title": "Main Quest",
            "description": "The hero's journey",
            "status": "active",
            "color": "#FF0000",
        },
    )
    assert resp.status_code == 201, resp.text
    plot_line = resp.json()
    plot_line_id = plot_line["id"]
    assert plot_line["title"] == "Main Quest"
    assert plot_line["status"] == "active"

    # List plot lines
    resp = client.get(f"{PREFIX}/projects/{project_id}/plot-lines")
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    # Update plot line
    resp = client.patch(
        f"{PREFIX}/plot-lines/{plot_line_id}",
        json={"status": "resolved"},
        headers={"expected-version": "1"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"

    # Create plot event
    resp = client.post(
        f"{PREFIX}/plot-lines/{plot_line_id}/events",
        json={
            "title": "Dragon Appears",
            "description": "The dragon descends on the village",
            "event_type": "conflict",
        },
    )
    assert resp.status_code == 201, resp.text
    event = resp.json()
    assert event["title"] == "Dragon Appears"
    assert event["event_type"] == "conflict"
    assert event["plot_line_id"] == plot_line_id

    # List plot events
    resp = client.get(f"{PREFIX}/plot-lines/{plot_line_id}/events")
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) == 1
    assert events[0]["title"] == "Dragon Appears"

    # Create plot hole
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-holes",
        json={
            "title": "Timeline inconsistency",
            "description": "Chapter 3 says 2 days but chapter 5 says 1 week",
            "severity": "high",
            "plot_line_id": plot_line_id,
        },
    )
    assert resp.status_code == 201, resp.text
    hole = resp.json()
    hole_id = hole["id"]
    assert hole["title"] == "Timeline inconsistency"
    assert hole["severity"] == "high"
    assert hole["status"] == "open"

    # List plot holes
    resp = client.get(f"{PREFIX}/projects/{project_id}/plot-holes")
    assert resp.status_code == 200
    holes = resp.json()
    assert len(holes) == 1

    # Update plot hole (resolve)
    resp = client.patch(
        f"{PREFIX}/plot-holes/{hole_id}",
        json={"status": "resolved", "resolution": "Changed chapter 5 to say 2 days"},
        headers={"expected-version": "1"},
    )
    assert resp.status_code == 200, resp.text
    resolved = resp.json()
    assert resolved["status"] == "resolved"
    assert resolved["resolution"] == "Changed chapter 5 to say 2 days"

    # Delete plot line (cascading is not tested here, just the line)
    resp = client.delete(
        f"{PREFIX}/plot-lines/{plot_line_id}",
        headers={"expected-version": "2"},
    )
    assert resp.status_code == 204

    # Delete plot hole
    resp = client.delete(
        f"{PREFIX}/plot-holes/{hole_id}",
        headers={"expected-version": "2"},
    )
    assert resp.status_code == 204


def test_plot_line_create_rejects_whitespace_title(client: TestClient):
    project = _create_project(client, "Plot Line Create Validation")
    project_id = project["id"]

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-lines",
        json={"title": "   "},
    )

    assert resp.status_code == 400, resp.text
    assert "title" in resp.json()["detail"].lower()


def test_plot_line_patch_null_description(client: TestClient):
    project = _create_project(client, "Plot Line Patch Validation")
    project_id = project["id"]

    create_resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-lines",
        json={"title": "Main Quest", "description": "Initial"},
    )
    assert create_resp.status_code == 201, create_resp.text
    plot_line_id = create_resp.json()["id"]

    resp = client.patch(
        f"{PREFIX}/plot-lines/{plot_line_id}",
        json={"description": None},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["description"] is None
    assert updated["version"] == 2


def test_plot_event_create_rejects_whitespace_title(client: TestClient):
    project = _create_project(client, "Plot Event Create Validation")
    project_id = project["id"]

    plot_line_resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-lines",
        json={"title": "Main Quest"},
    )
    assert plot_line_resp.status_code == 201, plot_line_resp.text
    plot_line_id = plot_line_resp.json()["id"]

    resp = client.post(
        f"{PREFIX}/plot-lines/{plot_line_id}/events",
        json={"title": "   ", "event_type": "plot"},
    )

    assert resp.status_code == 400, resp.text
    assert "title" in resp.json()["detail"].lower()


def test_plot_event_patch_allows_nullable_scene_links(client: TestClient):
    project = _create_project(client, "Plot Event Patch Validation")
    project_id = project["id"]
    chapter = _create_chapter(client, project_id)
    scene = _create_scene(client, chapter["id"])

    plot_line_resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-lines",
        json={"title": "Main Quest"},
    )
    assert plot_line_resp.status_code == 201, plot_line_resp.text
    plot_line_id = plot_line_resp.json()["id"]

    create_resp = client.post(
        f"{PREFIX}/plot-lines/{plot_line_id}/events",
        json={
            "title": "Dragon Appears",
            "scene_id": scene["id"],
            "chapter_id": chapter["id"],
            "event_type": "conflict",
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    event_id = create_resp.json()["id"]

    resp = client.patch(
        f"{PREFIX}/plot-events/{event_id}",
        json={"scene_id": None, "chapter_id": None},
        headers={"expected-version": "1"},
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["scene_id"] is None
    assert updated["chapter_id"] is None
    assert updated["version"] == 2


def test_plot_hole_create_rejects_whitespace_title(client: TestClient):
    project = _create_project(client, "Plot Hole Create Validation")
    project_id = project["id"]

    resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-holes",
        json={"title": "   "},
    )

    assert resp.status_code == 400, resp.text
    assert "title" in resp.json()["detail"].lower()


def test_plot_hole_patch_null_resolution(client: TestClient):
    project = _create_project(client, "Plot Hole Patch Validation")
    project_id = project["id"]

    create_resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-holes",
        json={"title": "Timeline gap", "severity": "high"},
    )
    assert create_resp.status_code == 201, create_resp.text
    hole_id = create_resp.json()["id"]

    first_patch = client.patch(
        f"{PREFIX}/plot-holes/{hole_id}",
        json={"status": "resolved", "resolution": "Temporary fix"},
        headers={"expected-version": "1"},
    )
    assert first_patch.status_code == 200, first_patch.text
    assert first_patch.json()["resolution"] == "Temporary fix"

    resp = client.patch(
        f"{PREFIX}/plot-holes/{hole_id}",
        json={"status": "resolved", "resolution": None},
        headers={"expected-version": "2"},
    )

    assert resp.status_code == 200, resp.text
    updated = resp.json()
    assert updated["resolution"] is None
    assert updated["status"] == "resolved"
    assert updated["version"] == 3


def test_deleted_project_hides_project_scoped_lists(client: TestClient):
    project = _create_project(client, "Ghost Project")
    project_id = project["id"]

    part_resp = client.post(f"{PREFIX}/projects/{project_id}/parts", json={"title": "Part I"})
    assert part_resp.status_code == 201, part_resp.text
    part = part_resp.json()

    chapter_resp = client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": "Chapter 1", "part_id": part["id"]},
    )
    assert chapter_resp.status_code == 201, chapter_resp.text

    char_resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={"name": "Aldric", "role": "protagonist"},
    )
    assert char_resp.status_code == 201, char_resp.text
    char = char_resp.json()

    other_resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={"name": "Brin", "role": "supporting"},
    )
    assert other_resp.status_code == 201, other_resp.text
    other = other_resp.json()

    rel_resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters/relationships",
        json={
            "from_character_id": char["id"],
            "to_character_id": other["id"],
            "relationship_type": "ally",
            "bidirectional": True,
        },
    )
    assert rel_resp.status_code == 201, rel_resp.text

    world_resp = client.post(
        f"{PREFIX}/projects/{project_id}/world-info",
        json={"kind": "location", "name": "Keep"},
    )
    assert world_resp.status_code == 201, world_resp.text

    plot_line_resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-lines",
        json={"title": "Main Quest"},
    )
    assert plot_line_resp.status_code == 201, plot_line_resp.text

    plot_hole_resp = client.post(
        f"{PREFIX}/projects/{project_id}/plot-holes",
        json={"title": "Gap", "description": "Missing", "severity": "high"},
    )
    assert plot_hole_resp.status_code == 201, plot_hole_resp.text

    delete_resp = client.delete(f"{PREFIX}/projects/{project_id}", headers={"expected-version": "1"})
    assert delete_resp.status_code == 204

    assert client.get(f"{PREFIX}/projects/{project_id}/parts").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/chapters").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/characters").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/characters/relationships").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/world-info").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/plot-lines").json() == []
    assert client.get(f"{PREFIX}/projects/{project_id}/plot-holes").json() == []


# -----------------------------------------------------------------------
# Scene-Character Linking
# -----------------------------------------------------------------------


def test_scene_character_linking(client: TestClient):
    """Link a character to a scene, list, verify join data, and unlink."""
    project = _create_project(client, "Linking Test")
    project_id = project["id"]
    chapter = _create_chapter(client, project_id)
    scene = _create_scene(client, chapter["id"])
    scene_id = scene["id"]

    # Create character
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/characters",
        json={"name": "Elara", "role": "protagonist"},
    )
    assert resp.status_code == 201
    char_id = resp.json()["id"]

    # Link character to scene
    resp = client.post(
        f"{PREFIX}/scenes/{scene_id}/characters",
        json={"character_id": char_id, "is_pov": True},
    )
    assert resp.status_code == 201, resp.text
    links = resp.json()
    assert len(links) >= 1
    link = next(lk for lk in links if lk["character_id"] == char_id)
    assert link["scene_id"] == scene_id
    assert link["name"] == "Elara"
    assert link["role"] == "protagonist"
    assert link["is_pov"] is True

    # List linked characters
    resp = client.get(f"{PREFIX}/scenes/{scene_id}/characters")
    assert resp.status_code == 200
    listed = resp.json()
    assert len(listed) == 1
    assert listed[0]["character_id"] == char_id

    # Unlink
    resp = client.delete(f"{PREFIX}/scenes/{scene_id}/characters/{char_id}")
    assert resp.status_code == 204

    # Verify unlinked
    resp = client.get(f"{PREFIX}/scenes/{scene_id}/characters")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


# -----------------------------------------------------------------------
# Scene-World Info Linking
# -----------------------------------------------------------------------


def test_scene_world_info_linking(client: TestClient):
    """Link world info to a scene, list, verify join data, and unlink."""
    project = _create_project(client, "World Link Test")
    project_id = project["id"]
    chapter = _create_chapter(client, project_id)
    scene = _create_scene(client, chapter["id"])
    scene_id = scene["id"]

    # Create world info
    resp = client.post(
        f"{PREFIX}/projects/{project_id}/world-info",
        json={"kind": "location", "name": "Dragon's Lair"},
    )
    assert resp.status_code == 201
    wi_id = resp.json()["id"]

    # Link to scene
    resp = client.post(
        f"{PREFIX}/scenes/{scene_id}/world-info",
        json={"world_info_id": wi_id},
    )
    assert resp.status_code == 201, resp.text
    links = resp.json()
    assert len(links) >= 1
    link = next(lk for lk in links if lk["world_info_id"] == wi_id)
    assert link["scene_id"] == scene_id
    assert link["name"] == "Dragon's Lair"
    assert link["kind"] == "location"

    # List linked world info
    resp = client.get(f"{PREFIX}/scenes/{scene_id}/world-info")
    assert resp.status_code == 200
    listed = resp.json()
    assert len(listed) == 1
    assert listed[0]["world_info_id"] == wi_id

    # Unlink
    resp = client.delete(f"{PREFIX}/scenes/{scene_id}/world-info/{wi_id}")
    assert resp.status_code == 204

    # Verify unlinked
    resp = client.get(f"{PREFIX}/scenes/{scene_id}/world-info")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


# -----------------------------------------------------------------------
# Citations
# -----------------------------------------------------------------------


def test_citations(client: TestClient):
    """Create a citation for a scene, list, and delete."""
    project = _create_project(client, "Citation Test")
    project_id = project["id"]
    chapter = _create_chapter(client, project_id)
    scene = _create_scene(client, chapter["id"])
    scene_id = scene["id"]

    # Create citation
    resp = client.post(
        f"{PREFIX}/scenes/{scene_id}/citations",
        json={
            "source_type": "rag",
            "source_title": "Medieval Fortress Architecture",
            "excerpt": "Fortresses were built with concentric walls...",
            "query_used": "fortress design medieval",
            "anchor_offset": 42,
        },
    )
    assert resp.status_code == 201, resp.text
    citation = resp.json()
    citation_id = citation["id"]
    assert citation["source_type"] == "rag"
    assert citation["source_title"] == "Medieval Fortress Architecture"
    assert citation["scene_id"] == scene_id
    assert citation["project_id"] == project_id
    assert citation["anchor_offset"] == 42
    assert citation["version"] == 1

    # List citations
    resp = client.get(f"{PREFIX}/scenes/{scene_id}/citations")
    assert resp.status_code == 200
    cits = resp.json()
    assert len(cits) == 1
    assert cits[0]["id"] == citation_id

    # Delete citation
    resp = client.delete(
        f"{PREFIX}/citations/{citation_id}",
        headers={"expected-version": "1"},
    )
    assert resp.status_code == 204

    # Verify deleted
    resp = client.get(f"{PREFIX}/scenes/{scene_id}/citations")
    assert resp.status_code == 200
    assert len(resp.json()) == 0


# -----------------------------------------------------------------------
# Research stub
# -----------------------------------------------------------------------


def test_research_stub(client: TestClient):
    """Verify the research stub endpoint returns the expected structure."""
    project = _create_project(client, "Research Test")
    project_id = project["id"]
    chapter = _create_chapter(client, project_id)
    scene = _create_scene(client, chapter["id"])
    scene_id = scene["id"]

    resp = client.post(
        f"{PREFIX}/scenes/{scene_id}/research",
        json={"query": "medieval castle defenses", "top_k": 10},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["query"] == "medieval castle defenses"
    assert isinstance(data["results"], list)
    # Stub returns empty results
    assert len(data["results"]) == 0


def test_research_nonexistent_scene(client: TestClient):
    """Research on a nonexistent scene should return 404."""
    resp = client.post(
        f"{PREFIX}/scenes/nonexistent-uuid/research",
        json={"query": "test"},
    )
    assert resp.status_code == 404
