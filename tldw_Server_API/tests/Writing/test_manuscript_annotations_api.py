"""Integration tests for manuscript annotation API endpoints."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper


pytestmark = pytest.mark.integration

PREFIX = "/api/v1/writing/manuscripts"


@pytest.fixture()
def api_context(tmp_path, monkeypatch):
    db_path = tmp_path / "manuscript_annotations_api.db"
    db = CharactersRAGDB(str(db_path), client_id="annotations_api_user")

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

    with TestClient(fastapi_app) as client:
        yield client, db

    fastapi_app.dependency_overrides.clear()


def _create_project(client: TestClient, title: str = "Annotation Project") -> dict:
    response = client.post(f"{PREFIX}/projects", json={"title": title})
    assert response.status_code == 201, response.text
    return response.json()


def _create_chapter(client: TestClient, project_id: str, title: str = "Chapter 1") -> dict:
    response = client.post(
        f"{PREFIX}/projects/{project_id}/chapters",
        json={"title": title},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_scene(client: TestClient, chapter_id: str, content_plain: str) -> dict:
    response = client.post(
        f"{PREFIX}/chapters/{chapter_id}/scenes",
        json={"title": "Scene 1", "content_plain": content_plain},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_scene_manuscript(
    client: TestClient,
    content_plain: str = "Alpha beta gamma delta.",
) -> tuple[dict, dict, dict]:
    project = _create_project(client)
    chapter = _create_chapter(client, project["id"])
    scene = _create_scene(client, chapter["id"], content_plain)
    return project, chapter, scene


def test_post_annotations_creates_manual_scene_range_annotation(api_context):
    client, _db = api_context
    _project, _chapter, scene = _create_scene_manuscript(client)
    selected_text = "beta"
    start = scene["content_plain"].index(selected_text)
    end = start + len(selected_text)

    response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "scene",
            "target_id": scene["id"],
            "category": "clarity",
            "body": "Clarify the image here.",
            "tags": ["line"],
            "suggested_fix": "Use a concrete detail.",
            "metadata": {"review_id": "manual-1"},
            "scene_version": scene["version"],
            "start": start,
            "end": end,
            "selected_text": selected_text,
        },
    )

    assert response.status_code == 201, response.text
    annotation = response.json()
    assert annotation["target_type"] == "scene"
    assert annotation["target_id"] == scene["id"]
    assert annotation["category"] == "clarity"
    assert annotation["source"] == "user"
    assert annotation["body"] == "Clarify the image here."
    assert annotation["tags"] == ["line"]
    assert annotation["metadata"] == {"review_id": "manual-1"}
    assert annotation["anchor_start"] == start
    assert annotation["anchor_end"] == end
    assert annotation["selected_text"] == selected_text
    assert annotation["anchor_status"] == "attached"
    assert annotation["derived_start"] == start
    assert annotation["derived_end"] == end
    assert annotation["version"] == 1


def test_get_annotation_returns_derived_anchor_state(api_context):
    client, _db = api_context
    _project, _chapter, scene = _create_scene_manuscript(client)
    selected_text = "gamma"
    start = scene["content_plain"].index(selected_text)
    create_response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "scene",
            "target_id": scene["id"],
            "category": "pacing",
            "body": "Check the emphasis.",
            "scene_version": scene["version"],
            "start": start,
            "end": start + len(selected_text),
            "selected_text": selected_text,
        },
    )
    assert create_response.status_code == 201, create_response.text
    annotation_id = create_response.json()["id"]

    revised = "Intro. Alpha beta gamma delta."
    update_response = client.patch(
        f"{PREFIX}/scenes/{scene['id']}",
        json={"content_plain": revised},
        headers={"expected-version": str(scene["version"])},
    )
    assert update_response.status_code == 200, update_response.text

    response = client.get(f"{PREFIX}/annotations/{annotation_id}")

    assert response.status_code == 200, response.text
    annotation = response.json()
    assert annotation["anchor_status"] == "reattached"
    assert annotation["derived_start"] == revised.index(selected_text)
    assert annotation["derived_end"] == revised.index(selected_text) + len(selected_text)
    assert annotation["scene_level"] is False


def test_project_annotations_list_returns_pagination_aliases(api_context):
    client, _db = api_context
    project = _create_project(client, title="Paginated Annotations")
    for body in ("First note.", "Second note."):
        response = client.post(
            f"{PREFIX}/annotations",
            json={
                "target_type": "project",
                "target_id": project["id"],
                "category": "other",
                "body": body,
            },
        )
        assert response.status_code == 201, response.text

    response = client.get(
        f"{PREFIX}/projects/{project['id']}/annotations",
        params={"limit": 1, "offset": 0},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["annotations"]) == 1
    assert payload["total"] == 2
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert payload["has_more"] is True
    assert payload["next_offset"] == 1
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": 2,
        "has_more": True,
        "next_offset": 1,
    }


def test_patch_annotation_requires_expected_version_header(api_context):
    client, _db = api_context
    project = _create_project(client, title="Patch Header")
    create_response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "project",
            "target_id": project["id"],
            "category": "other",
            "body": "Original note.",
        },
    )
    assert create_response.status_code == 201, create_response.text

    response = client.patch(
        f"{PREFIX}/annotations/{create_response.json()['id']}",
        json={"body": "Updated note."},
    )

    assert response.status_code == 422, response.text


def test_delete_annotation_soft_deletes_with_expected_version(api_context):
    client, _db = api_context
    project = _create_project(client, title="Delete Annotation")
    create_response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "project",
            "target_id": project["id"],
            "category": "other",
            "body": "Temporary note.",
        },
    )
    assert create_response.status_code == 201, create_response.text
    annotation = create_response.json()

    delete_response = client.delete(
        f"{PREFIX}/annotations/{annotation['id']}",
        headers={"expected-version": str(annotation["version"])},
    )

    assert delete_response.status_code == 204, delete_response.text
    get_response = client.get(f"{PREFIX}/annotations/{annotation['id']}")
    assert get_response.status_code == 404, get_response.text


def test_broad_anchor_status_filter_is_rejected_unless_bounded(api_context):
    client, _db = api_context
    project = _create_project(client, title="Broad Anchor Filter")
    create_response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "project",
            "target_id": project["id"],
            "category": "other",
            "body": "Project-level note.",
        },
    )
    assert create_response.status_code == 201, create_response.text

    broad_response = client.get(
        f"{PREFIX}/projects/{project['id']}/annotations",
        params={"anchor_status": "scene_level"},
    )

    assert broad_response.status_code == 400, broad_response.text
    assert "anchor_status" in broad_response.json()["detail"]


def test_soft_deleted_targets_return_not_found(api_context):
    client, _db = api_context
    _project, _chapter, scene = _create_scene_manuscript(client)

    delete_response = client.delete(
        f"{PREFIX}/scenes/{scene['id']}",
        headers={"expected-version": str(scene["version"])},
    )
    assert delete_response.status_code == 204, delete_response.text

    response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "scene",
            "target_id": scene["id"],
            "category": "clarity",
            "body": "Annotate deleted scene.",
            "scene_version": scene["version"],
            "start": 0,
            "end": 5,
            "selected_text": scene["content_plain"][:5],
        },
    )

    assert response.status_code == 404, response.text


def test_soft_deleted_scene_target_hides_existing_annotation(api_context):
    client, _db = api_context
    project, _chapter, scene = _create_scene_manuscript(client)
    selected_text = "beta"
    start = scene["content_plain"].index(selected_text)
    create_response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "scene",
            "target_id": scene["id"],
            "category": "clarity",
            "body": "Clarify this word.",
            "scene_version": scene["version"],
            "start": start,
            "end": start + len(selected_text),
            "selected_text": selected_text,
        },
    )
    assert create_response.status_code == 201, create_response.text
    annotation_id = create_response.json()["id"]

    delete_response = client.delete(
        f"{PREFIX}/scenes/{scene['id']}",
        headers={"expected-version": str(scene["version"])},
    )
    assert delete_response.status_code == 204, delete_response.text

    get_response = client.get(f"{PREFIX}/annotations/{annotation_id}")
    assert get_response.status_code == 404, get_response.text

    list_response = client.get(f"{PREFIX}/projects/{project['id']}/annotations")
    assert list_response.status_code == 200, list_response.text
    payload = list_response.json()
    assert payload["annotations"] == []
    assert payload["total"] == 0


def test_manual_range_offsets_are_unicode_code_point_offsets(api_context):
    client, _db = api_context
    scene_text = "Alpha 😀 beta 🌌 omega"
    _project, _chapter, scene = _create_scene_manuscript(client, scene_text)
    selected_text = "🌌"
    start = scene_text.index(selected_text)
    end = start + len(selected_text)
    byte_start = len(scene_text[:start].encode("utf-8"))

    response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "scene",
            "target_id": scene["id"],
            "category": "style",
            "body": "The symbol works as a range anchor.",
            "scene_version": scene["version"],
            "start": start,
            "end": end,
            "selected_text": selected_text,
        },
    )

    assert response.status_code == 201, response.text
    annotation = response.json()
    assert annotation["anchor_start"] == start
    assert annotation["anchor_end"] == end
    assert annotation["selected_text"] == selected_text

    bad_response = client.post(
        f"{PREFIX}/annotations",
        json={
            "target_type": "scene",
            "target_id": scene["id"],
            "category": "style",
            "body": "Byte offsets must not be accepted as character offsets.",
            "scene_version": scene["version"],
            "start": byte_start,
            "end": byte_start + len(selected_text.encode("utf-8")),
            "selected_text": selected_text,
        },
    )

    assert bad_response.status_code == 400, bad_response.text


def test_bounded_anchor_status_filter_is_allowed(api_context):
    client, db = api_context
    project, _chapter, scene = _create_scene_manuscript(client)
    helper = ManuscriptDBHelper(db)
    selected_text = "beta"
    start = scene["content_plain"].index(selected_text)
    annotation_id = helper.create_annotation(
        project_id=project["id"],
        target_type="scene",
        target_id=scene["id"],
        category="clarity",
        source="user",
        body="Bounded note.",
        scene_version=scene["version"],
        anchor_start=start,
        anchor_end=start + len(selected_text),
        selected_text=selected_text,
    )

    response = client.get(
        f"{PREFIX}/projects/{project['id']}/annotations",
        params={
            "target_type": "scene",
            "target_id": scene["id"],
            "anchor_status": "attached",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    assert payload["annotations"][0]["id"] == annotation_id
