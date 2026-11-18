"""Navigation revisions follow the active document version, including metadata writes."""

import hashlib
import json
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


@pytest.fixture
def versioned_media(tmp_path):
    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    with scoped_context(user_id=1, is_admin=False):
        db = create_media_database(db_path=str(tmp_path / "media.db"), client_id="1")
        content = "Alpha\n\nBravo"
        metadata = {
            "page_count": 2,
            "page_map_content_hash": hashlib.sha256(content.encode()).hexdigest(),
            "page_map": [
                {"page_number": 1, "start_char": 0, "end_char": 5},
                {"page_number": 2, "start_char": 7, "end_char": 12},
            ],
        }
        media_id, _, _ = db.add_media_with_keywords(
            url="revision.pdf", title="Revision", media_type="pdf", content=content,
            keywords=[], safe_metadata=json.dumps(metadata),
        )
        yield db, media_id, content, metadata
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["content_version", "metadata_revision"])
async def test_navigation_tree_and_content_cache_follow_document_revision(monkeypatch, versioned_media, change):
    from fastapi import HTTPException
    from tldw_Server_API.app.api.v1.endpoints.media import navigation as endpoint
    from tldw_Server_API.app.api.v1.schemas.media_navigation_schemas import (
        MediaNavigationContentQueryParams, MediaNavigationQueryParams,
    )
    from tldw_Server_API.app.core.DB_Management.media_db.api import get_document_version
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.safe_metadata_write_ops import (
        update_version_safe_metadata_in_transaction,
    )

    db, media_id, content, metadata = versioned_media
    nodes = [{
        "id": "chapter", "title": "Chapter", "level": 1, "order": 0, "source": "document_structure_index",
        "target_type": "char_range", "target_start": 0, "target_end": len(content),
    }]

    async def select_nodes(**kwargs):
        return nodes, ["document_structure_index"]

    cache = {}
    monkeypatch.setattr(endpoint, "_select_source_nodes", select_nodes)
    monkeypatch.setattr(endpoint, "get_cached_response", cache.get)
    monkeypatch.setattr(endpoint, "cache_response", lambda key, payload, **kwargs: cache.update({key: ("etag", payload)}))
    user = SimpleNamespace(id=1)
    before = await endpoint.get_media_navigation(media_id, MediaNavigationQueryParams(), db, user)
    await endpoint.get_media_navigation_content(
        media_id, "chapter", MediaNavigationContentQueryParams(navigation_version=before.navigation_version), db, user,
    )
    original_media_row = db.get_media_by_id(media_id)
    if change == "content_version":
        db.create_document_version(media_id=media_id, content="Bravo\n\nAlpha", safe_metadata=json.dumps(metadata))
    else:
        latest = get_document_version(db, media_id=media_id)
        metadata["page_map"][0]["printed_page_label"] = "i"
        with db.transaction() as connection:
            update_version_safe_metadata_in_transaction(db, latest["id"], json.dumps(metadata), metadata, connection)
    assert db.get_media_by_id(media_id)["version"] == original_media_row["version"]
    after = await endpoint.get_media_navigation(media_id, MediaNavigationQueryParams(), db, user)
    assert after.navigation_version != before.navigation_version
    with pytest.raises(HTTPException) as stale:
        await endpoint.get_media_navigation_content(
            media_id, "chapter", MediaNavigationContentQueryParams(navigation_version=before.navigation_version), db, user,
        )
    assert stale.value.status_code == 409


@pytest.mark.asyncio
async def test_equal_length_new_version_rejects_inherited_pdf_page_map(monkeypatch, versioned_media):
    from tldw_Server_API.app.core.Media import navigation

    db, media_id, content, metadata = versioned_media

    async def no_nodes(*args, **kwargs):
        return [], []

    monkeypatch.setattr(navigation, "_select_source_nodes", no_nodes)
    original = await navigation.load_navigation_sections(media_id, db)
    assert original["page_map"] == metadata["page_map"]
    db.create_document_version(media_id=media_id, content="Bravo\n\nAlpha", safe_metadata=json.dumps(metadata))
    edited = await navigation.load_navigation_sections(media_id, db)
    assert len(edited["content"]) == len(content)
    assert edited["page_map"] == []


def test_unmapped_education_node_does_not_match_special_education():
    from tldw_Server_API.app.api.v1.schemas.media_navigation_schemas import MediaNavigationNode
    from tldw_Server_API.app.core.Media.navigation import _derive_content_span

    node = MediaNavigationNode(
        id="toc:education", title="Education", level=1, order=0, source="generated_toc",
        target_type="page", target_start=319,
    )
    db = SimpleNamespace(lookup_section_by_heading=lambda *args: (10, 30))
    assert _derive_content_span(node, [node], {"type": "pdf"}, db, 1, 100) is None
