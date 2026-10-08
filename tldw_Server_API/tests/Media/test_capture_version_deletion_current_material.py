"""Version tombstones do not roll back the current capture retrieval projections."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.api.v1.endpoints.media.item import get_media_item
from tldw_Server_API.app.api.v1.endpoints.media.versions import delete_version
from tldw_Server_API.app.core.DB_Management.media_db import api
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_repository import MediaRepository
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MediaDBRetriever, RetrievalConfig


@pytest.mark.integration
@pytest.mark.asyncio
async def test_deleted_newest_capture_version_keeps_current_media_and_chunk_retrieval(tmp_path: Path) -> None:
    """The pre-Ask client must read current Media even when the active pin is head again."""
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="1")
    clip_id = "11111111-1111-4111-8111-111111111111"
    original = "Accepted original article alpha."
    changed = "Later changed current article beta."
    metadata = json.dumps(
        {
            "source": "web_clipper",
            "clip_type": "article",
            "clip_id": clip_id,
            "workspace_id": "capture-workspace",
            "source_url": "https://example.com/article",
            "capture_metadata": {
                "web_capture_v1": {
                    "mode": "server_article",
                    "requested_url": "https://example.com/article",
                    "captured_at": "2026-10-07T18:00:00Z",
                    "content_sha256": hashlib.sha256(original.encode()).hexdigest(),
                    "refresh_of": None,
                }
            },
        }
    )
    repo = MediaRepository.from_legacy_db(db)

    def save(content: str) -> int:
        """Use the existing overwrite producer for Media, versions, FTS and chunks."""
        media_id, _, _ = repo.add_text_media(
            url=f"web-clipper://{clip_id}",
            title="Capture",
            media_type="article",
            content=content,
            safe_metadata=metadata,
            owner_user_id=1,
            overwrite=True,
            deduplicate_content=False,
            chunks=[{"text": content, "start_char": 0, "end_char": len(content)}],
        )
        return media_id

    try:
        media_id = save(original)
        pin = api.list_document_versions(db, media_id, include_content=True)[0]
        assert save(changed) == media_id
        newest = max(api.list_document_versions(db, media_id), key=lambda row: row["version_number"])
        assert newest["version_number"] > pin["version_number"]
        response = await delete_version(media_id=media_id, version_number=newest["version_number"], db=db)
        assert response.status_code == 204
        active = api.list_document_versions(db, media_id, include_content=True)
        assert [(row["version_number"], row["uuid"], row["content"]) for row in active] == [
            (pin["version_number"], pin["uuid"], original)
        ]
        current = await get_media_item(
            request=Request({"type": "http", "headers": []}),
            response=Response(),
            media_id=media_id,
            include_content=True,
            include_versions=False,
            include_version_content=False,
            db=db,
            current_user=SimpleNamespace(id=1),
            if_none_match=None,
        )
        assert current["media_id"] == media_id
        assert current["content"]["text"] == changed
        assert (
            hashlib.sha256(current["content"]["text"].encode()).hexdigest()
            != hashlib.sha256(pin["content"].encode()).hexdigest()
        )
        # Exercise both ordinary retrieval consumers, not a SELECT over their tables.
        for level in ("media", "chunk"):
            retriever = MediaDBRetriever(
                db_path=None,
                media_db=db,
                user_id="1",
                config=RetrievalConfig(use_fts=True, use_vector=False, fts_level=level, min_score=0),
            )
            documents = await retriever.retrieve("beta", allowed_media_ids=[media_id])
            assert documents, level
            expected = f"Capture\n{changed}" if level == "media" else changed
            assert [document.content for document in documents] == [expected], level
    finally:
        db.close_connection()
