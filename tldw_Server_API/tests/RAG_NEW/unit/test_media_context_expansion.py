"""Canonical context expansion preserves source identity and exact locations."""

import asyncio
import hashlib
import threading
from contextlib import nullcontext

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


class ParentReader:
    def __init__(self, content, *, visible=True, mapped=True):
        self.content = content
        self.visible = visible
        self.rows = []
        self.reads = []
        midpoint = len(content) // 2
        self.metadata = {
            "page_map_content_hash": hashlib.sha256(content.encode()).hexdigest() if mapped else "stale",
            "page_map": [
                {"page_number": 1, "start_char": 0, "end_char": midpoint, "printed_page_label": "12"},
                {"page_number": 2, "start_char": midpoint, "end_char": len(content), "printed_page_label": "13"},
            ],
        }

    def get_media_by_id(self, media_id, **kwargs):
        self.reads.append(("parent", media_id))
        return {"id": media_id, "content": "obsolete parent text"} if self.visible else None

    def get_scoped_media_by_id(self, media_id, **kwargs):
        return self.get_media_by_id(media_id, **kwargs)

    def get_document_version(self, **kwargs):
        self.reads.append(("version", kwargs["media_id"]))
        return {"content": self.content, "safe_metadata": self.metadata}

    def get_unvectorized_chunks_in_range(self, media_id, start_index, end_index):
        self.reads.append(("siblings", start_index, end_index))
        return [row for row in self.rows if start_index <= row["chunk_index"] <= end_index]


class OwnedParentReader(ParentReader):
    def __init__(self, content, **overrides):
        super().__init__(content)
        self.parent = {"id": 1, "type": "pdf", "title": "Owned policy source", "url": "owned-source.pdf", **overrides}

    def get_media_by_id(self, media_id, **kwargs):
        self.reads.append(("parent", media_id))
        return self.parent


def seed(content, start, end, **metadata):
    return Document(
        id="chunk-5",
        content=content[start:end],
        score=0.8,
        chunk_index=5,
        start_char=start,
        end_char=end,
        page_number=88,
        metadata={
            "media_id": "1",
            "chunk_index": 5,
            "start_char": start,
            "end_char": end,
            "page_number": 88,
            "page_spans": [{"page_number": 88}],
            **metadata,
        },
    )


async def expand(documents, db, **options):
    from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
    from tldw_Server_API.app.core.RAG.rag_service.media_context_expansion import expand_media_context

    # Ordinary fixture cases run as the authenticated owner; visibility tests
    # supply their own narrowed scope and the missing-authority test calls directly.
    authority = nullcontext() if get_scope() is not None else scoped_context(user_id=1, is_admin=True)
    with authority:
        return await expand_media_context(
            documents,
            media_db=db,
            **{
                "enable_parent_expansion": True,
                "parent_context_size": 100,
                "include_sibling_chunks": False,
                "sibling_window": 1,
                **options,
            },
        )


@pytest.mark.asyncio
async def test_expands_truncated_sentence_from_current_version_with_exact_pages():
    content = "Introduction. " * 10 + "Move the office to the Department of Labor, subject to review. " + "After. " * 30
    start = content.index("Move")
    end = content.index(" Labor")
    original = seed(content, start, end)
    expanded, stats = await expand([original], ParentReader(content))
    result = expanded[0]
    assert "Department of Labor, subject to review." in result.content
    assert result.content == content[result.start_char : result.end_char]
    assert (result.id, result.score, result.chunk_index) == (original.id, original.score, original.chunk_index)
    assert result.metadata["matched_start_char"] == start and result.metadata["matched_end_char"] == end
    assert result.metadata["start_char"] == result.start_char and result.metadata["end_char"] == result.end_char
    assert result.page_number == result.metadata["page_spans"][0]["page_number"] != 88
    assert stats["context_expanded_count"] == 1
    assert original.content == content[start:end] and original.page_number == 88


@pytest.mark.asyncio
async def test_neighbors_extend_seed_without_repeating_or_crossing_section_bounds():
    content = "A" * 200 + "B" * 20 + "C" * 200
    db = ParentReader(content)
    db.rows = [
        {"chunk_index": index, "start_char": start, "end_char": end, "chunk_text": content[start:end]}
        for index, start, end in [(4, 0, 200), (5, 200, 220), (6, 220, 420)]
    ]
    result, stats = await expand(
        [seed(content, 200, 220, section_start=175, section_end=250)],
        db,
        enable_parent_expansion=False,
        include_sibling_chunks=True,
    )
    assert result[0].content == content[175:250]
    assert ("siblings", 4, 6) in db.reads
    assert stats["siblings_added_count"] == 2


@pytest.mark.asyncio
async def test_invalid_neighbor_offsets_do_not_expand():
    content = "A" * 200 + "B" * 20 + "C" * 200
    db = ParentReader(content)
    db.rows = [{"chunk_index": 6, "start_char": 220, "end_char": 420, "chunk_text": "outdated text"}]
    result, stats = await expand(
        [seed(content, 200, 220)], db, enable_parent_expansion=False, include_sibling_chunks=True
    )
    assert result[0].content == content[200:220] and stats["context_expanded_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("verification_only", [False, True])
async def test_hidden_parent_is_removed_before_any_version_or_neighbor_read(verification_only):
    db = ParentReader("secret", visible=False)
    result, stats = await expand([seed("secret", 0, 6)], db,
                                 enable_parent_expansion=not verification_only,
                                 include_sibling_chunks=not verification_only,
                                 verify_locations=verification_only)
    assert result == [] and stats["context_hidden_count"] == 1
    assert db.reads == [("parent", 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("verification_only", [False, True])
async def test_stale_seed_does_not_attach_unrelated_current_text(verification_only):
    db = ParentReader("Swapped pages have the same length.")
    original = seed("Original text has the same length!", 0, 20)
    result, stats = await expand([original], db, enable_parent_expansion=not verification_only,
                                 verify_locations=verification_only)
    assert result[0].content == original.content and stats["context_skipped_count"] == 1
    assert result[0].page_number is None and "page_spans" not in result[0].metadata


@pytest.mark.asyncio
async def test_stale_page_hash_drops_old_pages_when_expanding_valid_text():
    content = "Canonical source. " * 30
    result, _ = await expand([seed(content, 100, 120)], ParentReader(content, mapped=False))
    assert result[0].content == content[0:220]
    assert result[0].page_number is None and "page_number" not in result[0].metadata
    assert "page_spans" not in result[0].metadata


@pytest.mark.asyncio
async def test_disabled_flags_and_nonmedia_sources_do_not_read_parent():
    db = ParentReader("text")
    original = seed("text", 0, 4)
    result, _ = await expand([original], db, enable_parent_expansion=False)
    assert result == [original] and db.reads == []
    note = Document(id="note", content="private note", metadata={}, source=DataSource.NOTES)
    result, _ = await expand([note], db)
    assert result == [note] and db.reads == []


@pytest.mark.asyncio
async def test_notes_only_does_not_open_an_unavailable_media_database(monkeypatch):
    from tldw_Server_API.app.core.RAG.rag_service import media_context_expansion as mod

    def forbidden(*args, **kwargs):
        pytest.fail("Notes context must not open a media database")

    monkeypatch.setattr(mod, "managed_media_database", forbidden)
    note = Document(id="note", content="Note evidence", metadata={}, source=DataSource.NOTES)
    result, _ = await mod.expand_media_context(
        [note],
        media_db_path="unavailable.db",
        enable_parent_expansion=True,
        parent_context_size=100,
        include_sibling_chunks=True,
        sibling_window=1,
    )
    assert result == [note]


@pytest.mark.asyncio
async def test_cancellation_stops_before_loading_next_parent():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    db = ParentReader("a" * 400)
    read = db.get_media_by_id

    def held(media_id, **kwargs):
        started.set()
        release.wait(2)
        result = read(media_id, **kwargs)
        finished.set()
        return result

    db.get_media_by_id = held
    task = asyncio.create_task(expand([seed(db.content, 100, 120), seed(db.content, 200, 220, media_id="2")], db))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.to_thread(finished.wait, 2)
    assert db.reads == [("parent", 1)]


@pytest.mark.asyncio
async def test_existing_database_path_loads_canonical_parent(tmp_path):
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.RAG.rag_service.media_context_expansion import expand_media_context

    path = tmp_path / "media.db"
    db = MediaDatabase(db_path=str(path), client_id="context-test")
    try:
        content = "One complete source sentence. " * 20
        media_id, _, _ = db.add_media_with_keywords(title="Source", media_type="document", content=content, keywords=[])
        with scoped_context(user_id=1, is_admin=True):
            result, _ = await expand_media_context(
                [seed(content, 100, 120, media_id=str(media_id))],
                media_db_path=str(path),
                enable_parent_expansion=True,
                parent_context_size=100,
                include_sibling_chunks=False,
                sibling_window=1,
            )
        assert result[0].content == content[:220]
    finally:
        db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("visibility", "team_id", "org_id", "user_id", "is_admin", "allowed"),
    [
        ("personal", None, None, 1, False, False),
        ("team", 11, None, 1, False, False),
        ("org", None, 21, 1, False, False),
        ("personal", None, None, 2, False, True),
        ("team", 10, None, 1, False, True),
        ("org", None, 20, 1, False, True),
        ("personal", None, None, 1, True, True),
        ("personal", None, None, None, False, False),
    ],
)
async def test_cached_seed_rechecks_current_source_visibility_before_reading_context(
    media_database, monkeypatch, visibility, team_id, org_id, user_id, is_admin, allowed
):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.RAG.rag_service import media_context_expansion as mod

    content = "Previously shared excerpt. Newly private details. " * 10
    media_id, _, _ = media_database.add_media_with_keywords(
        title="Visibility fixture", media_type="document", content=content, keywords=[], owner_user_id=2
    )
    media_database.share_media(media_id, "team", team_id=10)
    cached_seed = seed(content, 0, 26, media_id=str(media_id))
    with scoped_context(user_id=1, team_ids=[10], org_ids=[20]):
        initial, _ = await expand([cached_seed], media_database)
    assert initial

    media_database.share_media(media_id, visibility, team_id=team_id, org_id=org_id)
    reads = []
    get_version = mod.get_document_version
    get_neighbors = mod.get_unvectorized_chunks_in_range

    def read_version(*args, **kwargs):
        reads.append("version")
        return get_version(*args, **kwargs)

    def read_neighbors(*args, **kwargs):
        reads.append("neighbors")
        return get_neighbors(*args, **kwargs)

    monkeypatch.setattr(mod, "get_document_version", read_version)
    monkeypatch.setattr(mod, "get_unvectorized_chunks_in_range", read_neighbors)
    with scoped_context(user_id=user_id, team_ids=[10], org_ids=[20], is_admin=is_admin):
        result, stats = await expand([cached_seed], media_database, include_sibling_chunks=True)

    assert [doc.id for doc in result] == ([cached_seed.id] if allowed else [])
    assert reads == (["version", "neighbors"] if allowed else [])
    assert stats["context_hidden_count"] == int(not allowed)


@pytest.mark.asyncio
async def test_context_without_authority_cannot_read_canonical_parent(media_database):
    from tldw_Server_API.app.core.DB_Management.scope_context import get_scope
    from tldw_Server_API.app.core.RAG.rag_service.media_context_expansion import expand_media_context

    assert get_scope() is None
    content = "Unscoped fixture content. " * 10
    media_id, _, _ = media_database.add_media_with_keywords(
        title="No authority", media_type="document", content=content, keywords=[], owner_user_id=2
    )
    result, stats = await expand_media_context(
        [seed(content, 0, 20, media_id=str(media_id))], media_db=media_database,
        enable_parent_expansion=True, parent_context_size=100, include_sibling_chunks=False, sibling_window=1,
    )
    assert result == [] and stats["context_hidden_count"] == 1


@pytest.mark.asyncio
async def test_custom_reader_cannot_silently_ignore_scoped_lookup():
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    reader = ParentReader("Fixture content. " * 30)
    reader.get_scoped_media_by_id = None
    with scoped_context(user_id=1, is_admin=True), pytest.raises(TypeError, match="scoped"):
        await expand([seed(reader.content, 100, 120)], reader)
    assert reader.reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize("verification_only", [False, True])
@pytest.mark.parametrize("owner_type", ["pdf", "video"])
@pytest.mark.parametrize("cached_fields", [{}, {"title": "Spoofed title", "type": "unknown", "media_type": "video", "url": "https://untrusted.example/",
    "source_id": "99", "source_type": "notes", "source": "notes", "evidence_origin": "web_search"}, {"type": "pdf", "media_type": "pdf",
    "source_id": "99", "source_type": "notes", "source": "notes", "evidence_origin": "web_search"}])
async def test_old_and_new_vector_metadata_uses_verified_owning_row_without_extra_reads(verification_only, owner_type, cached_fields):
    content = "Canonical PDF policy paragraph. " * 20
    db = OwnedParentReader(content, type=owner_type)
    first, second = seed(content, 100, 120, **cached_fields), seed(content, 150, 170, **cached_fields)
    results, _ = await expand([first, second], db, enable_parent_expansion=not verification_only, verify_locations=verification_only)
    for result in results:
        assert {key: result.metadata[key] for key in ("media_id", "source_id", "source_type", "source", "evidence_origin", "type", "media_type", "title", "url")} == {
            "media_id": "1", "source_id": "1", "source_type": "media_db", "source": "media_db", "evidence_origin": "local_library",
            "type": owner_type, "media_type": owner_type, "title": "Owned policy source", "url": "owned-source.pdf"}
        assert result.content == content[result.start_char:result.end_char]
    assert (results[0].id, results[0].score, results[0].chunk_index) == (first.id, first.score, first.chunk_index)
    assert db.reads == [("parent", 1), ("version", 1)]
    assert first.metadata.get("title") == cached_fields.get("title")


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_fields", [{"deleted": True}, {"is_trash": True}, {"system_operation_id": "internal"}, {"id": 2}])
async def test_unusable_or_mismatched_owner_never_enriches_or_reads_version(parent_fields):
    content = "Hidden source text. " * 20
    db = OwnedParentReader(content, **parent_fields)
    result, stats = await expand([seed(content, 100, 120)], db, verify_locations=True)
    assert result == []
    assert stats["context_hidden_count"] == 1
    assert db.reads == [("parent", 1)]


@pytest.mark.asyncio
async def test_mismatched_seed_is_not_given_verified_parent_metadata():
    db = OwnedParentReader("Changed canonical paragraph. " * 20)
    original = seed("Old paragraph. " * 40, 100, 120)
    results, stats = await expand([original], db, verify_locations=True)
    assert results[0].content == original.content
    assert results[0].page_number is None and "page_spans" not in results[0].metadata
    assert "media_type" not in results[0].metadata and "title" not in results[0].metadata
    assert stats["context_skipped_count"] == 1
    assert db.reads == [("parent", 1), ("version", 1)]


@settings(max_examples=20, deadline=None)
@given(st.text(min_size=220, max_size=600), st.integers(min_value=100, max_value=2000))
def test_expansion_always_preserves_exact_substring_and_bounded_sides(content, budget):
    result, _ = asyncio.run(expand([seed(content, 100, 120)], ParentReader(content), parent_context_size=budget))
    expanded = result[0]
    assert expanded.content == content[expanded.start_char : expanded.end_char]
    assert 0 <= 100 - expanded.start_char <= budget and 0 <= expanded.end_char - 120 <= budget
