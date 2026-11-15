"""Exact page and section offsets use the final stored text, never proportions."""

import hashlib

import pytest
from hypothesis import given, strategies as st


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("stored_content", ["Alpha\n\nBravo", "Bravo\n\nAlpha", "", "Alpha"])
def test_page_map_is_bound_to_exact_stored_content(stored_content):
    from tldw_Server_API.app.core.Media.page_provenance import page_map_for_content

    original = "Alpha\n\nBravo"
    page_map = [{"page_number": 1, "start_char": 0, "end_char": len(original)}]
    metadata = {"page_map": page_map, "page_map_content_hash": hashlib.sha256(original.encode()).hexdigest()}
    assert page_map_for_content(metadata, stored_content) == (page_map if stored_content == original else [])


def test_unbound_page_map_is_unavailable():
    from tldw_Server_API.app.core.Media.page_provenance import page_map_for_content

    assert page_map_for_content({"page_map": [{"page_number": 1, "start_char": 0, "end_char": 5}]}, "Alpha") == []


def test_page_spans_clip_cross_page_excerpt_and_ignore_blank_pages():
    from tldw_Server_API.app.core.Media.page_provenance import page_spans_for_range

    page_map = [
        {"page_number": 1, "start_char": 0, "end_char": 10, "printed_page_label": "iv"},
        {"page_number": 2, "start_char": 12, "end_char": 12},
        {"page_number": 3, "start_char": 14, "end_char": 100},
    ]
    assert page_spans_for_range(page_map, 8, 20) == [
        {"page_number": 1, "start_char": 8, "end_char": 10, "printed_page_label": "iv"},
        {"page_number": 3, "start_char": 14, "end_char": 20},
    ]


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=30))
def test_valid_map_preserves_all_physical_pages(lengths):
    from tldw_Server_API.app.core.Media.page_provenance import validate_page_map

    offset = 0
    page_map = []
    for number, length in enumerate(lengths, 1):
        page_map.append({"page_number": number, "start_char": offset, "end_char": offset + length})
        offset += length + 2
    assert validate_page_map(page_map, offset - 2) == page_map


@pytest.mark.parametrize(
    "bad_entry",
    [
        {"page_number": True, "start_char": 0, "end_char": 20},
        {"page_number": 0, "start_char": 0, "end_char": 20},
        {"page_number": 1, "start_char": -1, "end_char": 20},
        {"page_number": 1, "start_char": 10, "end_char": 5},
        {"page_number": 1, "start_char": 0, "end_char": 21},
    ],
)
def test_invalid_page_map_fails_closed(bad_entry):
    from tldw_Server_API.app.core.Media.page_provenance import validate_page_map

    assert validate_page_map([bad_entry], 20) == []


def test_exact_page_section_includes_children_and_uses_uneven_page_lengths():
    from tldw_Server_API.app.core.Media.page_provenance import resolve_navigation_span

    nodes = [
        {"id": "chapter", "level": 1, "target_type": "page", "target_start": 1, "source": "pdf_outline"},
        {"id": "child", "level": 2, "target_type": "page", "target_start": 2, "source": "pdf_outline"},
        {"id": "next", "level": 1, "target_type": "page", "target_start": 3, "source": "pdf_outline"},
    ]
    page_map = [
        {"page_number": 1, "start_char": 0, "end_char": 10},
        {"page_number": 2, "start_char": 12, "end_char": 70},
        {"page_number": 3, "start_char": 72, "end_char": 100},
    ]
    assert resolve_navigation_span(nodes[0], nodes, page_map, 100) == (0, 72)
    assert resolve_navigation_span(nodes[-1], nodes, page_map, 100) == (72, 100)
    assert resolve_navigation_span(nodes[0], nodes, [], 100) is None


def test_printed_toc_page_requires_explicit_matching_label():
    from tldw_Server_API.app.core.Media.page_provenance import resolve_navigation_span

    node = {"id": "toc", "level": 1, "target_type": "page", "target_start": 1, "source": "generated_toc"}
    page_map = [
        {"page_number": 1, "start_char": 0, "end_char": 10},
        {"page_number": 2, "start_char": 12, "end_char": 70, "printed_page_label": "1"},
    ]
    assert resolve_navigation_span(node, [node], page_map, 70) == (12, 70)
    page_map[1].pop("printed_page_label")
    assert resolve_navigation_span(node, [node], page_map, 70) is None


def test_missing_next_chapter_page_does_not_expand_scope_to_document_end():
    from tldw_Server_API.app.core.Media.page_provenance import resolve_navigation_span

    nodes = [
        {"id": "first", "level": 1, "target_type": "page", "target_start": 1, "source": "generated_toc"},
        {"id": "next", "level": 1, "target_type": "page", "target_start": 2, "source": "generated_toc"},
    ]
    page_map = [{"page_number": 1, "start_char": 0, "end_char": 100, "printed_page_label": "1"}]
    assert resolve_navigation_span(nodes[0], nodes, page_map, 100) is None


def test_char_section_keeps_explicit_end_when_no_child_is_truncated():
    from tldw_Server_API.app.core.Media.page_provenance import resolve_navigation_span

    node = {"id": "chunk", "level": 1, "target_type": "char_range", "target_start": 4, "target_end": 25}
    assert resolve_navigation_span(node, [node], [], 100) == (4, 25)


@pytest.mark.asyncio
async def test_pdf_pages_survive_persistence_and_chunk_retrieval(tmp_path):
    import asyncio
    import json
    import pymupdf
    from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database, get_document_version
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
    from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import process_document_like_item
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MediaDBRetriever, RetrievalConfig

    path = tmp_path / "mapped.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "The first chapter discusses history.")
        doc.new_page().insert_text((72, 72), "Quasar is the distinctive final chapter evidence.")
        doc.save(path)
    db_path = tmp_path / "media.db"
    with scoped_context(user_id=1, is_admin=False):
        result = await process_document_like_item(
            item_input_ref=path.name,
            processing_source=str(path),
            media_type="pdf",
            is_url=False,
            form_data=AddMediaForm(media_type="pdf", perform_analysis=False, perform_chunking=True),
            chunk_options={"method": "sentences", "max_size": 50, "overlap": 0},
            temp_dir=tmp_path,
            loop=asyncio.get_running_loop(),
            db_path=str(db_path),
            client_id="1",
            user_id=1,
        )
        assert result.get("db_id"), result.get("error")
        db = create_media_database(db_path=str(db_path), client_id="1")
        try:
            metadata = get_document_version(db, media_id=result["db_id"], version_number=None)["safe_metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            assert metadata["page_count"] == 2
            assert len(metadata["page_map"]) == 2
            retriever = MediaDBRetriever(
                str(db_path),
                user_id="1",
                media_db=db,
                config=RetrievalConfig(use_vector=False, fts_level="chunk"),
            )
            documents = await retriever.retrieve("Quasar", allowed_media_ids=[result["db_id"]])
            assert documents
            assert any(span["page_number"] == 2 for span in documents[0].metadata["page_spans"])
            assert documents[0].page_number is not None
        finally:
            db.close_connection()
