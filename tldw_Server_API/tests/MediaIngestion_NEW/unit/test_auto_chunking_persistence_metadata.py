import json
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def test_safe_metadata_subset_preserves_json_safe_chunking_plan():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import (
        build_safe_metadata_subset,
    )

    metadata = {
        "title": "Stored item",
        "chunking_plan": {
            "mode": "auto",
            "goal": "balanced",
            "used_llm": False,
            "method": "semantic",
            "max_size": 900,
            "overlap": 120,
            "derived_views": ["section_titles", None, "outline"],
            "fallback_reason": None,
            "profile": {"media_type": "document", "text_length": 20},
            "callable": lambda: None,
        },
        "externalIds": {
            "DOI": "10.1000/example",
            "ArXiv": "2301.00001",
            "PMID": None,
        },
        "unsafe_object": object(),
    }

    safe_meta = build_safe_metadata_subset(metadata)
    json.dumps(safe_meta)
    assert safe_meta["title"] == "Stored item"
    assert safe_meta["chunking_plan"]["mode"] == "auto"
    assert safe_meta["chunking_plan"]["profile"]["media_type"] == "document"
    assert safe_meta["chunking_plan"]["derived_views"] == [
        "section_titles",
        None,
        "outline",
    ]
    assert safe_meta["doi"] == "10.1000/example"
    assert safe_meta["arxiv_id"] == "2301.00001"
    assert "arxiv" not in safe_meta
    assert "pmid" not in safe_meta
    assert "callable" not in safe_meta["chunking_plan"]
    assert "unsafe_object" not in safe_meta


@pytest.mark.asyncio
async def test_persistence_auto_chunking_resolution_propagates_cancellation(monkeypatch):
    import asyncio

    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    async def _cancelled_resolver(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        persistence,
        "async_resolve_chunking_options_and_plan",
        _cancelled_resolver,
        raising=True,
    )

    with pytest.raises(asyncio.CancelledError):
        await persistence._resolve_auto_chunking_options_for_persistence(
            SimpleNamespace(chunking_mode="auto"),
            media_type="document",
            source_name="doc.md",
            extracted_text="content",
        )
