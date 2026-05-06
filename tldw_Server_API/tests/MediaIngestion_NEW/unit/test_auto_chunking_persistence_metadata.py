import json

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
            "derived_views": ["section_titles"],
            "fallback_reason": None,
            "profile": {"media_type": "document", "text_length": 20},
            "callable": lambda: None,
        },
        "unsafe_object": object(),
    }

    safe_meta = build_safe_metadata_subset(metadata)
    json.dumps(safe_meta)
    assert safe_meta["title"] == "Stored item"
    assert safe_meta["chunking_plan"]["mode"] == "auto"
    assert safe_meta["chunking_plan"]["profile"]["media_type"] == "document"
    assert "callable" not in safe_meta["chunking_plan"]
    assert "unsafe_object" not in safe_meta
