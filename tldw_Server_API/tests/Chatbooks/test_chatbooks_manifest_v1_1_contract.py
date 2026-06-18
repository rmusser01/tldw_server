import json
from pathlib import Path

import jsonschema

from tldw_Server_API.app.api.v1.schemas.chatbook_schemas import CreateChatbookRequest
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ChatbookVersion


def _load_v1_1_schema() -> dict:
    schema_path = Path(__file__).resolve().parents[3] / "Docs" / "Schemas" / "chatbooks_manifest_v1_1.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _minimal_v1_1_manifest() -> dict:
    return {
        "version": "1.1.0",
        "name": "v1.1 contract",
        "description": "contract",
        "author": None,
        "created_at": "2026-06-18T12:00:00+00:00",
        "updated_at": "2026-06-18T12:00:00+00:00",
        "export_id": "contract-export",
        "content_items": [],
        "relationships": [],
        "configuration": {
            "include_media": False,
            "include_embeddings": False,
            "include_generated_content": True,
            "media_quality": "compressed",
            "max_file_size_mb": 100,
        },
        "statistics": {
            "total_conversations": 0,
            "total_notes": 0,
            "total_characters": 0,
            "total_media_items": 0,
            "total_prompts": 0,
            "total_evaluations": 0,
            "total_embeddings": 0,
            "total_world_books": 0,
            "total_dictionaries": 0,
            "total_documents": 0,
            "total_explainer_sessions": 0,
            "total_size_bytes": 0,
        },
        "metadata": {
            "tags": [],
            "categories": [],
            "language": "en",
            "license": None,
        },
        "user_info": {"user_id": None},
        "features_used": [],
        "producer": {"name": "tldw_server"},
        "source_instance": {},
        "compatibility": {"min_reader_version": "1.0.0"},
        "file_inventory": [],
    }


def test_chatbook_version_accepts_v1_1():
    assert ChatbookVersion("1.1.0") is ChatbookVersion.V1_1


def test_create_chatbook_request_accepts_format_version_v1_1():
    request = CreateChatbookRequest(
        name="v1.1",
        description="v1.1",
        content_selections={},
        format_version="1.1.0",
    )

    assert request.format_version is ChatbookVersion.V1_1


def test_minimal_v1_1_manifest_matches_schema():
    jsonschema.validate(_minimal_v1_1_manifest(), _load_v1_1_schema())


def test_v1_1_manifest_allows_content_item_metadata_envelope():
    manifest = _minimal_v1_1_manifest()
    manifest["features_used"] = ["content_envelopes"]
    manifest["content_items"] = [
        {
            "id": "exp_123",
            "type": "explainer_session",
            "title": "Learn attention",
            "description": None,
            "created_at": None,
            "updated_at": None,
            "tags": [],
            "metadata": {
                "format": "tldw.explainer_session.v1",
                "envelope": {
                    "format": "tldw.explainer_session.v1",
                    "schema_version": 1,
                    "representations": [],
                    "integrity": {},
                    "lossiness": {},
                    "provenance": {},
                    "source_refs": [],
                },
            },
            "file_path": "content/explainer_sessions/session_exp_123.json",
            "checksum": "sha256:example",
        }
    ]

    jsonschema.validate(manifest, _load_v1_1_schema())
