import hashlib
import io
import json
import zipfile

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Chatbooks.chatbook_format_v1_1 import build_preview_report
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ChatbookManifest, ChatbookVersion
from tldw_Server_API.app.main import app


class _DummyAuditService:
    async def log_event(self, *args, **kwargs) -> None:
        return None


async def _override_user() -> User:
    return User(id=1, username="tester", email=None, is_active=True)


def _make_v1_1_chatbook_bytes() -> bytes:
    payload_path = "content/explainer_sessions/session_exp_1.json"
    payload = json.dumps({"id": "exp_1", "title": "Preview"}).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    manifest = {
        "version": "1.1.0",
        "name": "v1.1 Preview",
        "description": "Preview report fixture",
        "author": None,
        "created_at": "2026-06-18T12:00:00+00:00",
        "updated_at": "2026-06-18T12:00:00+00:00",
        "export_id": "preview-v11",
        "content_items": [
            {
                "id": "exp_1",
                "type": "explainer_session",
                "title": "Preview",
                "description": None,
                "created_at": None,
                "updated_at": None,
                "tags": [],
                "metadata": {
                    "format": "tldw.explainer_session.v1",
                    "envelope": {
                        "format": "tldw.explainer_session.v1",
                        "schema_version": 1,
                        "media_type": "application/json",
                        "representations": [
                            {
                                "kind": "structured",
                                "path": payload_path,
                                "media_type": "application/json",
                                "primary": True,
                                "role": "restore_payload",
                            }
                        ],
                        "integrity": {
                            "status": "verified",
                            "algorithm": "sha256",
                            "value": f"sha256:{digest}",
                            "scope": "primary_payload",
                        },
                        "lossiness": {"mode": "lossless", "reasons": []},
                        "provenance": {},
                        "source_refs": [{"resolution_status": "resolved"}],
                        "attachments": [],
                    },
                },
                "file_path": payload_path,
                "checksum": f"sha256:{digest}",
            }
        ],
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
            "total_explainer_sessions": 1,
            "total_size_bytes": len(payload),
        },
        "metadata": {"tags": [], "categories": [], "language": "en", "license": None},
        "user_info": {"user_id": "test"},
        "features_used": ["content_envelopes", "file_inventory", "integrity_metadata"],
        "producer": {"name": "tldw_server"},
        "source_instance": {},
        "compatibility": {"min_reader_version": "1.1.0"},
        "file_inventory": [
            {
                "path": payload_path,
                "media_type": "application/json",
                "size_bytes": len(payload),
                "integrity": {
                    "status": "verified",
                    "algorithm": "sha256",
                    "value": f"sha256:{digest}",
                },
                "role": "payload",
                "content_item_ids": ["exp_1"],
            }
        ],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        zf.writestr(payload_path, payload)
        zf.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


def test_preview_v1_1_returns_compatibility_feature_and_integrity_report(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    app.dependency_overrides[chatbooks_endpoints.get_request_user] = _override_user
    app.dependency_overrides[chatbooks_endpoints.get_audit_service_for_user] = lambda: _DummyAuditService()
    data = _make_v1_1_chatbook_bytes()
    files = {"file": ("v11.chatbook", data, "application/zip")}

    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/chatbooks/preview", files=files)
    finally:
        app.dependency_overrides.pop(chatbooks_endpoints.get_request_user, None)
        app.dependency_overrides.pop(chatbooks_endpoints.get_audit_service_for_user, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["compatibility"]["manifest_version"] == "1.1.0"
    assert "file_inventory" in body["features"]["supported"]
    assert body["integrity"]["verified_files"] >= 1


def test_preview_report_treats_non_string_feature_tokens_as_unsupported(tmp_path):
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1_1,
        name="malformed features",
        description="malformed-but-parseable preview",
        features_used=["file_inventory", {}],
        file_inventory=[],
    )

    report = build_preview_report(manifest, tmp_path)

    assert report["features"]["supported"] == ["file_inventory"]
    assert report["features"]["unsupported"] == ["{}"]
    assert report["warnings"]


def test_preview_report_treats_non_list_file_inventory_as_failed_item(tmp_path):
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1_1,
        name="malformed inventory",
        description="malformed-but-parseable preview",
        features_used=[],
        file_inventory=123,
    )

    report = build_preview_report(manifest, tmp_path)

    assert report["integrity"]["verified_files"] == 0
    assert report["integrity"]["failed_files"] == [
        {"path": None, "reason": "invalid_inventory"}
    ]
    assert report["errors"]


def test_preview_report_treats_non_string_inventory_path_as_failed_item(tmp_path):
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1_1,
        name="malformed inventory path",
        description="malformed-but-parseable preview",
        features_used=[],
        file_inventory=[{"path": 123}],
    )

    report = build_preview_report(manifest, tmp_path)

    assert report["integrity"]["verified_files"] == 0
    assert report["integrity"]["failed_files"] == [
        {"path": None, "reason": "invalid_path"}
    ]
    assert report["errors"]
