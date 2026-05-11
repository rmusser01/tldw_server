from pathlib import Path

import yaml


def test_openwebui_import_is_discoverable_from_user_guides() -> None:
    readme_text = Path("README.md").read_text(encoding="utf-8")
    guide = Path("Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md")
    guide_text = guide.read_text(encoding="utf-8")
    index_text = Path("Docs/User_Guides/index.md").read_text(encoding="utf-8")
    overview_text = Path("Docs/User_Guides/WebUI_Extension/User_Guide.md").read_text(
        encoding="utf-8"
    )

    assert "OpenWebUI Chat Import" in guide_text  # nosec B101 - pytest assertion
    assert "Export Chats" in guide_text  # nosec B101 - pytest assertion
    assert "message branches" in guide_text  # nosec B101 - pytest assertion
    assert "source metadata" in guide_text  # nosec B101 - pytest assertion
    assert "OpenWebUI database" in guide_text  # nosec B101 - pytest assertion
    assert "webui.db" in guide_text  # nosec B101 - pytest assertion
    assert "selected OpenWebUI user" in guide_text  # nosec B101 - pytest assertion
    assert "OpenWebUI / <selected user>" in guide_text  # nosec B101 - pytest assertion
    assert "OpenWebUI chat JSON migration" in readme_text  # nosec B101
    assert "OpenWebUI webui.db migration" in readme_text  # nosec B101
    assert "WebUI_Extension/Chatbook_User_Guide.md" in readme_text  # nosec B101
    assert "WebUI_Extension/Chatbook_User_Guide.md" in index_text  # nosec B101
    assert "OpenWebUI chat JSON and database import" in index_text  # nosec B101
    assert 'import OpenWebUI "Export Chats" JSON files and uploaded webui.db databases' in overview_text  # nosec B101


def test_openwebui_import_is_discoverable_from_api_docs() -> None:
    api_readme = Path("Docs/API-related/API_README.md").read_text(encoding="utf-8")
    api_tags = Path("Docs/API-related/API_Tags_Index.md").read_text(encoding="utf-8")
    api_doc = Path("Docs/API-related/Chatbook_API_Documentation.md").read_text(
        encoding="utf-8"
    )

    assert "Chatbooks - `/api/v1/chatbooks`" in api_readme  # nosec B101
    assert "source_format=openwebui_json" in api_readme  # nosec B101
    assert "source_format=openwebui_db" in api_readme  # nosec B101
    assert "Chatbook_API_Documentation.md" in api_readme  # nosec B101
    assert "| `chatbooks` | API-related/Chatbook_API_Documentation.md |" in api_tags  # nosec B101
    assert "OpenWebUI chat export JSON" in api_doc  # nosec B101
    assert "OpenWebUI webui.db database" in api_doc  # nosec B101
    assert "selected_openwebui_user_id" in api_doc  # nosec B101
    assert "OpenWebUI / <selected user>" in api_doc  # nosec B101


def test_openwebui_import_is_reflected_in_published_docs() -> None:
    published_guide = Path(
        "Docs/Published/User_Guides/WebUI_Extension/Chatbook_User_Guide.md"
    ).read_text(encoding="utf-8")
    published_api = Path("Docs/Published/API-related/API_README.md").read_text(
        encoding="utf-8"
    )
    feature_status = Path("Docs/Published/Overview/Feature_Status.md").read_text(
        encoding="utf-8"
    )

    assert "OpenWebUI Chat Import" in published_guide  # nosec B101
    assert "OpenWebUI database" in published_guide  # nosec B101
    assert "source_format=openwebui_json" in published_api  # nosec B101
    assert "source_format=openwebui_db" in published_api  # nosec B101
    assert "OpenWebUI JSON and database chat import" in feature_status  # nosec B101


def test_chatbook_import_docs_match_multipart_contract() -> None:
    source_api = Path("Docs/API-related/Chatbook_API_Documentation.md").read_text(
        encoding="utf-8"
    )
    published_api = Path(
        "Docs/Published/API-related/Chatbook_API_Documentation.md"
    ).read_text(encoding="utf-8")
    source_guide = Path(
        "Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md"
    ).read_text(encoding="utf-8")
    published_guide = Path(
        "Docs/Published/User_Guides/WebUI_Extension/Chatbook_User_Guide.md"
    ).read_text(encoding="utf-8")

    for api_text in (source_api, published_api):
        assert "**Request**: Multipart form data" in api_text  # nosec B101
        assert "source_format` (form field)" in api_text  # nosec B101
        assert "selected_openwebui_user_id` (form field)" in api_text  # nosec B101
        assert "chatbooks/import?conflict_resolution" not in api_text  # nosec B101
        assert '"import_media": false' in api_text  # nosec B101
        assert '"selected_openwebui_user_id": "user_abc123"' in api_text  # nosec B101

    for guide_text in (source_guide, published_guide):
        assert "**Import Media**: Not supported yet" in guide_text  # nosec B101
        assert "**Import Embeddings**: Not supported yet" in guide_text  # nosec B101
        assert "Keep this set to false" in guide_text  # nosec B101
        assert "Default: true" not in guide_text  # nosec B101
        assert "Files, images, and artifacts import as metadata references only" in guide_text  # nosec B101


def test_chatbook_openapi_documents_openwebui_multipart_fields() -> None:
    for path in (
        Path("Docs/API-related/chatbook_openapi.yaml"),
        Path("Docs/Published/API-related/chatbook_openapi.yaml"),
    ):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        import_properties = (
            spec["paths"]["/chatbooks/import"]["post"]["requestBody"]["content"][
                "multipart/form-data"
            ]["schema"]["properties"]
        )
        preview_properties = (
            spec["paths"]["/chatbooks/preview"]["post"]["requestBody"]["content"][
                "multipart/form-data"
            ]["schema"]["properties"]
        )

        assert import_properties["source_format"]["enum"] == [  # nosec B101
            "chatbook",
            "openwebui_json",
            "openwebui_db",
        ]
        assert "selected_openwebui_user_id" in import_properties  # nosec B101
        assert import_properties["content_selections"]["type"] == "string"  # nosec B101
        assert preview_properties["source_format"]["enum"] == [  # nosec B101
            "chatbook",
            "openwebui_json",
            "openwebui_db",
        ]
        assert "/chatbooks/import/jobs/{job_id}" in spec["paths"]  # nosec B101


def test_chatbook_openapi_database_import_result_matches_backend_schema() -> None:
    for path in (
        Path("Docs/API-related/chatbook_openapi.yaml"),
        Path("Docs/Published/API-related/chatbook_openapi.yaml"),
    ):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        properties = (
            spec["components"]["schemas"]["OpenWebUIDatabaseImportResult"]["allOf"][1]["properties"]
        )

        assert "selected_user_id" in properties  # nosec B101
        assert "selected_user_label" in properties  # nosec B101
        assert "mirrored_folders" in properties  # nosec B101
        assert "folder_links" in properties  # nosec B101
        assert "source_user_id" not in properties  # nosec B101
        assert "source_user_label" not in properties  # nosec B101
        assert "folder_count" not in properties  # nosec B101
        assert "selected_user" not in properties  # nosec B101
