from pathlib import Path


def test_openwebui_import_is_discoverable_from_user_guides() -> None:
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
    assert "WebUI_Extension/Chatbook_User_Guide.md" in index_text  # nosec B101
    assert "OpenWebUI chat JSON import" in index_text  # nosec B101
    assert 'import OpenWebUI "Export Chats" JSON files' in overview_text  # nosec B101


def test_openwebui_import_is_discoverable_from_api_docs() -> None:
    api_readme = Path("Docs/API-related/API_README.md").read_text(encoding="utf-8")
    api_tags = Path("Docs/API-related/API_Tags_Index.md").read_text(encoding="utf-8")
    api_doc = Path("Docs/API-related/Chatbook_API_Documentation.md").read_text(
        encoding="utf-8"
    )

    assert "Chatbooks - `/api/v1/chatbooks`" in api_readme  # nosec B101
    assert "source_format=openwebui_json" in api_readme  # nosec B101
    assert "Chatbook_API_Documentation.md" in api_readme  # nosec B101
    assert "| `chatbooks` | API-related/Chatbook_API_Documentation.md |" in api_tags  # nosec B101
    assert "OpenWebUI chat export JSON" in api_doc  # nosec B101


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
    assert "source_format=openwebui_json" in published_api  # nosec B101
    assert "OpenWebUI JSON chat import" in feature_status  # nosec B101
