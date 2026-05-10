import pytest

from tldw_Server_API.app.core.Chatbooks.openwebui_folders import (
    build_openwebui_namespace_segments,
    mirror_openwebui_folder_for_conversation,
    sanitize_openwebui_folder_segment,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


@pytest.fixture()
def folder_db(tmp_path):
    db = CharactersRAGDB(
        db_path=str(tmp_path / "openwebui-folder-mirroring.sqlite"),
        client_id="openwebui-folder-user",
    )
    character_id = db.add_character_card({"name": "Folder Assistant"})
    conversation_id = db.add_conversation({"character_id": character_id, "title": "Foldered Chat"})
    return db, conversation_id


def test_sanitize_openwebui_folder_segment_keeps_safe_readable_name():
    assert sanitize_openwebui_folder_segment(" Research / Papers ") == "Research _ Papers"
    assert sanitize_openwebui_folder_segment("") == "Untitled"
    assert sanitize_openwebui_folder_segment("   ") == "Untitled"


def test_build_openwebui_namespace_segments_uses_label_and_user_id():
    assert build_openwebui_namespace_segments("Alice Example", "user-a") == [
        "OpenWebUI",
        "Alice Example (user-a)",
    ]


def test_mirror_openwebui_folder_creates_namespace_path_and_links_conversation(folder_db):
    db, conversation_id = folder_db
    namespace = build_openwebui_namespace_segments("Alice", "user-a")

    result = mirror_openwebui_folder_for_conversation(
        db,
        conversation_id=conversation_id,
        namespace_segments=namespace,
        source_path_segments=["Research", "Papers"],
        source_folder_id="folder-papers",
        metadata={"source": "test"},
    )

    assert result.final_collection_id is not None
    assert result.created_collections == 4
    assert result.conversation_keyword_linked is True
    assert db.get_keyword_collection_by_name("OpenWebUI") is not None
    assert db.get_keyword_collection_by_name("Alice (user-a)") is not None
    final_collection = db.get_keyword_collection_by_id(result.final_collection_id)
    assert final_collection["name"] == "Papers"

    collection_keywords = db.get_keywords_for_collection(result.final_collection_id)
    conversation_keywords = db.get_keywords_for_conversation(conversation_id)
    collection_keyword_ids = {keyword["id"] for keyword in collection_keywords}
    conversation_keyword_ids = {keyword["id"] for keyword in conversation_keywords}
    assert collection_keyword_ids & conversation_keyword_ids

    rerun = mirror_openwebui_folder_for_conversation(
        db,
        conversation_id=conversation_id,
        namespace_segments=namespace,
        source_path_segments=["Research", "Papers"],
        source_folder_id="folder-papers",
        metadata={"source": "test"},
    )

    assert rerun.final_collection_id == result.final_collection_id
    assert rerun.created_collections == 0
    assert rerun.conversation_keyword_linked is False


def test_mirror_openwebui_folder_disambiguates_global_collection_name_collision(folder_db):
    db, conversation_id = folder_db
    existing_id = db.add_keyword_collection("Research")
    assert existing_id is not None

    result = mirror_openwebui_folder_for_conversation(
        db,
        conversation_id=conversation_id,
        namespace_segments=build_openwebui_namespace_segments("Alice", "user-a"),
        source_path_segments=["Research"],
        source_folder_id="folder-research",
        metadata={},
    )

    assert result.final_collection_id != existing_id
    final_collection = db.get_keyword_collection_by_id(result.final_collection_id)
    assert final_collection["name"].startswith("Research ")
    assert any("disambiguated" in warning for warning in result.warnings)
