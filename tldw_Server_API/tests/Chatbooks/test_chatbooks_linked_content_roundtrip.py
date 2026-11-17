"""Portable inline links must target newly imported content, never old destination IDs."""

import json
import zipfile
from contextlib import ExitStack
from pathlib import Path

import pytest
import yaml

from tldw_Server_API.app.core.Chat.document_generator import DocumentGeneratorService
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ChatbookVersion, ConflictResolution, ContentType
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration
NOTE = "\n# Rowan\n\nSelene\n\n"


@pytest.fixture
async def linked_archive(tmp_path, monkeypatch, request):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    with ExitStack() as stack:
        source = CharactersRAGDB(tmp_path / "source.db", client_id="1")
        destination = CharactersRAGDB(tmp_path / "destination.db", client_id="2")
        stack.callback(source.close_all_connections)
        stack.callback(destination.close_all_connections)
        destination.add_character_card({"name": "Destination only"})
        character = source.add_character_card({"name": "Rowan Archivist"})
        conversation = source.add_conversation({"title": "Rowan conversation", "character_id": character})
        question = source.add_message({"conversation_id": conversation, "sender": "user", "content": "Telescope?"})
        answer = source.add_message(
            {
                "conversation_id": conversation,
                "sender": "assistant",
                "content": "Selene",
                "parent_message_id": question,
            }
        )
        note = source.add_note(
            "Rowan Note", getattr(request, "param", NOTE), conversation_id=conversation, message_id=answer
        )
        document = DocumentGeneratorService(source, "1").create_manual_document(
            title="Rowan briefing",
            content="Selene",
            provider="manual",
            model="manual",
            conversation_id=conversation,
        )
        ids = {
            ContentType.CHARACTER: [str(character)],
            ContentType.CONVERSATION: [conversation],
            ContentType.NOTE: [note],
            ContentType.GENERATED_DOCUMENT: [str(document)],
        }
        service = ChatbookService(user_id="1", user_id_int=1, db=source)
        okay, message, path = await service.create_chatbook(
            name="Linked Rowan",
            description="Portable content",
            content_selections=ids,
            format_version=ChatbookVersion.V1_1,
            async_mode=False,
        )
        assert okay, message  # nosec B101 -- Pytest behavior assertion.
        restored = ChatbookService(user_id="2", user_id_int=2, db=destination)
        target = restored.import_dir / "linked.chatbook"
        target.write_bytes(Path(path).read_bytes())
        yield source, destination, restored, target, ids, question, answer


@pytest.mark.parametrize("contract", ["parent", "note_links"])
async def test_export_preserves_inline_relationships(linked_archive, contract):
    _source, _destination, _service, path, ids, question, answer = linked_archive
    with zipfile.ZipFile(path) as archive:
        if contract == "parent":
            conversation = json.loads(
                archive.read(f"content/conversations/conversation_{ids[ContentType.CONVERSATION][0]}.json")
            )
            messages = {row["id"]: row for row in conversation["messages"]}
            assert messages[answer].get("parent_message_id") == question  # nosec B101 -- Pytest behavior assertion.
        else:
            encoded = archive.read(f"content/notes/note_{ids[ContentType.NOTE][0]}.md").decode()
            metadata = yaml.safe_load(encoded.split("---", 2)[1])
            assert (metadata.get("conversation_id"), metadata.get("message_id")) == (  # nosec B101 -- Pytest behavior assertion.
                ids[ContentType.CONVERSATION][0],
                answer,
            )


async def test_public_roundtrip_remaps_links_and_preserves_note_whitespace(linked_archive):
    _source, destination, service, path, ids, _question, _answer = linked_archive
    okay, message, _result = await service.import_chatbook(
        str(path),
        content_selections=ids,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- Pytest behavior assertion.
    conversation = destination.search_conversations_by_title("Rowan conversation")[0]
    messages = {row["content"]: row for row in destination.get_messages_for_conversation(conversation["id"])}
    note = destination.list_notes()[0]
    documents = DocumentGeneratorService(destination, "2").list_documents()
    assert conversation["id"] != ids[ContentType.CONVERSATION][0]  # nosec B101 -- Pytest behavior assertion.
    assert messages["Selene"]["parent_message_id"] == messages["Telescope?"]["id"]  # nosec B101 -- Pytest behavior assertion.
    assert (note["content"], note["conversation_id"], note["message_id"]) == (  # nosec B101 -- Pytest behavior assertion.
        NOTE,
        conversation["id"],
        messages["Selene"]["id"],
    )
    assert len(documents) == 1 and documents[0]["conversation_id"] == conversation["id"]  # nosec B101 -- Pytest behavior assertion.


async def test_unselected_conversation_cannot_bind_notes_or_documents_to_old_destination_ids(linked_archive):
    _source, destination, service, path, ids, _question, answer = linked_archive
    old_id = ids[ContentType.CONVERSATION][0]
    unrelated = destination.add_conversation({"id": old_id, "title": "Unrelated", "character_id": 1})
    assert unrelated == old_id  # nosec B101 -- Pytest fixture collision guard.
    destination.add_message({"id": answer, "conversation_id": unrelated, "sender": "assistant", "content": "Private"})
    selections = {key: value for key, value in ids.items() if key in {ContentType.NOTE, ContentType.GENERATED_DOCUMENT}}
    okay, message, result = await service.import_chatbook(
        str(path),
        content_selections=selections,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- Pytest behavior assertion.
    note = destination.list_notes()[0]
    documents = DocumentGeneratorService(destination, "2").list_documents()
    assert (note["conversation_id"], note["message_id"]) == (None, None)  # nosec B101 -- Pytest behavior assertion.
    assert len(documents) == 1 and documents[0]["conversation_id"] is None  # nosec B101 -- Pytest behavior assertion.
    assert any("conversation" in warning.lower() for warning in result["warnings"])  # nosec B101 -- Pytest behavior assertion.
    assert destination.get_messages_for_conversation(unrelated)[0]["content"] == "Private"  # nosec B101 -- Pytest behavior assertion.


@pytest.mark.parametrize("case", ["child_first", "external_parent", "duplicate_id", "cycle", "write_failure"])
async def test_message_mapping_is_local_order_independent_and_atomic(linked_archive, tmp_path, monkeypatch, case):
    from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
        ChatbookManifest,
        ContentItem,
        ImportJob,
        ImportStatus,
    )

    _source, destination, service, path, ids, question, answer = linked_archive
    source_conversation = ids[ContentType.CONVERSATION][0]
    payload_path = f"content/conversations/conversation_{source_conversation}.json"
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read(payload_path))
    by_id = {row["id"]: row for row in payload["messages"]}
    by_id[answer]["parent_message_id"] = question
    if case == "child_first":
        payload["messages"].reverse()
    elif case == "external_parent":
        unrelated = destination.add_conversation({"title": "Unrelated parent", "character_id": 1})
        outside = destination.add_message({"conversation_id": unrelated, "sender": "user", "content": "Private parent"})
        by_id[answer]["parent_message_id"] = outside
    elif case == "duplicate_id":
        by_id[answer]["id"] = question
    elif case == "cycle":
        by_id[question]["parent_message_id"] = answer
    else:
        real_add = destination.add_message

        def fail_second(data):
            if data["content"] == "Selene":
                raise ValueError("Synthetic second message write failure")
            return real_add(data)

        monkeypatch.setattr(destination, "add_message", fail_second)
    work = tmp_path / "message-import"
    target = work / payload_path
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(payload))
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Message safety",
        description="",
        content_items=[
            ContentItem(
                id=source_conversation,
                type=ContentType.CONVERSATION,
                title="Rowan conversation",
                file_path=payload_path,
            )
        ],
    )
    status = ImportJob(job_id="message-import", user_id="2", status=ImportStatus.IN_PROGRESS, chatbook_path="local")
    conversations, messages = {}, {}
    service._import_conversations(
        work,
        manifest,
        [source_conversation],
        ConflictResolution.SKIP,
        False,
        status,
        conversation_id_map=conversations,
        message_id_map=messages,
    )
    if case in {"duplicate_id", "cycle", "write_failure"}:
        assert status.failed_items == 1 and not conversations and not messages  # nosec B101 -- Pytest behavior assertion.
        assert not destination.search_conversations_by_title("Rowan conversation")  # nosec B101 -- Pytest behavior assertion.
    else:
        imported = destination.get_messages_for_conversation(conversations[source_conversation])
        by_content = {row["content"]: row for row in imported}
        expected = by_content["Telescope?"]["id"] if case == "child_first" else None
        assert by_content["Selene"]["parent_message_id"] == expected  # nosec B101 -- Pytest behavior assertion.
        assert messages[(source_conversation, answer)] == by_content["Selene"]["id"]  # nosec B101 -- Pytest behavior assertion.
        if case == "external_parent":
            assert any("parent" in warning for warning in status.warnings)  # nosec B101 -- Pytest behavior assertion.


def test_note_links_survive_active_sync_capture(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Chatbooks.chatbook_models import ChatbookManifest, ImportJob, ImportStatus
    from tldw_Server_API.tests.Notes.test_notes_organization_sync_surfaces import (
        _conversation,
        _domains,
        _patch_active_service,
    )
    from tldw_Server_API.tests.Sync.notes_organization_test_support import build_ready_notes_sync_stack

    db, sync_store, sync_service = build_ready_notes_sync_stack(tmp_path)
    _patch_active_service(monkeypatch, sync_service)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    conversation = _conversation(db, sync_service, title="Destination Chat")
    message = db.add_message({"conversation_id": conversation, "sender": "user", "content": "Selene"})
    service = ChatbookService(user_id="user-1", db=db)
    note_path = tmp_path / "content/notes/note_source-note.md"
    note_path.parent.mkdir(parents=True)
    note_path.write_text(
        "---\ntitle: Linked capture\nconversation_id: source-chat\nmessage_id: source-message\n---\n\n" + NOTE
    )
    status = ImportJob(job_id="note-capture", user_id="user-1", status=ImportStatus.IN_PROGRESS, chatbook_path="local")
    service._import_notes(
        tmp_path,
        ChatbookManifest(version=ChatbookVersion.V1, name="Capture", description=""),
        ["source-note"],
        ConflictResolution.SKIP,
        False,
        status,
        conversation_id_map={"source-chat": conversation},
        message_id_map={("source-chat", "source-message"): message},
    )
    note = db.list_notes()[0]
    assert status.successful_items == 1 and not status.failed_items  # nosec B101 -- Pytest behavior assertion.
    assert (note["content"], note["conversation_id"], note["message_id"]) == (NOTE, conversation, message)  # nosec B101 -- Pytest behavior assertion.
    assert _domains(sync_store) == ["notes.note"]  # nosec B101 -- Pytest behavior assertion.


@pytest.mark.parametrize("linked_archive", ["\r\nRowan\r\n\r\n", "\rRowan\r\r"], indirect=True)
async def test_note_line_endings_round_trip_exactly(linked_archive, request):
    _source, destination, service, path, ids, _question, _answer = linked_archive
    okay, message, _result = await service.import_chatbook(
        str(path),
        content_selections=ids,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- Pytest behavior assertion.
    expected = request.node.callspec.params["linked_archive"]
    assert destination.list_notes()[0]["content"] == expected  # nosec B101 -- Pytest behavior assertion.
