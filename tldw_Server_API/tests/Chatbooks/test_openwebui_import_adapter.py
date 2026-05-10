import json

import pytest

from tldw_Server_API.app.core.Chatbooks.import_adapters.openwebui import (
    load_openwebui_export,
    preview_openwebui_export,
)


pytestmark = pytest.mark.unit


def _standard_export():
    return [
        {
            "id": "chat-standard",
            "chat": {
                "title": "Research thread",
                "models": ["gpt-4o"],
                "history": {
                    "currentId": "assistant-1",
                    "messages": {
                        "root-user": {
                            "id": "root-user",
                            "role": "user",
                            "content": "Explain retrieval augmented generation.",
                            "timestamp": 1700000000,
                            "childrenIds": ["assistant-1", "assistant-branch"],
                        },
                        "assistant-1": {
                            "id": "assistant-1",
                            "role": "assistant",
                            "content": "Main answer",
                            "parentId": "root-user",
                            "timestamp": 1700000001,
                            "model": "gpt-4o",
                        },
                        "assistant-branch": {
                            "id": "assistant-branch",
                            "role": "assistant",
                            "content": "Alternative answer",
                            "parentId": "root-user",
                            "timestamp": 1700000002,
                            "files": [{"id": "file-1", "name": "source.pdf"}],
                        },
                    },
                },
            },
        }
    ]


def test_load_openwebui_export_accepts_standard_wrapper_and_preserves_branches(tmp_path):
    export_path = tmp_path / "openwebui.json"
    export_path.write_text(json.dumps(_standard_export()), encoding="utf-8")

    parsed = load_openwebui_export(export_path)

    assert parsed.malformed_chat_count == 0
    assert len(parsed.chats) == 1
    chat = parsed.chats[0]
    assert chat.external_ref == "chat-standard"
    assert chat.title == "Research thread"
    assert chat.history_current_id == "assistant-1"
    assert chat.is_branched is True
    assert [message.source_id for message in chat.messages] == [
        "root-user",
        "assistant-1",
        "assistant-branch",
    ]
    assert chat.messages[2].attachment_refs == [{"id": "file-1", "name": "source.pdf"}]


def test_preview_counts_branches_attachments_and_duplicates(tmp_path):
    export_path = tmp_path / "openwebui.json"
    export_path.write_text(json.dumps(_standard_export()), encoding="utf-8")

    preview = preview_openwebui_export(
        export_path,
        duplicate_lookup=lambda external_ref: external_ref == "chat-standard",
    )

    assert preview.chat_count == 1
    assert preview.message_count == 3
    assert preview.branched_chat_count == 1
    assert preview.duplicate_chat_count == 1
    assert preview.attachment_reference_count == 1
    assert preview.malformed_chat_count == 0
    assert preview.items[0].duplicate is True
    assert preview.items[0].warning_count == 0


def test_load_openwebui_export_accepts_legacy_chat_objects(tmp_path):
    export_path = tmp_path / "legacy.json"
    export_path.write_text(
        json.dumps(
            [
                {
                    "title": "Legacy chat",
                    "history": {
                        "messages": {
                            "m1": {
                                "role": "user",
                                "content": "Legacy message",
                                "timestamp": 1700000100,
                            }
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    parsed = load_openwebui_export(export_path)

    assert parsed.malformed_chat_count == 0
    assert len(parsed.chats) == 1
    assert parsed.chats[0].title == "Legacy chat"
    assert parsed.chats[0].external_ref.startswith("openwebui:0:")
    assert parsed.chats[0].messages[0].source_id == "m1"


def test_load_openwebui_export_rejects_non_array_root(tmp_path):
    export_path = tmp_path / "bad.json"
    export_path.write_text(json.dumps({"chat": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="top-level JSON value must be an array"):
        load_openwebui_export(export_path)


def test_load_openwebui_export_rejects_non_utf8_json(tmp_path):
    export_path = tmp_path / "bad-encoding.json"
    export_path.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="Malformed OpenWebUI JSON export"):
        load_openwebui_export(export_path)
