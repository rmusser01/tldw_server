import hashlib

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_format_v1_1 import (
    _required_import_payload_paths,
    build_file_inventory,
    ensure_known_features,
)

pytestmark = pytest.mark.unit


def test_build_file_inventory_excludes_manifest_and_hashes_payload(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    payload = tmp_path / "content" / "notes" / "note_1.md"
    payload.parent.mkdir(parents=True)
    payload.write_text("hello", encoding="utf-8")

    inventory = build_file_inventory(tmp_path)

    assert [item["path"] for item in inventory] == ["content/notes/note_1.md"]
    assert inventory[0]["media_type"] == "text/markdown"
    assert inventory[0]["size_bytes"] == 5
    assert inventory[0]["integrity"]["status"] == "verified"
    assert inventory[0]["integrity"]["algorithm"] == "sha256"
    assert inventory[0]["integrity"]["value"] == (
        f"sha256:{hashlib.sha256(b'hello').hexdigest()}"
    )
    assert inventory[0]["role"] == "payload"
    assert inventory[0]["content_item_ids"] == []


def test_build_file_inventory_excludes_checksum_sidecars(tmp_path):
    payload = tmp_path / "content" / "notes" / "note_1.json"
    payload.parent.mkdir(parents=True)
    payload.write_text("{}", encoding="utf-8")
    (tmp_path / "archive.chatbook.zip.sha256").write_text("sha256:abc", encoding="utf-8")

    inventory = build_file_inventory(tmp_path)

    assert [item["path"] for item in inventory] == ["content/notes/note_1.json"]


def test_build_file_inventory_sorts_paths_and_assigns_roles(tmp_path):
    files = [
        tmp_path / "rendered" / "notes" / "note_1.md",
        tmp_path / "schemas" / "note.schema.json",
        tmp_path / "README.md",
        tmp_path / "content" / "notes" / "note_1.json",
    ]
    for file_path in files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(file_path.name, encoding="utf-8")

    inventory = build_file_inventory(tmp_path)

    assert [(item["path"], item["role"]) for item in inventory] == [
        ("README.md", "readme"),
        ("content/notes/note_1.json", "payload"),
        ("rendered/notes/note_1.md", "rendered"),
        ("schemas/note.schema.json", "schema"),
    ]


def test_build_file_inventory_uses_payload_role_for_uncategorized_paths(tmp_path):
    payload = tmp_path / "metadata.json"
    payload.write_text("{}", encoding="utf-8")

    inventory = build_file_inventory(tmp_path)

    assert [(item["path"], item["role"]) for item in inventory] == [
        ("metadata.json", "payload"),
    ]


def test_ensure_known_features_reports_unknown_tokens():
    report = ensure_known_features(["content_envelopes", "future_feature"])

    assert report["supported"] == ["content_envelopes"]
    assert report["unsupported"] == ["future_feature"]


def test_required_import_payload_paths_prefers_explicit_file_path_for_standard_types():
    paths = _required_import_payload_paths(
        [
            {
                "id": "1",
                "type": "note",
                "file_path": "content/custom/note-one.md",
            },
            {
                "id": "2",
                "type": "conversation",
                "file_path": "content/custom/conversation-two.json",
            },
        ]
    )

    assert paths == [
        "content/custom/note-one.md",
        "content/custom/conversation-two.json",
    ]
