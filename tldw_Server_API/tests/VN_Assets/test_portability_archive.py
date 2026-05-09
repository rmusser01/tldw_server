import io
import json
import zipfile

import pytest

from tldw_Server_API.app.core.VN_Assets.portability.archive import (
    validate_archive_members,
)
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import (
    canonical_json_bytes,
    canonical_payload_fingerprint,
    sha256_bytes,
    sha256_file,
)


REQUIRED_MEMBER_PAYLOADS = {
    "manifest.json": b'{"schema_version":"tldw.vnpack.v1"}',
    "metadata/pack.json": b"{}",
    "metadata/slots.json": b"[]",
    "metadata/items.json": b"[]",
    "checksums/sha256.json": b"{}",
}


def _zip_with_members(members: dict[str, bytes] | list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        if isinstance(members, dict):
            iterable = members.items()
        else:
            iterable = ((name, b"data") for name in members)
        for name, payload in iterable:
            zf.writestr(name, payload)
    return buffer.getvalue()


def _valid_archive_members(**overrides: bytes) -> dict[str, bytes]:
    return {**REQUIRED_MEMBER_PAYLOADS, **overrides}


def _with_encrypted_zip_flag(payload: bytes) -> bytes:
    data = bytearray(payload)
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        while (index := data.find(signature, start)) != -1:
            flag_start = index + flag_offset
            flag_end = flag_start + 2
            flags = int.from_bytes(data[flag_start:flag_end], "little")
            data[flag_start:flag_end] = (flags | 0x1).to_bytes(2, "little")
            start = index + len(signature)
    return bytes(data)


def _replace_raw_zip_member_name(payload: bytes, old_name: str, new_name: str) -> bytes:
    old = old_name.encode("utf-8")
    new = new_name.encode("utf-8")
    if len(old) != len(new):
        raise ValueError("zip_member_name_replacement_length_mismatch")
    return payload.replace(old, new)


def test_validate_archive_members_accepts_valid_contract_archive(tmp_path):
    archive_path = tmp_path / "valid.tldw-vnpack"
    archive_path.write_bytes(
        _zip_with_members(
            _valid_archive_members(
                **{
                    "README.md": b"# Demo\n",
                    "assets/items/sprite.happy.png": b"image-bytes",
                    "signatures/README.md": b"reserved",
                }
            )
        )
    )

    assert validate_archive_members(archive_path) == [  # nosec B101
        "manifest.json",
        "metadata/pack.json",
        "metadata/slots.json",
        "metadata/items.json",
        "checksums/sha256.json",
        "README.md",
        "assets/items/sprite.happy.png",
        "signatures/README.md",
    ]


def test_validate_archive_members_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "bad.tldw-vnpack"
    archive_path.write_bytes(
        _zip_with_members(_valid_archive_members(**{"../escape.png": b"data"}))
    )

    with pytest.raises(ValueError, match="unsafe_archive_member"):
        validate_archive_members(archive_path)


@pytest.mark.parametrize(
    "member_name",
    [
        "/manifest.json",
        "C:/temp/manifest.json",
        "assets\\items\\sprite.png",
        "assets/\x00sprite.png",
        "metadata//items.json",
        "assets/items/C:/sprite.png",
    ],
)
def test_validate_archive_members_rejects_unsafe_member_names(
    tmp_path, member_name: str
):
    archive_path = tmp_path / "bad-name.tldw-vnpack"
    archive_path.write_bytes(
        _zip_with_members(_valid_archive_members(**{member_name: b"data"}))
    )

    with pytest.raises(ValueError, match="unsafe_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_rejects_raw_null_byte_member_names(tmp_path):
    archive_path = tmp_path / "raw-null.tldw-vnpack"
    member_name = "assets/items/null-safe.png"
    archive_bytes = _zip_with_members(
        _valid_archive_members(**{member_name: b"image-bytes"})
    )
    archive_path.write_bytes(
        _replace_raw_zip_member_name(
            archive_bytes,
            member_name,
            "assets/items/null\x00safe.png",
        )
    )

    with pytest.raises(ValueError, match="unsafe_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_rejects_symlink_entries(tmp_path):
    archive_path = tmp_path / "symlink.tldw-vnpack"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, payload in REQUIRED_MEMBER_PAYLOADS.items():
            zf.writestr(name, payload)
        info = zipfile.ZipInfo("assets/items/link.png")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        zf.writestr(info, b"target.png")
    archive_path.write_bytes(buffer.getvalue())

    with pytest.raises(ValueError, match="unsafe_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_rejects_encrypted_entries(tmp_path):
    archive_path = tmp_path / "encrypted.tldw-vnpack"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, payload in REQUIRED_MEMBER_PAYLOADS.items():
            zf.writestr(name, payload)
    archive_path.write_bytes(_with_encrypted_zip_flag(buffer.getvalue()))

    with pytest.raises(ValueError, match="unsupported_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_rejects_duplicate_normalized_paths(tmp_path):
    archive_path = tmp_path / "duplicates.tldw-vnpack"
    archive_path.write_bytes(
        _zip_with_members(
            {
                **REQUIRED_MEMBER_PAYLOADS,
                "metadata/extra.json": b"one",
                "metadata/./extra.json": b"two",
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_rejects_unexpected_top_level_file(tmp_path):
    archive_path = tmp_path / "unexpected-file.tldw-vnpack"
    archive_path.write_bytes(
        _zip_with_members(_valid_archive_members(**{"unexpected.json": b"data"}))
    )

    with pytest.raises(ValueError, match="unexpected_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_rejects_unexpected_top_level_directory(tmp_path):
    archive_path = tmp_path / "unexpected-dir.tldw-vnpack"
    archive_path.write_bytes(
        _zip_with_members(_valid_archive_members(**{"content/item.json": b"data"}))
    )

    with pytest.raises(ValueError, match="unexpected_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_rejects_missing_required_files(tmp_path):
    archive_path = tmp_path / "missing.tldw-vnpack"
    members = dict(REQUIRED_MEMBER_PAYLOADS)
    del members["metadata/items.json"]
    archive_path.write_bytes(_zip_with_members(members))

    with pytest.raises(ValueError, match="missing_required_archive_member"):
        validate_archive_members(archive_path)


def test_validate_archive_members_rejects_member_over_size_limit(tmp_path):
    archive_path = tmp_path / "member-too-large.tldw-vnpack"
    archive_path.write_bytes(
        _zip_with_members(_valid_archive_members(**{"assets/items/big.png": b"12345"}))
    )

    with pytest.raises(ValueError, match="archive_member_too_large"):
        validate_archive_members(archive_path, max_member_size_bytes=4)


def test_validate_archive_members_rejects_archive_over_size_limit(tmp_path):
    archive_path = tmp_path / "archive-too-large.tldw-vnpack"
    archive_path.write_bytes(_zip_with_members(REQUIRED_MEMBER_PAYLOADS))

    with pytest.raises(ValueError, match="archive_too_large"):
        validate_archive_members(archive_path, max_archive_size_bytes=10)


def test_sha256_helpers_hash_bytes_and_files(tmp_path):
    payload = b"portable-vn-pack"
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(payload)

    assert sha256_bytes(payload) == sha256_file(payload_path)  # nosec B101


def test_canonical_json_bytes_are_deterministic_and_compact():
    payload = {"b": 2, "a": [3, {"d": 4, "c": 5}]}

    assert canonical_json_bytes(payload) == b'{"a":[3,{"c":5,"d":4}],"b":2}'  # nosec B101
    assert json.loads(canonical_json_bytes(payload)) == payload  # nosec B101


def test_canonical_payload_fingerprint_ignores_export_metadata():
    left = {
        "manifest": {
            "exported_at": "2026-01-01",
            "export_id": "a",
            "pack_title": "Demo",
            "archive_sha256": "left-archive",
            "canonical_payload_fingerprint": "left-payload",
        },
        "items": [{"slot_key": "sprite.happy", "checksum": "abc"}],
    }
    right = {
        "manifest": {
            "exported_at": "2026-02-01",
            "export_id": "b",
            "pack_title": "Demo",
            "archive_sha256": "right-archive",
            "canonical_payload_fingerprint": "right-payload",
        },
        "items": [{"checksum": "abc", "slot_key": "sprite.happy"}],
    }

    assert canonical_payload_fingerprint(left) == canonical_payload_fingerprint(right)  # nosec B101


def test_canonical_payload_fingerprint_sorts_semantic_lists():
    left = {
        "slots": [
            {"asset_type": "sprite", "slot_key": "sprite.neutral"},
            {"asset_type": "sprite", "slot_key": "sprite.happy"},
        ],
        "items": [
            {
                "slot_key": "sprite.neutral",
                "variant_index": 1,
                "checksum": "bbb",
                "source_item_fingerprint": "item-b",
            },
            {
                "slot_key": "sprite.happy",
                "variant_index": 0,
                "checksum": "aaa",
                "source_item_fingerprint": "item-a",
            },
        ],
    }
    right = {
        "items": [
            {
                "checksum": "aaa",
                "source_item_fingerprint": "item-a",
                "slot_key": "sprite.happy",
                "variant_index": 0,
            },
            {
                "checksum": "bbb",
                "source_item_fingerprint": "item-b",
                "slot_key": "sprite.neutral",
                "variant_index": 1,
            },
        ],
        "slots": [
            {"slot_key": "sprite.happy", "asset_type": "sprite"},
            {"slot_key": "sprite.neutral", "asset_type": "sprite"},
        ],
    }

    assert canonical_payload_fingerprint(left) == canonical_payload_fingerprint(right)  # nosec B101


def test_canonical_payload_fingerprint_preserves_semantic_volatile_named_fields():
    left = {
        "manifest": {"exported_at": "2026-01-01", "pack_title": "Demo"},
        "items": [{"slot_key": "sprite.happy", "archive_sha256": "semantic-a"}],
    }
    right = {
        "manifest": {"exported_at": "2026-02-01", "pack_title": "Demo"},
        "items": [{"slot_key": "sprite.happy", "archive_sha256": "semantic-b"}],
    }

    assert canonical_payload_fingerprint(left) != canonical_payload_fingerprint(right)  # nosec B101


def test_canonical_payload_fingerprint_preserves_order_sensitive_lists():
    left = {
        "manifest": {"pack_title": "Demo"},
        "story_beats": [
            {"id": "opening", "text": "Open the scene."},
            {"id": "choice", "text": "Offer the player a choice."},
        ],
        "runtime_sequence": ["opening", "choice"],
    }
    right = {
        "manifest": {"pack_title": "Demo"},
        "story_beats": [
            {"id": "choice", "text": "Offer the player a choice."},
            {"id": "opening", "text": "Open the scene."},
        ],
        "runtime_sequence": ["choice", "opening"],
    }

    assert canonical_payload_fingerprint(left) != canonical_payload_fingerprint(right)  # nosec B101


def test_canonical_payload_fingerprint_preserves_unknown_nested_sections_order():
    left = {
        "story": {
            "sections": [
                {"id": "opening", "text": "Open the scene."},
                {"id": "choice", "text": "Offer the player a choice."},
            ]
        }
    }
    right = {
        "story": {
            "sections": [
                {"id": "choice", "text": "Offer the player a choice."},
                {"id": "opening", "text": "Open the scene."},
            ]
        }
    }

    assert canonical_payload_fingerprint(left) != canonical_payload_fingerprint(right)  # nosec B101


def test_canonical_payload_fingerprint_changes_for_payload_content():
    left = {
        "manifest": {"pack_title": "Demo"},
        "items": [{"slot_key": "sprite.happy", "checksum": "abc"}],
    }
    right = {
        "manifest": {"pack_title": "Demo"},
        "items": [{"slot_key": "sprite.happy", "checksum": "def"}],
    }

    assert canonical_payload_fingerprint(left) != canonical_payload_fingerprint(right)  # nosec B101
