from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.RPG.errors import RPGValidationError
from tldw_Server_API.app.core.RPG.rules.refs import (
    RulesPackRef,
    normalize_rules_pack_ref_payloads,
    rules_pack_ref_from_dict,
    rules_pack_ref_to_dict,
)

pytestmark = pytest.mark.unit


def _now() -> datetime:
    return datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)


def test_normalize_rules_pack_refs_accepts_media_item_and_collection():
    refs = normalize_rules_pack_ref_payloads(
        [
            {"source_type": "media_item", "source_id": 7, "display_name": "Player Rules"},
            {"source_type": "media_collection", "source_id": 3},
        ],
        existing_refs=[],
        now=_now(),
    )

    assert [ref.ref_id for ref in refs] == ["media_item:7", "media_collection:3"]  # nosec B101
    assert refs[0].display_name == "Player Rules"  # nosec B101
    assert refs[1].display_name == "media_collection:3"  # nosec B101
    assert all(ref.enabled is True for ref in refs)  # nosec B101


def test_normalize_rules_pack_refs_rejects_unknown_source_type():
    with pytest.raises(RPGValidationError, match="invalid_rules_pack_ref_source_type"):
        normalize_rules_pack_ref_payloads(
            [{"source_type": "web", "source_id": 7}],
            existing_refs=[],
            now=_now(),
        )


@pytest.mark.parametrize("source_id", [0, -1, "7", True])
def test_normalize_rules_pack_refs_rejects_non_positive_source_id(source_id):
    with pytest.raises(RPGValidationError, match="invalid_rules_pack_ref_source_id"):
        normalize_rules_pack_ref_payloads(
            [{"source_type": "media_item", "source_id": source_id}],
            existing_refs=[],
            now=_now(),
        )


def test_normalize_rules_pack_refs_rejects_duplicate_ref_identity():
    with pytest.raises(RPGValidationError, match="duplicate_rules_pack_ref"):
        normalize_rules_pack_ref_payloads(
            [
                {"source_type": "media_item", "source_id": 7},
                {"source_type": "media_item", "source_id": 7, "display_name": "Duplicate"},
            ],
            existing_refs=[],
            now=_now(),
        )


def test_normalize_rules_pack_refs_ignores_client_timestamps():
    now = _now()

    refs = normalize_rules_pack_ref_payloads(
        [
            {
                "source_type": "media_item",
                "source_id": 7,
                "created_at": "2001-01-01T00:00:00Z",
                "updated_at": "2001-01-01T00:00:00Z",
            }
        ],
        existing_refs=[],
        now=now,
    )

    assert refs[0].created_at == now  # nosec B101
    assert refs[0].updated_at == now  # nosec B101


def test_normalize_rules_pack_refs_preserves_created_at_for_existing_ref():
    original_created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    existing = RulesPackRef(
        ref_id="media_item:7",
        source_type="media_item",
        source_id=7,
        display_name="Old",
        enabled=True,
        created_at=original_created_at,
        updated_at=original_created_at,
        metadata={},
    )

    refs = normalize_rules_pack_ref_payloads(
        [{"source_type": "media_item", "source_id": 7, "display_name": "New"}],
        existing_refs=[rules_pack_ref_to_dict(existing)],
        now=_now(),
    )

    assert refs[0].created_at == original_created_at  # nosec B101


def test_normalize_rules_pack_refs_updates_updated_at_for_existing_ref():
    original_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = _now()
    existing = RulesPackRef(
        ref_id="media_item:7",
        source_type="media_item",
        source_id=7,
        display_name="Old",
        enabled=True,
        created_at=original_time,
        updated_at=original_time,
        metadata={},
    )

    refs = normalize_rules_pack_ref_payloads(
        [{"source_type": "media_item", "source_id": 7}],
        existing_refs=[rules_pack_ref_to_dict(existing)],
        now=now,
    )

    assert refs[0].updated_at == now  # nosec B101


def test_normalize_rules_pack_refs_limits_metadata_to_json_object():
    refs = normalize_rules_pack_ref_payloads(
        [{"source_type": "media_item", "source_id": 7, "metadata": {"source_label": "user"}}],
        existing_refs=[],
        now=_now(),
    )

    assert refs[0].metadata == {"source_label": "user"}  # nosec B101

    with pytest.raises(RPGValidationError, match="invalid_rules_pack_ref_metadata"):
        normalize_rules_pack_ref_payloads(
            [{"source_type": "media_item", "source_id": 8, "metadata": ["not", "object"]}],
            existing_refs=[],
            now=_now(),
        )


@pytest.mark.parametrize("enabled", [None, 0, 1, "false", [], {}])
def test_normalize_rules_pack_refs_rejects_non_bool_enabled(enabled):
    with pytest.raises(RPGValidationError, match="invalid_rules_pack_ref_enabled"):
        normalize_rules_pack_ref_payloads(
            [{"source_type": "media_item", "source_id": 7, "enabled": enabled}],
            existing_refs=[],
            now=_now(),
        )


@pytest.mark.parametrize("enabled", [None, 0, 1, "false", [], {}])
def test_rules_pack_ref_from_dict_rejects_non_bool_enabled(enabled):
    with pytest.raises(RPGValidationError, match="invalid_rules_pack_ref_enabled"):
        rules_pack_ref_from_dict(
            {
                "ref_id": "media_item:7",
                "source_type": "media_item",
                "source_id": 7,
                "display_name": "Rules",
                "enabled": enabled,
                "created_at": "2026-06-25T12:00:00Z",
                "updated_at": "2026-06-25T12:00:00Z",
                "metadata": {},
            }
        )


def test_rules_pack_ref_from_dict_rejects_non_integer_ref_id_suffix():
    with pytest.raises(RPGValidationError, match="invalid_rules_pack_ref_source_id"):
        rules_pack_ref_from_dict(
            {
                "ref_id": "media_item:abc",
                "display_name": "Rules",
                "enabled": True,
                "created_at": "2026-06-25T12:00:00Z",
                "updated_at": "2026-06-25T12:00:00Z",
                "metadata": {},
            }
        )
