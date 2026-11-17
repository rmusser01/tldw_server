import json

import pytest

from tldw_Server_API.app.core.Character_Chat.ccv3_parser import validate_v3_card, parse_v3_card
from tldw_Server_API.app.core.Character_Chat.modules.character_io import import_character_card_from_json_string
from tldw_Server_API.app.core.Character_Chat.modules.character_validation import MAX_V2_TEXT_FIELD_LENGTHS


def test_ccv3_validate_and_parse_happy_path():
    card = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            "name": "V3 Hero",
            "description": "A brave v3 hero",
            "personality": "Bold",
            "scenario": "Epic quest",
            "first_mes": "Greetings from v3!",
            "mes_example": "USER: Hi\nASSISTANT: Hello",
        },
    }
    ok, errs = validate_v3_card(card)
    assert ok and not errs

    parsed = parse_v3_card(card)
    assert parsed is not None
    assert parsed["name"] == "V3 Hero"
    assert parsed["first_message"].startswith("Greetings")

    # Through import dispatcher
    parsed2 = import_character_card_from_json_string(json.dumps(card))
    assert parsed2 is not None
    assert parsed2["name"] == "V3 Hero"


def test_ccv3_missing_required_fields_returns_none():
    card = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {"name": "No First Message", "description": "Missing first_mes"},
    }
    ok, errs = validate_v3_card(card)
    assert not ok
    assert any("first_mes" in e for e in errs)

    parsed = import_character_card_from_json_string(json.dumps(card))
    # Validation fails; importer should fall through and not produce a usable card
    assert parsed is None


def test_ccv3_rejects_overlong_name():
    card = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            "name": "A" * (MAX_V2_TEXT_FIELD_LENGTHS["name"] + 1),
            "description": "Too much name",
            "first_mes": "Hello",
        },
    }

    ok, errs = validate_v3_card(card)
    assert not ok
    assert any("name" in err for err in errs)
    assert import_character_card_from_json_string(json.dumps(card)) is None


def test_ccv3_rejects_wrong_field_type():
    card = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            "name": "Bad Type",
            "description": ["not", "text"],
            "first_mes": "Hello",
        },
    }

    ok, errs = validate_v3_card(card)
    assert not ok
    assert any("description" in err for err in errs)
    assert import_character_card_from_json_string(json.dumps(card)) is None


@pytest.mark.parametrize("image_key", ["char_image", "image"])
def test_ccv3_parser_preserves_image_fields(image_key):
    sample_b64 = "aGVsbG8="
    card = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            "name": "ImageHero",
            "description": "Has an avatar",
            "personality": "Friendly",
            "scenario": "Testing",
            "first_mes": "Hello!",
            "mes_example": "User: Hi\nAssistant: Hello",
            image_key: sample_b64,
        },
    }

    parsed = parse_v3_card(card)
    assert parsed is not None
    assert parsed.get("image_base64") == sample_b64

    imported = import_character_card_from_json_string(json.dumps(card))
    assert imported is not None
    assert imported.get("image_base64") == sample_b64


OPTIONAL_V3_TEXT_FIELDS = (
    "personality", "scenario", "mes_example", "creator_notes", "system_prompt",
    "post_history_instructions", "creator", "character_version",
)


@pytest.mark.parametrize("field", OPTIONAL_V3_TEXT_FIELDS)
def test_ccv3_legacy_null_optional_text_uses_omitted_field_default(field):
    """Already-exported nullable DB text remains importable without mutating input."""
    card = {
        "spec": "chara_card_v3", "spec_version": "3.0",
        "data": {"name": "Legacy export", "description": "Saved", "first_mes": "Hi", field: None},
    }
    missing = {**card, "data": {key: value for key, value in card["data"].items() if key != field}}

    imported = import_character_card_from_json_string(json.dumps(card))

    assert imported == parse_v3_card(missing)  # nosec B101
    assert card["data"][field] is None  # nosec B101


@pytest.mark.parametrize("field", ("name", "description", "first_mes"))
def test_ccv3_null_required_text_remains_invalid(field):
    card = {"data": {"name": "Invalid", "description": "Saved", "first_mes": "Hi", field: None}}

    assert validate_v3_card(card)[0] is False  # nosec B101
    assert parse_v3_card(card) is None  # nosec B101


@pytest.mark.parametrize("field", OPTIONAL_V3_TEXT_FIELDS + ("alternate_greetings", "tags", "extensions"))
def test_ccv3_wrong_optional_type_remains_invalid(field):
    card = {"data": {"name": "Invalid", "description": "Saved", "first_mes": "Hi", field: 17}}

    assert validate_v3_card(card)[0] is False  # nosec B101
    assert parse_v3_card(card) is None  # nosec B101


@pytest.mark.parametrize("field", ("alternate_greetings", "tags", "extensions"))
def test_ccv3_null_collections_remain_invalid(field):
    card = {"data": {"name": "Invalid", "description": "Saved", "first_mes": "Hi", field: None}}

    assert validate_v3_card(card)[0] is False  # nosec B101
    assert parse_v3_card(card) is None  # nosec B101
