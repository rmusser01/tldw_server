from tldw_Server_API.app.core.VN_Play.gates import evaluate_character_safety


def test_unknown_character_metadata_warns_for_general_rating() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="general",
        settings={},
        trust_level="local",
    )

    assert result.allowed is True
    assert result.status == "unknown"
    assert result.warning_code == "character_safety_unknown"


def test_unknown_character_metadata_requires_override_for_mature_rating() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="mature",
        settings={},
        trust_level="local",
    )

    assert result.allowed is False
    assert result.error_code == "character_safety_unknown_requires_override"


def test_unknown_character_metadata_mature_override_still_warns() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="mature",
        settings={"allow_unknown_character_safety": True},
        trust_level="local",
    )

    assert result.allowed is True
    assert result.status == "unknown"
    assert result.warning_code == "character_safety_unknown"
