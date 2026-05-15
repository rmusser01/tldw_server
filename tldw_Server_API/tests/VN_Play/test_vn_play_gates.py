from tldw_Server_API.app.core.VN_Play.gates import evaluate_character_safety


def test_unknown_character_metadata_warns_for_general_rating() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="general",
        settings={},
        trust_level="local",
    )

    assert result.allowed is True
    assert result.status == "missing"
    assert result.warning_code == "character_safety_missing"


def test_unknown_character_metadata_requires_override_for_mature_rating() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="mature",
        settings={},
        trust_level="local",
    )

    assert result.allowed is False
    assert result.error_code == "character_safety_missing"


def test_strict_hosted_blocks_missing_character_metadata_for_general_rating() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="general",
        settings={"policy_profile_id": "strict_hosted"},
        trust_level="local",
    )

    assert result.allowed is False
    assert result.status == "missing"
    assert result.error_code == "character_safety_missing"


def test_untrusted_imported_character_metadata_warns_under_local_default() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira", "safety_metadata": {"age_status": "adult"}},
        content_rating="general",
        settings={"policy_profile_id": "local_default"},
        trust_level="untrusted_import",
    )

    assert result.allowed is True
    assert result.status == "imported_untrusted"
    assert result.warning_code == "character_safety_imported_untrusted"


def test_untrusted_imported_character_metadata_blocks_under_strict_hosted() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira", "safety_metadata": {"age_status": "adult"}},
        content_rating="general",
        settings={"policy_profile_id": "strict_hosted"},
        trust_level="untrusted_import",
    )

    assert result.allowed is False
    assert result.status == "imported_untrusted"
    assert result.error_code == "character_safety_imported_untrusted"


def test_minor_character_metadata_allows_general_rating() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira", "is_minor": True},
        content_rating="general",
        settings={"policy_profile_id": "local_default"},
        trust_level="local",
    )

    assert result.allowed is True
    assert result.status == "minor"


def test_empty_policy_profile_id_uses_local_default() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="general",
        settings={"policy_profile_id": ""},
        trust_level="local",
    )

    assert result.allowed is True
    assert result.status == "missing"
    assert result.warning_code == "character_safety_missing"


def test_custom_policy_profile_without_resolved_definition_fails_closed() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="general",
        settings={"policy_profile_id": "custom_strict"},
        trust_level="local",
    )

    assert result.allowed is False
    assert result.error_code == "policy_profile_unresolved"


def test_custom_policy_profile_uses_resolved_definition() -> None:
    result = evaluate_character_safety(
        character={"id": 1, "name": "Mira"},
        content_rating="general",
        settings={
            "policy_profile_id": "custom_strict",
            "policy_definition": {
                "character_safety": {
                    "missing": {"default": "block"},
                    "unknown_or_ambiguous": {"default": "block"},
                    "conflicting": {"default": "block"},
                    "imported_untrusted": {"default": "block"},
                }
            },
        },
        trust_level="local",
    )

    assert result.allowed is False
    assert result.error_code == "character_safety_missing"
