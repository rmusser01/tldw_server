from tldw_Server_API.app.core.Visual_Identities.expression_slots import (
    CANONICAL_EXPRESSION_SLOTS,
    CUSTOM_EXPRESSION_PREFIX,
    EXPRESSION_ALIASES,
    display_label_for_expression_key,
    is_custom_expression_key,
    normalize_expression_filename,
    normalize_expression_key,
)


def test_normalize_builtin_expression_aliases() -> None:
    assert normalize_expression_key("default") == "neutral"
    assert normalize_expression_key("normal") == "neutral"
    assert normalize_expression_key("joy") == "happy"


def test_unrecognized_filename_becomes_custom_expression() -> None:
    assert normalize_expression_filename("bashful smile.PNG") == "custom:bashful_smile"


def test_expression_slots_include_required_baseline_and_aliases() -> None:
    assert CANONICAL_EXPRESSION_SLOTS == (
        "neutral",
        "happy",
        "excited",
        "sad",
        "angry",
        "thinking",
        "confused",
        "surprised",
    )
    assert EXPRESSION_ALIASES["anger"] == "angry"
    assert normalize_expression_key("/emote anger".removeprefix("/emote ")) == "angry"


def test_normalize_expression_filename_maps_aliases_after_filename_cleanup() -> None:
    assert normalize_expression_filename("Joyful!!.webp") == "happy"
    assert normalize_expression_filename("  default portrait.gif") == "custom:default_portrait"


def test_custom_expression_helpers_and_labels() -> None:
    assert CUSTOM_EXPRESSION_PREFIX == "custom:"
    assert normalize_expression_key("bashful smile") == "custom:bashful_smile"
    assert is_custom_expression_key("custom:bashful_smile") is True
    assert display_label_for_expression_key("custom:bashful_smile") == "Bashful Smile"
    assert display_label_for_expression_key("joy") == "Happy"


def test_blank_expression_values_return_none() -> None:
    assert normalize_expression_key("") is None
    assert normalize_expression_filename(".png") is None
