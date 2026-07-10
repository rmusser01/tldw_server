from tldw_Server_API.app.core.Watchlists.briefing_contract import (
    briefing_selection_limit,
    get_briefing_contract,
    normalize_briefing_output_prefs,
)


def test_scheduled_legacy_audio_normalizes_to_required_text_and_reports():
    normalized = normalize_briefing_output_prefs(
        {
            "generate_audio": True,
            "target_audio_minutes": 20,
            "audio_cast": {
                "speaker_count": 2,
                "speakers": [
                    {"id": "host", "label": "Host", "voice": "alloy"},
                    {"id": "analyst", "label": "Analyst", "voice": "nova"},
                ],
            },
            "custom_future_key": {"keep": True},
        },
        scheduled=True,
    )
    contract = normalized.output_prefs["briefing_pipeline"]
    assert contract["version"] == 1
    assert contract["text"]["enabled"] is True
    assert contract["audio"]["enabled"] is True
    assert contract["audio"]["target_minutes"] == 20
    assert contract["delivery"]["reports"]["enabled"] is True
    assert normalized.output_prefs["custom_future_key"] == {"keep": True}


def test_delivery_is_not_enabled_by_normalization():
    normalized = normalize_briefing_output_prefs({}, scheduled=True)
    delivery = normalized.output_prefs["briefing_pipeline"]["delivery"]
    assert delivery["email"]["enabled"] is False
    assert delivery["chatbook"]["enabled"] is False


def test_selection_limit_uses_one_bounded_value(monkeypatch):
    monkeypatch.setenv("WATCHLIST_BRIEFING_MAX_ITEMS", "5000")
    contract = get_briefing_contract({}, scheduled=True)
    assert briefing_selection_limit(contract) == 1000


def test_malformed_canonical_sections_fall_back_to_safe_shapes():
    contract = get_briefing_contract(
        {
            "briefing_pipeline": {
                "selection": "automatic",
                "editorial": None,
                "text": [],
                "audio": "enabled",
                "delivery": 1,
                "test": ["external_delivery"],
            }
        },
        scheduled=False,
    )

    for section in ("selection", "editorial", "text", "audio", "delivery", "test"):
        assert isinstance(contract[section], dict)
    assert contract["text"]["enabled"] is False
    assert contract["audio"]["enabled"] is False
    assert contract["delivery"]["reports"]["enabled"] is True
    assert contract["delivery"]["email"]["enabled"] is False
    assert contract["delivery"]["chatbook"]["enabled"] is False
    assert contract["test"]["external_delivery"] is False


def test_false_and_invalid_boolean_strings_fail_closed():
    contract = get_briefing_contract(
        {
            "briefing_pipeline": {
                "text": {"enabled": "false", "show_notes": "invalid"},
                "audio": {"enabled": "0", "persona_summarize": "off"},
                "delivery": {
                    "email": {"enabled": "no"},
                    "chatbook": {"enabled": "invalid"},
                },
                "test": {"external_delivery": "true"},
            }
        },
        scheduled=False,
    )

    assert contract["text"]["enabled"] is False
    assert contract["text"]["show_notes"] is False
    assert contract["audio"]["enabled"] is False
    assert contract["audio"]["persona_summarize"] is False
    assert contract["delivery"]["email"]["enabled"] is False
    assert contract["delivery"]["chatbook"]["enabled"] is False
    assert contract["test"]["external_delivery"] is False


def test_legacy_and_canonical_boolean_representations_are_deterministic():
    legacy = get_briefing_contract(
        {
            "auto_output": {"enabled": "false"},
            "generate_audio": "false",
            "deliveries": {
                "email": {"enabled": "0"},
                "chatbook": {"enabled": 0},
            },
        },
        scheduled=False,
    )
    enabled = get_briefing_contract(
        {
            "briefing_pipeline": {
                "audio": {"enabled": "yes"},
                "delivery": {
                    "email": {"enabled": "1"},
                    "chatbook": {"enabled": 1},
                },
            }
        },
        scheduled=False,
    )

    assert legacy["text"]["enabled"] is False
    assert legacy["audio"]["enabled"] is False
    assert legacy["delivery"]["email"]["enabled"] is False
    assert legacy["delivery"]["chatbook"]["enabled"] is False
    assert enabled["text"]["enabled"] is True
    assert enabled["audio"]["enabled"] is True
    assert enabled["delivery"]["email"]["enabled"] is True
    assert enabled["delivery"]["chatbook"]["enabled"] is True
