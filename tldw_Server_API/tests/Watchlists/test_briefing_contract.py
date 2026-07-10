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
