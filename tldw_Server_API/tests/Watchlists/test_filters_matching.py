import pytest

pytestmark = pytest.mark.unit

from tldw_Server_API.app.core.Watchlists.filters import (
    _REGEX_TIMEOUT_ERROR,
    _compile_regex,
    _safe_regex_search,
    evaluate_filters,
    normalize_filters,
)


def test_keyword_any_and_all():


    payload = {
        "filters": [
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["ai", "ml"], "match": "any"}, "priority": 5},
            {"type": "keyword", "action": "flag", "value": {"keywords": ["research", "paper"], "match": "all"}, "priority": 1},
        ]
    }
    flt = normalize_filters(payload)

    decision, meta = evaluate_filters(flt, {"title": "New AI breakthroughs", "summary": "..."})
    assert decision == "exclude"
    assert meta and "key" in meta

    decision, _ = evaluate_filters(flt, {"title": "Interesting research", "summary": "..."})
    # 'all' requires both keywords; should not match when only one present
    assert decision is None

    decision, _ = evaluate_filters(flt, {"title": "Great research paper", "summary": "..."})
    assert decision == "flag"


def test_regex_flags_and_field():


    payload = {
        "filters": [
            {"type": "regex", "action": "exclude", "value": {"pattern": "(?i)breaking", "field": "title"}},
            {"type": "regex", "action": "flag", "value": {"pattern": "^Author:.*Doe$", "flags": "i", "field": "author"}},
        ]
    }
    flt = normalize_filters(payload)

    decision, _ = evaluate_filters(flt, {"title": "Breaking News", "author": "Jane"})
    assert decision == "exclude"

    decision, _ = evaluate_filters(flt, {"title": "Regular", "author": "author: John Doe"})
    assert decision == "flag"


def test_regex_nested_quantifier_pattern_is_rejected():
    assert _compile_regex(r"(a+)+$", None) is None


def test_regex_oversized_pattern_is_skipped():
    pattern = "a" * 1001
    flt = normalize_filters(
        {
            "filters": [
                {
                    "type": "regex",
                    "action": "exclude",
                    "value": {"pattern": pattern, "field": "title"},
                }
            ]
        }
    )

    decision, _ = evaluate_filters(flt, {"title": pattern})

    assert decision is None


def test_safe_regex_search_handles_regex_timeout_variant():
    class TimeoutRegex:
        def search(self, text, timeout=None):
            raise _REGEX_TIMEOUT_ERROR("timed out")

    assert _safe_regex_search(TimeoutRegex(), "aaaaaaaa") is False


def test_date_range_max_age_days():


    from datetime import datetime, timezone, timedelta

    recent_iso = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    old_iso = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

    payload = {
        "filters": [
            {"type": "date_range", "action": "exclude", "value": {"max_age_days": 7}},
        ]
    }
    flt = normalize_filters(payload)

    decision, _ = evaluate_filters(flt, {"published_at": old_iso})
    assert decision != "exclude"  # too old should not match include-only filter

    decision, _ = evaluate_filters(flt, {"published_at": recent_iso})
    assert decision == "exclude"


def test_keyword_field_and_regex_default_flags():


    payload = {
        "filters": [
            {"type": "keyword", "action": "include", "value": {"keywords": ["alpha"], "field": "summary"}, "priority": 1},
            {"type": "regex", "action": "exclude", "value": {"pattern": "breaking", "field": "title"}, "priority": 10},
        ]
    }
    flt = normalize_filters(payload)

    decision, _ = evaluate_filters(flt, {"title": "Routine Update", "summary": "Alpha details"})
    assert decision == "include"

    decision, _ = evaluate_filters(flt, {"title": "Breaking News", "summary": "Alpha details"})
    assert decision == "exclude"


def test_date_range_since_until():


    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    in_window = (now - timedelta(days=1)).isoformat()
    too_old = (now - timedelta(days=10)).isoformat()
    since = (now - timedelta(days=3)).isoformat()
    until = (now + timedelta(days=3)).isoformat()

    payload = {
        "filters": [
            {"type": "date_range", "action": "include", "value": {"since": since, "until": until}},
        ]
    }
    flt = normalize_filters(payload)

    decision, _ = evaluate_filters(flt, {"published_at": in_window})
    assert decision == "include"

    decision, _ = evaluate_filters(flt, {"published_at": too_old})
    assert decision is None
