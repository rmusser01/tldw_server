"""Edge-case unit tests for watchlist filter evaluation.

Covers: overlapping include+exclude at same priority, empty filter sets,
require_include semantics, priority ordering, regex special chars,
date_range boundaries, all-type catch-all, flag passthrough, and
multiple keyword match modes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Watchlists.filters import evaluate_filters, normalize_filters

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(**kwargs) -> dict:
    """Build a minimal candidate dict with sensible defaults."""
    base = {
        "title": kwargs.get("title", "Untitled"),
        "summary": kwargs.get("summary", ""),
        "content": kwargs.get("content", ""),
        "author": kwargs.get("author", ""),
        "published_at": kwargs.get(
            "published_at",
            datetime.now(timezone.utc).isoformat(),
        ),
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. Overlapping include + exclude at the same priority
# ---------------------------------------------------------------------------

def test_overlapping_include_exclude_same_priority_first_wins():
    """When include and exclude have the same priority, the first in input
    order wins because normalize_filters uses a stable sort."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "include", "value": {"keywords": ["AI"]}, "priority": 5},
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["AI"]}, "priority": 5},
        ]
    }
    flt = normalize_filters(payload)
    decision, meta = evaluate_filters(flt, _make_candidate(title="AI news"))
    # Stable sort preserves input order → include evaluated first
    assert decision == "include"
    assert meta.get("type") == "keyword"


def test_overlapping_exclude_include_same_priority_first_wins():
    """Reversed order: exclude first → exclude wins."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["AI"]}, "priority": 5},
            {"type": "keyword", "action": "include", "value": {"keywords": ["AI"]}, "priority": 5},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="AI news"))
    assert decision == "exclude"


# ---------------------------------------------------------------------------
# 2. Empty filter set
# ---------------------------------------------------------------------------

def test_empty_filter_set_returns_none():
    """With no filters and no require_include, all items pass (decision=None)."""
    flt = normalize_filters({"filters": []})
    decision, meta = evaluate_filters(flt, _make_candidate(title="Anything"))
    assert decision is None
    assert meta == {}


def test_bare_empty_list_returns_none():
    """Passing an empty list directly also works."""
    flt = normalize_filters([])
    decision, _ = evaluate_filters(flt, _make_candidate(title="Anything"))
    assert decision is None


# ---------------------------------------------------------------------------
# 3. Empty filter set + require_include
#    (Note: require_include gating is applied at the pipeline level,
#     not inside evaluate_filters. With zero include rules the pipeline
#     disables gating, so evaluate_filters correctly returns None.)
# ---------------------------------------------------------------------------

def test_empty_filters_with_require_include_flag():
    """normalize_filters ignores the require_include key but does not break.
    evaluate_filters returns None since there are no filter rules."""
    payload = {"filters": [], "require_include": True}
    flt = normalize_filters(payload)
    assert flt == []
    decision, _ = evaluate_filters(flt, _make_candidate(title="Anything"))
    assert decision is None


# ---------------------------------------------------------------------------
# 4. Include-only with require_include=true
#    (Pipeline-level semantics: only items matching an include rule pass.
#     At the evaluate_filters level we just verify the include match.)
# ---------------------------------------------------------------------------

def test_include_only_matching_item():
    """When a single include rule matches, decision is 'include'."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "include", "value": {"keywords": ["python"]}},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Python 3.12 release"))
    assert decision == "include"


def test_include_only_non_matching_item():
    """When a single include rule does NOT match, decision is None (pipeline
    would then filter it if require_include is true)."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "include", "value": {"keywords": ["python"]}},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Rust 1.80 release"))
    assert decision is None


# ---------------------------------------------------------------------------
# 5. Priority ordering — high-priority exclude beats low-priority include
# ---------------------------------------------------------------------------

def test_high_priority_exclude_beats_low_priority_include():
    payload = {
        "filters": [
            {"type": "keyword", "action": "include", "value": {"keywords": ["tech"]}, "priority": 1},
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["tech"]}, "priority": 10},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Tech news"))
    assert decision == "exclude"


def test_high_priority_include_beats_low_priority_exclude():
    payload = {
        "filters": [
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["tech"]}, "priority": 1},
            {"type": "keyword", "action": "include", "value": {"keywords": ["tech"]}, "priority": 10},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Tech news"))
    assert decision == "include"


# ---------------------------------------------------------------------------
# 6. Regex with special characters
# ---------------------------------------------------------------------------

def test_regex_word_boundary():
    r"""Regex \bbreaking\b with case-insensitive flag matches whole word."""
    payload = {
        "filters": [
            {
                "type": "regex",
                "action": "flag",
                "value": {"pattern": r"\bbreaking\b", "flags": "i", "field": "title"},
            },
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="BREAKING: Major event"))
    assert decision == "flag"

    # Should NOT match partial word
    decision, _ = evaluate_filters(flt, _make_candidate(title="Heartbreaking story"))
    assert decision is None


def test_regex_special_chars_in_pattern():
    """Regex with parentheses and quantifiers."""
    payload = {
        "filters": [
            {
                "type": "regex",
                "action": "exclude",
                "value": {"pattern": r"v\d+\.\d+\.\d+", "flags": "i", "field": "title"},
            },
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Release v1.2.3"))
    assert decision == "exclude"

    decision, _ = evaluate_filters(flt, _make_candidate(title="No version here"))
    assert decision is None


def test_regex_invalid_pattern_skipped():
    """Invalid regex pattern should not match (compile fails gracefully)."""
    payload = {
        "filters": [
            {
                "type": "regex",
                "action": "exclude",
                "value": {"pattern": "[invalid(", "field": "title"},
            },
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Test item"))
    assert decision is None


# ---------------------------------------------------------------------------
# 7. date_range with max_age_days=0
# ---------------------------------------------------------------------------

def test_date_range_max_age_zero_rejects_everything():
    """max_age_days=0 means delta must be <= 0 days. Any item published even
    milliseconds ago has a positive delta, so nothing matches. This is the
    correct behavior — it's an extremely tight boundary."""
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(seconds=1)).isoformat()
    old = (now - timedelta(hours=25)).isoformat()

    payload = {
        "filters": [
            {"type": "date_range", "action": "include", "value": {"max_age_days": 0}},
        ]
    }
    flt = normalize_filters(payload)

    # Even a "just now" item has a tiny positive delta → no match
    decision_recent, _ = evaluate_filters(flt, _make_candidate(published_at=recent))
    assert decision_recent is None

    # Definitely too old
    decision_old, _ = evaluate_filters(flt, _make_candidate(published_at=old))
    assert decision_old is None


def test_date_range_max_age_one_matches_recent():
    """max_age_days=1 should match items from the last 24 hours."""
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(hours=2)).isoformat()
    old = (now - timedelta(days=3)).isoformat()

    payload = {
        "filters": [
            {"type": "date_range", "action": "include", "value": {"max_age_days": 1}},
        ]
    }
    flt = normalize_filters(payload)

    decision_recent, _ = evaluate_filters(flt, _make_candidate(published_at=recent))
    assert decision_recent == "include"

    decision_old, _ = evaluate_filters(flt, _make_candidate(published_at=old))
    assert decision_old is None


def test_date_range_no_published_at():
    """Candidate without published_at should not match date_range filter."""
    payload = {
        "filters": [
            {"type": "date_range", "action": "exclude", "value": {"max_age_days": 7}},
        ]
    }
    flt = normalize_filters(payload)
    candidate = {"title": "No date", "summary": "..."}
    decision, _ = evaluate_filters(flt, candidate)
    assert decision is None


# ---------------------------------------------------------------------------
# 8. All-type filter as catch-all
# ---------------------------------------------------------------------------

def test_all_type_catch_all_excludes_everything():
    """An 'all' type filter at lowest priority catches everything not already matched."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "include", "value": {"keywords": ["important"]}, "priority": 10},
            {"type": "all", "action": "exclude", "value": {}, "priority": 1},
        ]
    }
    flt = normalize_filters(payload)

    # Important item should be included (higher priority)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Important update"))
    assert decision == "include"

    # Everything else should be excluded by the all-type catch-all
    decision, _ = evaluate_filters(flt, _make_candidate(title="Routine update"))
    assert decision == "exclude"


def test_all_type_flag_catches_all():
    """'all' type with 'flag' action flags every item."""
    payload = {"filters": [{"type": "all", "action": "flag", "value": {}}]}
    flt = normalize_filters(payload)
    decision, meta = evaluate_filters(flt, _make_candidate(title="Anything"))
    assert decision == "flag"
    assert meta["type"] == "all"


# ---------------------------------------------------------------------------
# 9. Flag action passthrough
# ---------------------------------------------------------------------------

def test_flag_does_not_exclude():
    """Flag action means the item is flagged AND ingested (not excluded)."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "flag", "value": {"keywords": ["urgent"]}},
        ]
    }
    flt = normalize_filters(payload)
    decision, meta = evaluate_filters(flt, _make_candidate(title="Urgent: server down"))
    assert decision == "flag"
    assert meta.get("key") is not None


def test_flag_with_no_match_returns_none():
    """Flag filter that doesn't match returns None (item ingested, not flagged)."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "flag", "value": {"keywords": ["urgent"]}},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Normal update"))
    assert decision is None


# ---------------------------------------------------------------------------
# 10. Multiple keyword match modes
# ---------------------------------------------------------------------------

def test_keyword_match_all_requires_both():
    """match='all' requires ALL keywords to be present."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "include", "value": {"keywords": ["AI", "safety"], "match": "all"}},
        ]
    }
    flt = normalize_filters(payload)

    # Both present
    decision, _ = evaluate_filters(flt, _make_candidate(title="AI safety research"))
    assert decision == "include"

    # Only one present
    decision, _ = evaluate_filters(flt, _make_candidate(title="AI research"))
    assert decision is None

    # Neither present
    decision, _ = evaluate_filters(flt, _make_candidate(title="Climate change"))
    assert decision is None


def test_keyword_match_any_requires_one():
    """match='any' (default) requires at least ONE keyword to be present."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["spam", "ad"], "match": "any"}},
        ]
    }
    flt = normalize_filters(payload)

    decision, _ = evaluate_filters(flt, _make_candidate(title="Great ad for products"))
    assert decision == "exclude"

    decision, _ = evaluate_filters(flt, _make_candidate(title="No issues here"))
    assert decision is None


def test_keyword_default_match_is_any():
    """When match mode is not specified, default is 'any'."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "flag", "value": {"keywords": ["breaking", "alert"]}},
        ]
    }
    flt = normalize_filters(payload)

    decision, _ = evaluate_filters(flt, _make_candidate(title="Breaking news"))
    assert decision == "flag"


# ---------------------------------------------------------------------------
# 11. Inactive filters are skipped
# ---------------------------------------------------------------------------

def test_inactive_filter_skipped():
    """A filter with is_active=False should be ignored."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["test"]}, "is_active": False},
            {"type": "keyword", "action": "flag", "value": {"keywords": ["test"]}, "is_active": True},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(title="Test item"))
    assert decision == "flag"


# ---------------------------------------------------------------------------
# 12. Filter meta key format
# ---------------------------------------------------------------------------

def test_meta_key_uses_id_when_available():
    """Meta key should use 'id:{id}' format when filter has an id."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["test"]}, "id": 42},
        ]
    }
    flt = normalize_filters(payload)
    _, meta = evaluate_filters(flt, _make_candidate(title="Test item"))
    assert meta["key"] == "id:42"
    assert meta["id"] == 42


def test_meta_key_uses_idx_when_no_id():
    """Meta key should use 'idx:{index}' format when filter has no id."""
    payload = {
        "filters": [
            {"type": "keyword", "action": "exclude", "value": {"keywords": ["test"]}},
        ]
    }
    flt = normalize_filters(payload)
    _, meta = evaluate_filters(flt, _make_candidate(title="Test item"))
    assert meta["key"] == "idx:0"
    assert meta["id"] is None


# ---------------------------------------------------------------------------
# 13. normalize_filters validation
# ---------------------------------------------------------------------------

def test_normalize_rejects_invalid_type():
    """Invalid filter type is silently dropped."""
    payload = {"filters": [{"type": "invalid", "action": "include", "value": {}}]}
    flt = normalize_filters(payload)
    assert flt == []


def test_normalize_rejects_invalid_action():
    """Invalid action is silently dropped."""
    payload = {"filters": [{"type": "keyword", "action": "delete", "value": {}}]}
    flt = normalize_filters(payload)
    assert flt == []


def test_normalize_handles_missing_value():
    """Missing or non-dict value defaults to empty dict."""
    payload = {"filters": [{"type": "all", "action": "flag", "value": "not_a_dict"}]}
    flt = normalize_filters(payload)
    assert len(flt) == 1
    assert flt[0]["value"] == {}


def test_normalize_handles_none_priority():
    """None priority defaults to 0."""
    payload = {"filters": [{"type": "all", "action": "flag", "value": {}}]}
    flt = normalize_filters(payload)
    assert flt[0]["priority"] == 0


# ---------------------------------------------------------------------------
# 14. Author filter edge cases
# ---------------------------------------------------------------------------

def test_author_filter_any_mode():
    payload = {
        "filters": [
            {"type": "author", "action": "include", "value": {"names": ["Doe", "Smith"], "match": "any"}},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(author="Jane Doe"))
    assert decision == "include"

    decision, _ = evaluate_filters(flt, _make_candidate(author="Alice Johnson"))
    assert decision is None


def test_author_filter_all_mode():
    """match='all' for author requires all names as substrings."""
    payload = {
        "filters": [
            {"type": "author", "action": "flag", "value": {"names": ["John", "Doe"], "match": "all"}},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(author="John Doe"))
    assert decision == "flag"

    decision, _ = evaluate_filters(flt, _make_candidate(author="John Smith"))
    assert decision is None


def test_author_filter_empty_author():
    """Empty author in candidate should not match."""
    payload = {
        "filters": [
            {"type": "author", "action": "exclude", "value": {"names": ["anyone"]}},
        ]
    }
    flt = normalize_filters(payload)
    decision, _ = evaluate_filters(flt, _make_candidate(author=""))
    assert decision is None
