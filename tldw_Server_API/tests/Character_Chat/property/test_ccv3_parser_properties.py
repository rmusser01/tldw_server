"""Property tests for the V3 character-card parser (RA4).

``ccv3_parser`` exposes ``validate_v3_card`` and ``parse_v3_card`` only — there
is NO serializer, so a strict ``parse(parse(x)) == parse(x)`` round-trip is not
possible (parse renames ``first_mes`` -> ``first_message``). The real
invariants are:

* totality — neither function raises on ANY dict input
* determinism — same input -> same output
* validate<->parse consistency — parse returns a card iff validate passes and
  the name is non-empty
* field preservation — a valid card's known fields survive parsing
* deterministic rejection — a card missing a required field is always rejected
"""
from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from tldw_Server_API.app.core.Character_Chat.ccv3_parser import (
    parse_v3_card,
    validate_v3_card,
)

pytestmark = [pytest.mark.unit, pytest.mark.property]

_COMMON = hyp_settings(max_examples=200, deadline=None)

# arbitrary JSON-ish scalars/containers, to stress the parser with junk
_json_scalars = st.one_of(
    st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False), st.text(max_size=20)
)
_arbitrary_dicts = st.dictionaries(
    keys=st.text(max_size=12),
    values=st.recursive(
        _json_scalars,
        lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=6), children, max_size=4),
        max_leaves=8,
    ),
    max_size=8,
)


@st.composite
def _valid_card(draw: st.DrawFn) -> dict[str, Any]:
    """A card that passes validation: required string fields, non-blank name."""
    data: dict[str, Any] = {
        "name": draw(st.text(min_size=1, max_size=30).filter(lambda s: s.strip() != "")),
        "description": draw(st.text(max_size=50)),
        "first_mes": draw(st.text(max_size=50)),
    }
    if draw(st.booleans()):
        data["tags"] = draw(st.lists(st.text(max_size=10), max_size=5))
    if draw(st.booleans()):
        data["personality"] = draw(st.text(max_size=40))
    # exercise both the flat and {"data": ...} envelope shapes
    return {"data": data} if draw(st.booleans()) else data


class TestParserTotality:
    @_COMMON
    @given(card=_arbitrary_dicts)
    def test_validate_never_raises_and_returns_bool_and_errors(self, card: dict) -> None:
        ok, errors = validate_v3_card(card)
        assert isinstance(ok, bool)
        assert isinstance(errors, list)
        assert ok == (len(errors) == 0)

    @_COMMON
    @given(card=_arbitrary_dicts)
    def test_parse_never_raises(self, card: dict) -> None:
        result = parse_v3_card(card)
        assert result is None or isinstance(result, dict)

    @_COMMON
    @given(card=_arbitrary_dicts)
    def test_parse_is_deterministic(self, card: dict) -> None:
        assert parse_v3_card(card) == parse_v3_card(card)

    @_COMMON
    @given(card=_arbitrary_dicts)
    def test_validate_parse_consistency(self, card: dict) -> None:
        """parse yields a card only when validate passes; if it rejects a valid
        card, that is solely because the resolved name is falsy."""
        ok, _errors = validate_v3_card(card)
        parsed = parse_v3_card(card)
        if not ok:
            assert parsed is None, "parse accepted a card validate rejected"
        elif parsed is None:
            data = card.get("data", card)
            assert not (isinstance(data, dict) and data.get("name"))


class TestValidCardInvariants:
    @_COMMON
    @given(card=_valid_card())
    def test_valid_card_parses_and_preserves_fields(self, card: dict) -> None:
        parsed = parse_v3_card(card)
        assert parsed is not None, "a card with required fields failed to parse"
        data = card.get("data", card)
        assert parsed["name"] == data["name"]
        assert parsed["description"] == data.get("description", "")
        # the documented rename: first_mes -> first_message
        assert parsed["first_message"] == data.get("first_mes", "")
        if "tags" in data:
            assert parsed["tags"] == data["tags"]

    @_COMMON
    @given(
        card=_valid_card(),
        drop=st.sampled_from(["name", "description", "first_mes"]),
    )
    def test_dropping_a_required_field_is_always_rejected(self, card: dict, drop: str) -> None:
        data = card.get("data", card)
        data.pop(drop, None)
        ok, errors = validate_v3_card(card)
        assert ok is False and errors, f"missing required field '{drop}' was accepted"
        assert parse_v3_card(card) is None

    @_COMMON
    @given(card=_valid_card())
    def test_blank_name_is_always_rejected(self, card: dict) -> None:
        data = card.get("data", card)
        data["name"] = "   "  # whitespace-only
        ok, _errors = validate_v3_card(card)
        assert ok is False
        assert parse_v3_card(card) is None
