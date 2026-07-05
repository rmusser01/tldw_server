"""Property tests for the config-section scalar parsers (RA4).

``chunking._parse_int`` / ``_parse_bool`` are the audit-named pure parsers
(chunking.py:31/:41). They must be TOTAL — never raise on arbitrary input —
default on garbage, and be idempotent on their own valid output.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from tldw_Server_API.app.core.config_sections.chunking import (
    _TRUE_VALUES,
    _parse_bool,
    _parse_int,
)

pytestmark = [pytest.mark.unit, pytest.mark.property]

_COMMON = hyp_settings(max_examples=300, deadline=None)

# anything a config value could be: strings (incl. numeric/whitespace), None, ints, floats, objects
_arbitrary_raw = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=30),
    st.binary(max_size=10),
    st.lists(st.integers(), max_size=3),
)


class TestParseIntProperties:
    @_COMMON
    @given(raw=_arbitrary_raw, default=st.integers(min_value=-1000, max_value=1000))
    def test_never_raises_and_returns_int(self, raw: object, default: int) -> None:
        result = _parse_int(raw, default)
        assert isinstance(result, int)

    @staticmethod
    def _is_unparseable_int(s: str) -> bool:
        """True if int(s.strip()) would raise — the robust 'non-numeric' filter
        (int() accepts leading +/-/whitespace, so a naive isdigit() check is wrong)."""
        try:
            int(s.strip())
        except (TypeError, ValueError):
            return True
        return False

    @_COMMON
    @given(raw=st.text(max_size=30), default=st.integers())
    def test_non_numeric_and_empty_yield_default(self, raw: str, default: int) -> None:
        # a non-integer string (or empty) must fall back to the default
        if not self._is_unparseable_int(raw):
            return  # skip strings that ARE valid integers (covered elsewhere)
        assert _parse_int(raw, default) == default

    @_COMMON
    @given(value=st.integers(min_value=-(10**9), max_value=10**9), default=st.integers())
    def test_round_trips_a_real_integer_string(self, value: int, default: int) -> None:
        assert _parse_int(str(value), default) == value

    @_COMMON
    @given(value=st.integers(min_value=-(10**6), max_value=10**6))
    def test_idempotent_on_its_own_output(self, value: int) -> None:
        once = _parse_int(str(value), 0)
        assert _parse_int(str(once), 0) == once


class TestParseBoolProperties:
    @_COMMON
    @given(raw=_arbitrary_raw, default=st.booleans())
    def test_never_raises_and_returns_bool(self, raw: object, default: bool) -> None:
        assert isinstance(_parse_bool(raw, default), bool)

    @_COMMON
    @given(default=st.booleans())
    def test_empty_yields_default(self, default: bool) -> None:
        assert _parse_bool("", default) is default
        assert _parse_bool("   ", default) is default

    @_COMMON
    @given(token=st.sampled_from(sorted(_TRUE_VALUES)), default=st.booleans(), upper=st.booleans())
    def test_truthy_tokens_are_true_case_insensitively(self, token: str, default: bool, upper: bool) -> None:
        rendered = token.upper() if upper else token
        assert _parse_bool(rendered, default) is True

    @_COMMON
    @given(
        raw=st.text(min_size=1, max_size=20).filter(lambda s: s.strip().lower() not in _TRUE_VALUES and s.strip() != ""),
        default=st.booleans(),
    )
    def test_non_empty_non_truthy_is_false(self, raw: str, default: bool) -> None:
        # chunking's _parse_bool returns False (not the default) for a non-empty
        # value that is not a recognized truthy token
        assert _parse_bool(raw, default) is False
