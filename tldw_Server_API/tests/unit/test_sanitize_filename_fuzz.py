"""Property-based tests for sanitize_filename (audit F10)."""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from tldw_Server_API.app.core.Utils.Utils import sanitize_filename

FORBIDDEN = set('<>:"/\\|?*')


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
@given(st.text(max_size=500))
def test_never_raises_and_never_empty(raw):
    out = sanitize_filename(raw)
    assert isinstance(out, str)
    assert out  # never empty: falls back to "untitled"


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
@given(st.text(max_size=500))
def test_output_contains_no_forbidden_characters(raw):
    out = sanitize_filename(raw)
    assert not (set(out) & FORBIDDEN), f"forbidden chars survived: {set(out) & FORBIDDEN}"


@pytest.mark.unit
@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
@given(st.text(min_size=1, max_size=300), st.integers(min_value=10, max_value=100))
def test_length_cap_is_respected(raw, cap):
    out = sanitize_filename(raw, max_total_length=cap, extension=".txt")
    assert len(out) + len(".txt") <= cap or out == "untitled"
