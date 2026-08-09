from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Claims_Extraction.claims_analytics_exports import (
    apply_export_transition,
)


@given(
    st.lists(
        st.sampled_from(["queued", "processing", "failed", "ready", "unknown"]),
        max_size=30,
    )
)
def test_ready_state_is_monotonic(candidate_states: list[str]) -> None:
    state = "queued"
    for candidate in candidate_states:
        state = apply_export_transition(state, candidate)
        if state == "ready":
            assert apply_export_transition(state, "processing") == "ready"
            assert apply_export_transition(state, "failed") == "ready"
            assert apply_export_transition(state, "queued") == "ready"


@given(
    current=st.sampled_from(["queued", "processing", "failed", "ready"]),
    requested=st.sampled_from(["queued", "processing", "failed", "ready", "invalid"]),
)
def test_transition_helper_matches_declared_database_map(
    current: str,
    requested: str,
) -> None:
    allowed = {
        ("queued", "processing"),
        ("queued", "failed"),
        ("processing", "ready"),
        ("processing", "failed"),
        ("failed", "processing"),
        ("ready", "ready"),
    }

    expected = requested if (current, requested) in allowed else current
    assert apply_export_transition(current, requested) == expected
