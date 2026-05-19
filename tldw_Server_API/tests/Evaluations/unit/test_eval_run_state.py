import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import (
    RunResultsResponse,
    RunStatus,
)
from tldw_Server_API.app.core.Evaluations.run_state import (
    can_transition_run_status,
    normalize_run_status,
)


@pytest.mark.unit
def test_normalize_run_status_maps_legacy_canceled_to_cancelled():
    assert normalize_run_status("canceled") == "cancelled"


@pytest.mark.unit
def test_can_transition_run_status_rejects_terminal_status_rewrite():
    assert can_transition_run_status("completed", "cancelled") is False


@pytest.mark.unit
def test_can_transition_run_status_allows_normalized_same_terminal_status():
    assert can_transition_run_status("canceled", "cancelled") is True


@pytest.mark.unit
def test_run_results_response_normalizes_legacy_status_to_canonical_enum():
    result = RunResultsResponse(
        id="run_1",
        eval_id="eval_1",
        status="canceled",
        started_at=1,
        completed_at=2,
        results={},
        duration_seconds=1.0,
    )

    assert result.status is RunStatus.CANCELLED


@pytest.mark.unit
def test_run_results_response_rejects_unknown_status_values():
    with pytest.raises(ValidationError):
        RunResultsResponse(
            id="run_1",
            eval_id="eval_1",
            status="queued",
            started_at=1,
            completed_at=2,
            results={},
            duration_seconds=1.0,
        )
