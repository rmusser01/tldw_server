"""Property tests for the Jobs operation-contract invariants (RA4).

Targets the top merged validation defect (a9b6a2c310 "harden operation
contract invariants"): ``AdmissionResult`` / ``LifecycleResult`` use
``__post_init__`` to reject impossible ``(outcome, fields)`` combinations. If a
guard is dropped, an inconsistent result would construct silently.

These properties assert the contract holds over arbitrary field combinations:
an impossible state ALWAYS raises, and any object that *does* construct
satisfies every invariant (an independent oracle, so a dropped guard fails).
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    LifecycleResult,
    NoTransitionReason,
    OperationOutcome,
)

pytestmark = [pytest.mark.unit, pytest.mark.property]

_COMMON = hyp_settings(max_examples=200, deadline=None)

_OUTCOMES = list(OperationOutcome)
_NO_TRANS = [None, *list(NoTransitionReason)]
_REJECTIONS = [None, *list(AdmissionRejectionReason)]


def _admission_is_consistent(
    outcome: OperationOutcome,
    row: dict | None,
    was_inserted: bool,
    no_transition_reason: NoTransitionReason | None,
    admission_rejection_reason: AdmissionRejectionReason | None,
    durable_events: tuple,
) -> bool:
    """Independent oracle mirroring the AdmissionResult contract."""
    if outcome is OperationOutcome.APPLIED:
        if not was_inserted or row is None:
            return False
    else:
        if was_inserted or durable_events:
            return False
    if outcome is OperationOutcome.NO_TRANSITION and no_transition_reason is None:
        return False
    if outcome is not OperationOutcome.NO_TRANSITION and no_transition_reason is not None:
        return False
    if outcome is OperationOutcome.ADMISSION_REJECTED and admission_rejection_reason is None:
        return False
    if outcome is not OperationOutcome.ADMISSION_REJECTED and admission_rejection_reason is not None:
        return False
    return True


def _lifecycle_is_consistent(
    outcome: OperationOutcome,
    row: dict | None,
    transition_applied: bool,
    no_transition_reason: NoTransitionReason | None,
    durable_events: tuple,
) -> bool:
    """Independent oracle mirroring the LifecycleResult contract."""
    if outcome is OperationOutcome.APPLIED:
        if not transition_applied or row is None:
            return False
    else:
        if transition_applied or durable_events:
            return False
    if outcome is OperationOutcome.NO_TRANSITION and no_transition_reason is None:
        return False
    if outcome is not OperationOutcome.NO_TRANSITION and no_transition_reason is not None:
        return False
    return True


class TestAdmissionResultContract:
    @_COMMON
    @given(
        outcome=st.sampled_from(_OUTCOMES),
        has_row=st.booleans(),
        was_inserted=st.booleans(),
        no_transition_reason=st.sampled_from(_NO_TRANS),
        admission_rejection_reason=st.sampled_from(_REJECTIONS),
        has_events=st.booleans(),
    )
    def test_constructor_never_yields_an_inconsistent_result(
        self,
        outcome: OperationOutcome,
        has_row: bool,
        was_inserted: bool,
        no_transition_reason: NoTransitionReason | None,
        admission_rejection_reason: AdmissionRejectionReason | None,
        has_events: bool,
    ) -> None:
        row = {"id": "r1"} if has_row else None
        events = ({"type": "created"},) if has_events else ()
        expected_ok = _admission_is_consistent(
            outcome, row, was_inserted, no_transition_reason,
            admission_rejection_reason, events,
        )
        try:
            result = AdmissionResult(
                outcome=outcome,
                row=row,
                was_inserted=was_inserted,
                no_transition_reason=no_transition_reason,
                admission_rejection_reason=admission_rejection_reason,
                durable_events=events,
            )
        except ValueError:
            assert not expected_ok, "contract rejected a state the oracle deems consistent"
            return
        # Constructed => the oracle must also deem it consistent (no impossible state slipped through)
        assert expected_ok, "contract admitted an impossible state (dropped invariant?)"
        # and the constructed object still reflects its inputs
        assert result.outcome is outcome
        assert result.was_inserted is was_inserted

    def test_minimal_valid_constructions_succeed(self) -> None:
        AdmissionResult(outcome=OperationOutcome.APPLIED, was_inserted=True, row={"id": 1})
        AdmissionResult(
            outcome=OperationOutcome.NO_TRANSITION,
            no_transition_reason=NoTransitionReason.MISSING,
        )
        AdmissionResult(
            outcome=OperationOutcome.ADMISSION_REJECTED,
            admission_rejection_reason=AdmissionRejectionReason.QUEUE_PAUSED,
        )
        AdmissionResult(outcome=OperationOutcome.BACKEND_CONFLICT)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"outcome": OperationOutcome.APPLIED, "was_inserted": False},  # applied needs insert
            {"outcome": OperationOutcome.APPLIED, "was_inserted": True},  # applied needs row
            {"outcome": OperationOutcome.NO_TRANSITION},  # needs a reason
            {"outcome": OperationOutcome.ADMISSION_REJECTED},  # needs a rejection reason
            {"outcome": OperationOutcome.BACKEND_ERROR, "was_inserted": True},  # only applied inserts
            {
                "outcome": OperationOutcome.BACKEND_ERROR,
                "no_transition_reason": NoTransitionReason.MISSING,
            },  # only no_transition carries a reason
        ],
    )
    def test_impossible_states_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            AdmissionResult(**kwargs)


class TestLifecycleResultContract:
    @_COMMON
    @given(
        outcome=st.sampled_from(_OUTCOMES),
        has_row=st.booleans(),
        transition_applied=st.booleans(),
        no_transition_reason=st.sampled_from(_NO_TRANS),
        has_events=st.booleans(),
    )
    def test_constructor_never_yields_an_inconsistent_result(
        self,
        outcome: OperationOutcome,
        has_row: bool,
        transition_applied: bool,
        no_transition_reason: NoTransitionReason | None,
        has_events: bool,
    ) -> None:
        row = {"id": "r1"} if has_row else None
        events = ({"type": "moved"},) if has_events else ()
        expected_ok = _lifecycle_is_consistent(
            outcome, row, transition_applied, no_transition_reason, events
        )
        try:
            result = LifecycleResult(
                outcome=outcome,
                row=row,
                transition_applied=transition_applied,
                no_transition_reason=no_transition_reason,
                durable_events=events,
            )
        except ValueError:
            # a ValueError is correct ONLY for a genuinely-inconsistent state;
            # raising on a valid one would be a bug the oracle catches here.
            assert not expected_ok, "contract rejected a state the oracle deems consistent"
            return
        assert expected_ok, "contract admitted an impossible lifecycle state (dropped invariant?)"
        assert result.outcome is outcome
        assert result.transition_applied is transition_applied

    def test_minimal_valid_constructions_succeed(self) -> None:
        LifecycleResult(outcome=OperationOutcome.APPLIED, transition_applied=True, row={"id": 1})
        LifecycleResult(
            outcome=OperationOutcome.NO_TRANSITION,
            transition_applied=False,
            no_transition_reason=NoTransitionReason.WRONG_STATUS,
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"outcome": OperationOutcome.APPLIED, "transition_applied": False},
            {"outcome": OperationOutcome.APPLIED, "transition_applied": True},  # needs row
            {"outcome": OperationOutcome.NO_TRANSITION, "transition_applied": False},  # needs reason
            {"outcome": OperationOutcome.BACKEND_ERROR, "transition_applied": True},  # only applied transitions
        ],
    )
    def test_impossible_states_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            LifecycleResult(**kwargs)
