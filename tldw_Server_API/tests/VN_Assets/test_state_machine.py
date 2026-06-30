from tldw_Server_API.app.core.VN_Assets.models import SlotReadiness
from tldw_Server_API.app.core.VN_Assets.state import (
    derive_pack_readiness,
    derive_slot_status,
)


def test_slot_status_prefers_reviewing_over_approved_when_drafts_remain() -> None:
    status = derive_slot_status(
        has_active_job=False,
        has_queued_job=False,
        is_skipped=False,
        requested_variants=2,
        failed_variants=0,
        review_statuses=["approved", "draft"],
        required_for_runtime=True,
    )

    assert status == "reviewing"


def test_slot_status_active_beats_queued_and_skipped() -> None:
    status = derive_slot_status(
        has_active_job=True,
        has_queued_job=True,
        is_skipped=True,
        requested_variants=1,
        failed_variants=0,
        review_statuses=[],
        required_for_runtime=False,
    )

    assert status == "generating"


def test_slot_status_queued_beats_skipped() -> None:
    status = derive_slot_status(
        has_active_job=False,
        has_queued_job=True,
        is_skipped=True,
        requested_variants=1,
        failed_variants=0,
        review_statuses=[],
        required_for_runtime=False,
    )

    assert status == "queued"


def test_slot_status_supports_cancelled_before_planned() -> None:
    status = derive_slot_status(
        has_active_job=False,
        has_queued_job=False,
        is_skipped=False,
        is_cancelled=True,
        requested_variants=1,
        failed_variants=0,
        review_statuses=[],
        required_for_runtime=False,
    )

    assert status == "cancelled"


def test_slot_status_returns_planned_for_unstarted_slot() -> None:
    status = derive_slot_status(
        has_active_job=False,
        has_queued_job=False,
        is_skipped=False,
        requested_variants=1,
        failed_variants=0,
        review_statuses=[],
        required_for_runtime=False,
    )

    assert status == "planned"


def test_slot_status_one_approved_item_without_drafts_is_approved() -> None:
    status = derive_slot_status(
        has_active_job=False,
        has_queued_job=False,
        is_skipped=False,
        requested_variants=3,
        failed_variants=0,
        review_statuses=["approved", "rejected"],
        required_for_runtime=True,
    )

    assert status == "approved"


def test_slot_status_required_generated_without_approved_item_is_reviewing() -> None:
    status = derive_slot_status(
        has_active_job=False,
        has_queued_job=False,
        is_skipped=False,
        requested_variants=2,
        failed_variants=1,
        review_statuses=["hidden"],
        required_for_runtime=True,
    )

    assert status == "reviewing"


def test_slot_status_zero_request_hidden_or_rejected_remains_reviewing() -> None:
    status = derive_slot_status(
        has_active_job=False,
        has_queued_job=False,
        is_skipped=False,
        requested_variants=0,
        failed_variants=0,
        review_statuses=["hidden", "rejected"],
        required_for_runtime=False,
    )

    assert status == "reviewing"


def test_optional_failed_slot_does_not_block_pack_readiness() -> None:
    readiness = derive_pack_readiness(
        required_slots=[SlotReadiness(slot_id=1, status="approved", warnings=())],
        optional_slots=[SlotReadiness(slot_id=2, status="failed", warnings=("depth_unavailable",))],
        active_jobs=0,
        approved_item_errors=[],
    )

    assert readiness.ready is True
    assert readiness.warnings == ("depth_unavailable",)
