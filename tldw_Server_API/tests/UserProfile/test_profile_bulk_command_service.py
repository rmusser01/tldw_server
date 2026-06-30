from __future__ import annotations

from tldw_Server_API.app.core.UserProfiles.bulk_command_service import (
    ProfileBulkCommandService,
)


def test_filter_visible_targets_keeps_candidate_order() -> None:
    service = ProfileBulkCommandService()

    visible = service.filter_visible_targets(
        candidate_user_ids=[1, 2, 3],
        visible_user_ids={1, 3},
    )

    assert visible == [1, 3]


def test_filter_visible_targets_coerces_ids_and_drops_duplicates() -> None:
    service = ProfileBulkCommandService()

    visible = service.filter_visible_targets(
        candidate_user_ids=["3", 2, "3", 1],
        visible_user_ids={"1", 3},
    )

    assert visible == [3, 1]


def test_requires_confirmation_only_for_non_dry_run_above_threshold() -> None:
    service = ProfileBulkCommandService()

    assert service.requires_confirmation(
        dry_run=False,
        total_targets=3,
        threshold=2,
        confirmed=False,
    ) is True
    assert service.requires_confirmation(
        dry_run=True,
        total_targets=3,
        threshold=2,
        confirmed=False,
    ) is False
    assert service.requires_confirmation(
        dry_run=False,
        total_targets=2,
        threshold=2,
        confirmed=False,
    ) is False
    assert service.requires_confirmation(
        dry_run=False,
        total_targets=3,
        threshold=2,
        confirmed=True,
    ) is False


def test_build_diffs_defaults_missing_before_values_to_none() -> None:
    service = ProfileBulkCommandService()

    diffs = service.build_diffs(
        updates=[
            ("limits.storage_quota_mb", 2048),
            ("preferences.ui.theme", "paper"),
        ],
        applied_keys={"limits.storage_quota_mb"},
        before_values={},
        mask_value=lambda key, value: f"{key}:{value}",
    )

    assert len(diffs) == 1
    assert diffs == [
        {
            "key": "limits.storage_quota_mb",
            "before": None,
            "after": "limits.storage_quota_mb:2048",
        }
    ]
