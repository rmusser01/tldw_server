"""
Bulk UserProfiles command helpers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any


class ProfileBulkCommandService:
    """Pure helper surface for bulk profile command orchestration."""

    @staticmethod
    def filter_visible_targets(
        *,
        candidate_user_ids: Iterable[int | str],
        visible_user_ids: Iterable[int | str],
    ) -> list[int]:
        visible = {int(user_id) for user_id in visible_user_ids}
        filtered: list[int] = []
        seen: set[int] = set()
        for user_id in candidate_user_ids:
            coerced = int(user_id)
            if coerced not in visible or coerced in seen:
                continue
            filtered.append(coerced)
            seen.add(coerced)
        return filtered

    @staticmethod
    def requires_confirmation(
        *,
        dry_run: bool,
        total_targets: int,
        threshold: int,
        confirmed: bool,
    ) -> bool:
        return not dry_run and total_targets > threshold and not confirmed

    @staticmethod
    def build_diffs(
        *,
        updates: Sequence[tuple[str, Any]],
        applied_keys: Iterable[str],
        before_values: dict[str, Any],
        mask_value: Callable[[str, Any], Any],
    ) -> list[dict[str, Any]]:
        applied = set(applied_keys)
        return [
            {
                "key": key,
                "before": before_values.get(key),
                "after": mask_value(key, value),
            }
            for key, value in updates
            if key in applied
        ]
