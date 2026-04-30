"""
In-memory startup warning registry for the current process boot.
"""

from __future__ import annotations

from collections import Counter

from tldw_Server_API.app.services.startup_warning_models import StartupWarningRecord


class StartupWarningRegistry:
    """Collect and summarize startup warnings for the current process."""

    def __init__(self, *, startup_id: str) -> None:
        self.startup_id = startup_id
        self._warnings: list[StartupWarningRecord] = []

    def add_warning(self, record: StartupWarningRecord) -> None:
        self._warnings.append(record)

    def list_warnings(self) -> list[StartupWarningRecord]:
        return sorted(
            self._warnings,
            key=lambda item: (
                item.component,
                item.code,
                item.startup_action,
                item.detected_at.isoformat(),
            ),
        )

    def summary(self) -> dict[str, object]:
        warnings = self.list_warnings()
        blocking_total = sum(
            1 for item in warnings if item.startup_action == "block_startup"
        )
        component_counts = Counter(item.component for item in warnings)
        action_counts = Counter(item.startup_action for item in warnings)
        return {
            "startup_id": self.startup_id,
            "total": len(warnings),
            "blocking_total": blocking_total,
            "has_blocking": blocking_total > 0,
            "by_component": {
                component: component_counts[component]
                for component in sorted(component_counts)
            },
            "by_action": {
                action: action_counts[action] for action in sorted(action_counts)
            },
        }

    def clear(self) -> None:
        self._warnings.clear()

    def should_block_startup(self) -> bool:
        return any(
            item.startup_action == "block_startup" for item in self._warnings
        )
