"""
Shared startup warning models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class StartupWarningRecord:
    """Immutable startup warning record stored for the current process boot."""

    component: str
    severity: str
    startup_action: str
    code: str
    summary: str
    remediation: str
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
