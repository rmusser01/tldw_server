"""Narrow caller identity and optional-read permission decisions."""

from pydantic import BaseModel


class UserCapabilities(BaseModel):
    """Current caller affordances; protected endpoints remain authoritative."""

    user_id: int | None
    can_read_scheduled_tasks: bool
    can_read_notifications: bool
    can_read_monitoring_alerts: bool
    can_run_audio_diagnostics: bool
