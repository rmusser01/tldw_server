"""Per-user SQLite-backed orchestration DB factory."""
from __future__ import annotations

import functools

from tldw_Server_API.app.core.DB_Management.Orchestration_DB import OrchestrationDB


@functools.lru_cache(maxsize=64)
def get_orchestration_db(user_id: int) -> OrchestrationDB:
    """Get or create the current user's durable orchestration DB."""
    return OrchestrationDB.for_user(int(user_id))
