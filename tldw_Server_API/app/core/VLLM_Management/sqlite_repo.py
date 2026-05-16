"""Compatibility import for the DB-owned managed vLLM SQLite repository."""

from __future__ import annotations

from tldw_Server_API.app.core.DB_Management.VLLM_Management_DB import (
    SqliteVLLMInstanceRepository,
)

__all__ = ["SqliteVLLMInstanceRepository"]
