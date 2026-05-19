"""Shared rollout constants for run-first chat and ACP tests."""

from __future__ import annotations

PHASE2C_RUN_FIRST_COHORT = [
    "openai:gpt-4o-mini",
    "anthropic:claude-3-7-sonnet",
    "openai:gpt-4o",
    "google:gemini-2.5-flash",
]

PHASE2C_RUN_FIRST_CSV = ",".join(PHASE2C_RUN_FIRST_COHORT)
