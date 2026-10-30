"""Validated Research phase authority, without persisted lease credentials."""

from __future__ import annotations

import re
import uuid
from typing import Any

PHASES = frozenset({"drafting_plan", "collecting", "synthesizing", "packaging"})
_FIELDS = frozenset(
    {
        "schema_version",
        "authorization",
        "dispatch_id",
        "phase",
        "checkpoint_id",
        "policy_version",
        "approved_outline_locked",
        "predecessor_job_uuid",
        "job_uuid",
    }
)


def read_phase_dispatch(value: Any) -> dict[str, Any]:
    """Refuse corrupt or unknown authorization instead of guessing phase identity."""
    if not isinstance(value, dict) or value.keys() != _FIELDS:
        raise ValueError("research_phase_dispatch_invalid")
    try:
        valid_id = (
            isinstance(value["dispatch_id"], str) and str(uuid.UUID(value["dispatch_id"])) == value["dispatch_id"]
        )
    except ValueError:
        valid_id = False
    if not (
        type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and isinstance(value["authorization"], str)
        and value["authorization"] in {"initial", "continuation", "checkpoint", "resume"}
        and valid_id
        and isinstance(value["phase"], str)
        and value["phase"] in PHASES
        and type(value["policy_version"]) is int
        and 1 <= value["policy_version"] < 2**31
        and type(value["approved_outline_locked"]) is bool
    ):
        raise ValueError("research_phase_dispatch_invalid")
    for key in ("checkpoint_id", "predecessor_job_uuid", "job_uuid"):
        entry = value[key]
        if entry is not None and (not isinstance(entry, str) or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", entry) is None):
            raise ValueError("research_phase_dispatch_invalid")
    return dict(value)


def new_phase_dispatch(
    phase: str,
    *,
    authorization: str = "initial",
    checkpoint_id: str | None = None,
    policy_version: int = 1,
    approved_outline_locked: bool = False,
    predecessor_job_uuid: str | None = None,
) -> dict[str, Any]:
    """Create one stable delivery identity while retaining the original checkpoint policy."""
    return read_phase_dispatch(
        {
            "schema_version": 1,
            "authorization": authorization,
            "dispatch_id": str(uuid.uuid4()),
            "phase": phase,
            "checkpoint_id": checkpoint_id,
            "policy_version": policy_version,
            "approved_outline_locked": approved_outline_locked,
            "predecessor_job_uuid": predecessor_job_uuid,
            "job_uuid": None,
        }
    )
