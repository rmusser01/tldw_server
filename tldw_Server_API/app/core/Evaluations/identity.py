"""Canonical identity helpers for evaluations ownership, routing, and webhook subjects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationIdentity:
    """Stable identity fields used across evaluations routes, services, and webhooks."""

    user_scope: str
    created_by: str
    rate_limit_subject: str
    webhook_user_id: str


def canonical_evaluations_user_scope(
    user_or_id: Any,
    *,
    fallback: str | int | None = None,
) -> str:
    """Return the canonical string scope for evaluations storage and cache keys."""

    raw = ""
    if user_or_id is not None and hasattr(user_or_id, "id_str"):
        raw = str(user_or_id.id_str or "").strip()
    if not raw and user_or_id is not None and hasattr(user_or_id, "id"):
        raw = str(user_or_id.id or "").strip()
    elif not raw and user_or_id is not None:
        raw = str(user_or_id).strip()

    if not raw and fallback is not None:
        raw = str(fallback).strip()

    if not raw:
        raise ValueError("Evaluations user scope is required")

    return raw


def evaluations_identity_from_user(
    user: Any,
    *,
    fallback: str | int | None = None,
) -> EvaluationIdentity:
    """Build the canonical evaluations identity for an authenticated user-like object."""

    user_scope = canonical_evaluations_user_scope(user, fallback=fallback)
    webhook_user_id = user_scope if user_scope.startswith("user_") else f"user_{user_scope}"
    return EvaluationIdentity(
        user_scope=user_scope,
        created_by=user_scope,
        rate_limit_subject=user_scope,
        webhook_user_id=webhook_user_id,
    )
