"""The ChatOps policy schema, normaliser and shared response envelope.

``discord_support.py`` and ``slack_support.py`` carried one copy each of the policy
default and its normaliser -- a 187-line identical run once the provider vocabulary
is normalised away -- together with ``_error_response`` and ``_metric_labels``,
which were byte-identical **at the same line numbers** (``:241`` and ``:248``) in
both files.

The vocabulary has already drifted, which is what a second copy buys:

=========================  ==========================  ================================
field                      Discord                     Slack
=========================  ==========================  ================================
scope quota key            ``guild_quota_per_minute``   ``workspace_quota_per_minute``
``status_scope`` values    ``guild``/``guild_and_user`` ``workspace``/``workspace_and_user``
``default_response_mode``  ``ephemeral``/``channel``    ``ephemeral``/``thread``/``channel``
=========================  ==========================  ================================

Those are **public request and response fields**, so they are parameterised rather
than unified: renaming them would break every existing caller and stored policy.
:class:`ChatOpsPolicySpec` carries each provider's spelling so one normaliser can
serve both. The ``thread`` mode is a real Slack capability, not drift.

See Docs/ADR/050-chatops-shared-shell.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from fastapi import status  # noqa: F401  (re-exported for call sites)
from fastapi.responses import JSONResponse

__all__ = [
    "ChatOpsPolicySpec",
    "default_policy",
    "normalize_policy_payload",
    "error_response",
    "metric_labels",
    "normalize_string_list",
]


@dataclass(frozen=True)
class ChatOpsPolicySpec:
    """One provider's spelling of the shared policy schema."""

    #: Supported command names, e.g. ("help", "ask", "rag", ...).
    supported_actions: Sequence[str]
    #: Public field holding the per-workspace quota, e.g. "guild_quota_per_minute".
    scope_quota_field: str
    #: Callable returning the default for that quota.
    scope_quota_default: Callable[[], int]
    #: Callable returning the default per-user quota.
    user_quota_default: Callable[[], int]
    #: Accepted status_scope values, first entry being the default.
    status_scope_values: Sequence[str]
    #: Accepted default_response_mode values.
    response_modes: Sequence[str]


def normalize_string_list(raw: Any, *, coerce: Callable[[Any], str | None]) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        cleaned = coerce(item)
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values


def default_policy(spec: ChatOpsPolicySpec) -> dict[str, Any]:
    """Return the provider's default policy, in the provider's own vocabulary."""
    return {
        "allowed_commands": list(spec.supported_actions),
        "channel_allowlist": [],
        "channel_denylist": [],
        "default_response_mode": "ephemeral",
        "strict_user_mapping": False,
        "service_user_id": None,
        "user_mappings": {},
        spec.scope_quota_field: spec.scope_quota_default(),
        "user_quota_per_minute": spec.user_quota_default(),
        "status_scope": spec.status_scope_values[0],
    }


def normalize_policy_payload(
    spec: ChatOpsPolicySpec,
    payload: dict[str, Any] | None,
    *,
    base: dict[str, Any] | None = None,
    coerce: Callable[[Any], str | None],
    safe_int: Callable[[Any], int | None],
) -> dict[str, Any]:
    """Merge a policy patch onto a base, dropping anything the schema rejects.

    Unknown keys and invalid values are ignored rather than rejected, which is the
    behaviour both copies already had.
    """
    merged = dict(base or default_policy(spec))
    data = payload if isinstance(payload, dict) else {}

    if "allowed_commands" in data:
        allowed: list[str] = []
        for candidate in normalize_string_list(data.get("allowed_commands"), coerce=coerce):
            lowered = candidate.lower()
            if lowered in spec.supported_actions and lowered not in allowed:
                allowed.append(lowered)
        merged["allowed_commands"] = allowed or list(spec.supported_actions)

    if "channel_allowlist" in data:
        merged["channel_allowlist"] = normalize_string_list(
            data.get("channel_allowlist"), coerce=coerce
        )
    if "channel_denylist" in data:
        merged["channel_denylist"] = normalize_string_list(
            data.get("channel_denylist"), coerce=coerce
        )

    if "default_response_mode" in data:
        mode = coerce(data.get("default_response_mode"))
        if mode and mode.lower() in spec.response_modes:
            merged["default_response_mode"] = mode.lower()

    if "strict_user_mapping" in data:
        merged["strict_user_mapping"] = bool(data.get("strict_user_mapping"))
    if "service_user_id" in data:
        merged["service_user_id"] = coerce(data.get("service_user_id"))

    if "user_mappings" in data and isinstance(data.get("user_mappings"), dict):
        normalized_mappings: dict[str, str] = {}
        for raw_key, raw_value in dict(data.get("user_mappings") or {}).items():
            key = coerce(raw_key)
            value = coerce(raw_value)
            if key and value:
                normalized_mappings[key] = value
        merged["user_mappings"] = normalized_mappings

    if spec.scope_quota_field in data:
        value = safe_int(data.get(spec.scope_quota_field))
        if value is not None and value > 0:
            merged[spec.scope_quota_field] = value
    if "user_quota_per_minute" in data:
        value = safe_int(data.get("user_quota_per_minute"))
        if value is not None and value > 0:
            merged["user_quota_per_minute"] = value

    if "status_scope" in data:
        scope = coerce(data.get("status_scope"))
        if scope and scope.lower() in spec.status_scope_values:
            merged["status_scope"] = scope.lower()

    return merged


def error_response(status_code: int, error: str, message: str) -> JSONResponse:
    """The ChatOps error envelope.

    Was byte-identical in both support modules, at the same line number.
    """
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error": error, "message": message},
    )


def metric_labels(**labels: Any) -> dict[str, str]:
    """Stringify metric labels, dropping None.

    Was byte-identical in both support modules, at the same line number.
    """
    normalized: dict[str, str] = {}
    for key, value in labels.items():
        if value is None:
            continue
        normalized[str(key)] = str(value)
    return normalized
