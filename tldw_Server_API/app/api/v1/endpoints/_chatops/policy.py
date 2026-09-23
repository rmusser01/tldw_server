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

import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from fastapi import status  # noqa: F401  (re-exported for call sites)
from fastapi.responses import JSONResponse

__all__ = [
    "ChatOpsPolicySpec",
    "ChatOpsPolicyRuntime",
    "evaluate_policy",
    "policy_error_response",
    "action_route",
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


@dataclass(frozen=True)
class ChatOpsPolicyRuntime:
    """The provider-facing strings and collaborators the evaluator needs.

    ``_evaluate_*_policy``, ``_*_policy_error_response`` and ``_*_action_route`` were
    identical across the two support modules once the provider vocabulary was
    normalised away -- 103 lines, character for character. What varies is only the
    public vocabulary, all of which is API or metric contract: the rate-limiter key
    prefix, the quota error code and message, and the metric label names.
    """

    #: "discord" / "slack" -- rate limiter key prefix and route namespace.
    name: str
    #: "guild" / "workspace" -- the scope word in keys, codes and messages.
    scope_word: str
    #: Public field holding the per-workspace quota.
    scope_quota_field: str
    #: Callable returning the configured default for that quota.
    scope_quota_default: Callable[[], int]
    #: Callable returning the configured default per-user quota.
    user_quota_default: Callable[[], int]
    #: Metric label naming the scope, e.g. "guild_id" / "team_id".
    scope_label_field: str
    #: Counter emitted when a request is rejected for exceeding a quota.
    quota_rejection_counter: str
    #: Counter emitted for any other policy denial.
    denial_counter: str


def evaluate_policy(
    runtime: ChatOpsPolicyRuntime,
    *,
    policy: dict[str, Any],
    scope_id: str | None,
    channel_id: str | None,
    actor_user_id: str | None,
    action: str,
    rate_limiter: Any,
    coerce: Callable[[Any], str | None],
    safe_int: Callable[[Any], int | None],
    http_status: Any,
) -> dict[str, Any] | None:
    """Return a denial dict, or None when the request is allowed."""
    allowed_commands = policy.get("allowed_commands")
    if isinstance(allowed_commands, list) and allowed_commands:
        if action not in {str(item).lower() for item in allowed_commands}:
            return {
                "status_code": http_status.HTTP_403_FORBIDDEN,
                "error": "command_blocked_by_policy",
                "message": f"Command '{action}' is not allowed for this {runtime.scope_word}",
            }

    deny_channels = set(normalize_string_list(policy.get("channel_denylist"), coerce=coerce))
    allow_channels = set(normalize_string_list(policy.get("channel_allowlist"), coerce=coerce))
    if channel_id and channel_id in deny_channels:
        return {
            "status_code": http_status.HTTP_403_FORBIDDEN,
            "error": "channel_blocked_by_policy",
            "message": f"Channel '{channel_id}' is blocked by policy",
        }
    if allow_channels and channel_id and channel_id not in allow_channels:
        return {
            "status_code": http_status.HTTP_403_FORBIDDEN,
            "error": "channel_not_allowed_by_policy",
            "message": f"Channel '{channel_id}' is not in the allowlist",
        }

    scope_limit = safe_int(policy.get(runtime.scope_quota_field)) or runtime.scope_quota_default()
    scope_key = coerce(scope_id) or "unknown"
    allowed_scope, retry_after_scope = rate_limiter.allow(
        f"{runtime.name}:{runtime.scope_word}:{scope_key}",
        max(1, scope_limit),
    )
    if not allowed_scope:
        return {
            "status_code": http_status.HTTP_429_TOO_MANY_REQUESTS,
            "error": f"{runtime.scope_word}_quota_exceeded",
            "message": f"{runtime.scope_word.capitalize()} command quota exceeded",
            "retry_after_seconds": retry_after_scope,
        }

    if actor_user_id:
        user_limit = safe_int(policy.get("user_quota_per_minute")) or runtime.user_quota_default()
        allowed_user, retry_after_user = rate_limiter.allow(
            f"{runtime.name}:user:{scope_key}:{actor_user_id}",
            max(1, user_limit),
        )
        if not allowed_user:
            return {
                "status_code": http_status.HTTP_429_TOO_MANY_REQUESTS,
                "error": "user_quota_exceeded",
                "message": "User command quota exceeded",
                "retry_after_seconds": retry_after_user,
            }

    return None


def policy_error_response(
    runtime: ChatOpsPolicyRuntime,
    policy_error: dict[str, Any],
    *,
    scope_id: str | None,
    action: str | None,
    emit_counter: Callable[..., None],
    log: Any,
    safe_int: Callable[[Any], int | None],
    http_status: Any,
) -> JSONResponse:
    """Render a denial, emitting the quota or generic counter and a warning."""
    status_code = int(policy_error.get("status_code") or http_status.HTTP_403_FORBIDDEN)
    response_payload = {k: v for k, v in policy_error.items() if k != "status_code"}
    headers: dict[str, str] = {}
    retry_after = safe_int(policy_error.get("retry_after_seconds"))
    counter = runtime.denial_counter
    if retry_after is not None and retry_after > 0:
        headers["Retry-After"] = str(retry_after)
        counter = runtime.quota_rejection_counter
    emit_counter(
        counter,
        **{runtime.scope_label_field: scope_id or "na"},
        action=action or "na",
        error=response_payload.get("error"),
    )
    log.warning(
        "{} policy denied request: {}={} action={} error={}",
        runtime.name.capitalize(),
        runtime.scope_label_field,
        scope_id or "na",
        action or "na",
        response_payload.get("error"),
    )
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={"ok": False, **response_payload},
    )


def action_route(runtime: ChatOpsPolicyRuntime, action: str) -> str:
    """Map a ChatOps command to its internal route."""
    routes = {
        "help": f"{runtime.name}.help",
        "ask": "chat.ask",
        "rag": "rag.search",
        "summarize": "summarize.run",
        "status": "jobs.status",
    }
    return routes.get(action, "chat.ask")


POLICY_DEFAULT_KEY = "__default__"


class PolicyStore:
    """Per-tenant policies over one default, normalised on every read and write.

    The tenant is a Discord guild or a Slack workspace; ``None`` means the default.
    """

    def __init__(
        self,
        *,
        normalize: Callable[..., dict[str, Any]],
        default_policy: Callable[[], dict[str, Any]],
        coerce: Callable[[Any], str | None],
    ) -> None:
        self._normalize = normalize
        self._default_policy = default_policy
        self._coerce = coerce
        self._lock = threading.Lock()
        self._policies: dict[str, dict[str, Any]] = {}

    def _key(self, scope_id: str | None) -> str:
        return self._coerce(scope_id) or POLICY_DEFAULT_KEY

    def _default_locked(self) -> dict[str, Any]:
        raw = self._policies.get(POLICY_DEFAULT_KEY)
        return self._normalize(raw if isinstance(raw, dict) else None)

    def get(self, scope_id: str | None) -> dict[str, Any]:
        key = self._key(scope_id)
        with self._lock:
            default = self._default_locked()
            if key == POLICY_DEFAULT_KEY:
                return default
            selected = self._policies.get(key)
            return self._normalize(selected, base=default) if isinstance(selected, dict) else default

    def set(self, scope_id: str | None, payload: dict[str, Any] | None) -> tuple[str | None, dict[str, Any]]:
        key = self._key(scope_id)
        with self._lock:
            base = self._default_locked() if key != POLICY_DEFAULT_KEY else self._default_policy()
            normalized = self._normalize(payload, base=base)
            self._policies[key] = dict(normalized)
        return self._coerce(scope_id), normalized

    def clear(self) -> None:
        with self._lock:
            self._policies.clear()


def resolve_actor_id(
    policy: dict[str, Any],
    requested_user_id: str | None,
    *,
    provider_label: str,
    coerce: Callable[[Any], str | None],
    http_status: Any,
) -> tuple[str | None, dict[str, Any] | None]:
    """Map a platform user to a local user id, or return a policy error."""
    requested = coerce(requested_user_id)
    user_mappings = policy.get("user_mappings") if isinstance(policy.get("user_mappings"), dict) else {}
    mapped_user = user_mappings.get(requested) if requested else None
    if mapped_user:
        return coerce(mapped_user), None
    service_user_id = coerce(policy.get("service_user_id"))
    if bool(policy.get("strict_user_mapping")) and not service_user_id:
        return None, {
            "status_code": http_status.HTTP_403_FORBIDDEN,
            "error": "unknown_user_mapping",
            "message": f"{provider_label} user is not mapped to a local user and strict mapping is enabled",
        }
    return requested or service_user_id, None
