"""Security stages for MCP tool execution."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from mcp_unified.interfaces.path_scope import (
    PathScopeCandidate,
    normalize_path_scope_candidates,
)
from pydantic import BaseModel

from ..auth.authnz_rbac import Action, Resource
from ..execution_outcomes import ExpectedToolFailure, ExpectedToolFailureReason
from ..modules.base import BaseModule
from ..protocol_types import (
    ApprovalRequiredError,
    AuthenticatedExecutionScope,
    GovernanceDeniedError,
    InvalidParamsException,
    PreparedToolCall,
    RequestContext,
    _has_trusted_compat_claims,
)
from ..tool_observability import ensure_tool_definition_eval_metadata
from .canonical import (
    ARGUMENTS_MAX_BYTES,
    PREPARED_HMAC_PAYLOAD_MAX_BYTES,
    SCOPE_REPORTING_MAX_BYTES,
    TOOL_DEFINITION_MAX_BYTES,
    JsonValue,
    canonical_json_bytes,
    decode_canonical_json_object_or_none,
)
from .dependencies import ToolExecutionDependencies
from .models import (
    CanonicalJsonSnapshot,
    IdempotencyExecutionPolicy,
    PreparedExecutionPolicy,
)

_TOOL_AUTHORIZATION_ALIASES = {
    "bash": "run",
    "shell": "run",
}
_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}
_IDEMPOTENCY_TTL_HARD_MAX_SECONDS = 604_800
_IDEMPOTENCY_LOCK_HARD_MAX_SECONDS = 604_800
_IDEMPOTENCY_MAX_ENTRIES_HARD_MAX = 100_000
_RATE_LIMIT_CATEGORIES = frozenset(
    {
        "ingestion",
        "management",
        "read",
        "utility",
        "browser",
        "code",
        "filesystem",
        "rag_generation",
        "shell",
        "search",
        "tool_discovery",
    }
)

_AsyncBoolCallback = Callable[..., bool | Awaitable[bool]]
_SyncBoolCallback = Callable[..., bool]


def _is_truthy(value: Any) -> bool:
    """Parse host-neutral truthy flags without importing protocol helpers."""
    try:
        return str(value or "").strip().lower() in _TRUTHY_VALUES
    except Exception:  # noqa: BLE001 - best-effort normalization must fail closed.
        return False


def _is_unexpected_keyword_type_error(exc: TypeError, keyword: str) -> bool:
    """Return True when a TypeError reports an unsupported keyword argument."""

    message = str(exc)
    return (
        f"unexpected keyword argument '{keyword}'" in message
        or f'unexpected keyword argument "{keyword}"' in message
    )


class ToolExecutionSecurity:
    """Perform validation, authorization, policy, and preparation for MCP tools/call."""

    def __init__(
        self,
        *,
        dependencies: ToolExecutionDependencies,
        tool_name_re: re.Pattern[str],
        prepared_call_secret: bytes,
        noncritical_exceptions: tuple[type[BaseException], ...],
    ) -> None:
        """Store security dependencies and mutable governance state."""

        self.dependencies = dependencies
        self.module_registry = dependencies.module_registry
        self.rbac_policy = dependencies.rbac_policy
        self.metrics = dependencies.metrics
        self._tool_name_re = tool_name_re
        self._prepared_call_secret = prepared_call_secret
        self._noncritical_exceptions = noncritical_exceptions
        self._governance_service: Any | None = None
        self._governance_store: Any | None = None
        self._governance_lock = asyncio.Lock()
        self._prepare_compatibility_callbacks: dict[str, Callable[..., Any]] = {}

    def configure_prepare_compatibility_callbacks(self, **callbacks: Callable[..., Any]) -> None:
        """Install late-bound protocol compatibility callbacks for prepare_tool_call."""

        self._prepare_compatibility_callbacks = {
            name: callback
            for name, callback in callbacks.items()
            if callable(callback)
        }

    def _prepare_callback(self, name: str, default: Callable[..., Any]) -> Callable[..., Any]:
        """Return a late-bound compatibility callback or the extracted default."""

        callback = self._prepare_compatibility_callbacks.get(name)
        return callback if callable(callback) else default

    async def rbac_check(
        self,
        user_id: str | None,
        resource: Resource,
        action: Action,
        resource_id: str | None = None,
        *,
        rbac_policy: Any | None = None,
    ) -> bool:
        """Evaluate RBAC permissions through the configured policy adapter."""

        if not user_id:
            return False
        fn = getattr(rbac_policy or self.rbac_policy, "check_permission", None)
        if not fn:
            return False
        try:
            if inspect.iscoroutinefunction(fn):
                return await fn(user_id, resource, action, resource_id)
            return fn(user_id, resource, action, resource_id)
        except self._noncritical_exceptions:
            return False

    def scoped_permissions(self, context: RequestContext) -> list[str]:
        metadata = getattr(context, "metadata", {})
        if not isinstance(metadata, dict):
            return []
        raw = metadata.get("permissions") or []
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(item) for item in raw if isinstance(item, str)]
        return []

    def mcp_scopes(
        self,
        context: RequestContext,
        *,
        scoped_permissions: Callable[[RequestContext], list[str]] | None = None,
    ) -> list[str]:
        scopes: list[str] = []
        scoped_permissions_fn = scoped_permissions or self.scoped_permissions
        for scope in scoped_permissions_fn(context):
            try:
                if scope.lower().startswith("mcp:"):
                    scopes.append(scope)
            except self._noncritical_exceptions:
                continue
        return scopes

    def api_key_scopes(self, context: RequestContext) -> set[str] | None:
        """Return normalized API key scopes when present on the request context."""
        metadata = getattr(context, "metadata", {})
        if not isinstance(metadata, dict):
            return None
        raw = metadata.get("api_key_scopes")
        if raw is None:
            return None

        normalizer = getattr(self.dependencies, "api_key_scope_normalizer", None)
        normalize = getattr(normalizer, "normalize", None) if normalizer is not None else None
        if callable(normalize):
            try:
                return set(normalize(raw))
            except self._noncritical_exceptions as exc:
                logger.debug(
                    "MCP API key scope normalization failed; using local fallback: {}",
                    exc.__class__.__name__,
                )

        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, list):
                        return {
                            item.strip().lower()
                            for item in parsed
                            if isinstance(item, str) and item.strip()
                        }
            return {stripped.lower()} if stripped else set()
        if isinstance(raw, (list, tuple, set)):
            return {str(item).strip().lower() for item in raw if str(item).strip()}
        return set()

    def api_key_scope_level(
        self,
        context: RequestContext,
        *,
        api_key_scopes: Callable[[RequestContext], set[str] | None] | None = None,
    ) -> str | None:
        scopes = (api_key_scopes or self.api_key_scopes)(context)
        if not scopes:
            return None
        if "admin" in scopes or "service" in scopes:
            return "admin"
        if "write" in scopes:
            return "write"
        if "read" in scopes:
            return "read"
        return None

    def api_key_allows(
        self,
        context: RequestContext,
        *,
        is_write: bool | None = None,
        api_key_scope_level: Callable[[RequestContext], str | None] | None = None,
    ) -> bool:
        """Gate MCP operations by API key scopes when present."""
        level = (api_key_scope_level or self.api_key_scope_level)(context)
        if level is None:
            return True
        if level == "admin":
            return True
        if is_write is None:
            return level in {"read", "write"}
        if is_write:
            return level == "write"
        return level in {"read", "write"}

    @staticmethod
    def scope_matches(scope: str, resource_kind: str, identifier: str | None) -> bool:
        scope = scope.strip().lower()
        if not scope.startswith("mcp:"):
            return False
        parts = scope.split(":")
        if len(parts) == 2 and parts[1] == "*":
            return True
        if len(parts) < 3:
            return False
        kind = parts[1]
        value = ":".join(parts[2:])
        if kind == "*":
            return True
        if kind != resource_kind:
            return False
        if value in {"*", ""}:
            return True
        if identifier is None:
            return False
        return value == identifier.lower()

    def scope_allows(
        self,
        context: RequestContext,
        resource_kind: str,
        identifier: str | None,
        *,
        mcp_scopes: Callable[[RequestContext], list[str]] | None = None,
        scope_matches: _SyncBoolCallback | None = None,
    ) -> bool:
        scopes = (mcp_scopes or self.mcp_scopes)(context)
        if not scopes:
            return True
        identifier_norm = identifier.lower() if isinstance(identifier, str) else None
        if identifier_norm is None:
            # Allow listing/browsing when any scoped permission exists for this resource kind.
            for scope in scopes:
                try:
                    parts = scope.strip().lower().split(":")
                except self._noncritical_exceptions:
                    continue
                if len(parts) >= 2 and parts[0] == "mcp":
                    if parts[1] == "*" or parts[1] == resource_kind:
                        return True
        scope_matches_fn = scope_matches or self.scope_matches
        return any(scope_matches_fn(scope, resource_kind, identifier_norm) for scope in scopes)

    @staticmethod
    async def _call_bool(callback: _AsyncBoolCallback, *args: Any, **kwargs: Any) -> bool:
        result = callback(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)

    async def has_module_permission(
        self,
        context: RequestContext,
        module_id: str | None,
        *,
        rbac_check: _AsyncBoolCallback | None = None,
        scope_allows: _SyncBoolCallback | None = None,
    ) -> bool:
        module_id_norm = module_id or ""
        scope_allows_fn = scope_allows or self.scope_allows
        if _has_trusted_compat_claims(context):
            return scope_allows_fn(context, Resource.MODULE.value, module_id_norm or None)
        rbac_check_fn = rbac_check or self.rbac_check
        if not await self._call_bool(rbac_check_fn, context.user_id, Resource.MODULE, Action.READ, module_id_norm):
            return False
        return scope_allows_fn(context, Resource.MODULE.value, module_id_norm or None)

    async def has_tool_permission(
        self,
        context: RequestContext,
        tool_name: str,
        *,
        is_write: bool | None = None,
        rbac_check: _AsyncBoolCallback | None = None,
        scope_allows: _SyncBoolCallback | None = None,
        api_key_allows: Callable[..., bool] | None = None,
        tool_authorization_names: Callable[[str], tuple[str, ...]] | None = None,
    ) -> bool:
        scope_allows_fn = scope_allows or self.scope_allows
        api_key_allows_fn = api_key_allows or self.api_key_allows
        tool_authorization_names_fn = tool_authorization_names or self.tool_authorization_names
        if _has_trusted_compat_claims(context):
            for auth_name in tool_authorization_names_fn(tool_name):
                if scope_allows_fn(context, Resource.TOOL.value, auth_name):
                    return api_key_allows_fn(context, is_write=is_write)
            return False
        has_named_permission = False
        rbac_check_fn = rbac_check or self.rbac_check
        for auth_name in tool_authorization_names_fn(tool_name):
            if not await self._call_bool(rbac_check_fn, context.user_id, Resource.TOOL, Action.EXECUTE, auth_name):
                continue
            if not scope_allows_fn(context, Resource.TOOL.value, auth_name):
                continue
            has_named_permission = True
            break
        if not has_named_permission:
            return False
        return api_key_allows_fn(context, is_write=is_write)

    def extract_allowed_tools(self, context: RequestContext) -> list[str] | None:
        """Extract allowed-tools list from request context metadata."""
        try:
            metadata = context.metadata or {}
            allowed = metadata.get("allowed_tools")
        except self._noncritical_exceptions:
            return None

        if allowed is None:
            return None
        if isinstance(allowed, list):
            cleaned = [str(item).strip() for item in allowed if str(item).strip()]
            return cleaned or None
        if isinstance(allowed, str):
            try:
                parsed = json.loads(allowed)
                if isinstance(parsed, list):
                    cleaned = [str(item).strip() for item in parsed if str(item).strip()]
                    return cleaned or None
            except json.JSONDecodeError:
                pass
            cleaned = [part.strip() for part in allowed.split(",") if part.strip()]
            return cleaned or None
        return None

    @staticmethod
    def extract_tool_command(tool_args: Any) -> str | None:
        """Extract command-like string from tool arguments for pattern matching."""
        if not isinstance(tool_args, dict):
            return None
        for key in ("command", "cmd", "args", "arguments"):
            if key not in tool_args:
                continue
            value = tool_args.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return " ".join(str(part) for part in value)
        return None

    def matches_allowed_tool_pattern(
        self,
        tool_name: str,
        tool_args: Any,
        pattern: str,
        *,
        extract_tool_command: Callable[[Any], str | None] | None = None,
    ) -> bool:
        """Check if tool invocation matches an allowed-tools pattern."""
        pattern = str(pattern or "").strip()
        if not pattern:
            return False
        if "(" not in pattern:
            return tool_name == pattern
        if not pattern.endswith(")"):
            return False

        base_name, cmd_pattern = pattern.split("(", 1)
        cmd_pattern = cmd_pattern[:-1]
        base_name = base_name.strip()
        if tool_name != base_name:
            return False

        command = (extract_tool_command or self.extract_tool_command)(tool_args)
        if command is None:
            return False

        regex_pattern = re.escape(cmd_pattern)
        regex_pattern = regex_pattern.replace(r"\*", ".*")
        try:
            return bool(re.match(f"^{regex_pattern}$", command.strip()))
        except re.error:
            return False

    @staticmethod
    def tool_authorization_names(tool_name: str) -> tuple[str, ...]:
        """Return invoked and canonical names that may authorize a tool call."""

        canonical_name = _TOOL_AUTHORIZATION_ALIASES.get(tool_name)
        if canonical_name and canonical_name != tool_name:
            return (tool_name, canonical_name)
        return (tool_name,)

    def matches_tool_authorization_pattern(
        self,
        tool_name: str,
        tool_args: Any,
        pattern: str,
        *,
        matches_allowed_tool_pattern: Callable[[str, Any, str], bool] | None = None,
        tool_authorization_names: Callable[[str], tuple[str, ...]] | None = None,
    ) -> bool:
        """Match a policy pattern against the invoked name and any canonical alias."""

        matches_allowed_tool_pattern_fn = matches_allowed_tool_pattern or self.matches_allowed_tool_pattern
        tool_authorization_names_fn = tool_authorization_names or self.tool_authorization_names
        return any(
            matches_allowed_tool_pattern_fn(auth_name, tool_args, pattern)
            for auth_name in tool_authorization_names_fn(tool_name)
        )

    def scope_allows_tool_name(
        self,
        context: RequestContext,
        tool_name: str,
        *,
        scope_allows: _SyncBoolCallback | None = None,
        tool_authorization_names: Callable[[str], tuple[str, ...]] | None = None,
    ) -> bool:
        """Return True when scopes allow the invoked tool name or canonical alias."""

        scope_allows_fn = scope_allows or self.scope_allows
        tool_authorization_names_fn = tool_authorization_names or self.tool_authorization_names
        return any(
            scope_allows_fn(context, Resource.TOOL.value, auth_name)
            for auth_name in tool_authorization_names_fn(tool_name)
        )

    def is_tool_allowed_by_context(
        self,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        *,
        extract_allowed_tools: Callable[[RequestContext], list[str] | None] | None = None,
        matches_tool_authorization_pattern: Callable[[str, Any, str], bool] | None = None,
    ) -> bool:
        """Return True when tool usage is allowed by context metadata."""
        allowed_tools = (extract_allowed_tools or self.extract_allowed_tools)(context)
        if not allowed_tools:
            return True
        matches_tool_authorization_pattern_fn = (
            matches_tool_authorization_pattern or self.matches_tool_authorization_pattern
        )
        return any(
            matches_tool_authorization_pattern_fn(tool_name, tool_args, pattern)
            for pattern in allowed_tools
        )

    def hash_arguments(self, arguments: Any) -> str | None:
        return self.hash_arguments_with_exceptions(
            arguments,
            noncritical_exceptions=self._noncritical_exceptions,
        )

    @staticmethod
    def hash_arguments_with_exceptions(
        arguments: Any,
        *,
        noncritical_exceptions: tuple[type[BaseException], ...],
    ) -> str | None:
        try:
            payload = canonical_json_bytes(arguments, max_bytes=ARGUMENTS_MAX_BYTES)
            return hashlib.sha256(payload).hexdigest()
        except noncritical_exceptions:
            return None

    async def resolve_tool_definition(
        self,
        module: BaseModule,
        tool_name: str,
    ) -> dict[str, Any] | None:
        """Resolve a tool definition for a module/tool pair."""
        try:
            get_def = getattr(module, "get_tool_def", None)
            if callable(get_def):
                tool_def = await get_def(tool_name)  # type: ignore[misc]
                if isinstance(tool_def, dict):
                    return tool_def
            tool_defs = await module.get_tools()
            for candidate in tool_defs:
                if isinstance(candidate, dict) and candidate.get("name") == tool_name:
                    return candidate
        except asyncio.CancelledError:
            raise
        except self._noncritical_exceptions:
            return None
        return None

    @staticmethod
    def normalize_tool_definition(tool_def: dict[str, Any]) -> dict[str, Any]:
        """Normalize a resolved definition before preparation or live comparison."""

        return ensure_tool_definition_eval_metadata(tool_def)

    def classify_write_tool_call(
        self,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
    ) -> bool | None:
        """Best-effort write classification using per-call module hook."""
        try:
            normalized_args = tool_args if isinstance(tool_args, dict) else {}
            return module.is_write_tool_call(tool_name, normalized_args, tool_def=tool_def)
        except self._noncritical_exceptions:
            return None

    def resolve_write_classification(
        self,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
        *,
        fallback_to_name_heuristic: bool,
    ) -> bool:
        """Resolve write classification with optional legacy fallback."""
        is_write = self.classify_write_tool_call(module, tool_name, tool_args, tool_def)
        if is_write is not None:
            return bool(is_write)
        if fallback_to_name_heuristic:
            return bool(re.search(r"(ingest|update|delete|create|import)", str(tool_name).lower()))
        return False

    @staticmethod
    def strip_forbidden_tool_argument_overrides(tool_args: dict[str, Any]) -> dict[str, Any]:
        """Remove tool argument fields that could override request ownership/db scope."""
        forbidden = {"user_id", "db_path", "db_paths", "chacha_db", "media_db", "prompts_db"}
        sanitized = dict(tool_args)
        for key in forbidden:
            sanitized.pop(key, None)
        return sanitized

    def harden_and_sanitize_tool_arguments(
        self,
        module: BaseModule,
        tool_args: Any,
    ) -> Any:
        """Normalize tool arguments before policy and execution checks."""
        if not isinstance(tool_args, dict):
            return tool_args
        hardened_args = self.strip_forbidden_tool_argument_overrides(tool_args)
        try:
            return module.sanitize_input(hardened_args)
        except self._noncritical_exceptions as san_err:
            raise InvalidParamsException(f"Invalid arguments: {str(san_err)}") from san_err

    @staticmethod
    def normalized_idempotency_key_digest(normalized_key: str | None) -> str:
        """Return the full normalized-key digest, or empty for no key."""

        if normalized_key is None:
            return ""
        if type(normalized_key) is not str:
            raise TypeError("normalized idempotency key must be a string or None")
        return hashlib.sha256(normalized_key.encode("utf-8")).hexdigest()

    @staticmethod
    def build_canonical_snapshot(
        value: JsonValue,
        *,
        max_bytes: int,
    ) -> CanonicalJsonSnapshot:
        """Create immutable canonical snapshot bytes and digest."""

        encoded = canonical_json_bytes(value, max_bytes=max_bytes)
        return CanonicalJsonSnapshot(
            encoded=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    @staticmethod
    def prepared_tool_call_payload(
        *,
        tool_name: str,
        module_id: str | None,
        policy: PreparedExecutionPolicy,
        idempotency_cache_key: str | None,
        normalized_idempotency_key_digest: str,
        arguments_hash: str | None,
        context_fingerprint: str,
        idempotency_scope_fingerprint: str,
        tool_definition_sha256: str,
        scope_reporting_sha256: str,
    ) -> bytes:
        payload = {
            "version": 1,
            "tool_name": tool_name,
            "module_id": module_id or "",
            "policy": asdict(policy),
            "idempotency_cache_key": idempotency_cache_key or "",
            "normalized_idempotency_key_digest": normalized_idempotency_key_digest,
            "arguments_hash": arguments_hash or "",
            "context_fingerprint": context_fingerprint,
            "idempotency_scope_fingerprint": idempotency_scope_fingerprint,
            "tool_definition_sha256": tool_definition_sha256,
            "scope_reporting_sha256": scope_reporting_sha256,
        }
        return canonical_json_bytes(payload, max_bytes=PREPARED_HMAC_PAYLOAD_MAX_BYTES)

    def build_prepared_tool_call_integrity_tag(
        self,
        *,
        tool_name: str,
        module_id: str | None,
        policy: PreparedExecutionPolicy,
        idempotency_cache_key: str | None,
        normalized_idempotency_key_digest: str,
        arguments_hash: str | None,
        context_fingerprint: str,
        idempotency_scope_fingerprint: str,
        tool_definition_sha256: str,
        scope_reporting_sha256: str,
    ) -> str:
        payload = self.prepared_tool_call_payload(
            tool_name=tool_name,
            module_id=module_id,
            policy=policy,
            idempotency_cache_key=idempotency_cache_key,
            normalized_idempotency_key_digest=normalized_idempotency_key_digest,
            arguments_hash=arguments_hash,
            context_fingerprint=context_fingerprint,
            idempotency_scope_fingerprint=idempotency_scope_fingerprint,
            tool_definition_sha256=tool_definition_sha256,
            scope_reporting_sha256=scope_reporting_sha256,
        )
        return hmac.new(self._prepared_call_secret, payload, digestmod="sha256").hexdigest()

    @staticmethod
    def _integrity_failure(reason: str) -> InvalidParamsException:
        return InvalidParamsException(f"Prepared tool call integrity check failed: {reason}")

    @classmethod
    def _require_fixed_sha256(cls, value: Any, *, name: str) -> str:
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise cls._integrity_failure(f"invalid {name}")
        return value

    @classmethod
    def _verify_snapshot(
        cls,
        snapshot: CanonicalJsonSnapshot,
        *,
        max_bytes: int,
        name: str,
    ) -> None:
        if type(snapshot) is not CanonicalJsonSnapshot or type(snapshot.encoded) is not bytes:
            raise cls._integrity_failure(f"invalid {name} snapshot")
        stored_digest = cls._require_fixed_sha256(snapshot.sha256, name=f"{name} snapshot digest")
        actual_digest = hashlib.sha256(snapshot.encoded).hexdigest()
        if not hmac.compare_digest(stored_digest, actual_digest):
            raise cls._integrity_failure(f"{name} snapshot digest mismatch")
        try:
            decoded = decode_canonical_json_object_or_none(
                snapshot.encoded,
                max_bytes=max_bytes,
            )
            canonical = canonical_json_bytes(decoded, max_bytes=max_bytes)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise cls._integrity_failure(f"invalid {name} snapshot") from exc
        if not hmac.compare_digest(snapshot.encoded, canonical):
            raise cls._integrity_failure(f"non-canonical {name} snapshot")

    @classmethod
    def _validate_prepared_policy(cls, policy: PreparedExecutionPolicy) -> None:
        if type(policy) is not PreparedExecutionPolicy:
            raise cls._integrity_failure("invalid execution policy")
        if policy.version != 1 or policy.effect not in {"read", "write"}:
            raise cls._integrity_failure("invalid execution policy")
        if type(policy.rate_limit_category) is not str or not policy.rate_limit_category:
            raise cls._integrity_failure("invalid execution policy")
        if type(policy.rate_limit_fail_closed) is not bool:
            raise cls._integrity_failure("invalid execution policy")
        idempotency = policy.idempotency
        if type(idempotency) is not IdempotencyExecutionPolicy:
            raise cls._integrity_failure("invalid idempotency policy")
        if type(idempotency.inject_argument) is not bool:
            raise cls._integrity_failure("invalid idempotency policy")
        bounded_values = (
            (idempotency.ttl_seconds, 1, _IDEMPOTENCY_TTL_HARD_MAX_SECONDS),
            (idempotency.contention_wait_seconds, 1, 30),
            (idempotency.finalize_seconds, 1, 15),
            (idempotency.lock_ttl_seconds, 1, _IDEMPOTENCY_LOCK_HARD_MAX_SECONDS),
            (idempotency.max_entries, 1, _IDEMPOTENCY_MAX_ENTRIES_HARD_MAX),
            (idempotency.max_result_bytes, 1, 1_000_000),
        )
        if any(type(value) is not int or value < minimum or value > maximum for value, minimum, maximum in bounded_values):
            raise cls._integrity_failure("invalid idempotency policy")

    def verify_prepared_tool_call_integrity(
        self,
        prepared: PreparedToolCall,
    ) -> None:
        if not isinstance(prepared.tool_name, str) or not self._tool_name_re.match(prepared.tool_name):
            raise self._integrity_failure("invalid tool name")

        self._validate_prepared_policy(prepared.policy)
        self._verify_snapshot(
            prepared.tool_definition_snapshot,
            max_bytes=TOOL_DEFINITION_MAX_BYTES,
            name="tool definition",
        )
        self._verify_snapshot(
            prepared.scope_reporting_snapshot,
            max_bytes=SCOPE_REPORTING_MAX_BYTES,
            name="scope reporting",
        )

        expected_hash = self.hash_arguments(prepared.tool_args)
        if expected_hash is None:
            raise self._integrity_failure("invalid JSON arguments")
        prepared_hash = self._require_fixed_sha256(prepared.arguments_hash, name="argument fingerprint")
        if not hmac.compare_digest(expected_hash, prepared_hash):
            raise self._integrity_failure("argument fingerprint mismatch")

        try:
            expected_context_fingerprint = self.fingerprint_request_context(prepared.context)
            expected_scope_fingerprint = self.fingerprint_idempotency_scope(prepared.context)
            expected_key_digest = self.normalized_idempotency_key_digest(
                prepared.normalized_idempotency_key,
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise self._integrity_failure("invalid bound request state") from exc

        prepared_context_fingerprint = self._require_fixed_sha256(
            prepared.context_fingerprint,
            name="context fingerprint",
        )
        if not hmac.compare_digest(expected_context_fingerprint, prepared_context_fingerprint):
            raise self._integrity_failure("context fingerprint mismatch")

        if expected_scope_fingerprint:
            prepared_scope_fingerprint = self._require_fixed_sha256(
                prepared.idempotency_scope_fingerprint,
                name="idempotency scope fingerprint",
            )
            if not hmac.compare_digest(expected_scope_fingerprint, prepared_scope_fingerprint):
                raise self._integrity_failure("idempotency scope fingerprint mismatch")
        elif prepared.idempotency_scope_fingerprint != "":
            raise self._integrity_failure("idempotency scope fingerprint mismatch")

        if expected_key_digest:
            prepared_key_digest = self._require_fixed_sha256(
                prepared.normalized_idempotency_key_digest,
                name="normalized idempotency key digest",
            )
            if not hmac.compare_digest(expected_key_digest, prepared_key_digest):
                raise self._integrity_failure("normalized idempotency key digest mismatch")
        elif prepared.normalized_idempotency_key_digest != "":
            raise self._integrity_failure("normalized idempotency key digest mismatch")

        expected_cache_key = None
        if prepared.policy.effect == "write" and prepared.normalized_idempotency_key is not None:
            expected_cache_key = self.make_idempotency_cache_key(
                prepared.context,
                prepared.module_id or getattr(prepared.module, "name", "unknown"),
                prepared.tool_name,
                prepared.normalized_idempotency_key,
            )
        if not hmac.compare_digest(
            expected_cache_key or "",
            prepared.idempotency_cache_key or "",
        ):
            raise self._integrity_failure("idempotency cache key mismatch")

        try:
            expected_tag = self.build_prepared_tool_call_integrity_tag(
                tool_name=prepared.tool_name,
                module_id=prepared.module_id,
                policy=prepared.policy,
                idempotency_cache_key=prepared.idempotency_cache_key,
                normalized_idempotency_key_digest=prepared.normalized_idempotency_key_digest,
                arguments_hash=prepared.arguments_hash,
                context_fingerprint=prepared.context_fingerprint,
                idempotency_scope_fingerprint=prepared.idempotency_scope_fingerprint,
                tool_definition_sha256=prepared.tool_definition_snapshot.sha256,
                scope_reporting_sha256=prepared.scope_reporting_snapshot.sha256,
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise self._integrity_failure("invalid signed payload") from exc
        prepared_tag = self._require_fixed_sha256(prepared.integrity_tag, name="signature")
        if not hmac.compare_digest(prepared_tag, expected_tag):
            raise self._integrity_failure("signature mismatch")

    async def verify_prepared_tool_call(
        self,
        prepared: PreparedToolCall,
        *,
        require_live_binding: bool,
    ) -> None:
        """Verify signed state and optionally require its current registry binding."""

        self.verify_prepared_tool_call_integrity(prepared)
        if not require_live_binding:
            return

        try:
            module_registry = self._prepare_callback(
                "module_registry",
                lambda: self.module_registry,
            )()
            current_module = await module_registry.find_module_for_tool(prepared.tool_name)
            if current_module is not prepared.module:
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.STALE_PREPARED_CALL,
                )

            current_tool_def = await self.resolve_tool_definition(
                current_module,
                prepared.tool_name,
            )
            if not isinstance(current_tool_def, dict):
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.STALE_PREPARED_CALL,
                )
            get_module = getattr(module_registry, "get_module", None)
            if callable(get_module):
                operational_module = await get_module(prepared.module_id)
            else:
                # Compatibility registries must still prove operational membership.
                get_all_modules = getattr(module_registry, "get_all_modules", None)
                if not callable(get_all_modules):
                    raise ExpectedToolFailure(
                        ExpectedToolFailureReason.STALE_PREPARED_CALL,
                    )
                operational_modules = await get_all_modules()
                operational_module = (
                    operational_modules.get(prepared.module_id)
                    if isinstance(operational_modules, dict)
                    else None
                )

            try:
                current_tool_def = self.normalize_tool_definition(current_tool_def)
            except asyncio.CancelledError:
                raise
            except self._noncritical_exceptions:
                pass
            current_snapshot = self.build_canonical_snapshot(
                current_tool_def,
                max_bytes=TOOL_DEFINITION_MAX_BYTES,
            )

            if current_module is not prepared.module or operational_module is not prepared.module:
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.STALE_PREPARED_CALL,
                )

            current_module_id = module_registry.get_module_id_for_tool(prepared.tool_name)
            if current_module_id != prepared.module_id:
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.STALE_PREPARED_CALL,
                )

            current_config = getattr(current_module, "config", None)
            if getattr(current_config, "enabled", None) is False:
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.STALE_PREPARED_CALL,
                )

            if not hmac.compare_digest(
                current_snapshot.sha256,
                prepared.tool_definition_snapshot.sha256,
            ):
                raise ExpectedToolFailure(
                    ExpectedToolFailureReason.STALE_PREPARED_CALL,
                )

            self.verify_prepared_tool_call_integrity(prepared)
        except asyncio.CancelledError:
            raise
        except InvalidParamsException:
            raise
        except ExpectedToolFailure:
            raise
        except Exception:  # noqa: BLE001 - live resolution must fail closed.
            raise ExpectedToolFailure(
                ExpectedToolFailureReason.STALE_PREPARED_CALL,
            ) from None

    def fingerprint_request_context(self, context: RequestContext) -> str:
        payload = {
            "request_id": str(context.request_id or ""),
            "user_id": str(context.user_id or ""),
            "client_id": str(context.client_id or ""),
            "session_id": str(context.session_id or ""),
            "metadata": self.context_json_safe(getattr(context, "metadata", {})),
            "db_paths": self.context_json_safe(getattr(context, "db_paths", {})),
            "server_auth_scope": self.authenticated_scope_object(
                getattr(context, "server_auth_scope", None),
            ),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def authenticated_scope_object(
        scope: AuthenticatedExecutionScope | None,
    ) -> dict[str, JsonValue] | None:
        """Return only the explicit server-authenticated active scope object."""

        if scope is None:
            return None
        if type(scope) is not AuthenticatedExecutionScope:
            raise TypeError("Invalid authenticated execution scope")
        validated_scope = AuthenticatedExecutionScope(
            active_org_id=scope.active_org_id,
            active_team_id=scope.active_team_id,
        )
        return validated_scope.canonical_object()

    def fingerprint_idempotency_scope(self, context: RequestContext) -> str:
        """Return empty for personal scope or a fixed lowercase scope digest."""

        scope_object = self.authenticated_scope_object(
            getattr(context, "server_auth_scope", None),
        )
        if scope_object is None:
            return ""
        encoded = canonical_json_bytes(
            scope_object,
            max_bytes=PREPARED_HMAC_PAYLOAD_MAX_BYTES,
        )
        return hashlib.sha256(encoded).hexdigest()

    def context_json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): self.context_json_safe(item)
                for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        if isinstance(value, (list, tuple, set)):
            return [self.context_json_safe(item) for item in value]
        if is_dataclass(value):
            return self.context_json_safe(asdict(value))
        return str(value)

    @staticmethod
    def normalize_idempotency_key(
        params: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> str | None:
        """Normalize idempotency key from explicit argument or request params."""
        raw_idempotency_key = idempotency_key
        if raw_idempotency_key is None:
            raw_idempotency_key = params.get("idempotencyKey")
            if raw_idempotency_key is None:
                raw_idempotency_key = params.get("idempotency_key")
        if raw_idempotency_key is None:
            arguments = params.get("arguments")
            if isinstance(arguments, dict):
                raw_idempotency_key = arguments.get("idempotencyKey")
                if raw_idempotency_key is None:
                    raw_idempotency_key = arguments.get("idempotency_key")

        if raw_idempotency_key is None:
            return None
        if not isinstance(raw_idempotency_key, str):
            raise InvalidParamsException("idempotencyKey must be a string")

        normalized = raw_idempotency_key.strip()
        if not normalized:
            raise InvalidParamsException("idempotencyKey must not be empty")
        return normalized

    def validate_input_schema(self, schema: dict[str, Any], args: dict[str, Any]) -> None:
        """Quick JSON Schema checks: required keys, primitive types, unknown fields.

        Only applies when schema.type == object.
        """
        try:
            if not isinstance(schema, dict):
                return
            if schema.get("type") != "object":
                return
            if not isinstance(args, dict):
                raise InvalidParamsException("Arguments must be an object")
            props = schema.get("properties") or {}
            required = schema.get("required") or []
            addl = schema.get("additionalProperties", True)

            # Required
            for key in required:
                if key not in args or args.get(key) is None:
                    raise InvalidParamsException(f"Missing required parameter: {key}")

            # Unknown fields
            if addl is False:
                unknown = [k for k in args if k not in props]
                if unknown:
                    raise InvalidParamsException(f"Unknown parameters: {', '.join(unknown)}")

            # Primitive type checks
            def _type_ok(expected: str, value: Any) -> bool:
                mapping = {
                    "string": str,
                    "number": (int, float),
                    "integer": int,
                    "boolean": bool,
                    "object": dict,
                    "array": list,
                }
                py = mapping.get(expected)
                if py is None:
                    return True
                # number should not reject ints; python isinstance(True, int) caveat
                if expected in {"number", "integer"} and isinstance(value, bool):
                    return False
                return isinstance(value, py)

            for k, v in args.items():
                if k in props:
                    p = props.get(k) or {}
                    t = p.get("type")
                    if isinstance(t, str) and not _type_ok(t, v):
                        raise InvalidParamsException(f"Invalid type for '{k}': expected {t}")
        except InvalidParamsException:
            raise
        except self._noncritical_exceptions:
            # Be forgiving on schema format errors
            return

    async def _resolve_effective_tool_policy(self, context: RequestContext) -> dict[str, Any] | None:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        if not _is_truthy(metadata.get("mcp_policy_context_enabled")):
            return None
        cached = metadata.get("_mcp_effective_tool_policy")
        if isinstance(cached, dict):
            return cached
        try:
            policy = await self.dependencies.effective_policy_resolver.resolve_for_context(
                user_id=context.user_id,
                metadata=metadata,
            )
        except self._noncritical_exceptions as exc:
            logger.warning("Failed to resolve MCP Hub effective policy: {}", exc)
            policy = {
                "enabled": True,
                "allowed_tools": [],
                "denied_tools": [],
                "capabilities": [],
                "sources": [],
                "resolution_error": "policy_resolution_failed",
            }
        if policy is not None:
            metadata["_mcp_effective_tool_policy"] = policy
        return policy

    def _is_tool_allowed_by_effective_policy(
        self,
        tool_name: str,
        tool_args: Any,
        policy: dict[str, Any] | None,
        *,
        matches_tool_authorization_pattern: Callable[[str, Any, str], bool] | None = None,
    ) -> bool:
        if not isinstance(policy, dict) or not bool(policy.get("enabled", False)):
            return True
        if str(policy.get("resolution_error") or "").strip():
            return False
        denied_tools = [
            str(pattern).strip()
            for pattern in (policy.get("denied_tools") or [])
            if str(pattern).strip()
        ]
        matcher = matches_tool_authorization_pattern or self.matches_tool_authorization_pattern
        if any(matcher(tool_name, tool_args, pattern) for pattern in denied_tools):
            return False
        allowed_tools = [
            str(pattern).strip()
            for pattern in (policy.get("allowed_tools") or [])
            if str(pattern).strip()
        ]
        if not allowed_tools:
            return True
        return any(matcher(tool_name, tool_args, pattern) for pattern in allowed_tools)

    async def _evaluate_runtime_approval(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        tool_def: dict[str, Any] | None,
        is_write: bool | None,
        within_effective_policy: bool,
        force_approval: bool = False,
        approval_reason: str | None = None,
        scope_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        policy = dict(effective_policy or {})
        if not bool(policy.get("enabled", False)):
            return {"status": "allow", "reason": "policy_disabled"}
        if str(policy.get("resolution_error") or "").strip():
            return {"status": "deny", "reason": "policy_unavailable"}
        try:
            return await self.dependencies.approval_evaluator.evaluate_tool_call(
                effective_policy=policy,
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
                tool_def=tool_def,
                is_write=is_write,
                within_effective_policy=within_effective_policy,
                force_approval=force_approval,
                approval_reason=approval_reason,
                scope_payload=scope_payload,
            )
        except self._noncritical_exceptions as exc:
            logger.debug("Failed to evaluate MCP Hub runtime approval: {}", exc)
            if policy.get("approval_policy_id") is not None or policy.get("approval_mode"):
                return {"status": "deny", "reason": "approval_unavailable"}
            return {"status": "allow" if within_effective_policy else "deny", "reason": "approval_not_configured"}

    async def _evaluate_path_scope(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        tool_def: dict[str, Any] | None,
        path_scope_candidates: list[PathScopeCandidate] | None = None,
    ) -> dict[str, Any]:
        policy = dict(effective_policy or {})
        policy_document = dict(policy.get("policy_document") or {})
        path_scope_mode = str(policy_document.get("path_scope_mode") or "").strip()
        if not bool(policy.get("enabled", False)) or not path_scope_mode or path_scope_mode == "none":
            return {
                "enabled": False,
                "within_scope": True,
                "reason": None,
                "force_approval": False,
                "normalized_paths": [],
                "scope_payload": None,
            }
        if str(policy.get("resolution_error") or "").strip():
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "policy_unavailable",
                "force_approval": False,
                "normalized_paths": [],
                "scope_payload": {"path_scope_mode": path_scope_mode, "reason": "policy_unavailable"},
            }
        try:
            try:
                return await self.dependencies.path_scope_enforcer.evaluate_tool_call(
                    effective_policy=policy,
                    context=context,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_def=tool_def,
                    path_scope_candidates=path_scope_candidates,
                )
            except TypeError as exc:
                if not _is_unexpected_keyword_type_error(exc, "path_scope_candidates"):
                    raise
                if path_scope_candidates:
                    raise PermissionError("path_scope_candidates_unsupported") from exc
                return await self.dependencies.path_scope_enforcer.evaluate_tool_call(
                    effective_policy=policy,
                    context=context,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_def=tool_def,
                )
        except PermissionError:
            raise
        except TypeError:
            raise
        except self._noncritical_exceptions as exc:
            logger.debug("Failed to evaluate MCP Hub path scope: {}", exc)
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "path_scope_unavailable",
                "force_approval": True,
                "normalized_paths": [],
                "scope_payload": {"path_scope_mode": path_scope_mode, "reason": "path_scope_unavailable"},
            }

    async def _extract_path_scope_candidates(
        self,
        *,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        context: RequestContext,
        tool_def: dict[str, Any] | None,
    ) -> list[PathScopeCandidate] | None:
        metadata = tool_def.get("metadata") if isinstance(tool_def, dict) else None
        if not isinstance(metadata, dict) or metadata.get("path_scope_candidate_source") != "module":
            return None
        if not isinstance(tool_args, dict):
            raise PermissionError("path_scope_candidates_unavailable")
        extractor = getattr(module, "extract_path_scope_candidates", None)
        if not callable(extractor):
            raise PermissionError("path_scope_candidates_unavailable")
        try:
            candidates = normalize_path_scope_candidates(
                await extractor(tool_name, tool_args, context)
            )
        except NotImplementedError as exc:
            raise PermissionError("path_scope_candidates_unavailable") from exc
        except self._noncritical_exceptions as exc:
            raise InvalidParamsException(str(exc)) from exc
        if not candidates:
            raise PermissionError("path_scope_candidates_unavailable")
        return candidates

    async def _evaluate_external_access(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        tool_name: str,
        context: RequestContext,
    ) -> dict[str, Any]:
        deny_only_reasons = {
            "external_access_unavailable",
            "external_server_not_bound",
            "invalid_external_tool_name",
            "required_slot_not_granted",
            "required_slot_secret_missing",
        }
        if not str(tool_name or "").startswith("ext."):
            return {
                "enabled": False,
                "within_scope": True,
                "reason": None,
                "scope_payload": None,
                "hard_deny": False,
            }
        policy = dict(effective_policy or {})
        if not bool(policy.get("enabled", False)):
            return {
                "enabled": False,
                "within_scope": True,
                "reason": None,
                "scope_payload": None,
                "hard_deny": False,
            }
        sources = policy.get("sources")
        if not isinstance(sources, list):
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "external_access_unavailable",
                "scope_payload": {
                    "server_id": tool_name.split(".", 2)[1],
                    "reason": "external_access_unavailable",
                    "blocked_reason": "external_access_unavailable",
                    "requested_slots": [],
                    "missing_bound_slots": [],
                    "missing_secret_slots": [],
                },
                "hard_deny": True,
            }
        parts = str(tool_name or "").split(".", 2)
        if len(parts) != 3 or not parts[1]:
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "invalid_external_tool_name",
                "scope_payload": {
                    "reason": "invalid_external_tool_name",
                    "blocked_reason": "invalid_external_tool_name",
                    "requested_slots": [],
                    "missing_bound_slots": [],
                    "missing_secret_slots": [],
                },
                "hard_deny": True,
            }
        server_id = parts[1]
        metadata = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
        cached = metadata.get("_mcp_effective_external_access")
        if not isinstance(cached, dict):
            try:
                cached = await self.dependencies.external_access_evaluator.resolve_for_sources(
                    sources=[dict(item) for item in sources if isinstance(item, dict)],
                    effective_policy=policy,
                )
                metadata["_mcp_effective_external_access"] = cached
            except self._noncritical_exceptions as exc:
                logger.debug("Failed to evaluate MCP Hub external access: {}", exc)
                return {
                    "enabled": True,
                    "within_scope": False,
                    "reason": "external_access_unavailable",
                    "scope_payload": {
                        "server_id": server_id,
                        "reason": "external_access_unavailable",
                        "blocked_reason": "external_access_unavailable",
                        "requested_slots": [],
                        "missing_bound_slots": [],
                        "missing_secret_slots": [],
                    },
                    "hard_deny": True,
                }

        rows = cached.get("servers") if isinstance(cached, dict) else None
        server_row = next(
            (
                row for row in rows
                if isinstance(row, dict) and str(row.get("server_id") or "") == server_id
            ),
            None,
        ) if isinstance(rows, list) else None
        if not isinstance(server_row, dict):
            return {
                "enabled": True,
                "within_scope": False,
                "reason": "external_server_not_bound",
                "scope_payload": {
                    "server_id": server_id,
                    "reason": "external_server_not_bound",
                    "blocked_reason": "external_server_not_bound",
                    "requested_slots": [],
                    "missing_bound_slots": [],
                    "missing_secret_slots": [],
                },
                "hard_deny": True,
            }
        runtime_executable = bool(server_row.get("runtime_executable"))
        reason = str(server_row.get("blocked_reason") or "").strip() or None
        requested_slots = [
            str(slot).strip()
            for slot in (server_row.get("requested_slots") or [])
            if str(slot).strip()
        ]
        bound_slots = [
            str(slot).strip()
            for slot in (server_row.get("bound_slots") or [])
            if str(slot).strip()
        ]
        missing_bound_slots = [
            str(slot).strip()
            for slot in (server_row.get("missing_bound_slots") or [])
            if str(slot).strip()
        ]
        missing_secret_slots = [
            str(slot).strip()
            for slot in (server_row.get("missing_secret_slots") or [])
            if str(slot).strip()
        ]
        scope_payload = {
            "server_id": server_id,
            "server_name": str(server_row.get("server_name") or "").strip() or None,
            "reason": reason or ("external_server_allowed" if runtime_executable else "external_server_not_bound"),
            "blocked_reason": reason or ("external_server_allowed" if runtime_executable else "external_server_not_bound"),
            "requested_slots": requested_slots,
            "bound_slots": bound_slots,
            "missing_bound_slots": missing_bound_slots,
            "missing_secret_slots": missing_secret_slots,
        }
        if not runtime_executable:
            return {
                "enabled": True,
                "within_scope": False,
                "reason": reason or "external_server_not_bound",
                "scope_payload": scope_payload,
                "hard_deny": (reason or "external_server_not_bound") in deny_only_reasons,
            }
        return {
            "enabled": True,
            "within_scope": True,
            "reason": None,
            "scope_payload": scope_payload,
            "hard_deny": False,
        }

    @staticmethod
    def _governance_preflight_bypassed(tool_name: str, context: RequestContext) -> bool:
        if str(tool_name or "").startswith("governance."):
            return True

        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return False

        raw = metadata.get("governance_bypass")
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            return _is_truthy(raw)
        return False

    def _governance_summary(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        rendered_args = ""
        try:
            rendered_args = json.dumps(tool_args or {}, sort_keys=True, default=str)
        except self._noncritical_exceptions:
            rendered_args = str(tool_args)
        if len(rendered_args) > 1200:
            rendered_args = rendered_args[:1200]
        return f"tool={tool_name}; args={rendered_args}"

    @staticmethod
    def _resolve_governance_category(tool_name: str, tool_def: dict[str, Any] | None) -> str:
        try:
            if isinstance(tool_def, dict):
                meta = tool_def.get("metadata")
                if isinstance(meta, dict):
                    category = str(meta.get("category") or "").strip().lower()
                    if category:
                        return category
        except Exception as exc:  # noqa: BLE001 - category fallback should not fail preflight.
            logger.debug("Falling back to tool-name governance category: {error_type}", error_type=exc.__class__.__name__)

        if isinstance(tool_name, str) and "." in tool_name:
            prefix = tool_name.split(".", 1)[0].strip().lower()
            if prefix:
                return prefix
        return "general"

    def _resolve_governance_rollout_mode(self, metadata: dict[str, Any] | None = None) -> str:
        """Resolve governance rollout mode from metadata override and server config."""

        raw_mode = None
        if isinstance(metadata, dict):
            raw_mode = metadata.get("governance_rollout_mode")

        try:
            from tldw_Server_API.app.core import config as app_config

            return app_config.resolve_governance_rollout_mode(
                str(raw_mode) if raw_mode is not None else None
            )
        except self._noncritical_exceptions as exc:
            logger.debug("Unable to resolve governance rollout mode from config: {}", exc)
            candidate = str(raw_mode or "").strip().lower()
            return candidate if candidate in {"off", "shadow", "enforce"} else "off"
        except Exception as exc:  # noqa: BLE001 - config resolver exceptions should not block tools.
            logger.debug(
                "Unable to resolve governance rollout mode from app config: {error_type}",
                error_type=exc.__class__.__name__,
            )
            candidate = str(raw_mode or "").strip().lower()
            return candidate if candidate in {"off", "shadow", "enforce"} else "off"

    def _record_governance_check(
        self,
        *,
        surface: str,
        category: str,
        status: str,
        rollout_mode: str,
    ) -> None:
        """Emit one governance check metric entry, failing open on metric errors."""
        with contextlib.suppress(self._noncritical_exceptions):
            self.metrics.record_governance_check(
                surface=surface,
                category=category,
                status=status,
                rollout_mode=rollout_mode,
            )

    @classmethod
    def _serialize_governance_decision(cls, decision: Any) -> dict[str, Any]:
        if decision is None:
            return {}
        if isinstance(decision, dict):
            return {str(k): v for k, v in decision.items()}
        if is_dataclass(decision):
            return cls._serialize_governance_decision(asdict(decision))
        dump = getattr(decision, "model_dump", None)
        if callable(dump):
            try:
                dumped = dump()
                if isinstance(dumped, dict):
                    return {str(k): v for k, v in dumped.items()}
            except Exception as exc:  # noqa: BLE001 - decision fallback handles noncritical model errors.
                logger.debug(
                    "Falling back to attribute governance decision serialization: {error_type}",
                    error_type=exc.__class__.__name__,
                )
        payload: dict[str, Any] = {}
        for key in ("action", "status", "category", "category_source", "fallback_reason", "matched_rules"):
            value = getattr(decision, key, None)
            if value is not None:
                payload[key] = value
        return payload

    async def _ensure_governance_service(self) -> Any | None:
        if self._governance_service is not None:
            return self._governance_service

        async with self._governance_lock:
            if self._governance_service is not None:
                return self._governance_service
            try:
                from tldw_Server_API.app.core.Governance.service import GovernanceService
                from tldw_Server_API.app.core.Governance.store import GovernanceStore
            except self._noncritical_exceptions as exc:
                logger.debug("MCP governance preflight unavailable (import failure): {}", exc)
                return None

            try:
                cfg = self.dependencies.config_provider()
                configured_path = getattr(cfg, "governance_db_path", None)
                sqlite_path = str(configured_path or "Databases/governance.db")
                db_path = Path(sqlite_path).expanduser()
                db_path.parent.mkdir(parents=True, exist_ok=True)

                self._governance_store = GovernanceStore(sqlite_path=str(db_path))
                await self._governance_store.ensure_schema()
                self._governance_service = GovernanceService(store=self._governance_store)
                return self._governance_service
            except self._noncritical_exceptions as exc:
                logger.debug("MCP governance preflight disabled (service init failure): {}", exc)
                self._governance_service = None
                self._governance_store = None
                return None

    async def _run_governance_preflight(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_def: dict[str, Any] | None,
        context: RequestContext,
        governance_preflight_bypassed: Callable[[str, RequestContext], bool] | None = None,
        resolve_governance_rollout_mode: Callable[[dict[str, Any] | None], str] | None = None,
        resolve_governance_category: Callable[[str, dict[str, Any] | None], str] | None = None,
        record_governance_check: Callable[..., None] | None = None,
        governance_summary: Callable[[str, dict[str, Any]], str] | None = None,
        serialize_governance_decision: Callable[[Any], dict[str, Any]] | None = None,
        ensure_governance_service: Callable[[], Awaitable[Any | None]] | None = None,
    ) -> dict[str, Any] | None:
        bypassed = governance_preflight_bypassed or self._governance_preflight_bypassed
        if bypassed(tool_name, context):
            return None

        metadata = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
        rollout_mode = (resolve_governance_rollout_mode or self._resolve_governance_rollout_mode)(metadata)
        category = (resolve_governance_category or self._resolve_governance_category)(tool_name, tool_def)
        record_check = record_governance_check or self._record_governance_check

        if rollout_mode == "off":
            record_check(
                surface="mcp_tool",
                category=category,
                status="unknown",
                rollout_mode=rollout_mode,
            )
            return {"status": "unknown", "rollout_mode": rollout_mode}

        service = await (ensure_governance_service or self._ensure_governance_service)()
        if service is None:
            record_check(
                surface="mcp_tool",
                category=category,
                status="error",
                rollout_mode=rollout_mode,
            )
            return None

        try:
            decision = await service.validate_change(
                surface="mcp_tool",
                summary=(governance_summary or self._governance_summary)(tool_name, tool_args),
                category=category,
                metadata=metadata,
            )
            payload = (serialize_governance_decision or self._serialize_governance_decision)(decision)
            payload.setdefault("rollout_mode", rollout_mode)
            if isinstance(context.metadata, dict):
                context.metadata["governance_preflight"] = payload
            action = str(payload.get("action") or payload.get("status") or "").strip().lower() or "unknown"
            record_check(
                surface="mcp_tool",
                category=category,
                status=action,
                rollout_mode=rollout_mode,
            )
            if action == "deny" and rollout_mode == "enforce":
                raise GovernanceDeniedError(
                    "Permission denied by governance policy",
                    governance=payload,
                )
            return payload
        except GovernanceDeniedError:
            raise
        except self._noncritical_exceptions as exc:
            record_check(
                surface="mcp_tool",
                category=category,
                status="error",
                rollout_mode=rollout_mode,
            )
            try:
                context.logger.debug(f"Governance preflight failed open: {exc}")
            except self._noncritical_exceptions:
                pass
            return None

    @staticmethod
    def _bounded_positive_int(value: Any, *, name: str, maximum: int) -> int:
        if type(value) is not int or value < 1 or value > maximum:
            raise ValueError(f"{name} must be a positive non-boolean integer no greater than {maximum}")
        return value

    @staticmethod
    def _resolve_rate_limit_category(
        tool_name: str,
        tool_def: dict[str, Any] | None,
        config: Any,
    ) -> str:
        metadata = tool_def.get("metadata") if isinstance(tool_def, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        category = str(metadata.get("category") or "").strip().lower().replace("-", "_")
        if bool(metadata.get("uses_network")) or category in {
            "web",
            "network",
            "external",
            "external_network",
        }:
            return "network"
        if category in _RATE_LIMIT_CATEGORIES:
            return category

        category_map = getattr(config, "tool_category_map", {})
        if isinstance(category_map, dict) and tool_name in category_map:
            mapped = str(category_map.get(tool_name) or "").strip()
            if mapped:
                return mapped
        if tool_name in {"ingest_media", "update_media", "delete_media"}:
            return "ingestion"
        return "read"

    def build_prepared_execution_policy(
        self,
        *,
        module: BaseModule,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
        is_write: bool,
        normalized_idempotency_key: str | None,
        config: Any,
    ) -> PreparedExecutionPolicy:
        """Freeze all execution decisions and bounded runtime settings."""

        ttl_seconds = self._bounded_positive_int(
            getattr(config, "idempotency_ttl_seconds", 300),
            name="idempotency TTL",
            maximum=_IDEMPOTENCY_TTL_HARD_MAX_SECONDS,
        )
        contention_wait_seconds = self._bounded_positive_int(
            getattr(config, "idempotency_wait_seconds", 5),
            name="idempotency contention wait",
            maximum=30,
        )
        finalize_seconds = self._bounded_positive_int(
            getattr(config, "idempotency_finalize_seconds", 5),
            name="idempotency finalize timeout",
            maximum=15,
        )
        max_entries = self._bounded_positive_int(
            getattr(config, "idempotency_cache_size", 512),
            name="idempotency max entries",
            maximum=_IDEMPOTENCY_MAX_ENTRIES_HARD_MAX,
        )
        max_result_bytes = self._bounded_positive_int(
            getattr(config, "idempotency_result_max_bytes", 256_000),
            name="idempotency result byte limit",
            maximum=1_000_000,
        )
        module_config = getattr(module, "config", None)
        module_timeout_value = getattr(
            module_config,
            "timeout_seconds",
            getattr(config, "module_timeout", 30),
        )
        module_timeout_seconds = self._bounded_positive_int(
            module_timeout_value,
            name="module timeout",
            maximum=_IDEMPOTENCY_LOCK_HARD_MAX_SECONDS,
        )
        lock_ttl_seconds = max(
            ttl_seconds,
            module_timeout_seconds * 2 + finalize_seconds,
        )
        lock_ttl_seconds = self._bounded_positive_int(
            lock_ttl_seconds,
            name="idempotency lock TTL",
            maximum=_IDEMPOTENCY_LOCK_HARD_MAX_SECONDS,
        )

        input_schema = tool_def.get("inputSchema") if isinstance(tool_def, dict) else None
        properties = input_schema.get("properties") if isinstance(input_schema, dict) else None
        inject_argument = bool(
            is_write
            and normalized_idempotency_key is not None
            and isinstance(tool_args, dict)
            and isinstance(properties, dict)
            and "idempotencyKey" in properties
            and "idempotencyKey" not in tool_args
        )
        metadata = tool_def.get("metadata") if isinstance(tool_def, dict) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        return PreparedExecutionPolicy(
            version=1,
            effect="write" if is_write else "read",
            rate_limit_category=self._resolve_rate_limit_category(tool_name, tool_def, config),
            rate_limit_fail_closed=metadata.get("rate_limit_fail_closed") is True,
            idempotency=IdempotencyExecutionPolicy(
                inject_argument=inject_argument,
                ttl_seconds=ttl_seconds,
                contention_wait_seconds=contention_wait_seconds,
                finalize_seconds=finalize_seconds,
                lock_ttl_seconds=lock_ttl_seconds,
                max_entries=max_entries,
                max_result_bytes=max_result_bytes,
            ),
        )

    def make_idempotency_cache_key(
        self,
        context: RequestContext,
        module_name: str,
        tool_name: str,
        idempotency_key: str,
    ) -> str:
        """Build the sole authoritative owner and active-scope replay key."""

        owner = (
            f"user:{context.user_id}"
            if context.user_id
            else (f"client:{context.client_id}" if context.client_id else "anon")
        )
        cache_key_prefix = f"{owner}|module:{module_name}|tool:{tool_name}"
        scope_fingerprint = self.fingerprint_idempotency_scope(context)
        if scope_fingerprint:
            return f"{cache_key_prefix}|scope:sha256:{scope_fingerprint}|key:{idempotency_key}"
        return f"{cache_key_prefix}|key:{idempotency_key}"

    def _make_idempotency_cache_key(
        self,
        context: RequestContext,
        module_name: str,
        tool_name: str,
        idempotency_key: str,
    ) -> str:
        """Compatibility delegate to the authoritative public key builder."""

        return self.make_idempotency_cache_key(
            context,
            module_name,
            tool_name,
            idempotency_key,
        )

    @staticmethod
    def _policy_document_path_scope_mode(policy_document: Any) -> str:
        if isinstance(policy_document, dict):
            return str(policy_document.get("path_scope_mode") or "").strip()
        if isinstance(policy_document, BaseModel):
            if hasattr(policy_document, "model_dump"):
                policy_document_payload = policy_document.model_dump()
            else:  # pragma: no cover - pydantic v1 compatibility
                policy_document_payload = policy_document.dict()
            return str(policy_document_payload.get("path_scope_mode") or "").strip()
        return ""

    async def prepare_tool_call(
        self,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
        hooks: Any | None = None,
    ) -> PreparedToolCall:
        """Prepare a tool invocation through protocol policy, validation, and governance checks."""

        tool_name = params.get("name")
        tool_args = params.get("arguments", {})
        normalized_idempotency_key = self.normalize_idempotency_key(
            params,
            idempotency_key=idempotency_key,
        )

        if not tool_name:
            raise InvalidParamsException("Tool name is required")

        # Strictly validate tool name
        if not self._tool_name_re.match(tool_name):
            raise InvalidParamsException("Invalid tool name")

        is_tool_allowed_by_context = self._prepare_callback(
            "is_tool_allowed_by_context",
            self.is_tool_allowed_by_context,
        )
        if not is_tool_allowed_by_context(tool_name, tool_args, context):
            raise PermissionError(f"Tool '{tool_name}' not allowed by execution context")

        resolve_effective_tool_policy = self._prepare_callback(
            "resolve_effective_tool_policy",
            self._resolve_effective_tool_policy,
        )
        effective_policy = await resolve_effective_tool_policy(context)

        is_tool_allowed_by_effective_policy = self._prepare_callback(
            "is_tool_allowed_by_effective_policy",
            self._is_tool_allowed_by_effective_policy,
        )
        within_effective_policy = is_tool_allowed_by_effective_policy(tool_name, tool_args, effective_policy)

        evaluate_external_access = self._prepare_callback(
            "evaluate_external_access",
            self._evaluate_external_access,
        )
        external_access_result = await evaluate_external_access(
            effective_policy=effective_policy,
            tool_name=tool_name,
            context=context,
        )
        external_block_reason = str(external_access_result.get("reason") or "").strip()
        if bool(external_access_result.get("hard_deny")) or external_block_reason in {
            "required_slot_not_granted",
            "required_slot_secret_missing",
        }:
            external_scope = (
                dict(external_access_result.get("scope_payload") or {})
                if isinstance(external_access_result.get("scope_payload"), dict)
                else {}
            )
            blocked_reason = external_block_reason or "external_access_denied"
            raise GovernanceDeniedError(
                "Blocked external credential use",
                governance={
                    "action": "deny",
                    "status": "deny",
                    "reason_code": blocked_reason,
                    "external_access": external_scope,
                },
            )

        module_registry = self._prepare_callback("module_registry", lambda: self.module_registry)()
        module = await module_registry.find_module_for_tool(tool_name)
        if not module:
            raise InvalidParamsException(f"Tool not found: {tool_name}")

        module_id = module_registry.get_module_id_for_tool(tool_name) or getattr(module, "name", None)

        # Look up tool definition early for scope gating and validation
        tool_def = await self.resolve_tool_definition(module, tool_name)
        if isinstance(tool_def, dict):
            try:
                tool_def = self.normalize_tool_definition(tool_def)
            except asyncio.CancelledError:
                raise
            except self._noncritical_exceptions as exc:
                context.logger.opt(exception=exc).debug(
                    "Failed to attach eval metadata to resolved tool definition: module_id={module_id} "
                    "tool_name={tool_name} error_type={error_type}",
                    module_id=module_id,
                    tool_name=tool_name,
                    error_type=exc.__class__.__name__,
                )
        tool_args = self.harden_and_sanitize_tool_arguments(module, tool_args)
        args_hash = self.hash_arguments(tool_args)
        if args_hash is None:
            raise InvalidParamsException("Tool arguments must contain only valid JSON values")

        # Determine write-capable status from sanitized arguments.
        is_write = self.resolve_write_classification(
            module,
            tool_name,
            tool_args,
            tool_def,
            fallback_to_name_heuristic=True,
        )

        has_module_permission = self._prepare_callback(
            "has_module_permission",
            self.has_module_permission,
        )
        has_tool_permission = self._prepare_callback(
            "has_tool_permission",
            self.has_tool_permission,
        )
        module_allowed = await has_module_permission(context, module_id)
        tool_allowed = await has_tool_permission(context, tool_name, is_write=is_write)

        if not module_allowed and not tool_allowed:
            raise PermissionError(f"Permission denied for module: {module_id}")

        if not tool_allowed:
            raise PermissionError(f"Permission denied for tool: {tool_name}")

        # Protocol-level pre-execution validation for write-capable tools
        # Ensures that modules validate arguments even if they forgot to call
        # validate_tool_arguments inside execute_tool.
        # Look up tool definition from module cache where possible
        if tool_def is None:
            tool_def = await self.resolve_tool_definition(module, tool_name)
            if isinstance(tool_def, dict):
                try:
                    tool_def = self.normalize_tool_definition(tool_def)
                except asyncio.CancelledError:
                    raise
                except self._noncritical_exceptions as exc:
                    context.logger.opt(exception=exc).debug(
                        "Failed to attach eval metadata to resolved tool definition: "
                        "module_id={module_id} tool_name={tool_name} error_type={error_type}",
                        module_id=module_id,
                        tool_name=tool_name,
                        error_type=exc.__class__.__name__,
                    )

        idempotency_cache_key = None
        try:
            cfg = self.dependencies.config_provider()
            if getattr(cfg, "validate_input_schema", False) and isinstance(tool_def, dict):
                schema = tool_def.get("inputSchema") or {}
                try:
                    self.validate_input_schema(schema, tool_args)
                except InvalidParamsException:
                    with contextlib.suppress(self._noncritical_exceptions):
                        self.metrics.record_tool_invalid_params(getattr(module, "name", "unknown"), str(tool_name))
                    raise

            if is_write:
                if getattr(cfg, "disable_write_tools", False):
                    raise PermissionError("Write tools are disabled by server policy")
                # Check module overrides validator
                if module.__class__.validate_tool_arguments is BaseModule.validate_tool_arguments:
                    with contextlib.suppress(self._noncritical_exceptions):
                        self.metrics.record_tool_validator_missing(getattr(module, "name", "unknown"), str(tool_name))
                    raise ValueError(
                        "Write-capable tool requires module.validate_tool_arguments override"
                    )
                # Run validator
                try:
                    module.validate_tool_arguments(tool_name, tool_args)
                except self._noncritical_exceptions as ve:
                    with contextlib.suppress(self._noncritical_exceptions):
                        self.metrics.record_tool_invalid_params(getattr(module, "name", "unknown"), str(tool_name))
                    raise ValueError(f"Invalid parameters for tool {tool_name}: {ve}") from ve

                if normalized_idempotency_key:
                    make_idempotency_cache_key = self._prepare_callback(
                        "make_idempotency_cache_key",
                        self.make_idempotency_cache_key,
                    )
                    idempotency_cache_key = make_idempotency_cache_key(
                        context,
                        module_id or getattr(module, "name", "unknown"),
                        tool_name,
                        normalized_idempotency_key,
                    )
        except ValueError as ve:
            # Surface as JSON-RPC INVALID_PARAMS at the protocol layer
            # by raising a sentinel exception handled by process_request
            raise InvalidParamsException(str(ve)) from ve

        policy_document = (effective_policy or {}).get("policy_document")
        path_scope_mode = self._policy_document_path_scope_mode(policy_document)
        path_scope_candidates = None
        if bool((effective_policy or {}).get("enabled")) and path_scope_mode not in {"", "none"}:
            extract_path_scope_candidates = self._prepare_callback(
                "extract_path_scope_candidates",
                self._extract_path_scope_candidates,
            )
            path_scope_candidates = await extract_path_scope_candidates(
                module=module,
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
                tool_def=tool_def if isinstance(tool_def, dict) else None,
            )
        evaluate_path_scope = self._prepare_callback(
            "evaluate_path_scope",
            self._evaluate_path_scope,
        )
        path_scope_result = await evaluate_path_scope(
            effective_policy=effective_policy,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            tool_def=tool_def if isinstance(tool_def, dict) else None,
            path_scope_candidates=path_scope_candidates,
        )
        within_resolved_scope = bool(path_scope_result.get("within_scope", True)) and bool(
            external_access_result.get("within_scope", True)
        )
        approval_reason = str(path_scope_result.get("reason") or "").strip() or None
        if approval_reason is None:
            approval_reason = str(external_access_result.get("reason") or "").strip() or None
        scope_payload: dict[str, Any] | None = None
        for payload in (
            path_scope_result.get("scope_payload"),
            external_access_result.get("scope_payload"),
        ):
            if isinstance(payload, dict):
                scope_payload = dict(scope_payload or {})
                scope_payload.update(payload)

        path_scope_block_reason = str(path_scope_result.get("reason") or "").strip()
        if path_scope_block_reason and not bool(path_scope_result.get("within_scope", True)):
            requires_approval = bool(path_scope_result.get("force_approval", False))
            if not requires_approval or path_scope_block_reason == "workspace_unresolvable_for_trust_source":
                raise GovernanceDeniedError(
                    "Blocked path-scoped tool use",
                    governance={
                        "action": "deny",
                        "status": "deny",
                        "reason_code": path_scope_block_reason,
                        "path_scope": dict(scope_payload or {}),
                    },
                )

        evaluate_runtime_approval = self._prepare_callback(
            "evaluate_runtime_approval",
            self._evaluate_runtime_approval,
        )
        approval_result = await evaluate_runtime_approval(
            effective_policy=effective_policy,
            tool_name=tool_name,
            tool_args=tool_args,
            context=context,
            tool_def=tool_def if isinstance(tool_def, dict) else None,
            is_write=is_write,
            within_effective_policy=within_effective_policy and within_resolved_scope,
            force_approval=bool(path_scope_result.get("force_approval", False)),
            approval_reason=approval_reason,
            scope_payload=scope_payload,
        )
        approval_status = str(approval_result.get("status") or "allow").strip().lower()
        if approval_status == "approval_required":
            raise ApprovalRequiredError(
                "Approval required by MCP Hub policy",
                approval=approval_result.get("approval") if isinstance(approval_result.get("approval"), dict) else None,
            )
        if approval_status != "allow":
            raise PermissionError(f"Tool '{tool_name}' not allowed by MCP Hub policy")

        run_governance_preflight = self._prepare_callback(
            "run_governance_preflight",
            self._run_governance_preflight,
        )
        await run_governance_preflight(
            tool_name=tool_name,
            tool_args=tool_args if isinstance(tool_args, dict) else {},
            tool_def=tool_def if isinstance(tool_def, dict) else None,
            context=context,
        )
        try:
            prepared_policy = self.build_prepared_execution_policy(
                module=module,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_def=tool_def if isinstance(tool_def, dict) else None,
                is_write=is_write,
                normalized_idempotency_key=normalized_idempotency_key,
                config=cfg,
            )
        except (TypeError, ValueError) as exc:
            raise InvalidParamsException("Invalid idempotency execution policy") from exc
        try:
            tool_definition_snapshot = self.build_canonical_snapshot(
                tool_def if isinstance(tool_def, dict) else None,
                max_bytes=TOOL_DEFINITION_MAX_BYTES,
            )
            scope_reporting_snapshot = self.build_canonical_snapshot(
                scope_payload if isinstance(scope_payload, dict) else None,
                max_bytes=SCOPE_REPORTING_MAX_BYTES,
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise InvalidParamsException(
                "Tool definition and scope reporting payloads must contain only valid bounded JSON values"
            ) from exc
        if hooks is None:
            raise RuntimeError("ToolExecutionHooks dependency is required")
        await hooks.run_pre_tool_hooks(
            tool_name=tool_name,
            tool_args=tool_args,
            module_id=module_id,
            tool_def=decode_canonical_json_object_or_none(
                tool_definition_snapshot.encoded,
                max_bytes=TOOL_DEFINITION_MAX_BYTES,
            ),
            is_write=is_write,
            arguments_hash=args_hash,
            context=context,
            scope_payload=decode_canonical_json_object_or_none(
                scope_reporting_snapshot.encoded,
                max_bytes=SCOPE_REPORTING_MAX_BYTES,
            ),
        )
        context_fingerprint = self.fingerprint_request_context(context)
        normalized_idempotency_key_digest = self.normalized_idempotency_key_digest(
            normalized_idempotency_key,
        )
        idempotency_scope_fingerprint = self.fingerprint_idempotency_scope(context)
        try:
            integrity_tag = self.build_prepared_tool_call_integrity_tag(
                tool_name=tool_name,
                module_id=module_id,
                policy=prepared_policy,
                idempotency_cache_key=idempotency_cache_key,
                normalized_idempotency_key_digest=normalized_idempotency_key_digest,
                arguments_hash=args_hash,
                context_fingerprint=context_fingerprint,
                idempotency_scope_fingerprint=idempotency_scope_fingerprint,
                tool_definition_sha256=tool_definition_snapshot.sha256,
                scope_reporting_sha256=scope_reporting_snapshot.sha256,
            )
        except (TypeError, ValueError, UnicodeError) as exc:
            raise InvalidParamsException("Unable to bind prepared execution policy") from exc

        return PreparedToolCall(
            tool_name=tool_name,
            tool_args=tool_args,
            module=module,
            module_id=module_id,
            policy=prepared_policy,
            tool_definition_snapshot=tool_definition_snapshot,
            scope_reporting_snapshot=scope_reporting_snapshot,
            normalized_idempotency_key=normalized_idempotency_key,
            normalized_idempotency_key_digest=normalized_idempotency_key_digest,
            idempotency_cache_key=idempotency_cache_key,
            arguments_hash=args_hash,
            context_fingerprint=context_fingerprint,
            idempotency_scope_fingerprint=idempotency_scope_fingerprint,
            integrity_tag=integrity_tag,
            context=context,
        )
