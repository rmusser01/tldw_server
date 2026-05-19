"""
supervised_policy.py

Layers guardian-configured supervised policies on top of the existing
ModerationService. Evaluates supervised block/notify rules for dependent
users and routes alerts to guardians.

Key capabilities:
- Governance policy filtering: schedule, chat-type scope, and transparent mode
  via linked GovernancePolicy objects (uses governance_utils.is_schedule_active
  and chat_type_matches)
- dispatch_guardian_notification(): routes guardian alerts through
  NotificationService.notify_or_batch() (JSONL sink, optional webhook/email)

Integration point: called by the chat pipeline after standard moderation
to apply guardian-imposed rules on supervised accounts.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Character_Chat.regex_safety import validate_regex_safety
from tldw_Server_API.app.core.DB_Management.Guardian_DB import (
    GovernancePolicy,
    GuardianDB,
    SupervisedPolicy,
)
from tldw_Server_API.app.core.DB_Management.guardian_db_resolver import (
    resolve_guardian_db_for_user_id,
)
from tldw_Server_API.app.core.Moderation.governance_utils import (
    chat_type_matches,
    is_schedule_active,
)
from tldw_Server_API.app.core.Moderation.conflict_resolution import resolve_conflicts
from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationPolicy,
    PatternRule,
)

_SUPERVISED_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    re.error,
)


@dataclass
class SupervisedCheckResult:
    """Result from evaluating supervised policies for a dependent user."""
    action: str = "pass"  # pass | block | redact | warn | notify
    matched_policy_id: str | None = None
    matched_category: str = ""
    matched_pattern: str = ""
    severity: str = "info"
    message_to_dependent: str | None = None
    notify_guardian: bool = False
    notify_context: str = "topic_only"
    context_snippet: str | None = None
    redacted_text: str | None = None
    rule_name_visible: str | None = None  # set when governance policy is transparent


@dataclass(frozen=True)
class GuardianModerationRuntime:
    """Resolved guardian moderation services for a dependent-user chat context."""

    dependent_user_id: str
    chat_type: str = "regular"
    guardian_db: GuardianDB | None = None
    supervised_engine: "SupervisedPolicyEngine | None" = None


class SupervisedPolicyEngine:
    """Evaluates supervised policies for a dependent user.

    Compiles patterns from GuardianDB policies and checks text
    in both input and output phases. Caches compiled patterns
    per-dependent for efficiency.
    """

    _DEFAULT_BLOCK_MESSAGE = (
        "This content has been restricted by your account administrator. "
        "If you have questions, please speak with your parent or guardian."
    )

    def __init__(self, guardian_db: GuardianDB) -> None:
        self._db = guardian_db
        self._lock = threading.RLock()
        # Cache: "<dependent_user_id>:<chat_type>" -> list of (policy, compiled_regex, governance_policy)
        self._compiled_cache: dict[str, list[tuple[SupervisedPolicy, re.Pattern, GovernancePolicy | None]]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl_seconds = 60.0  # refresh every minute

    def invalidate_cache(self, dependent_user_id: str | None = None) -> None:
        with self._lock:
            if dependent_user_id:
                prefix = f"{dependent_user_id}:"
                keys_to_remove = [k for k in self._compiled_cache if k.startswith(prefix)]
                for k in keys_to_remove:
                    self._compiled_cache.pop(k, None)
                    self._cache_timestamps.pop(k, None)
            else:
                self._compiled_cache.clear()
                self._cache_timestamps.clear()

    def _get_compiled_policies(
        self,
        dependent_user_id: str,
        chat_type: str | None = None,
    ) -> list[tuple[SupervisedPolicy, re.Pattern, GovernancePolicy | None]]:
        uid = str(dependent_user_id)
        cache_key = f"{uid}:{chat_type or ''}"
        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            cached_at = self._cache_timestamps.get(cache_key, 0.0)
            if cache_key in self._compiled_cache and (now - cached_at) < self._cache_ttl_seconds:
                return self._compiled_cache[cache_key]

        # Fetch and compile outside the lock
        policies = self._db.list_active_policies_for_dependent(uid)
        compiled: list[tuple[SupervisedPolicy, re.Pattern, GovernancePolicy | None]] = []
        for pol in policies:
            if not pol.pattern:
                continue

            # Governance policy filtering
            gp: GovernancePolicy | None = None
            if pol.governance_policy_id:
                try:
                    gp = self._db.get_governance_policy(pol.governance_policy_id)
                except _SUPERVISED_NONCRITICAL_EXCEPTIONS:
                    gp = None
                if gp:
                    if not gp.enabled:
                        continue
                    if not is_schedule_active(
                        gp.schedule_start, gp.schedule_end,
                        gp.schedule_days, gp.schedule_timezone,
                    ):
                        continue
                    if not chat_type_matches(gp.scope_chat_types, chat_type):
                        continue

            try:
                if pol.pattern_type == "regex":
                    is_safe, reason = validate_regex_safety(pol.pattern)
                    if not is_safe:
                        logger.warning("Unsafe supervised policy regex skipped")
                        continue
                    regex = re.compile(pol.pattern, re.IGNORECASE)
                else:
                    regex = re.compile(re.escape(pol.pattern), re.IGNORECASE)
                compiled.append((pol, regex, gp))
            except re.error:
                logger.warning("Invalid supervised policy pattern skipped")
                continue

        with self._lock:
            self._compiled_cache[cache_key] = compiled
            self._cache_timestamps[cache_key] = now
        return compiled

    def check_text(
        self,
        text: str,
        dependent_user_id: str,
        phase: str = "input",
        chat_type: str | None = None,
    ) -> SupervisedCheckResult:
        """Check text against supervised policies for a dependent user.

        Returns the highest-priority result (block > redact > warn > notify > pass).
        """
        if not text or not text.strip():
            return SupervisedCheckResult()

        compiled_policies = self._get_compiled_policies(dependent_user_id, chat_type=chat_type)
        if not compiled_policies:
            return SupervisedCheckResult()

        # Priority: block=4, redact=3, warn=2, notify=1
        action_priority = {"block": 4, "redact": 3, "warn": 2, "notify": 1}
        best_result = SupervisedCheckResult()
        best_priority = 0

        for pol, regex, gp in compiled_policies:
            # Phase filtering
            if pol.phase != "both" and pol.phase != phase:
                continue
            # Check for match
            match = regex.search(text)
            if not match:
                continue

            priority = action_priority.get(pol.action, 0)
            if priority <= best_priority:
                continue

            best_priority = priority
            snippet = self._build_snippet(text, match, pol)

            # Transparent mode: include rule/policy name in message
            rule_name_visible = None
            msg_to_dependent = pol.message_to_dependent or self._DEFAULT_BLOCK_MESSAGE
            if gp and gp.transparent:
                rule_name_visible = gp.name
                msg_to_dependent = (
                    f"[{gp.name}] Category: {pol.category or 'supervised'} — "
                    f"{pol.message_to_dependent or self._DEFAULT_BLOCK_MESSAGE}"
                )

            best_result = SupervisedCheckResult(
                action=pol.action,
                matched_policy_id=pol.id,
                matched_category=pol.category,
                matched_pattern=regex.pattern,
                severity=pol.severity,
                message_to_dependent=msg_to_dependent,
                notify_guardian=pol.notify_guardian,
                notify_context=pol.notify_context,
                context_snippet=snippet,
                rule_name_visible=rule_name_visible,
            )

            # If we hit a block, can't do worse — early exit
            if pol.action == "block":
                break

        # If action is redact, compute redacted text
        if best_result.action == "redact":
            redacted = text
            for pol, regex, _gp in compiled_policies:
                if pol.phase != "both" and pol.phase != phase:
                    continue
                if pol.action != "redact":
                    continue
                try:
                    redacted = regex.sub("[REDACTED]", redacted)
                except re.error:
                    continue
            best_result.redacted_text = redacted

        return best_result

    @staticmethod
    def _build_snippet(
        text: str,
        match: re.Match,
        policy: SupervisedPolicy,
    ) -> str | None:
        if policy.notify_context == "full_message":
            return text[:500] if len(text) > 500 else text
        if policy.notify_context == "snippet":
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            snippet = text[start:end].strip()
            return snippet[:200] if len(snippet) > 200 else snippet
        # topic_only — return just the category, no content
        return None

    def build_moderation_policy_overlay(
        self,
        dependent_user_id: str,
        base_policy: ModerationPolicy,
        chat_type: str | None = None,
    ) -> ModerationPolicy:
        """Create a ModerationPolicy that merges supervised rules with the base policy.

        This allows the standard moderation pipeline to enforce supervised
        rules without special-casing the check path.
        """
        compiled_policies = self._get_compiled_policies(dependent_user_id, chat_type=chat_type)
        if not compiled_policies:
            return base_policy

        # Merge supervised patterns into the base policy's block_patterns.
        # When multiple guardians define overlapping rules for a dependent,
        # apply deterministic strictest-wins conflict resolution per pattern.
        grouped_policies: dict[str, list[tuple[SupervisedPolicy, re.Pattern, GovernancePolicy | None]]] = {}
        for pol, regex, gp in compiled_policies:
            group_key = f"{regex.pattern}|{pol.phase}|{pol.category or 'supervised'}"
            grouped_policies.setdefault(group_key, []).append((pol, regex, gp))

        extra_rules: list[PatternRule] = []
        for group_key, entries in grouped_policies.items():
            winner = resolve_conflicts([entry[0] for entry in entries])
            if winner is None:
                continue

            selected_entry = next(
                (entry for entry in entries if entry[0].id == winner.id),
                entries[0],
            )
            pol, regex, _gp = selected_entry
            rule_action = pol.action if pol.action in ("block", "redact", "warn") else "warn"
            extra_rules.append(
                PatternRule(
                    regex=regex,
                    action=rule_action,
                    replacement="[REDACTED]" if pol.action == "redact" else None,
                    categories={pol.category} if pol.category else {"supervised"},
                )
            )
            if len(entries) > 1:
                logger.info(
                    "Resolved shared-dependent policy conflict for {}: winner={} candidates={}",
                    group_key,
                    pol.id,
                    [entry[0].id for entry in entries],
                )

        merged_patterns = list(base_policy.block_patterns or []) + extra_rules
        return ModerationPolicy(
            enabled=True,  # force enabled for supervised accounts
            input_enabled=base_policy.input_enabled,
            output_enabled=base_policy.output_enabled,
            input_action=base_policy.input_action,
            output_action=base_policy.output_action,
            redact_replacement=base_policy.redact_replacement,
            per_user_overrides=base_policy.per_user_overrides,
            block_patterns=merged_patterns,
            categories_enabled=base_policy.categories_enabled,
        )


def dispatch_guardian_notification(
    result: SupervisedCheckResult,
    dependent_user_id: str,
    guardian_user_id: str | None = None,
) -> str:
    """Send a guardian notification when a supervised policy triggers.

    Returns "sent", "skipped", or "failed".
    """
    if not result.notify_guardian or result.action == "pass":
        return "skipped"
    try:
        from tldw_Server_API.app.core.Monitoring.notification_service import (
            get_notification_service,
        )
        payload: dict[str, Any] = {
            "type": "guardian_alert",
            "severity": result.severity,
            "action": result.action,
            "category": result.matched_category,
            "pattern": result.matched_pattern,
            "dependent_user_id": dependent_user_id,
        }
        if guardian_user_id:
            payload["guardian_user_id"] = guardian_user_id
        if result.context_snippet:
            payload["context_snippet"] = result.context_snippet
        if result.rule_name_visible:
            payload["rule_name"] = result.rule_name_visible
        svc = get_notification_service()
        return svc.notify_or_batch(payload)
    except _SUPERVISED_NONCRITICAL_EXCEPTIONS:
        logger.debug("Guardian notification dispatch failed")
        return "failed"


class GuardianModerationProxy:
    """Wraps ModerationService to overlay guardian policies on get_effective_policy()."""

    def __init__(
        self,
        base: Any,
        engine: SupervisedPolicyEngine,
        dependent_user_id: str,
        *,
        chat_type: str | None = None,
    ) -> None:
        self._base = base
        self._engine = engine
        self._dep_uid = dependent_user_id
        self._chat_type = str(chat_type or "regular").strip().lower() or "regular"

    def get_effective_policy(self, user_id: str | None = None) -> ModerationPolicy:
        base_policy = self._base.get_effective_policy(user_id)
        try:
            return self._engine.build_moderation_policy_overlay(
                self._dep_uid,
                base_policy,
                chat_type=self._chat_type,
            )
        except _SUPERVISED_NONCRITICAL_EXCEPTIONS:
            logger.debug("Guardian policy overlay skipped in proxy")
            return base_policy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


# ── Factory accessor ──────────────────────────────────────────


def get_supervised_policy_engine(guardian_db: GuardianDB) -> SupervisedPolicyEngine:
    """Create a SupervisedPolicyEngine bound to the given DB.

    A new instance is returned each call so that multi-user deployments
    always query the correct per-user database.
    """
    return SupervisedPolicyEngine(guardian_db)


def bootstrap_guardian_moderation_runtime(
    *,
    user_id: object,
    dependent_user_id: str,
    chat_type: str = "regular",
) -> GuardianModerationRuntime:
    """Resolve guardian DB/services for a dependent-user moderation context."""
    normalized_chat_type = str(chat_type or "regular").strip().lower() or "regular"
    guardian_db = resolve_guardian_db_for_user_id(user_id)
    return GuardianModerationRuntime(
        dependent_user_id=str(dependent_user_id),
        chat_type=normalized_chat_type,
        guardian_db=guardian_db,
        supervised_engine=(
            get_supervised_policy_engine(guardian_db)
            if guardian_db is not None
            else None
        ),
    )
