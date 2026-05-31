"""
self_monitoring_service.py

Awareness-oriented self-monitoring for users. Unlike ModerationService
(which blocks/redacts), this service focuses on consciousness and
notification — helping users notice patterns in their AI interactions.

Features:
- Pattern-based topic detection with false-positive exclusions
- Configurable notification frequency with dedup
- Escalation: session-level and rolling-window threshold tracking
- Crisis resource integration (988 Lifeline, Crisis Text Line)
- Cooldown protection against impulsive rule disabling
- Multiple display modes (inline_banner, sidebar_note, etc.)
- Governance policy integration (schedule and chat-type scope filtering)
- Confirmation and partner_approval bypass deactivation flows

Integration: Called by chat pipeline after standard moderation to
evaluate user's self-monitoring rules.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.Character_Chat.regex_safety import validate_regex_safety
from tldw_Server_API.app.core.DB_Management.Guardian_DB import (
    GovernancePolicy,
    GuardianDB,
    SelfMonitoringRule,
)
from tldw_Server_API.app.core.Moderation.governance_utils import (
    chat_type_matches,
    is_schedule_active,
)

_SELFMON_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    re.error,
)


# ── Crisis Resources ──────────────────────────────────────────

CRISIS_RESOURCES = [
    {
        "name": "988 Suicide & Crisis Lifeline",
        "description": "Free, confidential support for people in distress. Call or text 988.",
        "contact": "988",
        "url": "https://988lifeline.org",
        "available_24_7": True,
    },
    {
        "name": "Crisis Text Line",
        "description": "Free crisis support via text message. Text HOME to 741741.",
        "contact": "Text HOME to 741741",
        "url": "https://www.crisistextline.org",
        "available_24_7": True,
    },
    {
        "name": "SAMHSA National Helpline",
        "description": "Free referral service for substance abuse and mental health. Call 1-800-662-4357.",
        "contact": "1-800-662-4357",
        "url": "https://www.samhsa.gov/find-help/national-helpline",
        "available_24_7": True,
    },
    {
        "name": "International Association for Suicide Prevention",
        "description": "Directory of crisis centers worldwide.",
        "contact": "See website for local numbers",
        "url": "https://www.iasp.info/resources/Crisis_Centres/",
        "available_24_7": False,
    },
]

CRISIS_DISCLAIMER = (
    "tldw is not a crisis service and cannot provide emergency assistance. "
    "If you are in immediate danger, please call emergency services (911 in the US) "
    "or contact the resources listed above. These resources are provided for "
    "informational purposes only. tldw assumes no liability for the availability "
    "or quality of external crisis services."
)


# ── Result dataclass ──────────────────────────────────────────

@dataclass
class SelfMonitoringCheckResult:
    """Result from evaluating self-monitoring rules against text."""
    triggered: bool = False
    alerts: list[dict[str, Any]] = field(default_factory=list)
    # Highest severity action to apply
    action: str = "pass"  # pass | notify | block | redact
    redacted_text: str | None = None
    block_message: str | None = None
    display_mode: str = "inline_banner"
    crisis_resources: list[dict[str, Any]] | None = None
    escalation_triggered: bool = False


# ── Service class ─────────────────────────────────────────────

class SelfMonitoringService:
    """Evaluates self-monitoring rules for a user and generates alerts."""

    def __init__(self, guardian_db: GuardianDB) -> None:
        self._db = guardian_db
        self._lock = threading.RLock()
        # Cache: user_id -> list of (rule, include_patterns, exclude_patterns)
        self._compiled_cache: dict[str, list[tuple[SelfMonitoringRule, list[re.Pattern], list[re.Pattern]]]] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl_seconds = 30.0

    def invalidate_cache(self, user_id: str | None = None) -> None:
        with self._lock:
            if user_id:
                prefix = f"{user_id}:"
                keys_to_remove = [k for k in self._compiled_cache if k.startswith(prefix)]
                for k in keys_to_remove:
                    self._compiled_cache.pop(k, None)
                    self._cache_timestamps.pop(k, None)
            else:
                self._compiled_cache.clear()
                self._cache_timestamps.clear()

    def _get_compiled_rules(
        self,
        user_id: str,
        chat_type: str | None = None,
    ) -> list[tuple[SelfMonitoringRule, list[re.Pattern], list[re.Pattern], GovernancePolicy | None]]:
        uid = str(user_id)
        cache_key = f"{uid}:{chat_type or ''}"
        now = datetime.now(timezone.utc).timestamp()
        with self._lock:
            cached_at = self._cache_timestamps.get(cache_key, 0.0)
            if cache_key in self._compiled_cache and (now - cached_at) < self._cache_ttl_seconds:
                return self._compiled_cache[cache_key]

        rules = self._db.list_self_monitoring_rules(uid, enabled_only=True)
        compiled: list[tuple[SelfMonitoringRule, list[re.Pattern], list[re.Pattern], GovernancePolicy | None]] = []

        for rule in rules:
            # Governance policy filtering
            gp: GovernancePolicy | None = None
            if rule.governance_policy_id:
                try:
                    gp = self._db.get_governance_policy(rule.governance_policy_id)
                except _SELFMON_NONCRITICAL_EXCEPTIONS:
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

            include_pats: list[re.Pattern] = []
            exclude_pats: list[re.Pattern] = []
            for pat_str in (rule.patterns or []):
                try:
                    if rule.pattern_type == "regex":
                        is_safe, reason = validate_regex_safety(pat_str)
                        if not is_safe:
                            logger.warning("Unsafe self-monitoring regex skipped")
                            continue
                        include_pats.append(re.compile(pat_str, re.IGNORECASE))
                    else:
                        include_pats.append(re.compile(re.escape(pat_str), re.IGNORECASE))
                except re.error:
                    logger.warning("Invalid self-monitoring pattern skipped")
            for exc_str in (rule.except_patterns or []):
                try:
                    if rule.pattern_type == "regex":
                        is_safe, reason = validate_regex_safety(exc_str)
                        if not is_safe:
                            logger.warning("Unsafe self-monitoring except regex skipped")
                            continue
                        exclude_pats.append(re.compile(exc_str, re.IGNORECASE))
                    else:
                        exclude_pats.append(re.compile(re.escape(exc_str), re.IGNORECASE))
                except re.error:
                    continue
            if include_pats:
                compiled.append((rule, include_pats, exclude_pats, gp))

        with self._lock:
            self._compiled_cache[cache_key] = compiled
            self._cache_timestamps[cache_key] = now
        return compiled

    def check_text(
        self,
        text: str,
        user_id: str,
        phase: str = "input",
        conversation_id: str | None = None,
        session_id: str | None = None,
        chat_type: str | None = None,
    ) -> SelfMonitoringCheckResult:
        """Evaluate self-monitoring rules against text.

        Returns a result with triggered alerts and the highest-priority action.
        """
        if not text or not text.strip():
            return SelfMonitoringCheckResult()

        compiled_rules = self._get_compiled_rules(user_id, chat_type=chat_type)
        if not compiled_rules:
            return SelfMonitoringCheckResult()

        result = SelfMonitoringCheckResult()
        action_priority = {"notify": 1, "redact": 2, "block": 3}
        best_priority = 0

        for rule, include_pats, exclude_pats, _gp in compiled_rules:
            # Phase filtering
            if rule.phase != "both" and rule.phase != phase:
                continue
            # Min context length check
            if rule.min_context_length > 0 and len(text) < rule.min_context_length:
                continue
            # Check for include pattern match
            matched_pattern = ""
            for pat in include_pats:
                m = pat.search(text)
                if m:
                    matched_pattern = pat.pattern
                    break
            if not matched_pattern:
                continue
            # Check for exclude pattern (false positive suppression)
            excluded = False
            for exc_pat in exclude_pats:
                if exc_pat.search(text):
                    excluded = True
                    break
            if excluded:
                continue
            # Dedup check
            if self._should_skip_dedup(rule, user_id, conversation_id, session_id):
                continue
            # Escalation check
            escalation_info = self._check_escalation(rule, user_id, session_id)
            effective_action = escalation_info.get("effective_action", rule.action)

            # Build alert
            snippet = self._build_context_snippet(text, rule)
            alert_info: dict[str, Any] = {
                "rule_id": rule.id,
                "rule_name": rule.name,
                "category": rule.category,
                "severity": rule.severity,
                "matched_pattern": matched_pattern,
                "action": effective_action,
                "display_mode": rule.display_mode,
                "context_note": rule.context_note,
                "notification_channels": rule.notification_channels,
                "crisis_resources_enabled": rule.crisis_resources_enabled,
            }
            if escalation_info.get("escalated"):
                alert_info["escalation_info"] = escalation_info
                result.escalation_triggered = True

            result.alerts.append(alert_info)
            result.triggered = True

            # Track highest priority action
            priority = action_priority.get(effective_action, 0)
            if priority > best_priority:
                best_priority = priority
                result.action = effective_action
                result.display_mode = rule.display_mode
                if effective_action == "block":
                    result.block_message = rule.block_message or (
                        f"Self-monitoring rule '{rule.name}' triggered. "
                        "You configured this rule to block this type of content."
                    )

            # Record the alert in DB
            try:
                channels_used = list(rule.notification_channels or [])
                self._db.create_self_monitoring_alert(
                    user_id=user_id,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    matched_pattern=matched_pattern,
                    context_snippet=snippet,
                    snippet_mode="full_snippet" if snippet else "none",
                    conversation_id=conversation_id,
                    session_id=session_id,
                    chat_type=chat_type,
                    phase=phase,
                    action_taken=effective_action,
                    notification_sent="in_app" in channels_used,
                    notification_channels_used=channels_used,
                    crisis_resources_shown=rule.crisis_resources_enabled,
                    display_mode=rule.display_mode,
                    escalation_info=escalation_info if escalation_info.get("escalated") else None,
                )
            except _SELFMON_NONCRITICAL_EXCEPTIONS:
                logger.warning("Failed to record self-monitoring alert")

            # Crisis resources
            if rule.crisis_resources_enabled:
                result.crisis_resources = CRISIS_RESOURCES

        # If any rule triggers redact, compute redacted text
        if result.action == "redact":
            redacted = text
            for rule, include_pats, _exclude_pats, _gp in compiled_rules:
                if rule.action != "redact":
                    continue
                for pat in include_pats:
                    try:
                        redacted = pat.sub("[SELF-REDACTED]", redacted)
                    except re.error:
                        continue
            result.redacted_text = redacted

        return result

    def _should_skip_dedup(
        self,
        rule: SelfMonitoringRule,
        user_id: str,
        conversation_id: str | None,
        session_id: str | None,
    ) -> bool:
        """Check notification frequency for dedup."""
        freq = rule.notification_frequency
        if freq == "every_message":
            return False
        try:
            if freq == "once_per_conversation" and conversation_id:
                return self._db.has_recent_alert(
                    user_id=user_id,
                    rule_id=rule.id,
                    conversation_id=conversation_id,
                )
            if freq == "once_per_day":
                since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
                return self._db.has_recent_alert(
                    user_id=user_id,
                    rule_id=rule.id,
                    since_iso=since,
                )
            if freq == "once_per_session" and session_id:
                return self._db.has_recent_alert(
                    user_id=user_id,
                    rule_id=rule.id,
                    session_id=session_id,
                )
        except _SELFMON_NONCRITICAL_EXCEPTIONS:
            logger.debug("Dedup check failed")
        return False

    def _check_escalation(
        self,
        rule: SelfMonitoringRule,
        user_id: str,
        session_id: str | None,
    ) -> dict[str, Any]:
        """Check and update escalation state for a rule trigger."""
        info: dict[str, Any] = {"escalated": False, "effective_action": rule.action}

        # Skip if no escalation configured
        if rule.escalation_session_threshold <= 0 and rule.escalation_window_threshold <= 0:
            return info

        try:
            state = self._db.get_escalation_state(rule.id, user_id)
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            session_count = 0

            if state:
                # Reset session counter if session changed
                if session_id and state.session_id != session_id:
                    session_count = 1
                else:
                    session_count = (state.session_trigger_count or 0) + 1
            else:
                session_count = 1

            # Rolling-window count: query actual alerts within the window
            # instead of monotonically incrementing a counter.
            # +1 because the current trigger's alert hasn't been recorded yet.
            if rule.escalation_window_days > 0 and rule.escalation_window_threshold > 0:
                window_count = self._db.count_alerts_in_window(
                    user_id, rule.id, rule.escalation_window_days,
                ) + 1  # include the current (not-yet-persisted) trigger
            elif state:
                window_count = (state.window_trigger_count or 0) + 1
            else:
                window_count = 1

            # Check cooldown-based auto de-escalation: if a previous
            # escalation has a cooldown_until in the past, clear it.
            if state and state.cooldown_until:
                try:
                    cooldown_dt = datetime.fromisoformat(state.cooldown_until)
                    if now >= cooldown_dt:
                        # Cooldown expired — reset escalation state
                        self._db.upsert_escalation_state(
                            rule_id=rule.id,
                            user_id=user_id,
                            session_id=session_id,
                            session_trigger_count=session_count,
                            window_trigger_count=window_count,
                            current_escalated_action=None,
                            escalated_at=None,
                            cooldown_until=None,
                        )
                        return info
                except (ValueError, TypeError):
                    pass  # malformed cooldown_until — ignore

            escalated = False
            escalated_action = None

            # Session-level escalation
            if (
                rule.escalation_session_threshold > 0
                and session_count >= rule.escalation_session_threshold
                and rule.escalation_session_action
            ):
                escalated = True
                escalated_action = rule.escalation_session_action

            # Window-level escalation (cross-session)
            if (
                rule.escalation_window_threshold > 0
                and window_count >= rule.escalation_window_threshold
                and rule.escalation_window_action
            ):
                escalated = True
                escalated_action = rule.escalation_window_action

            # Compute cooldown_until when escalation triggers
            cooldown_until_iso = None
            if escalated and rule.cooldown_minutes > 0:
                cooldown_until_iso = (now + timedelta(minutes=rule.cooldown_minutes)).isoformat()

            self._db.upsert_escalation_state(
                rule_id=rule.id,
                user_id=user_id,
                session_id=session_id,
                session_trigger_count=session_count,
                window_trigger_count=window_count,
                current_escalated_action=escalated_action if escalated else None,
                escalated_at=now_iso if escalated else None,
                cooldown_until=cooldown_until_iso,
            )

            if escalated and escalated_action:
                info["escalated"] = True
                info["effective_action"] = escalated_action
                info["session_trigger_count"] = session_count
                info["window_trigger_count"] = window_count
                info["escalated_action"] = escalated_action

        except _SELFMON_NONCRITICAL_EXCEPTIONS:
            logger.debug("Escalation check failed")

        return info

    @staticmethod
    def _build_context_snippet(text: str, rule: SelfMonitoringRule) -> str | None:
        """Build a context snippet based on the rule's display mode."""
        if rule.display_mode == "silent_log":
            return None
        if not text:
            return None
        max_len = 200
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def can_disable_rule(self, rule: SelfMonitoringRule) -> bool:
        """Check if a rule can be disabled (cooldown protection).

        The cooldown timer starts from when deactivation was requested
        (``deactivation_requested_at``), NOT from when the rule was created.
        If no deactivation has been requested yet, the rule cannot be
        disabled until a request is made and the cooldown elapses.
        """
        if rule.cooldown_minutes <= 0:
            return True
        if rule.bypass_protection == "none":
            return True
        # If a pending_deactivation_at is set, check if it has elapsed
        if rule.pending_deactivation_at:
            try:
                deactivation_at = datetime.fromisoformat(rule.pending_deactivation_at)
                return datetime.now(timezone.utc) >= deactivation_at
            except _SELFMON_NONCRITICAL_EXCEPTIONS:
                return True
        # If deactivation was requested, check requested_at + cooldown
        if rule.deactivation_requested_at:
            try:
                requested = datetime.fromisoformat(rule.deactivation_requested_at)
                cooldown_end = requested + timedelta(minutes=rule.cooldown_minutes)
                return datetime.now(timezone.utc) >= cooldown_end
            except _SELFMON_NONCRITICAL_EXCEPTIONS:
                return True
        # No deactivation requested yet — cannot disable
        return False

    def request_deactivation(self, rule_id: str, user_id: str) -> dict[str, Any]:
        """Request deactivation of a rule with bypass protection.

        Handles all bypass modes:
        - "none": disable immediately
        - "cooldown": start cooldown from *now*, disable after cooldown_minutes
        - "confirmation": generate token, require confirm_deactivation()
        - "partner_approval": generate token, require approve_deactivation()
        """
        rule = self._db.get_self_monitoring_rule(rule_id)
        if not rule:
            return {"ok": False, "error": "Rule not found"}
        if rule.user_id != str(user_id):
            return {"ok": False, "error": "Not your rule"}

        bypass = rule.bypass_protection or "none"

        # "none" — disable immediately regardless of cooldown
        if bypass == "none":
            self._db.update_self_monitoring_rule(rule_id, enabled=False)
            self.invalidate_cache(str(user_id))
            return {"ok": True, "status": "disabled_immediately"}

        # "cooldown" — cooldown timer starts from the deactivation request
        if bypass == "cooldown":
            # If a previous request is pending and cooldown has elapsed, disable
            if self.can_disable_rule(rule):
                self._db.update_self_monitoring_rule(
                    rule_id,
                    enabled=False,
                    pending_deactivation_at=None,
                    deactivation_requested_at=None,
                )
                self.invalidate_cache(str(user_id))
                return {"ok": True, "status": "disabled_immediately"}
            try:
                now = datetime.now(timezone.utc)
                now_iso = now.isoformat()
                deactivation_at = now + timedelta(minutes=rule.cooldown_minutes)
                self._db.update_self_monitoring_rule(
                    rule_id,
                    pending_deactivation_at=deactivation_at.isoformat(),
                    deactivation_requested_at=now_iso,
                )
                return {
                    "ok": True,
                    "status": "pending_deactivation",
                    "deactivation_at": deactivation_at.isoformat(),
                    "reason": (
                        f"This rule has a {rule.cooldown_minutes}-minute cooldown. "
                        f"It will be disabled at {deactivation_at.isoformat()}."
                    ),
                }
            except _SELFMON_NONCRITICAL_EXCEPTIONS:
                return {"ok": False, "error": "deactivation_request_failed"}

        # "confirmation" — generate token, user must confirm
        if bypass == "confirmation":
            token = uuid4().hex[:16]
            now_iso = datetime.now(timezone.utc).isoformat()
            self._db.update_self_monitoring_rule(
                rule_id,
                deactivation_confirmation_token=token,
                deactivation_requested_at=now_iso,
            )
            return {
                "ok": True,
                "status": "awaiting_confirmation",
                "confirmation_token": token,
            }

        # "partner_approval" — generate token, partner must approve
        if bypass == "partner_approval":
            token = uuid4().hex[:16]
            now_iso = datetime.now(timezone.utc).isoformat()
            self._db.update_self_monitoring_rule(
                rule_id,
                deactivation_confirmation_token=token,
                deactivation_requested_at=now_iso,
            )
            return {
                "ok": True,
                "status": "awaiting_partner_approval",
                "confirmation_token": token,
                "partner_user_id": rule.bypass_partner_user_id,
            }

        # Unknown bypass mode — fail safe
        return {"ok": False, "error": f"Unknown bypass_protection mode: {bypass}"}

    def confirm_deactivation(
        self, rule_id: str, user_id: str, token: str,
    ) -> dict[str, Any]:
        """Confirm deactivation of a rule (confirmation bypass mode).

        The user who owns the rule provides the token they received
        from request_deactivation().
        """
        rule = self._db.get_self_monitoring_rule(rule_id)
        if not rule:
            return {"ok": False, "error": "Rule not found"}
        if rule.user_id != str(user_id):
            return {"ok": False, "error": "Not your rule"}
        if rule.bypass_protection != "confirmation":
            return {"ok": False, "error": "Rule does not use confirmation bypass"}
        if not rule.deactivation_confirmation_token:
            return {"ok": False, "error": "No pending deactivation request"}
        if rule.deactivation_confirmation_token != token:
            return {"ok": False, "error": "Invalid confirmation token"}

        # Token matches — disable rule and clear token
        self._db.update_self_monitoring_rule(
            rule_id,
            enabled=False,
            deactivation_confirmation_token=None,
            deactivation_requested_at=None,
        )
        self.invalidate_cache(str(user_id))
        return {"ok": True, "status": "disabled"}

    def approve_deactivation(
        self, rule_id: str, approver_user_id: str, token: str,
    ) -> dict[str, Any]:
        """Approve deactivation of a rule (partner_approval bypass mode).

        The designated partner user provides the token to approve.
        """
        rule = self._db.get_self_monitoring_rule(rule_id)
        if not rule:
            return {"ok": False, "error": "Rule not found"}
        if rule.bypass_protection != "partner_approval":
            return {"ok": False, "error": "Rule does not use partner_approval bypass"}
        if rule.bypass_partner_user_id != str(approver_user_id):
            return {"ok": False, "error": "Not the designated partner"}
        if not rule.deactivation_confirmation_token:
            return {"ok": False, "error": "No pending deactivation request"}
        if rule.deactivation_confirmation_token != token:
            return {"ok": False, "error": "Invalid confirmation token"}

        # Token matches — disable rule and clear token
        self._db.update_self_monitoring_rule(
            rule_id,
            enabled=False,
            deactivation_confirmation_token=None,
            deactivation_requested_at=None,
        )
        self.invalidate_cache(rule.user_id)
        return {"ok": True, "status": "disabled"}

    def get_crisis_resources(self) -> dict[str, Any]:
        """Return crisis resources and disclaimer."""
        return {
            "resources": CRISIS_RESOURCES,
            "disclaimer": CRISIS_DISCLAIMER,
        }


# ── Factory accessor ──────────────────────────────────────────


def get_self_monitoring_service(guardian_db: GuardianDB) -> SelfMonitoringService:
    """Create a SelfMonitoringService bound to the given DB.

    A new instance is returned each call so that multi-user deployments
    always query the correct per-user database.
    """
    return SelfMonitoringService(guardian_db)
