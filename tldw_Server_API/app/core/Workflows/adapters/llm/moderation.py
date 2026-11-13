"""Moderation adapters: moderation, policy_check.

These adapters handle content moderation and policy checking.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation.async_execution import run_moderation_async
from tldw_Server_API.app.core.Moderation.execution_models import (
    ExecutionBounds,
    FailureMetadata,
    SubstitutionRuleSnapshot,
    execution_scope,
)
from tldw_Server_API.app.core.Moderation.execution_protocol import snapshot_substitution_request
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Utils.template_render import apply_template_to_string
from tldw_Server_API.app.core.Workflows.adapters._common import resolve_context_user_id
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.llm._config import (
    ModerationConfig,
    PolicyCheckConfig,
)


@registry.register(
    "moderation",
    category="llm",
    description="Check or redact text content using the moderation service",
    parallelizable=True,
    tags=["moderation", "safety"],
    config_model=ModerationConfig,
)
async def run_moderation_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Check or redact text content using the moderation service.

    Config:
      - action: Literal["check", "redact"] (default: "check")
      - text: Optional[str] (templated, defaults to last.text)
      - action_type: str = "generic" (context for check action)
      - patterns: Optional[List[str]] (for redact action, additional patterns)
    Output for "check":
      - {"allowed": bool, "reason": str, "matched_rules": [...]}
    Output for "redact":
      - {"redacted_text": str, "redaction_count": int, "text": str}
    """
    # Cancellation check
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    action = str(config.get("action") or "check").strip().lower()

    # Template rendering for text
    text_t = str(config.get("text") or "").strip()
    if text_t:
        text = apply_template_to_string(text_t, context) or text_t
    else:
        # Default to last.text
        text = None
        try:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                text = str(last.get("text") or last.get("content") or "")
        except Exception as context_error:
            logger.debug("Moderation adapter failed to read fallback text from context", exc_info=context_error)
    text = text or ""

    if not text.strip():
        return {"error": "missing_text"}

    def execute():
        with execution_scope():
            return _execute_moderation(config, context, action, text, simulated=is_test_mode())

    return await run_moderation_async(
        execute,
        metadata=FailureMetadata(
            "evaluate" if action == "check" else "custom_substitute",
            "input" if action == "check" else None,
            len(text),
        ),
    )


def _custom_substitute(text: str, patterns: list[str], replacement: str) -> tuple[str, int]:
    """Run one sequential stdlib template batch, including simulation rules."""
    from tldw_Server_API.app.core.Moderation.execution_runner import get_execution_runner

    rules = [
        SubstitutionRuleSnapshot(pattern.strip(), int(re.IGNORECASE), replacement)
        for pattern in patterns
        if isinstance(pattern, str) and pattern.strip()
    ]
    if not rules:
        return text, 0
    with execution_scope() as scope:
        request = snapshot_substitution_request(text, rules, None, ExecutionBounds())
        result = get_execution_runner().execute(request, scope).result
    if result.skipped_rule_indices:
        logger.warning("Invalid custom redaction pattern skipped count={}", len(result.skipped_rule_indices))
    return result.text, result.count


def _execute_moderation(
    config: dict[str, Any], context: dict[str, Any], action: str, text: str, *, simulated: bool
) -> dict[str, Any]:
    """Trusted orchestration only; all content regex execution is child-owned."""
    if simulated:
        if action == "check":
            # Simulate: flag if "blocked" appears in text
            is_blocked = "blocked" in text.lower() or "unsafe" in text.lower()
            return {
                "allowed": not is_blocked,
                "reason": "contains_blocked_term" if is_blocked else "passed",
                "matched_rules": ["test_blocked_term"] if is_blocked else [],
                "simulated": True,
            }
        if action == "redact":
            # Simulate: redact any occurrence of "secret" or "password"
            redacted, base_count = _custom_substitute(text, [r"\b(secret|password|blocked|unsafe)\b"], "[REDACTED]")
            # Apply custom patterns if provided
            custom_patterns = config.get("patterns")
            custom_count = 0
            if custom_patterns and isinstance(custom_patterns, list):
                redacted, custom_count = _custom_substitute(redacted, custom_patterns, "[REDACTED]")
            redaction_count = base_count + custom_count
            return {
                "redacted_text": redacted,
                "text": redacted,
                "redaction_count": redaction_count,
                "simulated": True,
            }
        return {"error": f"unknown_action:{action}", "simulated": True}

    try:
        from tldw_Server_API.app.core.Moderation.moderation_service import get_moderation_service

        service = get_moderation_service()

        # Get effective policy (no user_id required for moderation - stateless)
        user_id = resolve_context_user_id(context)
        policy = service.get_effective_policy(user_id)

        if action == "check":
            # Use check_text for basic flagging
            is_flagged, matched_sample = service.check_text(text, policy, phase="input")

            if is_flagged:
                # Get more details via evaluate_action
                eval_action, _, pattern, category, _ = service.evaluate_action_with_match(
                    text, policy, phase="input"
                )
                return {
                    "allowed": False,
                    "reason": f"matched:{category or pattern or 'rule'}",
                    "matched_rules": [pattern] if pattern else [],
                    "action_recommended": eval_action,
                    "sample": matched_sample,
                }
            return {
                "allowed": True,
                "reason": "passed",
                "matched_rules": [],
            }

        if action == "redact":
            if hasattr(service, "redact_text_with_count"):
                redacted, base_count = service.redact_text_with_count(text, policy)
            else:
                redacted = service.redact_text(text, policy)
                base_count = redacted.count("[REDACTED]") + redacted.count("[PII]")

            # Apply custom patterns if provided
            custom_patterns = config.get("patterns")
            custom_count = 0
            if custom_patterns and isinstance(custom_patterns, list):
                redacted, custom_count = _custom_substitute(
                    redacted, custom_patterns, policy.redact_replacement or "[REDACTED]"
                )

            redaction_count = int(base_count or 0) + custom_count
            return {
                "redacted_text": redacted,
                "text": redacted,  # Alias for chaining
                "redaction_count": redaction_count,
                "original_length": len(text),
                "redacted_length": len(redacted),
            }

        return {"error": f"unknown_action:{action}"}

    except ModerationExecutionError:
        raise
    except Exception:
        logger.exception("Moderation adapter error")
        return {"error": "moderation_error"}


@registry.register(
    "policy_check",
    category="llm",
    description="Policy/PII gate step for content validation",
    parallelizable=True,
    tags=["moderation", "pii", "policy"],
    config_model=PolicyCheckConfig,
)
async def run_policy_check_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Policy/PII gate step.

    Config:
      - text_source: 'last'|'inputs'|'field' (default: last)
      - field: path in context if text_source='field' (e.g., 'inputs.summary')
      - block_on_pii: bool (default false)
      - block_words: [str] (optional)
      - max_length: int (optional; characters)
      - redact_preview: bool (default false) include redacted text in outputs.preview

    Output:
      - { flags: { pii: {...}, block_words: [...], too_long: bool }, blocked: bool, reasons: [...], preview?: str }
    """
    source = str(config.get("text_source") or "last").strip().lower()
    field = str(config.get("field") or "").strip()
    block_on_pii = bool(config.get("block_on_pii") or False)
    block_words = config.get("block_words") or []
    max_length = config.get("max_length")
    redact_preview = bool(config.get("redact_preview") or False)

    text = ""
    try:
        if source == "inputs":
            if isinstance(context.get("inputs"), dict):
                text = str(context["inputs"].get("text") or context["inputs"].get("summary") or "")
        elif source == "field" and field:
            # Minimal dotted lookup
            obj = context
            for part in field.split('.'):
                obj = obj.get(part) if isinstance(obj, dict) else getattr(obj, part, None)
            if isinstance(obj, (str, bytes)):
                text = obj if isinstance(obj, str) else obj.decode("utf-8", errors="ignore")
            else:
                text = str(obj or "")
        else:
            last = context.get("prev") or context.get("last") or {}
            if isinstance(last, dict):
                text = str(last.get("text") or last.get("content") or "")
    except Exception:
        text = str(text or "")

    flags: dict[str, Any] = {"pii": {}, "block_words": [], "too_long": False}
    reasons: list[str] = []
    blocked = False

    # PII detection
    try:
        from tldw_Server_API.app.core.Audit.unified_audit_service import PIIDetector
        pii = PIIDetector().detect(text)
        if pii:
            flags["pii"] = pii
            if block_on_pii:
                blocked = True
                reasons.append("pii_detected")
    except Exception as pii_error:
        logger.debug("Moderation PII checks failed; continuing", exc_info=pii_error)

    # Block words
    if isinstance(block_words, list) and block_words:
        found = []
        low = (text or "").lower()
        for w in block_words:
            try:
                needle = str(w).lower() if w else ""
            except Exception as block_word_error:
                logger.debug("Moderation block-word normalization failed; skipping entry", exc_info=block_word_error)
                needle = ""
            if needle and needle in low:
                found.append(w)
        if found:
            flags["block_words"] = found
            blocked = True
            reasons.append("blocked_terms")

    # Max length
    try:
        if isinstance(max_length, int) and max_length > 0 and len(text or "") > max_length:
            flags["too_long"] = True
            blocked = True
            reasons.append("too_long")
    except Exception as length_error:
        logger.debug("Moderation length checks failed; continuing", exc_info=length_error)

    out: dict[str, Any] = {"flags": flags, "blocked": blocked, "reasons": reasons}
    if redact_preview and text:
        try:
            from tldw_Server_API.app.core.Audit.unified_audit_service import PIIDetector
            out["preview"] = PIIDetector().redact(text)
        except Exception:
            out["preview"] = text[:500]

    return out
