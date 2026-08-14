"""Shared canonical data models for Moderation policy compilation and evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_MODEL_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
    re.error,
)
_UNCATEGORIZED_CATEGORY = "uncategorized"


@dataclass
class ModerationPolicy:
    enabled: bool = False
    input_enabled: bool = True
    output_enabled: bool = True
    input_action: str = "block"  # block | redact | warn
    output_action: str = "redact"  # redact | block | warn (block only applies to non-streaming)
    redact_replacement: str = "[REDACTED]"
    per_user_overrides: bool = True
    # Compiled rules; each rule includes the regex and optional per-pattern action/replacement
    block_patterns: list[PatternRule] = field(default_factory=list)
    # Enabled categories filter (None or empty means allow all)
    categories_enabled: set[str] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of the policy (without raw regex objects)."""
        patterns: list[str] = []
        try:
            if self.block_patterns:
                # Backward-friendly: expose raw patterns as strings
                tmp: list[str] = []
                for p in self.block_patterns:
                    pat = getattr(p, "pattern", None)
                    if pat is None and isinstance(p, PatternRule):
                        pat = getattr(p.regex, "pattern", "")
                    tmp.append(pat or "")
                patterns = tmp
        except _MODEL_NONCRITICAL_EXCEPTIONS:
            patterns = []
        # Provide richer rule view
        rules: list[dict[str, str]] = []
        try:
            if self.block_patterns:
                for p in self.block_patterns:
                    if isinstance(p, PatternRule):
                        cats = p.categories if p.categories else {_UNCATEGORIZED_CATEGORY}
                        rules.append(
                            {
                                "pattern": p.regex.pattern,
                                "action": p.action or "",
                                "replacement": p.replacement or "",
                                "phase": p.phase or "both",
                                "categories": ",".join(sorted(cats)) if cats else "",
                            }
                        )
                    else:
                        rules.append(
                            {
                                "pattern": getattr(p, "pattern", ""),
                                "action": "",
                                "replacement": "",
                                "phase": "both",
                                "categories": "",
                            }
                        )
        except _MODEL_NONCRITICAL_EXCEPTIONS:
            rules = []
        return {
            "enabled": self.enabled,
            "input_enabled": self.input_enabled,
            "output_enabled": self.output_enabled,
            "input_action": self.input_action,
            "output_action": self.output_action,
            "redact_replacement": self.redact_replacement,
            "per_user_overrides": self.per_user_overrides,
            "blocklist_count": len(patterns),
            "block_patterns": patterns,
            "rules": rules,
            "categories_enabled": sorted(self.categories_enabled) if self.categories_enabled else [],
        }


@dataclass
class PatternRule:
    regex: re.Pattern
    action: str | None = None  # block | redact | warn | None
    replacement: str | None = None  # only used when action=redact
    categories: set[str] | None = None  # e.g., {"pii", "confidential"}
    phase: str = "both"  # input | output | both


@dataclass
class ModerationEvaluationResult:
    """Canonical moderation evaluation result."""

    action: str = "pass"
    redacted_text: str | None = None
    matched_pattern: str | None = None
    category: str | None = None
    match_span: tuple[int, int] | None = None
    sample: str | None = None
