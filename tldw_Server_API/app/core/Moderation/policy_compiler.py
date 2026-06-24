from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Moderation.moderation_service import (
        ModerationPolicy,
        PatternRule,
    )


@dataclass(frozen=True)
class ResolvedModerationConfig:
    enabled: bool = False
    input_enabled: bool = True
    output_enabled: bool = True
    input_action: str = "block"
    output_action: str = "redact"
    redact_replacement: str = "[REDACTED]"
    per_user_overrides: bool = True
    categories_enabled: set[str] | None = None
    pii_enabled: bool = False


@dataclass(frozen=True)
class PolicyCompilationIssue:
    source: str
    reason: str
    index: int | None = None
    detail: str | None = None


@dataclass
class PolicyCompilationReport:
    issues: list[PolicyCompilationIssue] = field(default_factory=list)

    def add(
        self,
        source: str,
        reason: str,
        *,
        index: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.issues.append(
            PolicyCompilationIssue(
                source=source,
                reason=reason,
                index=index,
                detail=detail,
            )
        )


@dataclass
class PolicyCompilationInput:
    config: ResolvedModerationConfig
    runtime_override: dict[str, object] = field(default_factory=dict)
    blocklist_lines: list[str] = field(default_factory=list)
    user_override: dict[str, object] | None = None
    pii_rules: list[PatternRule] = field(default_factory=list)


@dataclass
class PolicyCompilationResult:
    policy: ModerationPolicy
    report: PolicyCompilationReport


class PolicyCompiler:
    @staticmethod
    def policy_types() -> tuple[type[ModerationPolicy], type[PatternRule]]:
        from tldw_Server_API.app.core.Moderation.moderation_service import (
            ModerationPolicy,
            PatternRule,
        )

        return ModerationPolicy, PatternRule

    def compile_global(self, data: PolicyCompilationInput) -> PolicyCompilationResult:
        report = PolicyCompilationReport()
        config = data.config
        ModerationPolicy, PatternRule = self.policy_types()
        categories_enabled = self.resolve_runtime_categories(
            data.runtime_override,
            config.categories_enabled,
        )
        pii_enabled = self.resolve_runtime_pii(data.runtime_override, config.pii_enabled)
        block_patterns: list[PatternRule] = []
        if pii_enabled:
            block_patterns.extend(list(data.pii_rules or []))

        policy = ModerationPolicy(
            enabled=config.enabled,
            input_enabled=config.input_enabled,
            output_enabled=config.output_enabled,
            input_action=str(config.input_action).lower(),
            output_action=str(config.output_action).lower(),
            redact_replacement=config.redact_replacement,
            per_user_overrides=config.per_user_overrides,
            block_patterns=block_patterns,
            categories_enabled=categories_enabled,
        )
        return PolicyCompilationResult(policy=policy, report=report)

    @staticmethod
    def resolve_runtime_pii(runtime_override: dict[str, object], default: bool) -> bool:
        if "pii_enabled" in runtime_override:
            return bool(runtime_override.get("pii_enabled"))
        return bool(default)

    @staticmethod
    def resolve_runtime_categories(
        runtime_override: dict[str, object],
        default: set[str] | None,
    ) -> set[str] | None:
        if "categories_enabled" not in runtime_override:
            return set(default) if default is not None else None
        raw = runtime_override.get("categories_enabled") or []
        if isinstance(raw, (set, list, tuple)):
            parsed = {str(c).strip().lower() for c in raw if str(c).strip()}
            return parsed or None
        if isinstance(raw, str):
            parsed = {c.strip().lower() for c in raw.split(",") if c.strip()}
            return parsed or None
        return set(default) if default is not None else None
