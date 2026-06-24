"""
moderation_service.py
Description: Centralized, configurable moderation/guardrails for chat content

Features:
- Global moderation settings (from config.txt [Moderation])
- Optional per-user overrides via a JSON mapping file
- Simple, local rule-based checks using blocklist (literals or regex)
- Redaction or blocking actions for inputs and outputs

Notes:
- No network calls; designed to function offline by default
- Streaming supports redaction and, if a block is triggered mid-stream, an SSE error is emitted followed by a [DONE] sentinel for graceful termination.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from loguru import logger

from tldw_Server_API.app.core.config import load_and_log_configs, load_comprehensive_config
from tldw_Server_API.app.core.Moderation.policy_compiler import (
    PolicyCompilationInput,
    PolicyCompilationReport,
    PolicyCompiler,
    ResolvedModerationConfig,
)
from tldw_Server_API.app.core.testing import is_truthy

_MODERATION_NONCRITICAL_EXCEPTIONS = (
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
                    pat = getattr(p, 'pattern', None)
                    if pat is None and isinstance(p, PatternRule):
                        pat = getattr(p.regex, 'pattern', '')
                    tmp.append(pat or '')
                patterns = tmp
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            patterns = []
        # Provide richer rule view
        rules: list[dict[str, str]] = []
        try:
            if self.block_patterns:
                for p in self.block_patterns:
                    if isinstance(p, PatternRule):
                        cats = p.categories if p.categories else {ModerationService._UNCATEGORIZED_CATEGORY}
                        rules.append({
                            "pattern": p.regex.pattern,
                            "action": p.action or "",
                            "replacement": p.replacement or "",
                            "phase": p.phase or "both",
                            "categories": ",".join(sorted(cats)) if cats else "",
                        })
                    else:
                        rules.append(
                            {
                                "pattern": getattr(p, 'pattern', ''),
                                "action": "",
                                "replacement": "",
                                "phase": "both",
                                "categories": "",
                            }
                        )
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
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


@dataclass(frozen=True)
class _ResolvedModerationServiceState:
    compiler_config: ResolvedModerationConfig
    blocklist_path: str | None
    user_overrides_path: str | None
    runtime_overrides_path: str | None


class ModerationService:
    """Loads moderation configuration and evaluates content against policies."""
    _UNCATEGORIZED_CATEGORY = "uncategorized"
    _ALLOWED_REGEX_FLAGS = set(PolicyCompiler._ALLOWED_REGEX_FLAGS)
    _ALLOWED_ACTIONS = set(PolicyCompiler._ALLOWED_ACTIONS)

    def __init__(self) -> None:
        self._config = load_and_log_configs() or {}
        self._lock = threading.RLock()
        self._policy_compiler = PolicyCompiler()
        def _read_int_env(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except _MODERATION_NONCRITICAL_EXCEPTIONS:
                return default
        # Safety/performance limits (overridable via config or env)
        # NOTE: _max_scan_chars is used as the scan chunk size; the full text is scanned in chunks.
        self._max_scan_chars = _read_int_env("MODERATION_MAX_SCAN_CHARS", 200000)
        self._max_replacements_per_pattern = _read_int_env("MODERATION_MAX_REPLACEMENTS_PER_PATTERN", 1000)
        # Window extension to detect matches spanning chunk boundaries
        self._match_window_chars = _read_int_env("MODERATION_MATCH_WINDOW_CHARS", 4096)
        # Max text length for the full-text regex fallback (ReDoS guardrail)
        self._max_fallback_scan_chars = _read_int_env("MODERATION_MAX_FALLBACK_SCAN_CHARS", 800000)
        # Optional debounce for blocklist writes (ms); default disabled
        self._write_debounce_ms = _read_int_env("MODERATION_BLOCKLIST_WRITE_DEBOUNCE_MS", 0)
        self._last_blocklist_write: float = 0.0
        self._runtime_override: dict[str, object] = {}
        self._runtime_overrides_path: str | None = None
        self._pii_enabled: bool = False
        self._global_policy = self._load_global_policy()
        # Load runtime overrides file (if any) and re-apply policy
        try:
            self._load_runtime_overrides_file()
            self._global_policy = self._load_global_policy()
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            pass
        self._user_overrides: dict[str, dict[str, object]] = self._load_user_overrides()

    def _load_global_policy(self) -> ModerationPolicy:
        mod_cfg = self._load_moderation_config_section()
        resolved = self._resolve_moderation_config(mod_cfg)
        self._user_overrides_path = resolved.user_overrides_path
        self._blocklist_path = resolved.blocklist_path
        self._runtime_overrides_path = resolved.runtime_overrides_path
        return self._compile_global_policy_from_resolved_config(resolved.compiler_config)

    def _load_moderation_config_section(self) -> dict[str, object]:
        mod_cfg = (self._config.get("moderation") or {}) if isinstance(self._config, dict) else {}
        if mod_cfg:
            return dict(mod_cfg)
        try:
            parser = load_comprehensive_config()
            if parser and parser.has_section("Moderation"):
                return dict(parser.items("Moderation"))
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            return {}
        return {}

    def _resolve_moderation_config(self, mod_cfg: dict[str, object]) -> _ResolvedModerationServiceState:
        def _b(key: str, default: bool) -> bool:
            val = str(mod_cfg.get(key, default)).strip().lower()
            return is_truthy(val)

        def _anchor(path_value: str) -> str:
            try:
                from pathlib import Path as _Path
                path = _Path(str(path_value))
                if path.is_absolute():
                    return str(path)
                from tldw_Server_API.app.core.Utils.Utils import get_project_root
                return str((_Path(get_project_root()) / path).resolve())
            except _MODERATION_NONCRITICAL_EXCEPTIONS:
                return str(path_value)

        blocklist_path = (
            mod_cfg.get("blocklist_file")
            or os.getenv("MODERATION_BLOCKLIST_FILE")
            or "tldw_Server_API/Config_Files/moderation_blocklist.txt"
        )
        user_overrides_path = (
            mod_cfg.get("user_overrides_file")
            or os.getenv("MODERATION_USER_OVERRIDES_FILE")
            or "tldw_Server_API/Config_Files/moderation_user_overrides.json"
        )
        runtime_overrides_path = (
            mod_cfg.get("runtime_overrides_file")
            or os.getenv("MODERATION_RUNTIME_OVERRIDES_FILE")
            or "tldw_Server_API/Config_Files/moderation_runtime_overrides.json"
        )

        with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
            self._max_scan_chars = int(mod_cfg.get("max_scan_chars", self._max_scan_chars))
        with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
            self._max_replacements_per_pattern = int(
                mod_cfg.get("max_replacements_per_pattern", self._max_replacements_per_pattern)
            )
        with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
            self._match_window_chars = int(mod_cfg.get("match_window_chars", self._match_window_chars))
        with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
            if "blocklist_write_debounce_ms" in mod_cfg:
                self._write_debounce_ms = int(mod_cfg.get("blocklist_write_debounce_ms", self._write_debounce_ms) or 0)

        cats_val = mod_cfg.get("categories_enabled") if "categories_enabled" in mod_cfg else None
        if cats_val is None:
            cats_val = os.getenv("MODERATION_CATEGORIES_ENABLED", "")
        categories_enabled: set[str] = set()
        if isinstance(cats_val, (list, set, tuple)):
            categories_enabled = {str(c).strip().lower() for c in cats_val if str(c).strip()}
        elif isinstance(cats_val, str) and cats_val.strip():
            categories_enabled = {c.strip().lower() for c in cats_val.split(",") if c.strip()}
        elif cats_val:
            logger.warning("Invalid moderation categories_enabled type")

        pii_enabled = is_truthy(
            str(mod_cfg.get("pii_enabled", os.getenv("MODERATION_PII_ENABLED", "false"))).strip().lower()
        )

        compiler_config = ResolvedModerationConfig(
            enabled=_b("enabled", False),
            input_enabled=_b("input_enabled", True),
            output_enabled=_b("output_enabled", True),
            input_action=str(mod_cfg.get("input_action", "block")).lower(),
            output_action=str(mod_cfg.get("output_action", "redact")).lower(),
            redact_replacement=str(mod_cfg.get("redact_replacement", "[REDACTED]")),
            per_user_overrides=_b("per_user_overrides", True),
            categories_enabled=categories_enabled or None,
            pii_enabled=bool(pii_enabled),
        )
        return _ResolvedModerationServiceState(
            compiler_config=compiler_config,
            blocklist_path=_anchor(str(blocklist_path)) if blocklist_path else None,
            user_overrides_path=_anchor(str(user_overrides_path)) if user_overrides_path else None,
            runtime_overrides_path=_anchor(str(runtime_overrides_path)) if runtime_overrides_path else None,
        )

    @staticmethod
    def _read_blocklist_lines_from_path(path: str) -> list[str]:
        with open(path, encoding="utf-8") as f:
            return [ln.rstrip("\r\n") for ln in f.readlines()]

    def _read_blocklist_lines_for_compile(self, path: str | None) -> list[str]:
        if not path:
            return []
        if not os.path.exists(path):
            logger.warning("Moderation blocklist file not found")
            return []
        try:
            return self._read_blocklist_lines_from_path(path)
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            logger.error("Failed to load moderation blocklist")
            return []

    def _compile_global_policy_from_resolved_config(self, config: ResolvedModerationConfig) -> ModerationPolicy:
        effective_pii_enabled = self._policy_compiler.resolve_runtime_pii(
            self._runtime_override,
            config.pii_enabled,
        )
        self._pii_enabled = bool(effective_pii_enabled)
        pii_rules = self._load_builtin_pii_rules() if effective_pii_enabled else []
        lines = self._read_blocklist_lines_for_compile(getattr(self, "_blocklist_path", None))
        result = self._policy_compiler.compile_global(
            PolicyCompilationInput(
                config=config,
                runtime_override=self._runtime_override,
                blocklist_lines=lines,
                pii_rules=pii_rules,
            )
        )
        self._log_compilation_report(result.report)
        return result.policy

    @staticmethod
    def _log_compilation_report(report: PolicyCompilationReport) -> None:
        for issue in report.issues:
            if issue.source == "user_rule" and issue.reason == "invalid_is_regex":
                logger.warning("Skipped per-user rule with invalid is_regex")
            elif issue.source == "user_rule" and issue.reason == "dangerous_regex":
                logger.warning("Skipped dangerous per-user regex rule")
            elif issue.source == "user_rule" and issue.reason == "invalid_regex":
                logger.warning("Skipped invalid per-user regex rule")
            elif issue.reason == "invalid_action":
                logger.warning("Invalid moderation action in blocklist; skipping line")
            elif issue.reason == "dangerous_regex":
                logger.warning("Skipped dangerous regex in blocklist")
            elif issue.reason == "invalid_regex":
                logger.warning("Invalid blocklist pattern; skipping line")

    def _parse_rule_line(self, s: str) -> tuple[str | None, str | None, str | None, set[str] | None]:
        return self._policy_compiler.parse_rule_line(s)

    @staticmethod
    def _split_action_directive(text: str) -> tuple[str, str | None]:
        return PolicyCompiler.split_action_directive(text)

    @classmethod
    def _parse_regex_expr(cls, expr: str) -> tuple[str, str] | None:
        return PolicyCompiler.parse_regex_expr(expr)

    def _load_block_patterns(self, path: str | None) -> list[PatternRule]:
        report = PolicyCompilationReport()
        lines = self._read_blocklist_lines_for_compile(path)
        rules = self._policy_compiler.compile_blocklist_lines(lines, report)
        self._log_compilation_report(report)
        return rules

    def _build_block_patterns(self, path: str | None) -> list[PatternRule]:
        """Load blocklist patterns and optionally append built-in PII rules."""
        patterns = self._load_block_patterns(path)
        if self._pii_enabled:
            try:
                pii_rules = self._load_builtin_pii_rules()
                if pii_rules:
                    patterns.extend(pii_rules)
            except _MODERATION_NONCRITICAL_EXCEPTIONS:
                logger.warning("Failed to load builtin PII rules")
        return patterns

    def _load_builtin_pii_rules(self) -> list[PatternRule]:
        """Create PatternRule list for common PII if available and enabled."""
        rules: list[PatternRule] = []
        try:
            from tldw_Server_API.app.core.Audit.unified_audit_service import PIIDetector
            for name, compiled in getattr(PIIDetector, 'PII_PATTERNS', {}).items():
                try:
                    # Ensure it's a compiled regex
                    if isinstance(compiled, re.Pattern):
                        rules.append(PatternRule(regex=compiled, action='redact', replacement='[PII]', categories={'pii', name}))
                except _MODERATION_NONCRITICAL_EXCEPTIONS:
                    continue
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            # Fallback minimal PII patterns
            try:
                basic = {
                    'pii_email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', re.IGNORECASE),
                    'pii_phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
                }
                for name, pat in basic.items():
                    rules.append(PatternRule(regex=pat, action='redact', replacement='[PII]', categories={'pii', name}))
            except _MODERATION_NONCRITICAL_EXCEPTIONS:
                return []
        return rules

    @staticmethod
    def _has_nested_quantifiers(expr: str) -> bool:
        """Heuristic check for nested quantifiers like (.*)+ or (.+)* that can cause catastrophic backtracking."""
        try:
            return bool(re.search(r"\((?:[^)(]|\([^)(]*\))*[+*][^)]*\)\s*[+*]", expr))
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            return False

    @staticmethod
    def _too_many_groups(expr: str, limit: int = 100) -> bool:
        try:
            return expr.count("(") - expr.count("\\(") > limit
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            return False

    def _is_regex_dangerous(self, expr: str) -> bool:
        return self._policy_compiler.is_regex_dangerous(expr)

    def _load_user_overrides(self) -> dict[str, dict[str, object]]:
        overrides: dict[str, dict[str, object]] = {}
        p = getattr(self, "_user_overrides_path", None)
        if not p:
            return overrides
        try:
            if not os.path.exists(p):
                logger.info("Moderation user overrides file not found (optional)")
                return overrides
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    cleaned: dict[str, dict[str, object]] = {}
                    for k, v in data.items():
                        if not isinstance(v, dict):
                            continue
                        cleaned[str(k)] = self._sanitize_user_override(v)
                    overrides = cleaned
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            logger.error("Failed to load user overrides")
        return overrides

    def reload(self) -> None:
        """Reload global config and overrides from disk."""
        with self._lock:
            self._config = load_and_log_configs() or {}
            self._global_policy = self._load_global_policy()
            # Load runtime overrides from file and re-apply
            try:
                self._load_runtime_overrides_file()
                self._global_policy = self._load_global_policy()
            except _MODERATION_NONCRITICAL_EXCEPTIONS:
                pass
            self._user_overrides = self._load_user_overrides()

    # --------------- Settings helpers (runtime) ---------------
    def get_settings(self) -> dict[str, object]:
        pol = self._global_policy
        pii_effective = False
        try:
            for rule in (pol.block_patterns or []):
                if not isinstance(rule, PatternRule):
                    continue
                if not rule.categories or "pii" not in rule.categories:
                    continue
                if pol.categories_enabled:
                    if rule.categories & pol.categories_enabled:
                        pii_effective = True
                        break
                else:
                    pii_effective = True
                    break
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            pii_effective = False
        cats_override: list[str] | None = None
        if "categories_enabled" in self._runtime_override:
            cats_val = self._runtime_override.get("categories_enabled") or []
            if isinstance(cats_val, (set, list, tuple)):
                cats_override = sorted([str(c) for c in cats_val])
            else:
                cats_override = [str(cats_val)]
        return {
            "pii_enabled": bool(self._runtime_override.get("pii_enabled", None)) if ("pii_enabled" in self._runtime_override) else None,
            "categories_enabled": cats_override,
            "effective": {
                "pii_enabled": pii_effective,
                "categories_enabled": sorted(pol.categories_enabled) if pol.categories_enabled else [],
            }
        }

    def update_settings(
        self,
        pii_enabled: bool | None = None,
        categories_enabled: list[str] | None = None,
        persist: bool = False,
        clear_pii: bool = False,
        clear_categories: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            next_override = dict(self._runtime_override)
            if clear_pii:
                next_override.pop("pii_enabled", None)
            elif pii_enabled is not None:
                next_override["pii_enabled"] = bool(pii_enabled)
            if clear_categories:
                next_override.pop("categories_enabled", None)
            elif categories_enabled is not None:
                cats = [str(c).strip().lower() for c in categories_enabled if str(c).strip()]
                next_override["categories_enabled"] = set(cats)
            if persist:
                try:
                    self._persist_runtime_overrides(next_override)
                except _MODERATION_NONCRITICAL_EXCEPTIONS:
                    logger.warning("Failed to persist moderation overrides (continuing in-memory)")
            self._runtime_override = next_override
            # Recompute policy with overrides
            self._global_policy = self._load_global_policy()
            return self.get_settings()

    def _load_runtime_overrides_file(self) -> None:
        path = self._runtime_overrides_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                ro: dict[str, object] = {}
                if "pii_enabled" in data:
                    raw_val = data.get("pii_enabled")
                    parsed = self._parse_bool_value(raw_val)
                    if parsed is None:
                        if raw_val is not None:
                            logger.warning("Invalid pii_enabled override value")
                    else:
                        ro["pii_enabled"] = parsed
                cats = data.get("categories_enabled")
                if isinstance(cats, list):
                    ro["categories_enabled"] = {str(c).strip().lower() for c in cats if str(c).strip()}
                elif isinstance(cats, str):
                    ro["categories_enabled"] = {c.strip().lower() for c in cats.split(',') if c.strip()}
                self._runtime_override = ro
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            logger.warning("Failed to load runtime overrides file")

    @staticmethod
    def _write_json_atomic(path: str, payload: object) -> None:
        """Atomically write JSON payload to ``path`` using a temporary file."""
        dirpath = os.path.dirname(os.path.abspath(path))
        if not dirpath:
            dirpath = "."
        os.makedirs(dirpath, exist_ok=True)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=dirpath,
                prefix=".moderation.",
                suffix=".tmp",
            ) as tmp:
                tmp_path = tmp.name
                json.dump(payload, tmp, indent=2, ensure_ascii=False)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
                    os.unlink(tmp_path)

    def _runtime_overrides_payload(self, overrides: dict[str, object]) -> dict[str, object]:
        payload: dict[str, object] = {}
        if "pii_enabled" in overrides:
            payload["pii_enabled"] = bool(overrides.get("pii_enabled"))
        if "categories_enabled" in overrides:
            cats = overrides.get("categories_enabled")
            if isinstance(cats, set):
                payload["categories_enabled"] = sorted(cats)
            elif isinstance(cats, (list, tuple)):
                payload["categories_enabled"] = list(cats)
        return payload

    def _persist_runtime_overrides(self, overrides: dict[str, object]) -> bool:
        path = self._runtime_overrides_path
        if not path:
            return False
        self._write_json_atomic(path, self._runtime_overrides_payload(overrides))
        return True

    def _persist_user_overrides(self, overrides: dict[str, dict[str, object]]) -> bool:
        path = getattr(self, "_user_overrides_path", None)
        if not path:
            return False
        self._write_json_atomic(path, overrides)
        return True

    def _save_runtime_overrides_file(self) -> None:
        path = self._runtime_overrides_path
        if not path:
            return
        try:
            self._persist_runtime_overrides(self._runtime_override)
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            logger.warning("Failed to save runtime overrides file")

    def get_effective_policy(self, user_id: str | None) -> ModerationPolicy:
        """Return policy after applying per-user overrides if enabled."""
        policy = self._global_policy
        if not policy.per_user_overrides or not user_id:
            return policy
        override = self._user_overrides.get(str(user_id))
        if not override:
            return policy
        result = self._policy_compiler.compile_user_policy(policy, override)
        self._log_compilation_report(result.report)
        return result.policy

    def _resolve_categories_override(
        self,
        overrides: dict[str, object],
        default_categories: set[str] | None,
    ) -> set[str] | None:
        return self._policy_compiler.resolve_categories_override(overrides, default_categories)

    @staticmethod
    def _parse_categories_override(v: object | None) -> set[str] | None:
        return PolicyCompiler.parse_categories_override(v)

    @classmethod
    def _is_valid_action(cls, action: str) -> bool:
        return str(action).strip().lower() in cls._ALLOWED_ACTIONS

    @staticmethod
    def _normalize_override_actions(override: dict[str, object]) -> dict[str, object]:
        out = dict(override or {})
        for key in ("input_action", "output_action"):
            if key in out and out[key] is not None:
                out[key] = str(out[key]).strip().lower()
        return out

    def _validate_override_actions(self, override: dict[str, object]) -> str | None:
        for key in ("input_action", "output_action"):
            if key in (override or {}) and override.get(key) is not None:
                val = str(override.get(key)).strip().lower()
                if val not in self._ALLOWED_ACTIONS:
                    return f"invalid {key}: {override.get(key)}"
        return None

    def _validate_override_rules_strict(self, override: dict[str, object]) -> str | None:
        """Validate per-user override rules for API writes.

        Returns a descriptive validation error string when invalid, else ``None``.
        """
        rules_raw = (override or {}).get("rules")
        if rules_raw is None:
            return None
        if not isinstance(rules_raw, list):
            return "invalid rules: expected a list"
        for idx, raw in enumerate(rules_raw):
            if not isinstance(raw, dict):
                return f"invalid rule at index {idx}: expected object"
            rule_id = str(raw.get("id", "")).strip()
            pattern = str(raw.get("pattern", "")).strip()
            action = str(raw.get("action", "")).strip().lower()
            phase = str(raw.get("phase", "both")).strip().lower()
            is_regex_raw = raw.get("is_regex", False)
            if not rule_id:
                return f"invalid rule id at index {idx}"
            if not pattern:
                return f"invalid rule pattern at index {idx}"
            if action not in {"block", "warn"}:
                return f"invalid rule action: {raw.get('action')}"
            if phase not in {"input", "output", "both"}:
                return f"invalid rule phase: {raw.get('phase')}"
            if "is_regex" in raw and not isinstance(is_regex_raw, bool):
                return f"invalid rule is_regex at index {idx}: expected boolean"
            if bool(is_regex_raw):
                if self._is_regex_dangerous(pattern):
                    return f"dangerous regex in rule: {rule_id}"
                try:
                    re.compile(pattern, flags=re.IGNORECASE)
                except re.error:
                    return f"invalid regex in rule: {rule_id}"
        return None

    def _sanitize_user_override(self, override: dict[str, object]) -> dict[str, object]:
        out = self._normalize_override_actions(override)
        for key in ("input_action", "output_action"):
            if key in out and out.get(key) is not None:
                val = str(out.get(key)).strip().lower()
                if val not in self._ALLOWED_ACTIONS:
                    logger.warning("Invalid moderation override action; dropping value")
                    out.pop(key, None)
        rules_raw = out.get("rules")
        if rules_raw is None:
            return out
        if not isinstance(rules_raw, list):
            out.pop("rules", None)
            return out
        normalized_rules: list[dict[str, object]] = []
        for raw in rules_raw:
            if not isinstance(raw, dict):
                continue
            rule_id = str(raw.get("id", "")).strip()
            pattern = str(raw.get("pattern", "")).strip()
            action = str(raw.get("action", "")).strip().lower()
            phase = str(raw.get("phase", "both")).strip().lower()
            parsed_is_regex = self._parse_bool_value(raw.get("is_regex", False))
            if parsed_is_regex is None:
                continue
            is_regex = parsed_is_regex
            if not rule_id or not pattern or action not in {"block", "warn"}:
                continue
            if phase not in {"input", "output", "both"}:
                continue
            if is_regex:
                if self._is_regex_dangerous(pattern):
                    continue
                try:
                    re.compile(pattern, flags=re.IGNORECASE)
                except re.error:
                    continue
            normalized_rules.append(
                {
                    "id": rule_id,
                    "pattern": pattern,
                    "is_regex": is_regex,
                    "action": action,
                    "phase": phase,
                }
            )
        if not normalized_rules and rules_raw:
            logger.warning("Dropped invalid moderation override rules during sanitize")
        out["rules"] = normalized_rules
        return out

    def _compile_user_rule(self, raw_rule: object) -> PatternRule | None:
        """Compile a per-user override rule into a PatternRule."""
        report = PolicyCompilationReport()
        compiled = self._policy_compiler.compile_user_rule(raw_rule, report, 0)
        self._log_compilation_report(report)
        return compiled

    @classmethod
    def _effective_rule_categories(cls, rule: PatternRule) -> set[str]:
        cats = rule.categories or set()
        normalized = {str(c).strip().lower() for c in cats if str(c).strip()}
        return normalized if normalized else {cls._UNCATEGORIZED_CATEGORY}

    @staticmethod
    def _rule_applies_to_phase(rule: PatternRule, phase: str | None) -> bool:
        """Return whether a rule should run for the requested moderation phase."""
        if phase not in {"input", "output"}:
            return True
        rule_phase = str(getattr(rule, "phase", "both") or "both").strip().lower()
        if rule_phase not in {"input", "output", "both"}:
            rule_phase = "both"
        return rule_phase in {"both", phase}

    @classmethod
    def _rule_matches_enabled_categories(
        cls,
        rule: PatternRule,
        categories_enabled: set[str] | None,
    ) -> bool:
        """Return whether a rule is allowed by the active category filter."""
        if not categories_enabled:
            return True
        if "*" in categories_enabled:
            return True
        rcats = cls._effective_rule_categories(rule)
        if "*" in rcats:
            return True
        return bool(rcats & categories_enabled)

    def effective_policy_snapshot(self, user_id: str | None) -> dict[str, object]:
        """Return a serializable dict of the effective policy for inspection."""
        return self.get_effective_policy(user_id).to_dict()

    @staticmethod
    def _coalesce_bool(v: str | bool | None, default: bool) -> bool:
        return PolicyCompiler.coalesce_bool(v, default)

    @staticmethod
    def _parse_bool_value(v: object) -> bool | None:
        return PolicyCompiler.parse_bool_value(v)

    # --------------- Checking and transformations ---------------
    def check_text(self, text: str, policy: ModerationPolicy, phase: str | None = None) -> tuple[bool, str | None]:
        """Return (is_flagged, matched_sample)."""
        result = self._evaluate_text_core(text, policy, phase, include_redacted_text=False)
        return result.action != "pass", result.sample

    @staticmethod
    def _build_sanitized_snippet(text: str, match_span: tuple[int, int], replacement: str) -> str | None:
        if not text or not match_span:
            return None
        start, end = match_span
        if start < 0:
            start = 0
        if end < start:
            end = start
        if start > len(text):
            start = len(text)
        if end > len(text):
            end = len(text)
        left_start = max(0, start - 16)
        right_end = min(len(text), end + 16)
        left = text[left_start:start]
        right = text[end:right_end]
        snippet = (left + (replacement or "[REDACTED]") + right).strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        return snippet

    def build_sanitized_snippet(
        self,
        text: str,
        policy: ModerationPolicy,
        match_span: tuple[int, int] | None,
        pattern: str | None = None,
    ) -> str | None:
        """Create a sanitized snippet for a known match span and pattern."""
        if not text or not match_span:
            return None
        replacement = policy.redact_replacement or "[REDACTED]"
        if pattern and policy.block_patterns:
            for rule in policy.block_patterns:
                if not isinstance(rule, PatternRule):
                    continue
                try:
                    if getattr(rule.regex, "pattern", None) == pattern:
                        if rule.replacement:
                            replacement = rule.replacement
                        break
                except _MODERATION_NONCRITICAL_EXCEPTIONS:
                    continue
        return self._build_sanitized_snippet(text, match_span, replacement)

    def redact_text(self, text: str, policy: ModerationPolicy, phase: str | None = None) -> str:
        if not text or not policy.block_patterns:
            return text
        if phase == "input" and not policy.input_enabled:
            return text
        if phase == "output" and not policy.output_enabled:
            return text
        redacted = text
        for rule in policy.block_patterns:
            if isinstance(rule, PatternRule) and not self._rule_applies_to_phase(rule, phase):
                continue
            # Respect category gating similar to evaluate_action/check_text
            if isinstance(rule, PatternRule) and not self._rule_matches_enabled_categories(rule, policy.categories_enabled):
                continue
            pat = rule.regex if isinstance(rule, PatternRule) else rule
            repl = None
            if isinstance(rule, PatternRule) and rule.replacement:
                repl = rule.replacement
            try:
                replacement = repl or policy.redact_replacement
                limit_raw = self._max_replacements_per_pattern
                try:
                    limit_int = int(limit_raw) if limit_raw is not None else 0
                except _MODERATION_NONCRITICAL_EXCEPTIONS:
                    limit_int = 0
                # Treat non-positive values as unlimited (re.sub uses 0 for no limit)
                if limit_int <= 0:
                    limit_int = 0
                if len(redacted) <= self._max_scan_chars:
                    redacted = pat.sub(lambda _m, _r=replacement: _r, redacted, count=limit_int)
                else:
                    matches = self._collect_rule_matches(redacted, pat)
                    if matches:
                        redacted = self._apply_rule_redactions(redacted, matches, replacement)
            except re.error:
                # in case of unexpected regex issue, skip
                continue
        return redacted

    def redact_text_with_count(self, text: str, policy: ModerationPolicy, phase: str | None = None) -> tuple[str, int]:
        """Redact text and return (redacted_text, replacement_count)."""
        if not text or not policy.block_patterns:
            return text, 0
        if phase == "input" and not policy.input_enabled:
            return text, 0
        if phase == "output" and not policy.output_enabled:
            return text, 0
        redacted = text
        total_count = 0
        for rule in policy.block_patterns:
            if isinstance(rule, PatternRule) and not self._rule_applies_to_phase(rule, phase):
                continue
            # Respect category gating similar to evaluate_action/check_text
            if isinstance(rule, PatternRule) and not self._rule_matches_enabled_categories(rule, policy.categories_enabled):
                continue
            pat = rule.regex if isinstance(rule, PatternRule) else rule
            repl = None
            if isinstance(rule, PatternRule) and rule.replacement:
                repl = rule.replacement
            try:
                replacement = repl or policy.redact_replacement
                limit_raw = self._max_replacements_per_pattern
                try:
                    limit_int = int(limit_raw) if limit_raw is not None else 0
                except _MODERATION_NONCRITICAL_EXCEPTIONS:
                    limit_int = 0
                # Treat non-positive values as unlimited (re.sub uses 0 for no limit)
                if limit_int <= 0:
                    limit_int = 0
                if len(redacted) <= self._max_scan_chars:
                    redacted, count = pat.subn(lambda _m, _r=replacement: _r, redacted, count=limit_int)
                else:
                    matches = self._collect_rule_matches(redacted, pat)
                    count = len(matches)
                    if matches:
                        redacted = self._apply_rule_redactions(redacted, matches, replacement)
                total_count += count
            except re.error:
                # in case of unexpected regex issue, skip
                continue
        return redacted, total_count

    # --------------- Decision helpers ---------------
    def evaluate_text(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None = None,
    ) -> ModerationEvaluationResult:
        """Compute the canonical moderation result for text and a policy."""
        return self._evaluate_text_core(text, policy, phase, include_redacted_text=True)

    def _evaluate_text_core(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None,
        *,
        include_redacted_text: bool,
    ) -> ModerationEvaluationResult:
        """Shared moderation evaluation logic for probes and full result generation."""
        if not text:
            return ModerationEvaluationResult()
        if not policy.enabled:
            return ModerationEvaluationResult()
        enabled_phase = True
        if phase == "input":
            enabled_phase = policy.input_enabled
        elif phase == "output":
            enabled_phase = policy.output_enabled
        if not enabled_phase:
            return ModerationEvaluationResult()
        default_action = "warn"
        if phase == "input":
            default_action = policy.input_action
        elif phase == "output":
            default_action = policy.output_action
        best_action = "pass"
        best_rank = 0
        best_pattern = None
        best_category = None
        best_match_pos = None
        best_match_span: tuple[int, int] | None = None
        best_replacement: str | None = None
        for rule in policy.block_patterns or []:
            pat = rule.regex if isinstance(rule, PatternRule) else rule
            if isinstance(rule, PatternRule) and not self._rule_applies_to_phase(rule, phase):
                continue
            # Category gating
            if isinstance(rule, PatternRule) and not self._rule_matches_enabled_categories(rule, policy.categories_enabled):
                continue
            match_span = self._find_match_span(pat, text)
            if not match_span:
                continue
            # Prefer rule action if specified, else global
            action = None
            action = rule.action if isinstance(rule, PatternRule) and rule.action else default_action
            action = (action or 'warn').lower()
            if action not in {"block", "redact", "warn"}:
                action = "warn"
            rank = {"warn": 1, "redact": 2, "block": 3}.get(action, 1)
            match_pos = match_span[0]
            if rank > best_rank or (rank == best_rank and (best_match_pos is None or match_pos < best_match_pos)):
                best_action = action
                best_rank = rank
                best_match_pos = match_pos
                best_match_span = match_span
                best_pattern = pat.pattern
                if isinstance(rule, PatternRule) and rule.replacement:
                    best_replacement = rule.replacement
                else:
                    best_replacement = policy.redact_replacement
                if isinstance(rule, PatternRule):
                    try:
                        cats = self._effective_rule_categories(rule)
                        if policy.categories_enabled:
                            cats = cats & set(policy.categories_enabled)
                        if cats:
                            if "pii" in cats and len(cats) > 1:
                                cats = {c for c in cats if c != "pii"}
                            best_category = sorted(cats)[0]
                        else:
                            best_category = None
                    except _MODERATION_NONCRITICAL_EXCEPTIONS:
                        best_category = None
                else:
                    best_category = None
        if best_action == "pass" or best_match_span is None:
            return ModerationEvaluationResult()
        sanitized_sample = self._build_sanitized_snippet(
            text,
            best_match_span,
            best_replacement or policy.redact_replacement or "[REDACTED]",
        )
        redacted_text = None
        if include_redacted_text and best_action == "redact":
            redacted_text = self.redact_text(text, policy, phase=phase)
        return ModerationEvaluationResult(
            action=best_action,
            redacted_text=redacted_text,
            matched_pattern=best_pattern,
            category=best_category,
            match_span=best_match_span,
            sample=sanitized_sample,
        )

    def _evaluate_action_internal(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None,
    ) -> tuple[str, str | None, str | None, str | None, tuple[int, int] | None]:
        """Compatibility wrapper around evaluate_text()."""
        result = self.evaluate_text(text, policy, phase)
        return result.action, result.redacted_text, result.matched_pattern, result.category, result.match_span

    def evaluate_action(self, text: str, policy: ModerationPolicy, phase: str) -> tuple[str, str | None, str | None, str | None]:
        """Decide the action for a given text and phase."""
        result = self.evaluate_text(text, policy, phase)
        return result.action, result.redacted_text, result.matched_pattern, result.category

    def evaluate_action_with_match(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str,
    ) -> tuple[str, str | None, str | None, str | None, tuple[int, int] | None]:
        """Decide action and return the match span when available."""
        result = self.evaluate_text(text, policy, phase)
        return result.action, result.redacted_text, result.matched_pattern, result.category, result.match_span

    def _iter_scan_chunks(self, text: str) -> Iterator[tuple[int, int]]:
        if not text:
            return
        chunk_size = max(1, int(self._max_scan_chars))
        if len(text) <= chunk_size:
            yield 0, len(text)
            return
        overlap = min(1024, max(32, chunk_size // 10))
        if overlap >= chunk_size:
            overlap = max(0, chunk_size - 1)
        step = chunk_size - overlap if chunk_size > overlap else chunk_size
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(text_len, start + chunk_size)
            yield start, end
            if end == text_len:
                break
            start += step

    def _find_match_span(self, pat: re.Pattern, text: str) -> tuple[int, int] | None:
        try:
            chunk_limit = max(1, int(self._max_scan_chars))
            if len(text) <= chunk_limit:
                m = pat.search(text)
                if not m:
                    return None
                return m.start(), m.end()
            text_len = len(text)
            window = max(0, int(self._match_window_chars))
            for start, end in self._iter_scan_chunks(text):
                window_end = min(text_len, end + window)
                m = pat.search(text, start, window_end)
                if not m:
                    continue
                if m.start() < end:
                    return m.start(), m.end()
            # Full-text fallback with configurable length guardrail to
            # mitigate ReDoS risk on very large inputs.
            fallback_limit = max(1, int(self._max_fallback_scan_chars))
            if len(text) <= fallback_limit:
                m = pat.search(text)
                if m:
                    return m.start(), m.end()
            return None
        except re.error:
            return None

    def _collect_rule_matches(self, text: str, pat: re.Pattern) -> list[re.Match]:
        """Collect non-overlapping matches across scan chunks for soft-capped redaction."""
        if not text:
            return []
        limit = self._max_replacements_per_pattern
        if limit is not None and int(limit) <= 0:
            limit = None
        matches: list[re.Match] = []
        try:
            for m in pat.finditer(text):
                span = m.span()
                if span[0] == span[1]:
                    continue
                matches.append(m)
                if limit is not None and len(matches) >= limit:
                    break
        except re.error:
            return []
        return matches

    @staticmethod
    def _apply_rule_redactions(text: str, matches: list[re.Match], replacement: str) -> str:
        """Apply redactions using precomputed match objects."""
        if not matches:
            return text
        out_parts: list[str] = []
        last = 0
        for m in matches:
            start, end = m.span()
            if start < last:
                continue
            out_parts.append(text[last:start])
            out_parts.append(replacement)
            last = end
        out_parts.append(text[last:])
        return "".join(out_parts)

    # --------------- Persistence helpers ---------------
    def list_user_overrides(self) -> dict[str, dict[str, object]]:
        """Return a shallow copy of all user overrides."""
        return dict(self._user_overrides or {})

    def set_user_override(self, user_id: str, override: dict[str, object]) -> dict[str, object]:
        """Create or update a user override and persist to file if configured.

        Returns a dict ``{ok, persisted, error?, error_type?}``, where
        ``error_type`` is ``validation`` or ``persistence`` when ``ok`` is false.
        """
        if not user_id:
            return {"ok": False, "persisted": False, "error": "user_id required", "error_type": "validation"}
        err = self._validate_override_actions(override)
        if err:
            return {"ok": False, "persisted": False, "error": err, "error_type": "validation"}
        rule_err = self._validate_override_rules_strict(override)
        if rule_err:
            return {"ok": False, "persisted": False, "error": rule_err, "error_type": "validation"}
        with self._lock:
            normalized = self._sanitize_user_override(override)
            next_overrides = {str(k): dict(v) for k, v in self._user_overrides.items()}
            next_overrides[str(user_id)] = {str(k): v for k, v in normalized.items()}
            path = getattr(self, "_user_overrides_path", None)
            if not path:
                self._user_overrides = next_overrides
                logger.warning("User override path not configured; changes will not persist across restarts")
                return {"ok": True, "persisted": False}
            try:
                self._persist_user_overrides(next_overrides)
                self._user_overrides = next_overrides
                logger.info(f"Saved moderation user overrides to {path}")
                return {"ok": True, "persisted": True}
            except _MODERATION_NONCRITICAL_EXCEPTIONS as e:
                logger.error("Failed to save user overrides")
                return {
                    "ok": False,
                    "persisted": False,
                    "error": "Failed to persist user override.",
                    "error_type": "persistence",
                }

    def delete_user_override(self, user_id: str) -> dict[str, object]:
        """Delete a user override and persist to file if configured.

        Returns a dict {ok: bool, persisted: bool, error?: str}
        """
        with self._lock:
            key = str(user_id)
            if key in self._user_overrides:
                next_overrides = {str(k): dict(v) for k, v in self._user_overrides.items()}
                next_overrides.pop(key, None)
                path = getattr(self, "_user_overrides_path", None)
                try:
                    if path:
                        self._persist_user_overrides(next_overrides)
                        self._user_overrides = next_overrides
                        return {"ok": True, "persisted": True}
                    self._user_overrides = next_overrides
                    return {"ok": True, "persisted": False}
                except _MODERATION_NONCRITICAL_EXCEPTIONS as e:
                    logger.error("Failed to persist user override deletion")
                    return {
                        "ok": False,
                        "persisted": False,
                        "error": "Failed to delete user override.",
                        "error_type": "persistence",
                    }
            return {"ok": False, "persisted": False, "error": "not found"}

    def get_blocklist_lines(self) -> list[str]:
        """Read current blocklist file lines (without trailing newlines)."""
        path = getattr(self, "_blocklist_path", None)
        if not path or not os.path.exists(path):
            return []
        try:
            with self._lock, open(path, encoding="utf-8") as f:
                return [ln.rstrip("\r\n") for ln in f.readlines()]
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            logger.error("Failed to read blocklist")
            return []

    def set_blocklist_lines(self, lines: list[str]) -> bool:
        """Write blocklist lines to file and reload compiled patterns."""
        path = getattr(self, "_blocklist_path", None)
        if not path:
            logger.warning("Blocklist path not configured; cannot persist blocklist")
            return False
        try:
            with self._lock:
                # Optional debounce to coalesce bursts of writes
                if self._write_debounce_ms and self._write_debounce_ms > 0:
                    now = time.monotonic()
                    min_interval = float(self._write_debounce_ms) / 1000.0
                    elapsed = now - (self._last_blocklist_write or 0.0)
                    if elapsed < min_interval:
                        time.sleep(max(0.0, min_interval - elapsed))
                dirpath = os.path.dirname(os.path.abspath(path))
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                # Normalize line endings; ensure trailing newline for POSIX friendliness
                text = "\n".join(lines).rstrip("\n") + "\n" if lines else ""
                tmp_path = None
                try:
                    tmp_dir = dirpath if dirpath else None
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        delete=False,
                        dir=tmp_dir,
                        prefix=".moderation_blocklist.",
                        suffix=".tmp",
                    ) as tmp:
                        tmp.write(text)
                        tmp_path = tmp.name
                    os.replace(tmp_path, path)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
                            os.unlink(tmp_path)
                # Reload patterns (preserve built-in PII rules when enabled)
                self._global_policy.block_patterns = self._build_block_patterns(path)
                logger.info(f"Updated moderation blocklist at {path} ({len(lines)} lines)")
                # Record write time after successful write
                if self._write_debounce_ms and self._write_debounce_ms > 0:
                    self._last_blocklist_write = time.monotonic()
                return True
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            logger.error("Failed to write blocklist")
            return False

    # --------------- Managed blocklist with versioning ---------------
    @staticmethod
    def _normalize_lines(lines: list[str]) -> list[str]:
        return [str(ln).rstrip("\r\n") for ln in (lines or [])]

    @staticmethod
    def _compute_version(lines: list[str]) -> str:
        """Compute a stable version string (ETag) for the blocklist content."""
        norm = ModerationService._normalize_lines(lines)
        payload = ("\n".join(norm) + "\n").encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get_blocklist_state(self) -> dict[str, object]:
        """Return current blocklist with a content hash version and indexed items."""
        lines = self.get_blocklist_lines()
        version = self._compute_version(lines)
        items = [{"id": i, "line": ln} for i, ln in enumerate(lines)]
        return {"version": version, "items": items}

    def append_blocklist_line(self, expected_version: str, line: str) -> tuple[bool, dict[str, object]]:
        """Append a line with optimistic concurrency control. Returns (ok, state)."""
        if line is None:
            return False, {"error": "line required"}
        line_text = str(line)
        if "\n" in line_text or "\r" in line_text:
            return False, {"error": "line must be single-line"}
        with self._lock:
            current = self.get_blocklist_lines()
            cur_version = self._compute_version(current)
            if expected_version and cur_version != expected_version:
                return False, {"version": cur_version, "conflict": True}
            new_lines = current + [line_text.rstrip("\n")]
            ok = self.set_blocklist_lines(new_lines)
            state = self.get_blocklist_state() if ok else {"error": "persist failed"}
            return ok, state

    def delete_blocklist_index(self, expected_version: str, index: int) -> tuple[bool, dict[str, object]]:
        """Delete a line by index with optimistic concurrency control. Returns (ok, state)."""
        with self._lock:
            current = self.get_blocklist_lines()
            cur_version = self._compute_version(current)
            if expected_version and cur_version != expected_version:
                return False, {"version": cur_version, "conflict": True}
            if index < 0 or index >= len(current):
                return False, {"error": "index out of range", "count": len(current)}
            new_lines = current[:index] + current[index+1:]
            ok = self.set_blocklist_lines(new_lines)
            state = self.get_blocklist_state() if ok else {"error": "persist failed"}
            return ok, state

    # --------------- Lint helpers ---------------
    def lint_blocklist_lines(self, lines: list[str]) -> dict[str, object]:
        """Validate blocklist lines without persisting.

        Returns a dict with items [{index, line, ok, pattern_type, action, replacement, categories, error?, warning?, sample?}]
        and summary counts.
        """
        results: list[dict[str, object]] = []
        valid_count = 0
        invalid_count = 0
        for idx, raw in enumerate(lines or []):
            line = str(raw).rstrip("\n")
            item: dict[str, object] = {"index": idx, "line": line, "ok": False}
            try:
                if not line or not line.strip():
                    item.update({"ok": True, "pattern_type": "empty", "warning": "blank line (ignored)"})
                    results.append(item)
                    valid_count += 1
                    continue
                if line.lstrip().startswith("#"):
                    item.update({"ok": True, "pattern_type": "comment", "warning": "comment (ignored)"})
                    results.append(item)
                    valid_count += 1
                    continue
                expr, action, repl, cats = self._parse_rule_line(line)
                if expr is None or expr == "":
                    item.update({"ok": False, "error": "empty pattern after parsing"})
                    results.append(item)
                    invalid_count += 1
                    continue
                if action and not self._is_valid_action(action):
                    item.update({"ok": False, "error": f"invalid action: {action}"})
                    results.append(item)
                    invalid_count += 1
                    continue
                if not cats:
                    cats = {self._UNCATEGORIZED_CATEGORY}
                # Recognize /regex/flags form as regex
                regex_parts = self._parse_regex_expr(expr)
                is_regex = regex_parts is not None
                invalid_flags_warning = None
                if not is_regex and expr.startswith("/") and expr.rfind("/") > 0:
                    last_slash = expr.rfind("/")
                    flags_part = expr[last_slash + 1:]
                    if flags_part and flags_part.isalpha() and len(flags_part) <= len(self._ALLOWED_REGEX_FLAGS):
                        fs = flags_part.lower()
                        if any(ch not in self._ALLOWED_REGEX_FLAGS for ch in fs):
                            invalid_flags_warning = "invalid regex flags; treating as literal"
                item.update({
                    "action": action,
                    "replacement": repl,
                    "categories": sorted(cats) if cats else [],
                })
                if is_regex:
                    raw_pat, flags_part = regex_parts
                    if self._is_regex_dangerous(raw_pat):
                        item.update({"ok": False, "pattern_type": "regex", "error": "dangerous regex (nested quantifiers/too complex)"})
                        results.append(item)
                        invalid_count += 1
                        continue
                    try:
                        flags = re.IGNORECASE
                        flags_str = (flags_part or "").lower()
                        if 'i' in flags_str:
                            flags |= re.IGNORECASE
                        if 'm' in flags_str:
                            flags |= re.MULTILINE
                        if 's' in flags_str:
                            flags |= re.DOTALL
                        if 'x' in flags_str:
                            flags |= re.VERBOSE
                        re.compile(raw_pat, flags=flags)
                    except re.error as e:
                        item.update({"ok": False, "pattern_type": "regex", "error": f"invalid regex: {e}"})
                        results.append(item)
                        invalid_count += 1
                        continue
                    item.update({"ok": True, "pattern_type": "regex", "sample": raw_pat})
                    valid_count += 1
                else:
                    # For literal samples, present unescaped '#'
                    item.update({"ok": True, "pattern_type": "literal", "sample": expr.replace("\\#", "#")})
                    if invalid_flags_warning:
                        item["warning"] = invalid_flags_warning
                    valid_count += 1
                results.append(item)
            except _MODERATION_NONCRITICAL_EXCEPTIONS as e:
                item.update({"ok": False, "error": str(e)})
                results.append(item)
                invalid_count += 1
        return {"items": results, "valid_count": valid_count, "invalid_count": invalid_count}


# Singleton accessor
_moderation_service: ModerationService | None = None


def get_moderation_service() -> ModerationService:
    global _moderation_service
    if _moderation_service is None:
        _moderation_service = ModerationService()
    return _moderation_service
