from __future__ import annotations

"""
Topic Monitoring Service

Detects configured topics in text and emits non-blocking alerts for admins/owners.

Design goals:
- Local, offline-first; regex/literal patterns
- Safe regex compilation (reuse heuristics from moderation)
- Simple per-user scoping in Phase 1
- Bounded scanning and dedup window to avoid alert spam
"""

import asyncio
import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.monitoring_schemas import Watchlist, WatchlistRule
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import (
    TopicAlert,
    TopicMonitoringDB,
    WatchlistRecord,
    WatchlistRuleRecord,
)
from tldw_Server_API.app.core.Monitoring.notification_service import get_notification_service
from tldw_Server_API.app.core.testing import is_truthy


@dataclass
class CompiledRule:
    rule_id: str
    regex: re.Pattern
    category: str | None
    severity: str | None
    pattern_text: str  # original pattern or regex body
    note: str | None = None


def _find_project_root(start: Path) -> Path | None:
    """Best-effort search for the repository root starting from a file/dir path."""
    start_dir = start if start.is_dir() else start.parent

    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
        if (candidate / "AGENTS.md").is_file() and (candidate / "tldw_Server_API").is_dir():
            return candidate
        if (candidate / ".git").exists():
            return candidate
    return None


_TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


class TopicMonitoringService:
    _ALLOWED_SEVERITIES = {"info", "warning", "critical"}
    _ALLOWED_SCOPE_TYPES = {"global", "all", "user", "team", "org"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._config = load_and_log_configs() or {}
        monitoring_cfg = (
            self._config.get("monitoring") if isinstance(self._config, dict) else None
        ) or {}
        self._enabled = self._resolve_enabled(monitoring_cfg)
        self._max_scan_chars = self._resolve_max_scan_chars()
        self._dedup_window_seconds = self._coerce_int(
            os.getenv("TOPIC_MONITOR_DEDUP_SECONDS", monitoring_cfg.get("dedup_seconds", 300)),
            300,
        )
        self._simhash_distance = self._coerce_int(
            os.getenv("TOPIC_MONITOR_SIMHASH_DISTANCE", monitoring_cfg.get("simhash_distance", 3)),
            3,
        )
        self._watchlists_path, self._db_path = self._resolve_paths(monitoring_cfg)
        self._db = TopicMonitoringDB(db_path=self._db_path)
        self._watchlists: dict[str, Watchlist] = {}
        self._compiled: dict[str, list[CompiledRule]] = {}
        self._dedupe_state: dict[str, dict[str, deque[tuple[float, int]]]] = {}
        self._dedupe_stream_last_seen: dict[str, float] = {}
        self._dedupe_last_cleanup: float = 0.0
        self._seed_watchlists_from_file()
        self._load_watchlists_from_db()

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _resolve_enabled(self, monitoring_cfg: dict[str, Any]) -> bool:
        raw = monitoring_cfg.get("enabled") if isinstance(monitoring_cfg, dict) else None
        if raw is None:
            raw = os.getenv("MONITORING_ENABLED", "false")
        return is_truthy(str(raw))

    def _resolve_max_scan_chars(self) -> int:
        default = 200000
        mod_cfg = self._config.get("moderation") if isinstance(self._config, dict) else None
        if isinstance(mod_cfg, dict) and "max_scan_chars" in mod_cfg:
                try:
                    return int(mod_cfg.get("max_scan_chars", default))
                except (TypeError, ValueError):
                    pass
        raw = os.getenv("MODERATION_MAX_SCAN_CHARS")
        if raw is None:
            raw = os.getenv("TOPIC_MONITOR_MAX_SCAN_CHARS", str(default))
        return self._coerce_int(raw, default)

    def _resolve_paths(self, monitoring_cfg: dict[str, Any]) -> tuple[str, str]:
        configured_watchlists = monitoring_cfg.get("watchlists_file") if isinstance(monitoring_cfg, dict) else None
        if configured_watchlists:
            raw_watchlists = configured_watchlists
            watchlists_source = "monitoring.watchlists_file"
        else:
            raw_watchlists = os.getenv(
                "MONITORING_WATCHLISTS_FILE",
                "tldw_Server_API/Config_Files/monitoring_watchlists.json",
            )
            watchlists_source = "MONITORING_WATCHLISTS_FILE"
        raw_db_path = os.getenv("MONITORING_ALERTS_DB", "Databases/monitoring_alerts.db")
        # Anchor relative paths to project root to avoid creating dirs from CWD
        try:
            wl_p = Path(str(raw_watchlists))
        except (TypeError, ValueError):
            wl_p = Path("tldw_Server_API/Config_Files/monitoring_watchlists.json")
        try:
            db_p = Path(str(raw_db_path))
        except (TypeError, ValueError):
            db_p = Path("Databases/monitoring_alerts.db")

        if not wl_p.is_absolute() and wl_p.parent == Path("."):
            msg = (
                f"{watchlists_source} must include a directory when using a relative path "
                f"(got {raw_watchlists!r}). Use an absolute path or include a directory component."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if not db_p.is_absolute() and db_p.parent == Path("."):
            msg = (
                "MONITORING_ALERTS_DB must include a directory when using a relative path "
                f"(got {raw_db_path!r}). Use an absolute path or include a directory component."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if not wl_p.is_absolute() or not db_p.is_absolute():
            try:
                from tldw_Server_API.app.core.Utils.Utils import get_project_root as _gpr
            except Exception as exc:
                start = Path(__file__).resolve()
                root = _find_project_root(start)
                if root is None:
                    msg = (
                        "Unable to resolve monitoring paths: watchlists/db are relative, "
                        f"and importing get_project_root failed: {exc}. "
                        f"Searched parents of {start} for root markers (pyproject.toml, AGENTS.md, .git) "
                        "but none were found."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg) from exc
                logger.debug(
                    "monitoring: get_project_root unavailable ({}); using fallback root {}",
                    exc,
                    root,
                )
            else:
                root = Path(_gpr()).resolve()

            if not wl_p.is_absolute():
                wl_p = (root / wl_p).resolve()
            if not db_p.is_absolute():
                db_p = (root / db_p).resolve()
        return str(wl_p), str(db_p)

    # --------------------- File I/O (seed/import) ---------------------
    def _load_watchlists_file(self) -> list[Watchlist] | None:
        try:
            if not self._watchlists_path or not os.path.exists(self._watchlists_path):
                logger.info(
                    'Topic monitoring: watchlists file not found at {}, skipping seed',
                    self._watchlists_path,
                )
                return None
            with open(self._watchlists_path, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("watchlists") if isinstance(data, dict) else data
            if not isinstance(raw, list):
                raw = []
            watchlists: list[Watchlist] = []
            for it in raw:
                try:
                    wl = Watchlist(**it)
                    watchlists.append(wl)
                except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Skipping invalid watchlist entry: {e}")
            return watchlists
        except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to load watchlists: {e}")
            return None

    @staticmethod
    def _normalize_scope(scope_type: str, scope_id: str | None) -> tuple[str, str | None]:
        st = (scope_type or "user").strip().lower()
        if st not in TopicMonitoringService._ALLOWED_SCOPE_TYPES:
            raise ValueError(f"Unsupported scope_type '{scope_type}'")
        if st in {"all", "global"}:
            return "global", None
        return st, str(scope_id) if scope_id is not None else None

    @staticmethod
    def _normalize_tags(tags: Iterable[str] | None) -> list[str]:
        if tags is None:
            return []
        cleaned = {str(tag).strip() for tag in tags if tag is not None and str(tag).strip()}
        return sorted(cleaned)

    @staticmethod
    def _compute_rule_id(rule: WatchlistRule) -> str:
        payload = {
            "pattern": str(rule.pattern or ""),
            "category": str(rule.category or ""),
            "severity": str(rule.severity or "info"),
            "note": str(rule.note or ""),
            "tags": TopicMonitoringService._normalize_tags(rule.tags),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return digest

    def _coerce_watchlist_record(self, wl: Watchlist) -> WatchlistRecord:
        scope_type, scope_id = self._normalize_scope(wl.scope_type, wl.scope_id)
        managed_by = (wl.managed_by or "api").strip().lower()
        return WatchlistRecord(
            id=str(wl.id),
            name=wl.name,
            description=wl.description,
            enabled=bool(wl.enabled),
            scope_type=scope_type,
            scope_id=scope_id,
            managed_by=managed_by,
        )

    def _coerce_rule_records(
        self,
        wl_id: str,
        rules: list[WatchlistRule],
        *,
        strict: bool = False,
    ) -> list[WatchlistRuleRecord]:
        out: list[WatchlistRuleRecord] = []
        seen: set[str] = set()
        for rule in rules:
            severity = (rule.severity or "info").strip().lower()
            if severity not in self._ALLOWED_SEVERITIES:
                if strict:
                    raise ValueError(f"Invalid rule severity '{rule.severity}'")
                logger.warning("Topic monitoring: skipped rule with invalid severity '{}'", rule.severity)
                continue
            pattern = str(rule.pattern or "").strip()
            if not pattern:
                logger.warning("Topic monitoring: skipped rule with empty pattern")
                continue
            rule_id = str(rule.rule_id) if rule.rule_id else self._compute_rule_id(rule)
            if rule_id in seen:
                continue
            seen.add(rule_id)
            out.append(
                WatchlistRuleRecord(
                    rule_id=rule_id,
                    watchlist_id=str(wl_id),
                    pattern=pattern,
                    category=rule.category,
                    severity=severity,
                    note=rule.note,
                    tags=self._normalize_tags(rule.tags),
                )
            )
        return out

    def _seed_watchlists_from_file(
        self,
        *,
        delete_missing: bool = False,
        disable_missing: bool = False,
        include_unmanaged: bool = False,
    ) -> None:
        watchlists = self._load_watchlists_file()
        if watchlists is None:
            if delete_missing or disable_missing:
                logger.info("Topic monitoring: seed file missing; skipping delete/disable operations")
            return
        active_ids: set[str] = set()
        for wl in watchlists:
            try:
                scope_type, scope_id = self._normalize_scope(wl.scope_type, wl.scope_id)
                if scope_type != "global" and not scope_id:
                    logger.warning("Topic monitoring: skipping watchlist with missing scope_id: {}", wl.name)
                    continue
                wl.scope_type = scope_type
                wl.scope_id = scope_id
                wl.managed_by = "config"
                existing = None
                if wl.id:
                    existing = self._db.get_watchlist(str(wl.id))
                if existing is None:
                    existing = self._db.get_watchlist_by_key(wl.name, scope_type, scope_id)
                if existing:
                    if not include_unmanaged and str(existing.get("managed_by")) != "config":
                        logger.info(
                            'Topic monitoring: skipping seed update for unmanaged watchlist {}',
                            existing.get("id"),
                        )
                        continue
                    wl.id = str(existing.get("id"))
                else:
                    if not wl.id:
                        wl.id = str(uuid.uuid4())
                record = self._coerce_watchlist_record(wl)
                record.managed_by = "config"
                self._db.upsert_watchlist(record)
                rules = self._coerce_rule_records(record.id, wl.rules or [])
                self._db.replace_watchlist_rules(record.id, rules)
                active_ids.add(record.id)
            except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Topic monitoring: failed to seed watchlist {getattr(wl, 'name', '')}: {e}")
        if delete_missing or disable_missing:
            existing_rows = self._db.list_watchlists(include_rules=False)
            for row in existing_rows:
                if not include_unmanaged and str(row.get("managed_by")) != "config":
                    continue
                row_id = str(row.get("id"))
                if row_id in active_ids:
                    continue
                if delete_missing:
                    self._db.delete_watchlist(row_id)
                elif disable_missing:
                    record = WatchlistRecord(
                        id=row_id,
                        name=row.get("name") or "",
                        description=row.get("description"),
                        enabled=False,
                        scope_type=row.get("scope_type") or "user",
                        scope_id=row.get("scope_id"),
                        managed_by=row.get("managed_by") or "config",
                    )
                    self._db.upsert_watchlist(record)

    def _load_watchlists_from_db(self) -> None:
        rows = self._db.list_watchlists(include_rules=True)
        watchlists: dict[str, Watchlist] = {}
        wl_fields = {"id", "name", "description", "enabled", "scope_type", "scope_id", "managed_by"}
        rule_fields = {"rule_id", "pattern", "category", "severity", "note", "tags"}
        for row in rows:
            rules_raw = row.get("rules") or []
            wl_data = {k: row.get(k) for k in wl_fields}
            wl_data["enabled"] = bool(wl_data.get("enabled", True))
            if wl_data.get("scope_type") in {"global", "all"}:
                wl_data["scope_id"] = None
            rules: list[WatchlistRule] = []
            for rule in rules_raw:
                try:
                    rule_data = {k: rule.get(k) for k in rule_fields}
                    rule_data["tags"] = rule_data.get("tags") or []
                    rules.append(WatchlistRule(**rule_data))
                except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Topic monitoring: skipped invalid rule in DB: {e}")
            try:
                wl_data["rules"] = rules
                wl = Watchlist(**wl_data)
                if wl.id:
                    watchlists[str(wl.id)] = wl
            except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Topic monitoring: skipped invalid watchlist in DB: {e}")
        self._watchlists = watchlists
        self._recompile_all()

    # --------------------- Rule compilation ---------------------
    @staticmethod
    def _is_regex_dangerous(expr: str) -> bool:
        if not expr:
            return True
        if len(expr) > 2000:
            return True
        # Nested quantifiers heuristic
        try:
            if re.search(r"\((?:[^)(]|\([^)(]*\))*[+*][^)]*\)\s*[+*]", expr):
                return True
        except (re.error, TypeError, ValueError):
            return True
        # Too many groups
        try:
            if expr.count("(") - expr.count("\\(") > 100:
                return True
        except (TypeError, ValueError):
            return True
        return False

    def _compile_rule(self, rule: WatchlistRule) -> CompiledRule | None:
        pat_text = (rule.pattern or "").strip()
        if not pat_text:
            logger.warning("Topic monitor: skipped empty pattern")
            return None
        rule_id = str(rule.rule_id) if rule.rule_id else self._compute_rule_id(rule)
        try:
            regex_flags = re.IGNORECASE
            m = re.match(r"^/(.*)/([a-zA-Z]*)$", pat_text)
            if m:
                raw = m.group(1)
                flags_text = m.group(2)
                if self._is_regex_dangerous(raw):
                    logger.warning(f"Topic monitor: skipped dangerous regex: {raw}")
                    return None
                extra_flags = 0
                unknown_flags = set()
                for ch in flags_text:
                    if ch == "i":
                        extra_flags |= re.IGNORECASE
                    elif ch == "m":
                        extra_flags |= re.MULTILINE
                    elif ch == "s":
                        extra_flags |= re.DOTALL
                    elif ch == "x":
                        extra_flags |= re.VERBOSE
                    else:
                        unknown_flags.add(ch)
                if unknown_flags:
                    logger.warning(
                        f"Topic monitor: unsupported regex flags in pattern '{pat_text}': {''.join(sorted(unknown_flags))}"
                    )
                rgx = re.compile(raw, regex_flags | extra_flags)
                patt_for_log = raw
            else:
                rgx = re.compile(re.escape(pat_text), re.IGNORECASE)
                patt_for_log = pat_text
            return CompiledRule(
                rule_id=rule_id,
                regex=rgx,
                category=(rule.category or None),
                severity=(rule.severity or None),
                pattern_text=patt_for_log,
                note=rule.note,
            )
        except re.error as e:
            logger.warning(f"Invalid watchlist pattern '{pat_text}': {e}")
            return None

    def _recompile_all(self) -> None:
        self._compiled = {}
        for wid, wl in self._watchlists.items():
            compiled: list[CompiledRule] = []
            for r in wl.rules or []:
                cr = self._compile_rule(r)
                if cr is not None:
                    compiled.append(cr)
            self._compiled[wid] = compiled

    # --------------------- Public CRUD ---------------------
    def list_watchlists(self) -> list[Watchlist]:
        with self._lock:
            return list(self._watchlists.values())

    def upsert_watchlist(self, wl: Watchlist) -> Watchlist:
        scope_type, scope_id = self._normalize_scope(wl.scope_type, wl.scope_id)
        if scope_type != "global" and not scope_id:
            raise ValueError("scope_id required for non-global watchlists")
        existing = None
        if wl.id:
            existing = self._db.get_watchlist(str(wl.id))
        if not wl.id:
            existing = self._db.get_watchlist_by_key(wl.name, scope_type, scope_id)
            if existing:
                wl.id = str(existing.get("id"))
            else:
                wl.id = str(uuid.uuid4())
        wl.scope_type = scope_type
        wl.scope_id = scope_id
        if wl.managed_by is None:
            wl.managed_by = str(existing.get("managed_by")) if existing else "api"
        wl.managed_by = str(wl.managed_by).strip().lower()
        record = self._coerce_watchlist_record(wl)
        self._db.upsert_watchlist(record)
        rules = self._coerce_rule_records(record.id, wl.rules or [], strict=True)
        self._db.replace_watchlist_rules(record.id, rules)
        self._load_watchlists_from_db()
        return self._watchlists.get(record.id, wl)

    def delete_watchlist(self, watchlist_id: str) -> bool:
        ok = self._db.delete_watchlist(watchlist_id)
        if ok:
            self._load_watchlists_from_db()
        return ok

    def reload(
        self,
        *,
        delete_missing: bool = False,
        disable_missing: bool = False,
        include_unmanaged: bool = False,
    ) -> None:
        with self._lock:
            self._config = load_and_log_configs() or {}
            monitoring_cfg = (
                self._config.get("monitoring") if isinstance(self._config, dict) else None
            ) or {}
            self._enabled = self._resolve_enabled(monitoring_cfg)
            self._max_scan_chars = self._resolve_max_scan_chars()
            self._dedup_window_seconds = self._coerce_int(
                os.getenv("TOPIC_MONITOR_DEDUP_SECONDS", monitoring_cfg.get("dedup_seconds", 300)),
                300,
            )
            self._simhash_distance = self._coerce_int(
                os.getenv("TOPIC_MONITOR_SIMHASH_DISTANCE", monitoring_cfg.get("simhash_distance", 3)),
                3,
            )
            self._dedupe_state = {}
            self._dedupe_stream_last_seen = {}
            self._dedupe_last_cleanup = 0.0
            self._watchlists_path, self._db_path = self._resolve_paths(monitoring_cfg)
            self._db = TopicMonitoringDB(db_path=self._db_path)
            self._seed_watchlists_from_file(
                delete_missing=delete_missing,
                disable_missing=disable_missing,
                include_unmanaged=include_unmanaged,
            )
            self._load_watchlists_from_db()

    # --------------------- Evaluation helpers ---------------------
    def _monitoring_active(self) -> bool:
        if not self._enabled:
            return False
        with self._lock:
            return any(wl.enabled for wl in self._watchlists.values())

    @staticmethod
    def _snippet_around(text: str, start: int, end: int, max_len: int = 200) -> str:
        # Provide a bounded snippet around the match
        mid = (start + end) // 2
        half = max_len // 2
        s = max(0, mid - half)
        e = min(len(text), s + max_len)
        return text[s:e]

    def _iter_scan_chunks(self, text: str) -> Iterable[tuple[int, int]]:
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
        for start, end in self._iter_scan_chunks(text):
            try:
                m = pat.search(text, start, end)
            except re.error:
                return None
            if m:
                return m.start(), m.end()
        return None

    def _applicable_watchlists(
        self,
        user_id: str | None,
        team_ids: list[str] | None = None,
        org_ids: list[str] | None = None,
    ) -> list[tuple[str, Watchlist]]:
        out_global: list[tuple[str, Watchlist]] = []
        out_scoped: list[tuple[str, Watchlist]] = []
        with self._lock:
            items = list(self._watchlists.items())
        for wid, wl in items:
            if not wl.enabled:
                continue
            try:
                scope_type, scope_id = self._normalize_scope(wl.scope_type, wl.scope_id)
            except ValueError as exc:
                logger.warning("Topic monitoring: skipping watchlist {} due to scope error: {}", wid, exc)
                continue
            if scope_type == "global":
                out_global.append((wid, wl))
                continue
            if not user_id:
                continue
            elif scope_type == "user" and str(scope_id) == str(user_id):
                out_scoped.append((wid, wl))
            elif scope_type == "team" and team_ids:
                try:
                    if str(scope_id) in {str(t) for t in team_ids}:
                        out_scoped.append((wid, wl))
                except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS:
                    pass
            elif scope_type == "org" and org_ids:
                try:
                    if str(scope_id) in {str(o) for o in org_ids}:
                        out_scoped.append((wid, wl))
                except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS:
                    pass
        return out_global + out_scoped

    @staticmethod
    def _word_ngrams(words: list[str], n: int = 3) -> list[str]:
        if not words:
            return []
        if len(words) < n:
            return words
        return [" ".join(words[i : i + n]) for i in range(0, len(words) - n + 1)]

    def _simhash(self, text: str) -> int:
        words = re.findall(r"\w+", text)
        grams = self._word_ngrams(words, 3)
        if not grams:
            grams = [text]
        weights = [0] * 64
        for gram in grams:
            digest = hashlib.sha1(gram.encode("utf-8"), usedforsecurity=False).digest()
            h = int.from_bytes(digest[:8], "big")
            for i in range(64):
                if h & (1 << i):
                    weights[i] += 1
                else:
                    weights[i] -= 1
        out = 0
        for i, w in enumerate(weights):
            if w > 0:
                out |= (1 << i)
        return out

    @staticmethod
    def _hamming_distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def _prune_dedupe_state(self, now: float) -> None:
        if not self._dedupe_stream_last_seen:
            self._dedupe_last_cleanup = now
            return
        window = self._dedup_window_seconds
        stale = [
            stream_id
            for stream_id, last_seen in self._dedupe_stream_last_seen.items()
            if (now - last_seen) > window
        ]
        for stream_id in stale:
            self._dedupe_stream_last_seen.pop(stream_id, None)
            self._dedupe_state.pop(stream_id, None)
        self._dedupe_last_cleanup = now

    def _dedupe_should_skip(
        self,
        *,
        stream_id: str,
        rule_id: str,
        text: str,
    ) -> tuple[bool, dict[str, Any]]:
        now = time.monotonic()
        simhash = self._simhash(text)
        simhash_hex = f"{simhash:016x}"
        with self._lock:
            self._dedupe_stream_last_seen[stream_id] = now
            if (now - self._dedupe_last_cleanup) >= self._dedup_window_seconds:
                self._prune_dedupe_state(now)
            by_rule = self._dedupe_state.setdefault(stream_id, {})
            entries = by_rule.setdefault(rule_id, deque())
            while entries and (now - entries[0][0]) > self._dedup_window_seconds:
                entries.popleft()
            min_dist = None
            for _ts, existing in entries:
                dist = self._hamming_distance(simhash, existing)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                if dist <= self._simhash_distance:
                    return True, {
                        "dedupe_hash": simhash_hex,
                        "dedupe_algo": "simhash",
                        "dedupe_similarity": dist,
                        "dedupe_window_ms": int(self._dedup_window_seconds * 1000),
                    }
            entries.append((now, simhash))
        return False, {
            "dedupe_hash": simhash_hex,
            "dedupe_algo": "simhash",
            "dedupe_similarity": min_dist if min_dist is not None else 0,
            "dedupe_window_ms": int(self._dedup_window_seconds * 1000),
        }

    def evaluate_and_alert(
        self,
        user_id: str | None,
        text: str | None,
        source: str,
        *,
        scope_type: str = "user",
        scope_id: str | None = None,
        team_ids: list[str] | None = None,
        org_ids: list[str] | None = None,
        source_id: str | None = None,
        chunk_id: str | None = None,
        chunk_seq: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Scan text for any watchlist matches and emit alerts.
        Returns count of alerts created.
        """
        if not text:
            return 0
        if not self._monitoring_active():
            return 0
        total_created = 0
        applicable = self._applicable_watchlists(user_id, team_ids=team_ids, org_ids=org_ids)
        if not applicable:
            return 0
        with self._lock:
            compiled_map = {wid: list(self._compiled.get(wid) or []) for wid, _ in applicable}
        streaming_mode = bool(source_id and (chunk_id is not None or chunk_seq is not None))
        alert_user_id = str(user_id) if user_id is not None else None
        for wid, wl in applicable:
            compiled = compiled_map.get(wid, [])
            for cr in compiled:
                match_span = self._find_match_span(cr.regex, text)
                if not match_span:
                    continue
                if streaming_mode:
                    # Dedupe per watchlist to avoid suppressing alerts across distinct watchlists
                    rule_key = f"{wid}:{cr.rule_id}"
                    skip, dedupe_meta = self._dedupe_should_skip(
                        stream_id=str(source_id),
                        rule_id=rule_key,
                        text=text,
                    )
                    if skip:
                        logger.debug(
                            'Topic monitoring dedupe skipped stream={} rule={}',
                            source_id,
                            cr.rule_id,
                        )
                        continue
                else:
                    if self._db.recent_duplicate_exists(
                        user_id=alert_user_id,
                        watchlist_id=str(wid),
                        source=str(source),
                        rule_id=str(cr.rule_id),
                        pattern=str(cr.pattern_text),
                        window_seconds=self._dedup_window_seconds,
                    ):
                        continue
                    dedupe_meta = {}
                snippet = self._snippet_around(text, match_span[0], match_span[1], max_len=200)
                # Scope reflects the matched watchlist's scope
                combined_meta: dict[str, Any] = {}
                if metadata:
                    combined_meta.update(metadata)
                if cr.note:
                    combined_meta.setdefault("note", cr.note)
                if streaming_mode:
                    combined_meta.update(dedupe_meta)
                    combined_meta.setdefault("stream_id", source_id)
                    combined_meta.setdefault("chunk_id", chunk_id)
                    combined_meta.setdefault("chunk_seq", chunk_seq)
                alert_scope_type = str(wl.scope_type or scope_type)
                alert_scope_id = None
                if alert_scope_type != "global":
                    alert_scope_id = str(wl.scope_id) if getattr(wl, "scope_id", None) is not None else scope_id
                alert = TopicAlert(
                    user_id=alert_user_id,
                    scope_type=alert_scope_type,
                    scope_id=alert_scope_id,
                    source=source,
                    watchlist_id=str(wid),
                    rule_id=str(cr.rule_id),
                    rule_category=cr.category,
                    rule_severity=cr.severity,
                    pattern=cr.pattern_text,
                    source_id=str(source_id) if source_id is not None else None,
                    chunk_id=str(chunk_id) if chunk_id is not None else None,
                    chunk_seq=chunk_seq,
                    text_snippet=snippet,
                    metadata=combined_meta or None,
                )
                self._db.insert_alert(alert)
                # Notify if configured and severity threshold met
                try:
                    notifier = get_notification_service()
                    notifier.notify(alert)
                except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(
                        "Topic monitoring notify skipped after {}",
                        type(exc).__name__,
                    )
                total_created += 1
        return total_created

    def schedule_evaluate_and_alert(
        self,
        *,
        user_id: str | None,
        text: str | None,
        source: str,
        scope_type: str = "user",
        scope_id: str | None = None,
        team_ids: list[str] | None = None,
        org_ids: list[str] | None = None,
        source_id: str | None = None,
        chunk_id: str | None = None,
        chunk_seq: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Run evaluate_and_alert in the background to avoid blocking the caller."""
        if not text:
            return
        if not self._monitoring_active():
            return
        if not self._applicable_watchlists(user_id, team_ids=team_ids, org_ids=org_ids):
            return

        def _run_sync() -> None:
            try:
                self.evaluate_and_alert(
                    user_id=user_id,
                    text=text,
                    source=source,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    team_ids=team_ids,
                    org_ids=org_ids,
                    source_id=source_id,
                    chunk_id=chunk_id,
                    chunk_seq=chunk_seq,
                    metadata=metadata,
                )
            except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "Topic monitoring background evaluation failed after {}",
                    type(exc).__name__,
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            threading.Thread(target=_run_sync, daemon=True).start()
            return

        async def _runner() -> None:
            try:
                await asyncio.to_thread(_run_sync)
            except _TOPIC_MONITOR_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "Topic monitoring background evaluation failed after {}",
                    type(exc).__name__,
                )

        loop.create_task(_runner())


_service_singleton: TopicMonitoringService | None = None


def get_topic_monitoring_service() -> TopicMonitoringService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = TopicMonitoringService()
    return _service_singleton


def _reset_topic_monitoring_service() -> None:
    """
    Testing helper: reset the cached singleton so the next access picks up
    fresh environment configuration (e.g., temp DB paths).
    """
    global _service_singleton
    _service_singleton = None
