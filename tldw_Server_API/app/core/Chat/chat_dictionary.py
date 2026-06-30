# chat_dictionary.py
# Description: Chat dictionary utilities for keyword replacement and token budgeting.
"""
Helpers for parsing chat dictionary files, representing dictionary entries, and
transforming user input with probability, grouping, and token-budget controls.
"""

from __future__ import annotations

import os
import re
import secrets
import warnings
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.Chunking.regex_safety import check_pattern as check_regex_pattern
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Templating.template_renderer import (
    TemplateContext,
    TemplateEnv,
)
from tldw_Server_API.app.core.Templating.template_renderer import (
    render as render_template,
)
from tldw_Server_API.app.core.Utils.Utils import logging

_CHAT_DICT_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    re.error,
    UnicodeError,
)


def _truthy_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return is_truthy(val)


def _templates_enabled() -> bool:
    # Initial rollout default is false; may flip on after bake-in
    if _truthy_env("CHAT_DICT_TEMPLATES_ENABLED", False):
        return True
    try:
        cp = load_comprehensive_config()
        if cp and cp.has_section('Chat-Templating'):
            raw = cp.get('Chat-Templating', 'enable_templates', fallback='false')
            return is_truthy(raw)
    except _CHAT_DICT_NONCRITICAL_EXCEPTIONS:
        pass
    return False


def _has_template_syntax(text: str) -> bool:
    try:
        # Template auto-detection is expression-only; block tags are not
        # considered valid syntax for this feature.
        return "{{" in text
    except TypeError:
        return False


def _build_template_context(extra: dict[str, Any] | None = None) -> TemplateContext:
    tz = os.getenv("TEMPLATE_DEFAULT_TZ")
    if not tz:
        try:
            cp = load_comprehensive_config()
            tz = cp.get('Chat-Templating', 'default_timezone', fallback='UTC') if cp else 'UTC'
        except _CHAT_DICT_NONCRITICAL_EXCEPTIONS:
            tz = 'UTC'
    env = TemplateEnv(timezone=str(tz or 'UTC'))
    ex = dict(extra or {})
    # Hint the renderer for metrics labeling
    ex.setdefault("_metrics_source", "dict")
    return TemplateContext(env=env, extra=ex)


class _SafeMatch:
    """Read-only facade exposing a minimal API over re.Match for templates.

    Exposes only: group(idx|name), groups(), groupdict(), start(), end().
    """

    __slots__ = ("_m",)
    _tldw_template_safe_methods = frozenset({"group", "groups", "groupdict", "start", "end"})

    def __init__(self, match: re.Match):
        self._m = match

    def group(self, *args):  # type: ignore[no-untyped-def]
        return self._m.group(*args)

    def groups(self):  # type: ignore[no-untyped-def]
        return self._m.groups()

    def groupdict(self):  # type: ignore[no-untyped-def]
        return self._m.groupdict()

    def start(self, *args):  # type: ignore[no-untyped-def]
        return self._m.start(*args)

    def end(self, *args):  # type: ignore[no-untyped-def]
        return self._m.end(*args)


def parse_user_dict_markdown_file(file_path: str) -> dict[str, str]:
    """
    Parse a user-defined dictionary from a markdown-style file.

    Format:
        key: value                         # single line
        key: |                             # multi-line marker
        multi-line content
        ---@@@---                          # terminator (strip surrounding whitespace)

    Keys and single-line values are stripped of whitespace. Multi-line values
    preserve internal whitespace until the terminator is reached.
    """
    logger.debug(f"Parsing user dictionary file: {file_path}")
    replacement_dict: dict[str, str] = {}
    current_key: str | None = None
    current_value_lines: list[str] = []

    new_key_pattern = re.compile(r"^\s*([^:\n]+?)\s*:(.*)$")
    termination_pattern = re.compile(r"^\s*---@@@---\s*$")

    try:
        with open(file_path, encoding="utf-8") as file:
            for line_number, original_line in enumerate(file, 1):
                stripped_line = original_line.strip()

                if termination_pattern.match(stripped_line):
                    if current_key:
                        replacement_dict[current_key] = "\n".join(current_value_lines).strip()
                        logger.trace(f"L{line_number}: Terminated multi-line for '{current_key}'.")
                        current_key, current_value_lines = None, []
                    continue

                new_key_match = new_key_pattern.match(original_line)
                if new_key_match:
                    if current_key:
                        replacement_dict[current_key] = "\n".join(current_value_lines).strip()
                        logger.trace(f"L{line_number}: New key, finalized old '{current_key}'.")

                    potential_key = new_key_match.group(1).strip()
                    value_after_colon = new_key_match.group(2).strip()

                    if value_after_colon == "|":
                        current_key = potential_key
                        current_value_lines = []
                        logger.trace(f"L{line_number}: Starting multi-line for '{current_key}'.")
                    else:
                        replacement_dict[potential_key] = value_after_colon
                        logger.trace(f"L{line_number}: Parsed single-line key '{potential_key}'.")
                        current_key, current_value_lines = None, []
                    continue

                if current_key:
                    current_value_lines.append(original_line.rstrip("\n\r"))

            if current_key:
                replacement_dict[current_key] = "\n".join(current_value_lines).strip()
                logger.debug(f"Finalizing last multi-line key '{current_key}' at EOF.")

    except FileNotFoundError:
        logger.error(f"Chat dictionary file not found: {file_path}")
        return {}
    except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Error parsing chat dictionary file {file_path}: {exc}", exc_info=True)
        return {}

    logger.debug(f"Finished parsing chat dictionary. Keys: {list(replacement_dict.keys())}")
    return replacement_dict


class ChatDictionary:
    """
    Represents an entry in the chat dictionary for keyword replacement or expansion.

    Attributes:
        key_raw: Original key string (plain text or regex literal).
        key: Compiled key (str or compiled regex).
        content: Replacement or expansion text.
        probability: Trigger probability (0-100).
        group: Optional group name for scoring and selection logic.
        timed_effects: Sticky/cooldown/delay configuration (seconds).
        last_triggered: Timestamp when entry was last applied.
        max_replacements: Max replacements allowed per input.
    """

    def __init__(
        self,
        key: str,
        content: str,
        probability: int = 100,
        group: str | None = None,
        timed_effects: dict[str, int] | None = None,
        max_replacements: int = 1,
        enable_templates: bool | None = None,
    ):
        self.key_raw = key
        self.key = self.compile_key(key)
        self.content = content
        self.probability = probability
        self.group = group
        self.timed_effects = timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0}
        self.last_triggered: datetime | None = None
        self.max_replacements = max_replacements
        self.enable_templates = enable_templates

    @staticmethod
    def compile_key(key: str) -> str | re.Pattern:
        """Compile a key string into a regex pattern if wrapped in '/'."""
        if key.startswith("/") and key.endswith("/"):
            pattern = key[1:-1]
            safety_error = check_regex_pattern(pattern, max_len=256)
            if safety_error:
                raise ValueError(f"Unsafe chat dictionary regex: {safety_error}")
            return re.compile(pattern, re.IGNORECASE)
        return key

    def matches(self, text: str) -> bool:
        """Return True if the key matches the provided text."""
        if isinstance(self.key, re.Pattern):
            return self.key.search(text) is not None
        # Plain string keys rely on `match_whole_words` for final matching.
        return self.key in text


def apply_strategy(entries: list[ChatDictionary], strategy: str = "sorted_evenly") -> list[ChatDictionary]:
    """
    Sort entries according to the requested strategy.

    Strategies:
        - sorted_evenly: alphabetical by key_raw.
        - character_lore_first: priority for entries with group == "character".
        - global_lore_first: priority for entries with group == "global".
    """
    logging.debug(f"Applying strategy: {strategy}")
    if strategy == "sorted_evenly":
        return sorted(entries, key=lambda entry: str(entry.key_raw))
    if strategy == "character_lore_first":
        return sorted(entries, key=lambda entry: (entry.group != "character", str(entry.key_raw)))
    if strategy == "global_lore_first":
        return sorted(entries, key=lambda entry: (entry.group != "global", str(entry.key_raw)))
    return entries


def filter_by_probability(entries: list[ChatDictionary]) -> list[ChatDictionary]:
    """Filter entries by applying their probability thresholds."""
    return [entry for entry in entries if secrets.randbelow(100) + 1 <= entry.probability]


def group_scoring(entries: list[ChatDictionary]) -> list[ChatDictionary]:
    """
    Apply group scoring rules, selecting the best entry per named group while
    allowing all ungrouped entries to pass through.
    """
    logging.debug(f"Group scoring for {len(entries)} entries")
    if not entries:
        return []

    grouped_entries: dict[str | None, list[ChatDictionary]] = {}
    for entry in entries:
        grouped_entries.setdefault(entry.group, []).append(entry)

    selected_entries: list[ChatDictionary] = []
    for group_name, grouped in grouped_entries.items():
        if not grouped:
            continue
        if group_name is None:
            selected_entries.extend(grouped)
        else:
            best = max(grouped, key=lambda entry: len(str(entry.key_raw)) if entry.key_raw else 0)
            selected_entries.append(best)

    logging.debug(f"Selected {len(selected_entries)} entries after group scoring.")
    return selected_entries


def apply_timed_effects(entry: ChatDictionary, current_time: datetime) -> bool:
    """
    Evaluate delay/cooldown rules for a dictionary entry, updating `last_triggered`
    when the entry is eligible for use.
    """
    logging.debug(f"Applying timed effects for entry: {entry.key_raw}")
    if entry.timed_effects["delay"] > 0:
        base_time = datetime.min if entry.last_triggered is None else entry.last_triggered
        if current_time - base_time < timedelta(seconds=entry.timed_effects["delay"]):
            logging.debug(f"Entry '{entry.key_raw}' blocked by delay.")
            return False

    if entry.timed_effects["cooldown"] > 0 and entry.last_triggered:
        if current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["cooldown"]):
            logging.debug(f"Entry '{entry.key_raw}' still cooling down.")
            return False

    entry.last_triggered = current_time
    logging.debug(f"Entry '{entry.key_raw}' passed timed effects.")
    return True


def calculate_token_usage(entries: list[ChatDictionary]) -> int:
    """Estimate total token usage for the provided entries."""
    total_tokens = 0
    for entry in entries:
        total_tokens += len(entry.content.split())
    logging.debug(f"Calculated token usage: {total_tokens}")
    return total_tokens


class TokenBudgetExceededWarning(Warning):
    """Warning raised when the dictionary content exceeds the configured budget."""


def enforce_token_budget(entries: list[ChatDictionary], max_tokens: int) -> list[ChatDictionary]:
    """
    Trim entries to respect the token budget. Entries are assumed to be sorted
    by priority before this function executes.
    """
    filtered_entries: list[ChatDictionary] = []
    current_tokens = 0

    for entry in entries:
        entry_tokens = len(entry.content.split())
        if current_tokens + entry_tokens <= max_tokens:
            filtered_entries.append(entry)
            current_tokens += entry_tokens
        else:
            warning_msg = (
                f"Token budget exceeded while processing chat dictionary entries. "
                f"Max tokens: {max_tokens}, current tokens: {current_tokens}, "
                f"entry '{entry.key_raw}' would add {entry_tokens}."
            )
            warnings.warn(TokenBudgetExceededWarning(warning_msg), stacklevel=2)
            logging.warning(warning_msg)
            break

    logging.debug(f"Entries after enforcing token budget: {len(filtered_entries)}")
    return filtered_entries


def match_whole_words(entries: list[ChatDictionary], text: str) -> list[ChatDictionary]:
    """
    For plain-string keys, perform whole-word matching (case-insensitive) against
    the provided text. Regex keys are assumed to have matched earlier.
    """
    matched_entries: list[ChatDictionary] = []
    text_lower = text.lower()

    for entry in entries:
        if isinstance(entry.key, re.Pattern):
            matched_entries.append(entry)
        else:
            pattern = r"\b" + re.escape(str(entry.key)) + r"\b"
            if re.search(pattern, text_lower, flags=re.IGNORECASE):
                matched_entries.append(entry)

    logging.debug(f"Matched {len(matched_entries)} entries after whole-word matching.")
    return matched_entries


def alert_token_budget_exceeded(entries: list[ChatDictionary], max_tokens: int) -> None:
    """Emit a metric when token usage approaches the configured budget."""
    total_tokens = calculate_token_usage(entries)
    if total_tokens > max_tokens:
        logging.warning(
            "Chat dictionary token usage (%s) exceeds max budget (%s).", total_tokens, max_tokens
        )
        log_counter("chat_dict_token_budget_exceeded", labels={"max_tokens": str(max_tokens)})


def apply_replacement_once(text: str, entry: ChatDictionary) -> tuple[str, int]:
    """
    Apply a single pass replacement for the provided entry. Returns the new text
    plus the number of replacements performed in that pass.
    """
    replacements_done = 0

    # Templating precedence:
    # 1) Global feature flag off -> disabled for all entries.
    # 2) Entry/dictionary override False -> disabled for that entry.
    # 3) Entry/dictionary override True -> force rendering.
    # 4) Otherwise auto-detect with "{{" syntax.
    templating_globally_on = _templates_enabled()
    if not templating_globally_on:
        has_tpl = False
    elif entry.enable_templates is False:
        has_tpl = False
    elif entry.enable_templates is True:
        has_tpl = True
    else:
        has_tpl = _has_template_syntax(entry.content)

    if isinstance(entry.key, re.Pattern):
        if has_tpl:
            def replacement(match: re.Match) -> str:
                nonlocal replacements_done
                replacements_done += 1
                try:
                    extra_ctx = {
                        "match": _SafeMatch(match),
                        "matched_text": match.group(0),
                    }
                    ctx = _build_template_context(extra=extra_ctx)
                    rendered = render_template(entry.content, ctx)
                    return rendered
                except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as _e:
                    logger.debug(f"chat_dict.template_render_error(regex): {_e}")
                    return entry.content

            new_text, _ = entry.key.subn(replacement, text)
            return new_text, replacements_done
        else:
            def replacement_plain(match: re.Match) -> str:
                nonlocal replacements_done
                replacements_done += 1
                return entry.content

            new_text, _ = entry.key.subn(replacement_plain, text)
            return new_text, replacements_done

    # Literal path
    pattern = r"\b" + re.escape(str(entry.key)) + r"\b"
    if has_tpl:
        def replacement_lit(match: re.Match) -> str:
            nonlocal replacements_done
            replacements_done += 1
            try:
                extra_ctx = {"matched_text": match.group(0)}
                ctx = _build_template_context(extra=extra_ctx)
                return render_template(entry.content, ctx)
            except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as _e:
                logger.debug(f"chat_dict.template_render_error(literal): {_e}")
                return entry.content

        new_text, count = re.subn(pattern, replacement_lit, text, flags=re.IGNORECASE)
        return new_text, count
    else:
        new_text, count = re.subn(pattern, entry.content, text, flags=re.IGNORECASE)
        replacements_done += count
        return new_text, replacements_done


def process_user_input(
    user_input: str,
    entries: list[ChatDictionary] | None = None,
    max_tokens: int = 500,
    strategy: str = "sorted_evenly",
) -> str:
    """
    Transform user input by applying matching dictionary entries:
        1. Match entries by plain string or regex.
        2. Apply group scoring.
        3. Filter by probability.
        4. Enforce timed effects.
        5. Enforce token budget.
        6. Apply replacement strategy and perform replacements.
    """
    if entries is None:
        entries = []

    try:
        logging.debug("Chat Dictionary: Starting processing of user input.")

        matched_entries: list[ChatDictionary] = []
        temp_user_input = user_input

        # 1. Match entries (regex first, then whole-word for literal keys)
        try:
            logging.debug("Chat Dictionary: Matching entries.")
            for entry in entries:
                if isinstance(entry.key, re.Pattern):
                    if entry.key.search(user_input):
                        matched_entries.append(entry)
                else:
                    matched_entries.append(entry)
            matched_entries = match_whole_words(matched_entries, user_input)
        except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
            log_counter("chat_dict_match_error")
            logging.error(f"Error matching chat dictionary entries: {exc}")
            matched_entries = []

        # 2. Group scoring
        try:
            logging.debug("Chat Dictionary: Applying group scoring.")
            matched_entries = group_scoring(matched_entries)
        except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
            log_counter("chat_dict_group_scoring_error")
            logging.error(f"Error during chat dictionary group scoring: {exc}")
            matched_entries = []

        # 3. Probability filtering
        try:
            logging.debug("Chat Dictionary: Filtering by probability for %s entries", len(matched_entries))
            matched_entries = filter_by_probability(matched_entries)
        except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
            log_counter("chat_dict_probability_error")
            logging.error(f"Error in probability filtering: {exc}")
            matched_entries = []

        current_time = datetime.now()

        # 4. Timed effects
        active_timed_entries: list[ChatDictionary] = []
        try:
            logging.debug("Chat Dictionary: Applying timed effects.")
            for entry in matched_entries:
                if apply_timed_effects(entry, current_time):
                    active_timed_entries.append(entry)
            matched_entries = active_timed_entries
        except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
            log_counter("chat_dict_timed_effects_error")
            logging.error(f"Error applying timed effects: {exc}")
            matched_entries = []

        # 5. Token budget
        try:
            logging.debug("Chat Dictionary: Enforcing token budget for %s entries", len(matched_entries))
            matched_entries = enforce_token_budget(matched_entries, max_tokens)
        except TokenBudgetExceededWarning as warning:
            log_counter("chat_dict_token_limit")
            logging.warning(str(warning))
            matched_entries = []
        except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
            log_counter("chat_dict_token_budget_error")
            logging.error(f"Error enforcing token budget: {exc}")
            matched_entries = []

        try:
            alert_token_budget_exceeded(matched_entries, max_tokens)
        except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
            log_counter("chat_dict_token_alert_error")
            logging.error(f"Error in token budget alert: {exc}")

        # 6. Strategy and replacements
        try:
            logging.debug("Chat Dictionary: Applying replacement strategy.")
            matched_entries = apply_strategy(matched_entries, strategy)
        except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
            log_counter("chat_dict_strategy_error")
            logging.error(f"Error applying strategy: {exc}")
            matched_entries = []

        for entry in matched_entries:
            try:
                logging.debug("Chat Dictionary: Applying replacements.")
                replacements_done = 0
                remaining_replacements = entry.max_replacements
                while remaining_replacements > 0:
                    temp_user_input, replaced = apply_replacement_once(temp_user_input, entry)
                    if replaced > 0:
                        replacements_done += replaced
                        remaining_replacements -= 1
                        entry.last_triggered = current_time
                    else:
                        break
                if replacements_done > 0:
                    logging.debug(f"Replaced {replacements_done} occurrences of '{entry.key_raw}'.")
            except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as exc:
                log_counter("chat_dict_replacement_error", labels={"key": entry.key_raw})
                logging.error(
                    f"Error applying replacement for entry {entry.key_raw}: {exc}",
                    exc_info=True,
                )
                continue

        return temp_user_input

    except _CHAT_DICT_NONCRITICAL_EXCEPTIONS as critical_exc:
        log_counter("chat_dict_processing_error")
        logging.error(f"Critical error in process_user_input: {critical_exc}", exc_info=True)
        return user_input


__all__ = [
    "ChatDictionary",
    "TokenBudgetExceededWarning",
    "apply_replacement_once",
    "apply_strategy",
    "apply_timed_effects",
    "alert_token_budget_exceeded",
    "calculate_token_usage",
    "enforce_token_budget",
    "filter_by_probability",
    "group_scoring",
    "match_whole_words",
    "parse_user_dict_markdown_file",
    "process_user_input",
]
