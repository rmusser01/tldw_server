"""LLM-backed schema and regex rule generation."""

import re
from typing import Any, Optional

from ...safe_regex import SafeRegexLimits, search_untrusted
from ..dependencies import build_default_dependencies
from .llm import (
    _NONCRITICAL_EXCEPTIONS,
    call_llm_provider,
    extract_llm_response_text,
    extract_usage_from_response,
    parse_llm_json,
    record_llm_usage_metrics,
    resolve_llm_provider,
)


def _schema_prompt(html_text: str, *, url: str, query: Optional[str], example_json: Optional[str]) -> str:
    snippet = html_text.strip()[:8000]
    if len(html_text.strip()) > 8000:
        snippet += "\n...[truncated]"
    parts = [
        "Generate a schema DSL for extracting structured data from this HTML.",
        "Return JSON with key `schema` containing fields: name, baseSelector, baseFields, fields.",
        "Selectors should use XPath or prefix CSS with `css:`. Use `type` and `selector` per field.",
        f"URL: {url}",
    ]
    if query:
        parts.append(f"User query: {query}")
    if example_json:
        parts.append(f"Example JSON output: {example_json}")
    return "\n".join([*parts, f"HTML:\n{snippet}"])


def _regex_prompt(
    html_text: str, *, url: str, label: Optional[str], query: Optional[str], examples: Optional[list[str]]
) -> str:
    snippet = html_text.strip()[:8000]
    if len(html_text.strip()) > 8000:
        snippet += "\n...[truncated]"
    parts = [
        "Generate a regex pattern to extract the requested value from this HTML/text.",
        "Return JSON with keys: pattern (no delimiters), flags (e.g. 'i'), group (optional).",
        f"URL: {url}",
    ]
    if label:
        parts.append(f"Label: {label}")
    if query:
        parts.append(f"Query: {query}")
    if examples:
        parts.append(f"Examples: {examples}")
    return "\n".join([*parts, f"HTML:\n{snippet}"])


def _prepare_settings(settings: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(settings)
    if prepared.get("strict_json") and prepared.get("response_format") is None:
        prepared["response_format"] = {"type": "json_object"}
    return prepared


def _record_response(
    result: dict[str, Any], response: Any, *, provider: str, settings: dict[str, Any], dependencies: Any
) -> None:
    usage = extract_usage_from_response(response)
    model = str(response.get("model") if isinstance(response, dict) else settings.get("model") or "unknown")
    record_llm_usage_metrics(usage, provider=provider, model=model, dependencies=dependencies)
    result.update({"llm_usage": usage, "llm_provider": provider, "llm_model": model})


def generate_schema_rules_from_llm(
    html_text: str,
    url: str,
    *,
    llm_settings: Optional[dict[str, Any]] = None,
    query: Optional[str] = None,
    example_json: Optional[str] = None,
) -> dict[str, Any]:
    dependencies = build_default_dependencies()
    dependencies.cancellation_checkpoint()
    result: dict[str, Any] = {"success": False}
    if not html_text:
        result["error"] = "schema_llm_empty_html"
        return result
    settings = _prepare_settings(dict(llm_settings or {}))
    provider, app_config = resolve_llm_provider(settings)
    if not provider:
        result["error"] = "schema_llm_provider_missing"
        return result
    response, failed = call_llm_provider(
        provider=provider,
        settings=settings,
        messages=[
            {"role": "user", "content": _schema_prompt(html_text, url=url, query=query, example_json=example_json)}
        ],
        app_config=app_config,
        dependencies=dependencies,
        stage="schema_generation",
        url=url,
    )
    if failed:
        result["error"] = "provider_error"
        return result
    _record_response(result, response, provider=provider, settings=settings, dependencies=dependencies)
    obj, meta = parse_llm_json(extract_llm_response_text(response), strict=bool(settings.get("strict_json")))
    if obj is None:
        result["error"] = meta.get("error") or "schema_llm_parse_failed"
        return result
    schema = (
        obj.get("schema")
        if isinstance(obj, dict) and isinstance(obj.get("schema"), dict)
        else obj if isinstance(obj, dict) and ("fields" in obj or "baseFields" in obj) else None
    )
    if not isinstance(schema, dict):
        result["error"] = "schema_llm_no_schema"
        return result
    try:
        validation = dependencies.validate_selector_rules(schema, html_text=html_text)
    except _NONCRITICAL_EXCEPTIONS:
        validation = {"errors": [{"key": "validation", "error": "selector_invalid"}], "warnings": []}
    result["schema_rules"] = schema
    result["schema_validation"] = validation
    result["success"] = not bool(validation.get("errors"))
    return result


def _parse_flags(flags_spec: Any) -> int:
    if isinstance(flags_spec, int):
        return flags_spec
    if isinstance(flags_spec, list):
        flags_spec = "".join(str(item) for item in flags_spec)
    flags = 0
    if isinstance(flags_spec, str):
        for char in flags_spec:
            flags |= {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}.get(char.lower(), 0)
    return flags


def generate_regex_pattern_from_llm(
    html_text: str,
    url: str,
    *,
    llm_settings: Optional[dict[str, Any]] = None,
    label: Optional[str] = None,
    query: Optional[str] = None,
    examples: Optional[list[str]] = None,
) -> dict[str, Any]:
    dependencies = build_default_dependencies()
    dependencies.cancellation_checkpoint()
    result: dict[str, Any] = {"success": False}
    if not html_text:
        result["error"] = "regex_llm_empty_html"
        return result
    settings = _prepare_settings(dict(llm_settings or {}))
    provider, app_config = resolve_llm_provider(settings)
    if not provider:
        result["error"] = "regex_llm_provider_missing"
        return result
    response, failed = call_llm_provider(
        provider=provider,
        settings=settings,
        messages=[
            {"role": "user", "content": _regex_prompt(html_text, url=url, label=label, query=query, examples=examples)}
        ],
        app_config=app_config,
        dependencies=dependencies,
        stage="regex_generation",
        url=url,
    )
    if failed:
        result["error"] = "provider_error"
        return result
    _record_response(result, response, provider=provider, settings=settings, dependencies=dependencies)
    obj, meta = parse_llm_json(extract_llm_response_text(response), strict=bool(settings.get("strict_json")))
    if not isinstance(obj, dict):
        result["error"] = meta.get("error") or "regex_llm_parse_failed"
        return result
    pattern = obj.get("pattern") or obj.get("regex")
    if not isinstance(pattern, str) or not pattern.strip():
        result["error"] = "regex_llm_no_pattern"
        return result
    pattern = pattern.strip()
    flags = _parse_flags(obj.get("flags")) | (re.IGNORECASE if obj.get("ignore_case") is True else 0)
    group = obj.get("group")
    group_index = group if isinstance(group, int) else None
    sample_too_large = len(html_text) > 1_000_000
    safe_result = search_untrusted(
        pattern,
        "" if sample_too_large else html_text,
        flags=flags,
        limits=SafeRegexLimits(max_pattern_chars=4_096, max_input_chars=1_000_000, timeout_s=0.100),
    )
    if safe_result.code:
        result["error"] = safe_result.code
        return result
    if sample_too_large:
        result["sample_status"] = "skipped_input_too_large"
    elif safe_result.match:
        try:
            result["sample_match"] = (
                safe_result.match.group(group_index) if group_index is not None else safe_result.match.group(0)
            )
        except _NONCRITICAL_EXCEPTIONS:
            result["sample_match"] = safe_result.match.group(0)
        result["sample_span"] = [safe_result.match.start(), safe_result.match.end()]
    result.update({"pattern": pattern, "success": True})
    if obj.get("flags") is not None:
        result["flags"] = obj["flags"]
    if group_index is not None:
        result["group"] = group_index
    return result
