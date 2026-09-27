"""LLM-backed extraction strategy with bounded provider observability."""

import json
import os
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

from bs4 import BeautifulSoup
from loguru import logger

from ...observability import bounded_code, bounded_stage, sanitized_host
from .. import throttles
from ..dependencies import ExtractionDependencies, build_default_dependencies
from ..metrics import LLM_PROVIDER_LABEL_VALUES, emit_counter
from ..retry import cap_retry_delay

_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)


def _load_app_config() -> Optional[dict[str, Any]]:
    try:
        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import ensure_app_config

        return ensure_app_config(None)
    except _NONCRITICAL_EXCEPTIONS:
        return None


def resolve_llm_provider(settings: dict[str, Any]) -> tuple[str, Optional[dict[str, Any]]]:
    provider = str(settings.get("provider") or "").strip().lower()
    app_config = _load_app_config()
    if not provider and app_config:
        provider = str(app_config.get("RAG_DEFAULT_LLM_PROVIDER") or "").strip().lower()
    return provider, app_config


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except _NONCRITICAL_EXCEPTIONS:
        return None
    return parsed if parsed > 0 else None


def _coerce_non_negative_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except _NONCRITICAL_EXCEPTIONS:
        return default
    return parsed if parsed >= 0.0 else default


def _throttle_settings(settings: dict[str, Any]) -> tuple[Optional[int], float, float]:
    max_concurrency = _coerce_positive_int(
        settings.get("max_concurrency") if "max_concurrency" in settings else os.getenv("LLM_MAX_CONCURRENCY")
    )
    delay_ms = _coerce_non_negative_float(
        settings.get("delay_ms") if "delay_ms" in settings else os.getenv("LLM_DELAY_MS")
    )
    jitter_ms = _coerce_non_negative_float(
        settings.get("delay_jitter_ms")
        if "delay_jitter_ms" in settings
        else settings.get("delay_jitter") if "delay_jitter" in settings else os.getenv("LLM_DELAY_JITTER_MS")
    )
    return max_concurrency, delay_ms, jitter_ms


@contextmanager
def _llm_throttle(
    provider: str,
    settings: dict[str, Any],
    dependencies: ExtractionDependencies,
) -> Iterator[None]:
    max_concurrency, delay_ms, jitter_ms = _throttle_settings(settings)
    semaphore = throttles.get_llm_semaphore(provider, max_concurrency) if max_concurrency else None
    if semaphore is None:
        throttles.apply_llm_delay(
            provider,
            delay_ms,
            jitter_ms,
            wall_time=dependencies.wall_time,
            sleep=dependencies.sleep,
        )
        yield
        return
    with throttles.cancellable_semaphore(semaphore, dependencies.cancellation_checkpoint):
        throttles.apply_llm_delay(
            provider,
            delay_ms,
            jitter_ms,
            wall_time=dependencies.wall_time,
            sleep=dependencies.sleep,
        )
        yield


def _retry_settings() -> tuple[int, float, float]:
    return (
        _coerce_positive_int(os.getenv("EXTRACTOR_MAX_RETRIES")) or 0,
        _coerce_non_negative_float(os.getenv("EXTRACTOR_RETRY_BASE_MS")),
        _coerce_non_negative_float(os.getenv("EXTRACTOR_RETRY_JITTER_MS")),
    )


def _metric_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    return normalized if normalized in LLM_PROVIDER_LABEL_VALUES else "other"


def _metric_attempt(attempt: int) -> str:
    return str(attempt) if attempt <= 3 else "4_plus"


def _log_provider_failure(exc: Exception, *, stage: str, url: str) -> None:
    fields = {
        "code": bounded_code("provider_error"),
        "exception_class": exc.__class__.__name__,
        "stage": bounded_stage(stage),
        "host": sanitized_host(url),
    }
    logger.bind(**fields).warning("LLM provider call failed")


def _response_model(response: Any, settings: dict[str, Any]) -> str:
    """Resolve the response model with a configured and then stable fallback."""

    response_model = response.get("model") if isinstance(response, dict) else getattr(response, "model", None)
    return str(response_model or settings.get("model") or "unknown")


def call_llm_provider(
    *,
    provider: str,
    settings: dict[str, Any],
    messages: list[dict[str, str]],
    app_config: Optional[dict[str, Any]],
    dependencies: ExtractionDependencies,
    stage: str,
    url: str,
) -> tuple[Any | None, bool]:
    """Call the injected provider with shared throttling and bounded retries."""

    max_retries, base_delay_ms, jitter_ms = _retry_settings()
    attempt = 0
    while True:
        try:
            with _llm_throttle(provider, settings, dependencies):
                dependencies.cancellation_checkpoint()
                response = dependencies.perform_chat_api_call(
                    api_provider=provider,
                    messages=messages,
                    system_message=settings.get("system_message"),
                    model=settings.get("model"),
                    api_key=settings.get("api_key"),
                    temperature=settings.get("temperature"),
                    max_tokens=settings.get("max_tokens"),
                    response_format=settings.get("response_format"),
                    app_config=app_config,
                )
                dependencies.cancellation_checkpoint()
                return response, False
        except Exception as exc:  # noqa: BLE001 - provider SDKs use unrelated exception hierarchies
            dependencies.cancellation_checkpoint()
            _log_provider_failure(exc, stage=stage, url=url)
            if attempt >= max_retries:
                return None, True
            dependencies.cancellation_checkpoint()
            attempt += 1
            emit_counter(
                dependencies,
                "extraction_retry_total",
                labels={"strategy": "llm", "attempt": _metric_attempt(attempt)},
            )
            delay_s = (base_delay_ms / 1000.0) * (2 ** (attempt - 1))
            if jitter_ms:
                delay_s += random.uniform(0.0, jitter_ms / 1000.0)  # nosec B311
            delay_s = cap_retry_delay(delay_s)
            if delay_s > 0.0:
                dependencies.sleep(delay_s)


def extract_text_for_llm(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def split_llm_chunks(
    text: str,
    *,
    chunk_token_threshold: int,
    overlap_rate: float,
    word_token_rate: float,
) -> list[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    rate = max(0.1, float(word_token_rate))
    if len(words) * rate <= max(1, int(chunk_token_threshold)):
        return [" ".join(words)]
    chunk_words = max(50, int(chunk_token_threshold / rate))
    overlap = max(0, min(int(chunk_words * max(0.0, min(overlap_rate, 0.9))), chunk_words - 1))
    step = max(1, chunk_words - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def extract_llm_response_text(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]
        if isinstance(resp.get("content"), str):
            return resp["content"]
    return ""


def extract_usage_from_response(resp: Any) -> dict[str, int]:
    if not isinstance(resp, dict) or not isinstance(resp.get("usage"), dict):
        return {}
    return {
        key: int(value)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance((value := resp["usage"].get(key)), (int, float))
    }


def record_llm_usage_metrics(
    usage: dict[str, int], *, provider: str, model: str, dependencies: ExtractionDependencies
) -> None:
    for token_type, value in (("prompt", usage.get("prompt_tokens")), ("completion", usage.get("completion_tokens"))):
        if value:
            labels = {"provider": _metric_provider(provider), "model": "configured", "type": token_type}
            emit_counter(dependencies, "llm_tokens_used_total", value=float(value), labels=labels)
            emit_counter(
                dependencies,
                "llm_tokens_used_total_by_operation",
                value=float(value),
                labels={**labels, "operation": "extraction"},
            )


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*", "", stripped).strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    return stripped


def _json_candidates(text: str) -> list[str]:
    candidates = [match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)]
    candidates.extend(match.group(1).strip() for match in re.finditer(r"<json>(.*?)</json>", text, re.I | re.S))
    candidates.append(_strip_code_fences(text))
    return [candidate for candidate in candidates if candidate]


def _decode_all_json(payload: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    index = 0
    while index < len(payload):
        brace, bracket = payload.find("{", index), payload.find("[", index)
        if brace == bracket == -1:
            break
        start = bracket if brace == -1 or (bracket != -1 and bracket < brace) else brace
        try:
            obj, index = decoder.raw_decode(payload, start)
        except _NONCRITICAL_EXCEPTIONS:
            index = start + 1
            continue
        objects.append(obj)
    return objects


def parse_llm_json(text: str, *, strict: bool) -> tuple[Optional[Any], dict[str, Any]]:
    meta: dict[str, Any] = {"objects": []}
    if not text:
        return None, meta
    if strict:
        try:
            obj = json.loads(text.strip())
        except _NONCRITICAL_EXCEPTIONS:
            return None, {"objects": [], "error": "strict_json_failed"}
        return obj, {"objects": [obj]}
    for candidate in _json_candidates(text.strip()):
        objects = _decode_all_json(candidate)
        if not objects:
            try:
                objects = [json.loads(candidate)]
            except _NONCRITICAL_EXCEPTIONS:
                continue
        meta["objects"].extend(objects)
        return objects[0], meta
    return None, meta


def schema_rules_to_field_specs(schema_rules: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(schema_rules, dict):
        return []
    fields: list[dict[str, Any]] = []
    if isinstance(schema_rules.get("fields"), list) or isinstance(schema_rules.get("baseFields"), (list, dict)):

        def normalize_field_definitions(raw: Any) -> list[dict[str, Any]]:
            if isinstance(raw, list):
                return [field for field in raw if isinstance(field, dict)]
            if isinstance(raw, dict):
                normalized: list[dict[str, Any]] = []
                for name, spec in raw.items():
                    entry = dict(spec) if isinstance(spec, dict) else {"selector": spec}
                    entry.setdefault("name", str(name))
                    normalized.append(entry)
                return normalized
            return []

        for group in ("baseFields", "fields"):
            for field in normalize_field_definitions(schema_rules.get(group) or []):
                name = field.get("name")
                if isinstance(name, str) and name.strip():
                    fields.append({"name": name.strip(), "type": str(field.get("type") or "text").strip().lower()})
        return fields
    selector_fields = {
        "title": ("title_xpath", "title_selector"),
        "summary": ("summary_xpath", "summary_selector", "description_xpath"),
        "content": ("content_xpath", "content_selector"),
        "author": ("author_xpath", "author_selector"),
        "published": ("published_xpath", "date_xpath", "date_selector"),
    }
    return [
        {"name": name, "type": "text"}
        for name, keys in selector_fields.items()
        if any(schema_rules.get(key) for key in keys)
    ]


def _prompt_for_mode(
    *,
    mode: str,
    chunk: str,
    url: str,
    fields: list[dict[str, Any]],
    chunk_index: int,
    chunk_count: int,
    extra: Optional[str],
) -> str:
    header = "Extract structured information from the following webpage text. Return only JSON with nulls for unknown fields."
    note = f"Chunk {chunk_index + 1} of {chunk_count}."
    if mode == "schema":
        prompt = f"{header}\nURL: {url}\n{note}\nSchema fields (name/type): {json.dumps(fields, ensure_ascii=True)}\nReturn a JSON object with those fields at the top level."
    elif mode == "infer_schema":
        prompt = f'{header}\nURL: {url}\n{note}\nInfer a compact schema for the content and return:\n{{"schema": {{"fields": [...]}}, "data": {{...}}}}'
    else:
        prompt = f"{header}\nURL: {url}\n{note}\nReturn a JSON object with keys: title, author, date, content, blocks.\nBlocks should be a list of {{type, text}}."
    if extra:
        prompt = f"{prompt}\nAdditional instructions: {extra}"
    return f"{prompt}\n\nContent:\n{chunk}"


def _merge_data(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key not in base or base[key] in (None, "", [], {}):
            base[key] = value
        elif isinstance(base[key], list) and isinstance(value, list):
            base[key].extend(value)


def _merge_results(objects: list[dict[str, Any]], mode: str) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    merged: dict[str, Any] = {}
    schema: Optional[dict[str, Any]] = None
    for obj in objects:
        if mode == "infer_schema":
            schema = obj.get("schema") if isinstance(obj.get("schema"), dict) else schema
            _merge_data(merged, obj.get("data") if isinstance(obj.get("data"), dict) else obj)
        else:
            _merge_data(merged, obj)
    return merged, schema


def _has_content(data: dict[str, Any]) -> bool:
    for value in data.values():
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and value:
            return True
    return False


def _extract_llm_entities_with_dependencies(
    html_text: str,
    url: str,
    *,
    dependencies: ExtractionDependencies,
    llm_settings: Optional[dict[str, Any]] = None,
    schema_rules: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    dependencies.cancellation_checkpoint()
    result: dict[str, Any] = {
        "url": url,
        "title": "N/A",
        "author": "N/A",
        "content": "",
        "date": "N/A",
        "extraction_successful": False,
        "llm_mode": None,
    }
    if not html_text:
        return result
    settings = dict(llm_settings or {})
    provider, app_config = resolve_llm_provider(settings)
    if not provider:
        result["llm_error"] = "llm_provider_missing"
        return result
    text = extract_text_for_llm(html_text)
    if not text:
        result["llm_error"] = "llm_empty_text"
        return result
    mode = str(settings.get("mode") or ("schema" if schema_rules else "blocks")).strip().lower()
    if mode not in {"blocks", "schema", "infer_schema"}:
        mode = "blocks"
    strict_json = bool(settings.get("strict_json") or False)
    if strict_json and settings.get("response_format") is None:
        settings["response_format"] = {"type": "json_object"}
    chunks = split_llm_chunks(
        text,
        chunk_token_threshold=int(settings.get("chunk_token_threshold") or 1200),
        overlap_rate=float(settings.get("overlap_rate") or 0.1),
        word_token_rate=float(settings.get("word_token_rate") or 1.3),
    )
    if not chunks:
        result["llm_error"] = "llm_no_chunks"
        return result
    parsed: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    provider_failed = False
    parse_failed = False
    fields = schema_rules_to_field_specs(schema_rules)
    for index, chunk in enumerate(chunks):
        dependencies.cancellation_checkpoint()
        response, failed = call_llm_provider(
            provider=provider,
            settings=settings,
            messages=[
                {
                    "role": "user",
                    "content": _prompt_for_mode(
                        mode=mode,
                        chunk=chunk,
                        url=url,
                        fields=fields,
                        chunk_index=index,
                        chunk_count=len(chunks),
                        extra=str(settings["prompt"]) if settings.get("prompt") else None,
                    ),
                }
            ],
            app_config=app_config,
            dependencies=dependencies,
            stage="llm_extraction",
            url=url,
        )
        provider_failed = provider_failed or failed
        if failed:
            continue
        usage = extract_usage_from_response(response)
        model = _response_model(response, settings)
        record_llm_usage_metrics(usage, provider=provider, model=model, dependencies=dependencies)
        for key in usage_total:
            usage_total[key] += usage.get(key, 0)
        obj, meta = parse_llm_json(extract_llm_response_text(response), strict=strict_json)
        if isinstance(obj, dict):
            parsed.append(obj)
        else:
            parse_failed = parse_failed or meta.get("error") == "strict_json_failed"
    if not parsed:
        result["llm_error"] = (
            "provider_error" if provider_failed else "strict_json_failed" if parse_failed else "llm_no_parseable_output"
        )
        result["llm_mode"] = mode
        return result
    merged, inferred_schema = _merge_results(parsed, mode)
    result.update(
        {
            "llm_extraction": merged,
            "llm_schema": inferred_schema,
            "llm_provider": provider,
            "llm_mode": mode,
            "llm_usage": usage_total,
        }
    )
    for key in ("title", "author", "date", "summary", "content"):
        if merged.get(key) is not None:
            result[key] = merged[key]
    if not result["content"] and isinstance(merged.get("blocks"), list):
        parts = [
            block.get("text", "").strip() if isinstance(block, dict) else block.strip()
            for block in merged["blocks"]
            if (isinstance(block, dict) and isinstance(block.get("text"), str))
            or (isinstance(block, str) and block.strip())
        ]
        if parts:
            result["content"] = "\n\n".join(parts)
    result["extraction_successful"] = _has_content(merged)
    dependencies.cancellation_checkpoint()
    return result


def extract_llm_entities(
    html_text: str,
    url: str,
    *,
    llm_settings: Optional[dict[str, Any]] = None,
    schema_rules: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return _extract_llm_entities_with_dependencies(
        html_text,
        url,
        dependencies=build_default_dependencies(),
        llm_settings=llm_settings,
        schema_rules=schema_rules,
    )
