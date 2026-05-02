"""Writing Playground API endpoints and helpers."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from collections import Counter
from typing import Any, NoReturn

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Response, status
from fastapi.responses import JSONResponse
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep, get_request_user, RateLimiter, rbac_rate_limit, User

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.endpoints.llm_providers import get_configured_providers_async
from tldw_Server_API.app.api.v1.schemas.writing_schemas import (
    WritingCapabilitiesResponse,
    WritingContextCapabilities,
    WritingDefaultTemplate,
    WritingDefaultTheme,
    WritingDefaultsResponse,
    WritingDetokenizeRequest,
    WritingDetokenizeResponse,
    WritingProviderCapabilities,
    WritingRequestedCapabilities,
    WritingServerCapabilities,
    WritingSessionCloneRequest,
    WritingSessionCreate,
    WritingSessionListItem,
    WritingSessionListResponse,
    WritingSessionResponse,
    WritingSessionUpdate,
    WritingSnapshotCounts,
    WritingSnapshotExportResponse,
    WritingSnapshotImportRequest,
    WritingSnapshotImportResponse,
    WritingSnapshotSessionItem,
    WritingSnapshotTemplateItem,
    WritingSnapshotThemeItem,
    WritingTemplateCreate,
    WritingTemplateListResponse,
    WritingTemplateResponse,
    WritingTemplateUpdate,
    WritingThemeCreate,
    WritingThemeListResponse,
    WritingThemeResponse,
    WritingThemeUpdate,
    WritingTokenCountRequest,
    WritingTokenCountResponse,
    WritingTokenizeMeta,
    WritingTokenizeRequest,
    WritingTokenizeResponse,
    WritingTokenProbabilitiesCapabilities,
    WritingTokenizerSupport,
    WritingVersionResponse,
    WritingWordcloudMeta,
    WritingWordcloudOptions,
    WritingWordcloudRequest,
    WritingWordcloudResponse,
    WritingWordcloudResult,
    WritingWordcloudWord,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.exceptions import TokenizerUnavailable
from tldw_Server_API.app.core.LLM_Calls.capability_registry import get_allowed_fields
from tldw_Server_API.app.core.LLM_Calls.extra_body_compat_catalog import (
    get_model_extra_body_compat,
    get_provider_extra_body_compat,
)
from tldw_Server_API.app.core.LLM_Calls.tokenizer_resolver import (
    ProviderNativeTokenizerHTTPAdapter as _SharedProviderNativeTokenizerHTTPAdapter,
    normalize_provider_for_tokenizer as _shared_normalize_provider_for_tokenizer,
    resolve_provider_native_tokenizer as _resolve_provider_native_tokenizer_shared,
    resolve_tiktoken_encoding as _resolve_tiktoken_encoding_shared,
    resolve_tokenizer as _resolve_tokenizer_shared,
    strict_token_counting_enabled as _strict_token_counting_enabled_shared,
)
from tldw_Server_API.app.core.testing import is_test_mode

router = APIRouter()

_WRITING_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    CharactersRAGDBError,
    ConflictError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    InputError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TokenizerUnavailable,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)

TOKENIZER_UNAVAILABLE_DETAIL = "Tokenizer unavailable for provider/model"
_TRUE_VALUES = {"1", "true", "yes", "on"}


async def _enforce_rate_limit(rate_limiter: RateLimiter, user_id: int, scope: str) -> None:
    """Enforce a rate limit for the given user and scope."""
    try:
        allowed, meta = await rate_limiter.check_user_rate_limit(int(user_id), scope)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        retry_after = 60
        logger.error("Rate limiter check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiter unavailable",
            headers={"Retry-After": str(retry_after)},
        ) from exc
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {scope}",
            headers={"Retry-After": str(meta.get("retry_after", 60))},
        )


def _handle_db_errors(exc: Exception, entity_label: str) -> NoReturn:
    """Translate database exceptions into HTTP errors."""
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, InputError):
        logger.warning("Input error for {}: {}", entity_label, exc)
        raise map_db_error_to_http(exc) from exc
    if isinstance(exc, ConflictError):
        message = str(exc)
        lowered = message.lower()
        if "not found" in lowered or "soft-deleted" in lowered or "soft deleted" in lowered:
            logger.debug("Entity not found for {}: {}", entity_label, exc)
            raise map_db_error_to_http(
                exc,
                conflict_status_code=status.HTTP_404_NOT_FOUND,
                conflict_detail=f"{entity_label} not found",
            ) from exc
        logger.warning("Conflict error for {}: {}", entity_label, exc)
        raise map_db_error_to_http(exc) from exc
    if isinstance(exc, CharactersRAGDBError):
        logger.error("Database error while processing writing entity")
        raise map_db_error_to_http(
            exc,
            default_detail=f"Database error while processing {entity_label}",
        ) from exc
    logger.error("Unexpected error while processing writing entity")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Unexpected error while processing {entity_label}",
    ) from exc


def _resolve_tiktoken_encoding(model: str) -> Any:
    """Resolve a tiktoken encoding for the given model name."""
    return _resolve_tiktoken_encoding_shared(model)


def _normalize_provider_for_tokenizer(provider: str) -> str:
    return _shared_normalize_provider_for_tokenizer(provider)


_ProviderNativeTokenizerHTTPAdapter = _SharedProviderNativeTokenizerHTTPAdapter


def _strict_token_counting_enabled() -> bool:
    return _strict_token_counting_enabled_shared(default=False)


def _resolve_provider_native_tokenizer(
    provider: str,
    model: str,
) -> tuple[Any, str, str, str, bool]:
    resolution = _resolve_provider_native_tokenizer_shared(
        provider,
        model,
        strict_mode_effective=_strict_token_counting_enabled(),
        config_loader=load_comprehensive_config,
        adapter_cls=_ProviderNativeTokenizerHTTPAdapter,
        runtime_probe_exact=True,
        runtime_probe_timeout_seconds=2.0,
        runtime_probe_text="",
    )
    if not resolution.available or resolution.encoding is None:
        raise TokenizerUnavailable(resolution.error or "Provider-native tokenizer is not configured for provider")
    return (
        resolution.encoding,
        resolution.tokenizer or "unknown",
        resolution.kind or "provider-native",
        resolution.source or "provider-native.tokenize",
        bool(resolution.detokenize_available),
    )


def _resolve_tokenizer(provider: str, model: str) -> tuple[Any, str, str, str, bool, str, bool]:
    """Resolve tokenizer metadata via shared resolver with compatibility overrides."""
    if not provider or not provider.strip():
        raise TokenizerUnavailable("Provider is required")
    if not model or not model.strip():
        raise TokenizerUnavailable("Model is required")

    strict_mode_effective = _strict_token_counting_enabled()
    normalized_provider = _normalize_provider_for_tokenizer(provider)
    normalized_model = model.strip()

    # Keep local wrapper path first for test monkeypatch compatibility.
    if normalized_provider != "mlx":
        try:
            encoding, tokenizer_name, tokenizer_kind, tokenizer_source, detok_available = (
                _resolve_provider_native_tokenizer(normalized_provider, normalized_model)
            )
            # Defensive runtime probe so local monkeypatch paths cannot claim exactness when
            # the underlying provider-native endpoint is unavailable.
            if tokenizer_kind in {"provider-native", "provider-native-count"}:
                _tokenizer_count(encoding, "")
            return (
                encoding,
                tokenizer_name,
                tokenizer_kind,
                tokenizer_source,
                detok_available,
                "exact",
                strict_mode_effective,
            )
        except TokenizerUnavailable:
            pass

    resolution = _resolve_tokenizer_shared(
        provider,
        model,
        strict_mode_effective=strict_mode_effective,
        config_loader=load_comprehensive_config,
        adapter_cls=_ProviderNativeTokenizerHTTPAdapter,
        runtime_probe_exact=True,
        runtime_probe_timeout_seconds=2.0,
        runtime_probe_text="",
    )
    if not resolution.available or resolution.encoding is None:
        raise TokenizerUnavailable(resolution.error or "Tokenizer not available for provider/model")
    return (
        resolution.encoding,
        resolution.tokenizer or "unknown",
        resolution.kind or "tiktoken",
        resolution.source or "tiktoken.encoding_for_model",
        bool(resolution.detokenize_available),
        resolution.count_accuracy or "unavailable",
        bool(resolution.strict_mode_effective),
    )


def _resolve_tokenizer_details(
    provider: str,
    model: str,
) -> tuple[Any, str, str, str, bool, str, bool]:
    """Resolve tokenizer plus compatibility metadata.

    Keeps backward compatibility with tests that monkeypatch `_resolve_tokenizer`
    to return a 2-tuple `(encoding, tokenizer_name)`.
    """
    resolved = _resolve_tokenizer(provider, model)
    if not isinstance(resolved, tuple) or len(resolved) < 2:
        raise TokenizerUnavailable("Tokenizer resolver returned invalid payload")

    encoding = resolved[0]
    tokenizer_name = str(resolved[1])
    tokenizer_kind = "tiktoken"
    tokenizer_source = "tiktoken.encoding_for_model"
    detokenize_available = True
    count_accuracy = "unavailable"
    strict_mode_effective = _strict_token_counting_enabled()

    if len(resolved) >= 3 and isinstance(resolved[2], str) and resolved[2].strip():
        tokenizer_kind = resolved[2].strip()
    if len(resolved) >= 4 and isinstance(resolved[3], str) and resolved[3].strip():
        tokenizer_source = resolved[3].strip()
    if len(resolved) >= 5:
        detokenize_available = bool(resolved[4])
    if len(resolved) >= 6 and isinstance(resolved[5], str) and resolved[5].strip():
        count_accuracy = resolved[5].strip()
    elif tokenizer_kind in {"provider-native", "provider-native-count"}:
        count_accuracy = "exact"
    if len(resolved) >= 7:
        strict_mode_effective = bool(resolved[6])
    if count_accuracy not in {"exact", "unavailable"}:
        count_accuracy = "unavailable"

    return (
        encoding,
        tokenizer_name,
        tokenizer_kind,
        tokenizer_source,
        detokenize_available,
        count_accuracy,
        strict_mode_effective,
    )


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encode_fn = getattr(tokenizer, "encode", None)
    if not callable(encode_fn):
        raise TokenizerUnavailable("Tokenize not available for provider/model")
    try:
        encoded = encode_fn(text, disallowed_special=())
    except TypeError:
        try:
            encoded = encode_fn(text)
        except Exception as exc:
            raise TokenizerUnavailable(f"Unable to tokenize text: {exc}") from exc
    except Exception as exc:
        raise TokenizerUnavailable(f"Unable to tokenize text: {exc}") from exc
    if not isinstance(encoded, list):
        try:
            encoded = list(encoded)
        except Exception as exc:
            raise TokenizerUnavailable("Tokenizer returned non-iterable token ids") from exc
    try:
        return [int(token_id) for token_id in encoded]
    except Exception as exc:
        raise TokenizerUnavailable("Tokenizer returned invalid token ids") from exc


def _tokenizer_decode(tokenizer: Any, token_ids: list[int]) -> str:
    decode_fn = getattr(tokenizer, "decode", None)
    if not callable(decode_fn):
        raise TokenizerUnavailable("Detokenize not available for provider/model")
    try:
        decoded = decode_fn(token_ids)
    except Exception as exc:
        raise TokenizerUnavailable(f"Unable to detokenize ids: {exc}") from exc
    if not isinstance(decoded, str):
        raise TokenizerUnavailable("Tokenizer returned invalid decoded text")
    return decoded


def _tokenizer_decode_single(tokenizer: Any, token_id: int) -> str:
    return _tokenizer_decode(tokenizer, [int(token_id)])


def _tokenizer_count(tokenizer: Any, text: str) -> int:
    count_fn = getattr(tokenizer, "count_tokens", None)
    if callable(count_fn):
        try:
            counted = count_fn(text)
        except Exception as exc:
            raise TokenizerUnavailable(f"Unable to count tokens: {exc}") from exc
        if isinstance(counted, bool):
            raise TokenizerUnavailable("Tokenizer count returned invalid payload")
        if isinstance(counted, (int, float)):
            token_count = int(counted)
            if token_count < 0:
                raise TokenizerUnavailable("Tokenizer count returned invalid payload")
            return token_count
        raise TokenizerUnavailable("Tokenizer count returned invalid payload")

    return len(_tokenizer_encode(tokenizer, text))


def _provider_features(provider: str) -> dict[str, bool]:
    """Build a provider feature map from allowed capability fields."""
    fields = get_allowed_fields(provider)
    return {
        "logprobs": "logprobs" in fields,
        "logit_bias": "logit_bias" in fields,
        "tools": "tools" in fields,
        "tool_choice": "tool_choice" in fields,
        "response_format": "response_format" in fields,
        "seed": "seed" in fields,
        "top_k": "top_k" in fields,
        "min_p": "min_p" in fields,
        "top_p": "top_p" in fields,
        "temperature": "temperature" in fields,
        "presence_penalty": "presence_penalty" in fields,
        "frequency_penalty": "frequency_penalty" in fields,
    }


def _coerce_model_name(model: Any) -> str | None:
    """Coerce a provider model payload into a normalized model name."""
    if isinstance(model, str):
        return model.strip() or None
    if isinstance(model, dict):
        for key in ("name", "id", "model"):
            value = model.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _runtime_context() -> dict[str, Any]:
    strict_openai_compat = False
    try:
        config_parser = load_comprehensive_config()
        if config_parser.has_section("Local-API") and config_parser.has_option("Local-API", "strict_openai_compat"):
            strict_openai_compat = _truthy(
                config_parser.get("Local-API", "strict_openai_compat", fallback="false")
            )
    except Exception:  # noqa: BLE001 - capabilities metadata should fail open
        strict_openai_compat = False

    env_override = os.getenv("LOCAL_LLM_STRICT_OPENAI_COMPAT")
    if env_override is not None:
        strict_openai_compat = _truthy(env_override)

    return {"strict_openai_compat": strict_openai_compat}


def _fallback_extra_body_compat() -> dict[str, Any]:
    return {
        "supported": False,
        "effective_reason": "unsupported for deployment/runtime configuration",
        "known_params": [],
        "param_groups": [],
        "notes": "No extra_body compatibility metadata registered for this provider/model.",
        "example": {"extra_body": {}},
        "source": "catalog+runtime",
    }


def _safe_provider_extra_body_compat(provider: str, runtime_context: dict[str, Any]) -> dict[str, Any]:
    try:
        return get_provider_extra_body_compat(provider, runtime_context=runtime_context)
    except Exception:  # noqa: BLE001 - capabilities metadata should fail open
        return _fallback_extra_body_compat()


def _safe_model_extra_body_compat(provider: str, model: str, runtime_context: dict[str, Any]) -> dict[str, Any]:
    try:
        return get_model_extra_body_compat(provider, model, runtime_context=runtime_context)
    except Exception:  # noqa: BLE001 - capabilities metadata should fail open
        return _fallback_extra_body_compat()


def _normalize_theme_response(theme: dict[str, Any]) -> dict[str, Any]:
    """Normalize DB theme dict to API response format."""
    theme["order"] = theme.pop("order_index", 0)
    return theme


def _list_all_writing_sessions(db: CharactersRAGDB, *, batch_size: int = 500) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = db.list_writing_sessions(limit=batch_size, offset=offset)
        if not batch:
            break
        sessions.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return sessions


def _list_all_writing_templates(db: CharactersRAGDB, *, batch_size: int = 500) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = db.list_writing_templates(limit=batch_size, offset=offset)
        if not batch:
            break
        templates.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return templates


def _list_all_writing_themes(db: CharactersRAGDB, *, batch_size: int = 500) -> list[dict[str, Any]]:
    themes: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = db.list_writing_themes(limit=batch_size, offset=offset)
        if not batch:
            break
        themes.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return themes


def _restore_soft_deleted_writing_session(
    db: CharactersRAGDB,
    *,
    session_id: str,
    name: str,
    payload: dict[str, Any],
    schema_version: int,
    version_parent_id: str | None,
) -> None:
    """Restore a soft-deleted writing session while preserving its ID.

    Delegates to ``db.restore_writing_session()`` so that all SQL stays
    in the DB-management layer.  The *conn* parameter is accepted for
    call-site compatibility but is no longer used directly.
    """
    db.restore_writing_session(
        session_id,
        name=name,
        payload=payload,
        schema_version=schema_version,
        version_parent_id=version_parent_id,
    )


def _tokenizer_support(provider: str, model: str) -> WritingTokenizerSupport:
    """Build tokenizer support metadata for a provider/model pair."""
    try:
        (
            _encoding,
            tokenizer_name,
            tokenizer_kind,
            tokenizer_source,
            detokenize_available,
            count_accuracy,
            strict_mode_effective,
        ) = _resolve_tokenizer_details(provider, model)
        return WritingTokenizerSupport(
            available=True,
            tokenizer=tokenizer_name,
            kind=tokenizer_kind,
            source=tokenizer_source,
            detokenize=detokenize_available,
            count_accuracy=count_accuracy,
            strict_mode_effective=strict_mode_effective,
        )
    except TokenizerUnavailable:
        return WritingTokenizerSupport(
            available=False,
            count_accuracy="unavailable",
            strict_mode_effective=_strict_token_counting_enabled(),
            error=TOKENIZER_UNAVAILABLE_DETAIL,
        )


WORDCLOUD_ALGO_VERSION = 1
WORDCLOUD_STATUS_QUEUED = "queued"
WORDCLOUD_STATUS_RUNNING = "running"
WORDCLOUD_STATUS_READY = "ready"
WORDCLOUD_STATUS_FAILED = "failed"
WORDCLOUD_GENERATION_FAILED_DETAIL = "Wordcloud generation failed"
WORDCLOUD_TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
DEFAULT_WORDCLOUD_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}

DEFAULT_WRITING_TEMPLATE_CATALOG: list[WritingDefaultTemplate] = [
    WritingDefaultTemplate(
        name="default",
        payload={},
        schema_version=1,
        is_default=True,
    )
]

DEFAULT_WRITING_THEME_CATALOG: list[WritingDefaultTheme] = [
    WritingDefaultTheme(
        name="default",
        class_name="",
        css="",
        schema_version=1,
        is_default=True,
        order=0,
    )
]


def _is_test_mode() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST") or is_test_mode())


def _normalize_wordcloud_options(options: WritingWordcloudOptions | None) -> WritingWordcloudOptions:
    return options or WritingWordcloudOptions()


def _wordcloud_options_payload(options: WritingWordcloudOptions) -> dict[str, Any]:
    payload = options.model_dump() if hasattr(options, "model_dump") else options.dict()
    payload["algo_version"] = WORDCLOUD_ALGO_VERSION
    return payload


def _resolve_stopwords(options: WritingWordcloudOptions) -> set[str]:
    if options.stopwords is None:
        return set(DEFAULT_WORDCLOUD_STOPWORDS)
    custom = {word.strip().casefold() for word in options.stopwords if isinstance(word, str) and word.strip()}
    return custom


def _hash_wordcloud_input(text: str, options_payload: dict[str, Any]) -> tuple[str, str]:
    options_json = json.dumps(options_payload, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    digest.update(b"\n")
    digest.update(options_json.encode("utf-8"))
    return digest.hexdigest(), options_json


def _compute_wordcloud(text: str, options: WritingWordcloudOptions) -> tuple[list[WritingWordcloudWord], WritingWordcloudMeta]:
    normalized = text.casefold()
    tokens = WORDCLOUD_TOKEN_RE.findall(normalized)
    stopwords = _resolve_stopwords(options)
    counts: Counter[str] = Counter()
    for token in tokens:
        if not token or len(token) < options.min_word_length:
            continue
        if not options.keep_numbers and token.isnumeric():
            continue
        if not token.strip("-_"):
            continue
        if stopwords and token in stopwords:
            continue
        counts[token] += 1
    most_common = counts.most_common(options.max_words)
    words = [WritingWordcloudWord(text=word, weight=count) for word, count in most_common]
    total_tokens = sum(counts.values())
    meta = WritingWordcloudMeta(
        input_chars=len(text),
        total_tokens=total_tokens,
        top_n=len(words),
    )
    return words, meta


def _model_dump(obj: Any) -> dict[str, Any]:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    dump = getattr(obj, "dict", None)
    if callable(dump):
        return dump()
    return dict(obj)


def _run_wordcloud_job(
    db: CharactersRAGDB,
    wordcloud_id: str,
    text: str,
    options: WritingWordcloudOptions,
) -> None:
    try:
        db.set_writing_wordcloud_status(wordcloud_id, WORDCLOUD_STATUS_RUNNING)
        words, meta = _compute_wordcloud(text, options)
        words_payload = [_model_dump(word) for word in words]
        meta_payload = _model_dump(meta)
        db.set_writing_wordcloud_result(
            wordcloud_id,
            status=WORDCLOUD_STATUS_READY,
            words=words_payload,
            meta=meta_payload,
            error=None,
        )
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Wordcloud job failed")
        try:
            db.set_writing_wordcloud_result(
                wordcloud_id,
                status=WORDCLOUD_STATUS_FAILED,
                error=WORDCLOUD_GENERATION_FAILED_DETAIL,
            )
        except _WRITING_NONCRITICAL_EXCEPTIONS:
            logger.error("Failed to persist wordcloud failure")


def _build_wordcloud_response_from_row(row: dict[str, Any], *, cached: bool) -> WritingWordcloudResponse:
    status_value = row.get("status") or WORDCLOUD_STATUS_QUEUED
    result = None
    words_payload = row.get("words")
    meta_payload = row.get("meta")
    if isinstance(words_payload, list) and isinstance(meta_payload, dict):
        words = [WritingWordcloudWord(**word) for word in words_payload if isinstance(word, dict)]
        meta = WritingWordcloudMeta(**meta_payload)
        result = WritingWordcloudResult(words=words, meta=meta)
    return WritingWordcloudResponse(
        id=str(row.get("id") or ""),
        status=status_value,
        cached=cached,
        result=result,
        error=row.get("error"),
    )


@router.get(
    "/version",
    response_model=WritingVersionResponse,
    summary="Writing Playground API version",
    tags=["writing"],
)
async def get_writing_version(
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.version")),
) -> WritingVersionResponse:
    """Return the writing API version."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.version")
    return WritingVersionResponse(version=1)


@router.get(
    "/capabilities",
    response_model=WritingCapabilitiesResponse,
    summary="Writing Playground capabilities",
    tags=["writing"],
)
async def get_writing_capabilities(
    provider: str | None = Query(None, description="Optional provider to resolve"),
    model: str | None = Query(None, description="Optional model to resolve"),
    include_providers: bool = Query(True, description="Include configured providers list"),
    include_deprecated: bool = Query(False, description="Include deprecated models"),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.capabilities")),
) -> WritingCapabilitiesResponse:
    """Return writing capabilities and provider metadata."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.capabilities")

    server = WritingServerCapabilities(
        sessions=True,
        templates=True,
        themes=True,
        defaults_catalog=True,
        snapshots=True,
        tokenize=True,
        detokenize=True,
        token_count=True,
        wordclouds=True,
        token_probabilities=WritingTokenProbabilitiesCapabilities(
            inline_reroll=True,
        ),
        context=WritingContextCapabilities(
            author_note_depth_mode="insertion",
            context_order=True,
            context_budget=True,
        ),
    )

    providers_payload: list[WritingProviderCapabilities] | None = None
    default_provider = None
    runtime_ctx = _runtime_context()
    if include_providers:
        providers_info = await get_configured_providers_async(include_deprecated=include_deprecated)
        default_provider = providers_info.get("default_provider")
        providers_payload = []
        for provider_info in providers_info.get("providers", []):
            name = provider_info.get("name") or "unknown"
            raw_models = provider_info.get("models") or []
            models: list[str] = []
            for raw_model in raw_models:
                model_name = _coerce_model_name(raw_model)
                if model_name:
                    models.append(model_name)
            capabilities = provider_info.get("capabilities") or {}
            supported_fields = sorted(get_allowed_fields(name))
            tokenizers = None
            if models:
                tokenizers = {model: _tokenizer_support(name, model) for model in models}
            model_extra_body_compat = {
                model_name: _safe_model_extra_body_compat(name, model_name, runtime_ctx)
                for model_name in models
            }
            providers_payload.append(
                WritingProviderCapabilities(
                    name=name,
                    models=models,
                    capabilities=capabilities,
                    supported_fields=supported_fields,
                    features=_provider_features(name),
                    tokenizers=tokenizers,
                    extra_body_compat=_safe_provider_extra_body_compat(name, runtime_ctx),
                    model_extra_body_compat=model_extra_body_compat or None,
                )
            )

    requested: WritingRequestedCapabilities | None = None
    if provider or model:
        provider_name = (provider or "").strip()
        model_name = (model or "").strip() or None
        supported_fields = sorted(get_allowed_fields(provider_name)) if provider_name else []
        features = _provider_features(provider_name) if provider_name else {}
        tokenizer_available = False
        tokenizer_name = None
        tokenizer_kind = None
        tokenizer_source = None
        detokenize_available = False
        count_accuracy = "unavailable"
        strict_mode_effective = _strict_token_counting_enabled()
        tokenization_error = None
        extra_body_compat = None
        if provider_name and model_name:
            try:
                (
                    _encoding,
                    tokenizer_name,
                    tokenizer_kind,
                    tokenizer_source,
                    detokenize_available,
                    count_accuracy,
                    strict_mode_effective,
                ) = _resolve_tokenizer_details(provider_name, model_name)
                tokenizer_available = True
            except TokenizerUnavailable:
                tokenization_error = TOKENIZER_UNAVAILABLE_DETAIL
            extra_body_compat = _safe_model_extra_body_compat(provider_name, model_name, runtime_ctx)
        elif provider_name:
            extra_body_compat = _safe_provider_extra_body_compat(provider_name, runtime_ctx)
        requested = WritingRequestedCapabilities(
            provider=provider_name,
            model=model_name,
            supported_fields=supported_fields,
            features=features,
            tokenizer_available=tokenizer_available,
            tokenizer=tokenizer_name,
            tokenizer_kind=tokenizer_kind,
            tokenizer_source=tokenizer_source,
            detokenize_available=detokenize_available,
            count_accuracy=count_accuracy,
            strict_mode_effective=strict_mode_effective,
            tokenization_error=tokenization_error,
            extra_body_compat=extra_body_compat,
        )

    return WritingCapabilitiesResponse(
        version=1,
        server=server,
        default_provider=default_provider,
        providers=providers_payload,
        requested=requested,
    )


@router.get(
    "/defaults",
    response_model=WritingDefaultsResponse,
    summary="Writing Playground default template/theme catalog",
    tags=["writing"],
)
async def get_writing_defaults(
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.defaults.get")),
) -> WritingDefaultsResponse:
    """Return server-advertised default templates/themes for restore actions."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.defaults.get")
    return WritingDefaultsResponse(
        version=1,
        templates=DEFAULT_WRITING_TEMPLATE_CATALOG,
        themes=DEFAULT_WRITING_THEME_CATALOG,
    )


@router.get(
    "/sessions",
    response_model=WritingSessionListResponse,
    summary="List writing sessions",
    tags=["writing"],
)
async def list_writing_sessions(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.sessions.list")),
) -> WritingSessionListResponse:
    """List writing sessions for the current user."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.sessions.list")
    try:
        sessions = db.list_writing_sessions(limit=limit, offset=offset)
        total = db.count_writing_sessions()
        items = [WritingSessionListItem(**item) for item in sessions]
        return WritingSessionListResponse(
            sessions=items,
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(items),
            ),
        )
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing sessions")


@router.post(
    "/sessions",
    response_model=WritingSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a writing session",
    tags=["writing"],
)
async def create_writing_session(
    payload: WritingSessionCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.sessions.create")),
) -> WritingSessionResponse:
    """Create a new writing session."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.sessions.create")
    session_name = payload.name.strip()
    if not session_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session name cannot be empty")
    try:
        session_id = db.add_writing_session(
            name=session_name,
            payload=payload.payload,
            schema_version=payload.schema_version,
            session_id=payload.id,
            version_parent_id=payload.version_parent_id,
        )
        session = db.get_writing_session(session_id)
        if not session:
            raise CharactersRAGDBError("Session created but could not be retrieved")
        session["payload"] = session.get("payload") or {}
        return WritingSessionResponse(**session)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing session")


@router.get(
    "/sessions/{session_id}",
    response_model=WritingSessionResponse,
    summary="Get a writing session",
    tags=["writing"],
)
async def get_writing_session(
    session_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.sessions.get")),
) -> WritingSessionResponse:
    """Fetch a writing session by id."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.sessions.get")
    try:
        session = db.get_writing_session(session_id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        session["payload"] = session.get("payload") or {}
        return WritingSessionResponse(**session)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing session")


@router.patch(
    "/sessions/{session_id}",
    response_model=WritingSessionResponse,
    summary="Update a writing session",
    tags=["writing"],
)
async def update_writing_session(
    session_id: str,
    payload: WritingSessionUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.sessions.update")),
) -> WritingSessionResponse:
    """Update a writing session with optimistic locking."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.sessions.update")
    update_data: dict[str, Any] = {}
    if payload.name is not None:
        update_data["name"] = payload.name.strip()
        if not update_data["name"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session name cannot be empty")
    if payload.payload is not None:
        update_data["payload"] = payload.payload
    if payload.schema_version is not None:
        update_data["schema_version"] = payload.schema_version
    if payload.version_parent_id is not None:
        update_data["version_parent_id"] = payload.version_parent_id
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update")
    try:
        db.update_writing_session(session_id, update_data, expected_version)
        session = db.get_writing_session(session_id)
        if not session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        session["payload"] = session.get("payload") or {}
        return WritingSessionResponse(**session)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing session")


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a writing session",
    tags=["writing"],
)
async def delete_writing_session(
    session_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.sessions.delete")),
) -> Response:
    """Soft-delete a writing session."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.sessions.delete")
    try:
        db.soft_delete_writing_session(session_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing session")


@router.post(
    "/sessions/{session_id}/clone",
    response_model=WritingSessionResponse,
    summary="Clone a writing session",
    tags=["writing"],
)
async def clone_writing_session(
    session_id: str,
    payload: WritingSessionCloneRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.sessions.clone")),
) -> WritingSessionResponse:
    """Clone a writing session."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.sessions.clone")
    try:
        cloned = db.clone_writing_session(session_id, name=payload.name)
        cloned["payload"] = cloned.get("payload") or {}
        return WritingSessionResponse(**cloned)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing session")


@router.get(
    "/templates",
    response_model=WritingTemplateListResponse,
    summary="List writing templates",
    tags=["writing"],
)
async def list_writing_templates(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.templates.list")),
) -> WritingTemplateListResponse:
    """List writing templates for the current user."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.templates.list")
    try:
        templates = db.list_writing_templates(limit=limit, offset=offset)
        total = db.count_writing_templates()
        for tmpl in templates:
            tmpl["payload"] = tmpl.get("payload") or {}
        return WritingTemplateListResponse(
            templates=[WritingTemplateResponse(**tmpl) for tmpl in templates],
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(templates),
            ),
        )
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing templates")


@router.post(
    "/templates",
    response_model=WritingTemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a writing template",
    tags=["writing"],
)
async def create_writing_template(
    payload: WritingTemplateCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.templates.create")),
) -> WritingTemplateResponse:
    """Create a new writing template."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.templates.create")
    template_name = payload.name.strip()
    if not template_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Template name cannot be empty")
    try:
        db.add_writing_template(
            name=template_name,
            payload=payload.payload,
            schema_version=payload.schema_version,
            version_parent_id=payload.version_parent_id,
            is_default=payload.is_default,
        )
        template = db.get_writing_template_by_name(template_name)
        if not template:
            raise CharactersRAGDBError("Template created but could not be retrieved")
        template["payload"] = template.get("payload") or {}
        return WritingTemplateResponse(**template)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing template")


@router.get(
    "/templates/{name}",
    response_model=WritingTemplateResponse,
    summary="Get a writing template",
    tags=["writing"],
)
async def get_writing_template(
    name: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.templates.get")),
) -> WritingTemplateResponse:
    """Fetch a writing template by name."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.templates.get")
    try:
        template = db.get_writing_template_by_name(name)
        if not template:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")
        template["payload"] = template.get("payload") or {}
        return WritingTemplateResponse(**template)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing template")


@router.patch(
    "/templates/{name}",
    response_model=WritingTemplateResponse,
    summary="Update a writing template",
    tags=["writing"],
)
async def update_writing_template(
    name: str,
    payload: WritingTemplateUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.templates.update")),
) -> WritingTemplateResponse:
    """Update a writing template with optimistic locking."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.templates.update")
    update_data: dict[str, Any] = {}
    if payload.name is not None:
        update_data["name"] = payload.name.strip()
        if not update_data["name"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Template name cannot be empty")
    if payload.payload is not None:
        update_data["payload"] = payload.payload
    if payload.schema_version is not None:
        update_data["schema_version"] = payload.schema_version
    if payload.version_parent_id is not None:
        update_data["version_parent_id"] = payload.version_parent_id
    if payload.is_default is not None:
        update_data["is_default"] = payload.is_default
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update")
    try:
        db.update_writing_template(name, update_data, expected_version)
        template_name = update_data.get("name", name)
        template = db.get_writing_template_by_name(template_name)
        if not template:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")
        template["payload"] = template.get("payload") or {}
        return WritingTemplateResponse(**template)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing template")


@router.delete(
    "/templates/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a writing template",
    tags=["writing"],
)
async def delete_writing_template(
    name: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.templates.delete")),
) -> Response:
    """Soft-delete a writing template."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.templates.delete")
    try:
        db.soft_delete_writing_template(name, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing template")


@router.get(
    "/themes",
    response_model=WritingThemeListResponse,
    summary="List writing themes",
    tags=["writing"],
)
async def list_writing_themes(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.themes.list")),
) -> WritingThemeListResponse:
    """List writing themes for the current user."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.themes.list")
    try:
        themes = db.list_writing_themes(limit=limit, offset=offset)
        total = db.count_writing_themes()
        for theme in themes:
            _normalize_theme_response(theme)
        return WritingThemeListResponse(
            themes=[WritingThemeResponse(**theme) for theme in themes],
            total=total,
            limit=limit,
            offset=offset,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=offset,
                count=len(themes),
            ),
        )
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing themes")


@router.post(
    "/themes",
    response_model=WritingThemeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a writing theme",
    tags=["writing"],
)
async def create_writing_theme(
    payload: WritingThemeCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.themes.create")),
) -> WritingThemeResponse:
    """Create a new writing theme."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.themes.create")
    theme_name = payload.name.strip()
    if not theme_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Theme name cannot be empty")
    try:
        db.add_writing_theme(
            name=theme_name,
            class_name=payload.class_name,
            css=payload.css,
            schema_version=payload.schema_version,
            version_parent_id=payload.version_parent_id,
            is_default=payload.is_default,
            order_index=payload.order,
        )
        theme = db.get_writing_theme_by_name(theme_name)
        if not theme:
            raise CharactersRAGDBError("Theme created but could not be retrieved")
        _normalize_theme_response(theme)
        return WritingThemeResponse(**theme)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing theme")


@router.get(
    "/themes/{name}",
    response_model=WritingThemeResponse,
    summary="Get a writing theme",
    tags=["writing"],
)
async def get_writing_theme(
    name: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.themes.get")),
) -> WritingThemeResponse:
    """Fetch a writing theme by name."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.themes.get")
    try:
        theme = db.get_writing_theme_by_name(name)
        if not theme:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Theme not found")
        _normalize_theme_response(theme)
        return WritingThemeResponse(**theme)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing theme")


@router.patch(
    "/themes/{name}",
    response_model=WritingThemeResponse,
    summary="Update a writing theme",
    tags=["writing"],
)
async def update_writing_theme(
    name: str,
    payload: WritingThemeUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.themes.update")),
) -> WritingThemeResponse:
    """Update a writing theme with optimistic locking."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.themes.update")
    update_data: dict[str, Any] = {}
    if payload.name is not None:
        update_data["name"] = payload.name.strip()
        if not update_data["name"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Theme name cannot be empty")
    if payload.class_name is not None:
        update_data["class_name"] = payload.class_name
    if payload.css is not None:
        update_data["css"] = payload.css
    if payload.schema_version is not None:
        update_data["schema_version"] = payload.schema_version
    if payload.version_parent_id is not None:
        update_data["version_parent_id"] = payload.version_parent_id
    if payload.is_default is not None:
        update_data["is_default"] = payload.is_default
    if payload.order is not None:
        update_data["order_index"] = payload.order
    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update")
    try:
        db.update_writing_theme(name, update_data, expected_version)
        theme_name = update_data.get("name", name)
        theme = db.get_writing_theme_by_name(theme_name)
        if not theme:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Theme not found")
        _normalize_theme_response(theme)
        return WritingThemeResponse(**theme)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing theme")


@router.delete(
    "/themes/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a writing theme",
    tags=["writing"],
)
async def delete_writing_theme(
    name: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.themes.delete")),
) -> Response:
    """Soft-delete a writing theme."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.themes.delete")
    try:
        db.soft_delete_writing_theme(name, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing theme")


@router.get(
    "/snapshot/export",
    response_model=WritingSnapshotExportResponse,
    summary="Export writing snapshot",
    tags=["writing"],
)
async def export_writing_snapshot(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.snapshot.export")),
) -> WritingSnapshotExportResponse:
    """Export sessions, templates, and themes as a single snapshot payload."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.snapshot.export")
    try:
        session_items: list[WritingSnapshotSessionItem] = []
        for session_summary in _list_all_writing_sessions(db):
            session_id = str(session_summary.get("id") or "").strip()
            if not session_id:
                continue
            session = db.get_writing_session(session_id)
            if not session:
                continue
            session_items.append(
                WritingSnapshotSessionItem(
                    id=session_id,
                    name=str(session.get("name") or ""),
                    payload=session.get("payload") or {},
                    schema_version=int(session.get("schema_version") or 1),
                    version_parent_id=session.get("version_parent_id"),
                )
            )

        template_items = [
            WritingSnapshotTemplateItem(
                name=str(template.get("name") or ""),
                payload=template.get("payload") or {},
                schema_version=int(template.get("schema_version") or 1),
                version_parent_id=template.get("version_parent_id"),
                is_default=bool(template.get("is_default")),
            )
            for template in _list_all_writing_templates(db)
        ]

        theme_items: list[WritingSnapshotThemeItem] = []
        for theme in _list_all_writing_themes(db):
            normalized = dict(theme)
            _normalize_theme_response(normalized)
            theme_items.append(
                WritingSnapshotThemeItem(
                    name=str(normalized.get("name") or ""),
                    class_name=normalized.get("class_name"),
                    css=normalized.get("css"),
                    schema_version=int(normalized.get("schema_version") or 1),
                    version_parent_id=normalized.get("version_parent_id"),
                    is_default=bool(normalized.get("is_default")),
                    order=int(normalized.get("order") or 0),
                )
            )

        counts = WritingSnapshotCounts(
            sessions=len(session_items),
            templates=len(template_items),
            themes=len(theme_items),
        )
        return WritingSnapshotExportResponse(
            version=1,
            counts=counts,
            sessions=session_items,
            templates=template_items,
            themes=theme_items,
        )
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing snapshot export")


@router.post(
    "/snapshot/import",
    response_model=WritingSnapshotImportResponse,
    summary="Import writing snapshot",
    tags=["writing"],
)
async def import_writing_snapshot(
    payload: WritingSnapshotImportRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.snapshot.import")),
) -> WritingSnapshotImportResponse:
    """Import sessions, templates, and themes from a snapshot payload."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.snapshot.import")
    try:
        # Validate all session names up front before touching the DB.
        for session_item in payload.snapshot.sessions:
            if not session_item.name.strip():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Session name cannot be empty")
        for template_item in payload.snapshot.templates:
            if not template_item.name.strip():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Template name cannot be empty")
        for theme_item in payload.snapshot.themes:
            if not theme_item.name.strip():
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Theme name cannot be empty")

        # Delegate to the DB layer which runs everything in a single atomic
        # transaction using one connection — no partial imports on failure.
        imported = db.import_writing_snapshot(
            mode=payload.mode,
            sessions=[
                {
                    "id": s.id,
                    "name": s.name,
                    "payload": s.payload,
                    "schema_version": s.schema_version,
                    "version_parent_id": s.version_parent_id,
                }
                for s in payload.snapshot.sessions
            ],
            templates=[
                {
                    "name": t.name,
                    "payload": t.payload,
                    "schema_version": t.schema_version,
                    "version_parent_id": t.version_parent_id,
                    "is_default": t.is_default,
                }
                for t in payload.snapshot.templates
            ],
            themes=[
                {
                    "name": th.name,
                    "class_name": th.class_name,
                    "css": th.css,
                    "schema_version": th.schema_version,
                    "version_parent_id": th.version_parent_id,
                    "is_default": th.is_default,
                    "order_index": th.order,
                }
                for th in payload.snapshot.themes
            ],
        )

        return WritingSnapshotImportResponse(
            mode=payload.mode,
            imported=WritingSnapshotCounts(
                sessions=imported["sessions"],
                templates=imported["templates"],
                themes=imported["themes"],
            ),
        )
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing snapshot import")


@router.post(
    "/tokenize",
    response_model=WritingTokenizeResponse,
    summary="Tokenize text for a provider/model",
    description="Uses provider-native tokenizers when available with tiktoken fallback. Special tokens are allowed for tiktoken paths (disallowed_special=()).",
    tags=["writing"],
)
async def tokenize_writing_text(
    payload: WritingTokenizeRequest,
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.tokenize")),
) -> WritingTokenizeResponse:
    """Tokenize text for the requested provider/model."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.tokenize")
    include_strings = True
    if payload.options is not None:
        include_strings = bool(payload.options.include_strings)
    try:
        (
            encoding,
            tokenizer_name,
            tokenizer_kind,
            tokenizer_source,
            detokenize_available,
            count_accuracy,
            strict_mode_effective,
        ) = _resolve_tokenizer_details(payload.provider, payload.model)
    except TokenizerUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if strict_mode_effective and count_accuracy != "exact":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Exact tokenizer unavailable for provider/model",
        )
    try:
        token_ids = _tokenizer_encode(encoding, payload.text)
    except TokenizerUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    token_strings = None
    if include_strings:
        try:
            token_strings = [
                _tokenizer_decode_single(encoding, token_id) for token_id in token_ids
            ]
        except TokenizerUnavailable as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    meta = WritingTokenizeMeta(
        provider=payload.provider,
        model=payload.model,
        tokenizer=tokenizer_name,
        tokenizer_kind=tokenizer_kind,
        tokenizer_source=tokenizer_source,
        detokenize_available=detokenize_available,
        count_accuracy=count_accuracy,
        strict_mode_effective=strict_mode_effective,
        input_chars=len(payload.text),
        token_count=len(token_ids),
        warnings=[],
    )
    return WritingTokenizeResponse(ids=token_ids, strings=token_strings, meta=meta)


@router.post(
    "/detokenize",
    response_model=WritingDetokenizeResponse,
    summary="Detokenize token IDs for a provider/model",
    description="Decodes token IDs into text fragments using the provider/model tokenizer.",
    tags=["writing"],
)
async def detokenize_writing_tokens(
    payload: WritingDetokenizeRequest,
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.tokenize")),
) -> WritingDetokenizeResponse:
    """Detokenize IDs for the requested provider/model."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.tokenize")
    try:
        (
            encoding,
            tokenizer_name,
            tokenizer_kind,
            tokenizer_source,
            detokenize_available,
            count_accuracy,
            strict_mode_effective,
        ) = _resolve_tokenizer_details(payload.provider, payload.model)
    except TokenizerUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if strict_mode_effective and count_accuracy != "exact":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Exact tokenizer unavailable for provider/model",
        )

    if not detokenize_available:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Detokenize not available for provider/model",
        )

    try:
        normalized_ids = [int(token_id) for token_id in payload.ids]
        token_strings = [_tokenizer_decode_single(encoding, token_id) for token_id in normalized_ids]
        text = _tokenizer_decode(encoding, normalized_ids)
    except TokenizerUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - invalid token id payloads
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unable to detokenize ids: {exc}",
        ) from exc

    meta = WritingTokenizeMeta(
        provider=payload.provider,
        model=payload.model,
        tokenizer=tokenizer_name,
        tokenizer_kind=tokenizer_kind,
        tokenizer_source=tokenizer_source,
        detokenize_available=detokenize_available,
        count_accuracy=count_accuracy,
        strict_mode_effective=strict_mode_effective,
        input_chars=len(text),
        token_count=len(normalized_ids),
        warnings=[],
    )
    return WritingDetokenizeResponse(text=text, strings=token_strings, meta=meta)


@router.post(
    "/token-count",
    response_model=WritingTokenCountResponse,
    summary="Count tokens for a provider/model",
    description="Uses provider-native tokenizers when available with tiktoken fallback. Special tokens are allowed for tiktoken paths (disallowed_special=()).",
    tags=["writing"],
)
async def count_writing_tokens(
    payload: WritingTokenCountRequest,
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.token_count")),
) -> WritingTokenCountResponse:
    """Count tokens for the requested provider/model."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.token_count")
    try:
        (
            encoding,
            tokenizer_name,
            tokenizer_kind,
            tokenizer_source,
            detokenize_available,
            count_accuracy,
            strict_mode_effective,
        ) = _resolve_tokenizer_details(payload.provider, payload.model)
    except TokenizerUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    if strict_mode_effective and count_accuracy != "exact":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Exact tokenizer unavailable for provider/model",
        )
    try:
        token_count = _tokenizer_count(encoding, payload.text)
    except TokenizerUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    meta = WritingTokenizeMeta(
        provider=payload.provider,
        model=payload.model,
        tokenizer=tokenizer_name,
        tokenizer_kind=tokenizer_kind,
        tokenizer_source=tokenizer_source,
        detokenize_available=detokenize_available,
        count_accuracy=count_accuracy,
        strict_mode_effective=strict_mode_effective,
        input_chars=len(payload.text),
        token_count=token_count,
        warnings=[],
    )
    return WritingTokenCountResponse(count=token_count, meta=meta)


@router.post(
    "/wordclouds",
    response_model=WritingWordcloudResponse,
    summary="Create a wordcloud from text",
    tags=["writing"],
)
async def create_wordcloud(
    payload: WritingWordcloudRequest,
    background_tasks: BackgroundTasks,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.wordclouds.create")),
) -> WritingWordcloudResponse:
    """Queue or return a cached wordcloud for the provided text."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.wordclouds.create")
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text cannot be empty")

    options = _normalize_wordcloud_options(payload.options)
    options_payload = _wordcloud_options_payload(options)
    cache_key, _ = _hash_wordcloud_input(text, options_payload)

    try:
        existing = db.get_writing_wordcloud(cache_key)
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing wordcloud")

    if existing:
        status_value = existing.get("status") or WORDCLOUD_STATUS_QUEUED
        words_payload = existing.get("words")
        meta_payload = existing.get("meta")
        if status_value == WORDCLOUD_STATUS_READY and isinstance(words_payload, list) and isinstance(meta_payload, dict):
            return _build_wordcloud_response_from_row(existing, cached=True)
        if status_value == WORDCLOUD_STATUS_FAILED:
            return _build_wordcloud_response_from_row(existing, cached=False)
        response = _build_wordcloud_response_from_row(existing, cached=False)
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=_model_dump(response))

    try:
        db.add_writing_wordcloud_job(
            cache_key,
            options_payload,
            input_chars=len(text),
            status=WORDCLOUD_STATUS_QUEUED,
        )
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing wordcloud")

    if _is_test_mode():
        _run_wordcloud_job(db, cache_key, text, options)
        try:
            refreshed = db.get_writing_wordcloud(cache_key)
            if refreshed:
                return _build_wordcloud_response_from_row(refreshed, cached=False)
        except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
            _handle_db_errors(exc, "writing wordcloud")
        return WritingWordcloudResponse(id=cache_key, status=WORDCLOUD_STATUS_FAILED, error="Wordcloud job failed")

    background_tasks.add_task(_run_wordcloud_job, db, cache_key, text, options)
    response = WritingWordcloudResponse(id=cache_key, status=WORDCLOUD_STATUS_QUEUED, cached=False)
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=_model_dump(response))


@router.get(
    "/wordclouds/{wordcloud_id}",
    response_model=WritingWordcloudResponse,
    summary="Get a wordcloud job status/result",
    tags=["writing"],
)
async def get_wordcloud(
    wordcloud_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.wordclouds.get")),
) -> WritingWordcloudResponse:
    """Fetch a wordcloud job by ID."""
    await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.wordclouds.get")
    try:
        existing = db.get_writing_wordcloud(wordcloud_id)
        if not existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Wordcloud not found")
        return _build_wordcloud_response_from_row(existing, cached=False)
    except HTTPException:
        raise
    except _WRITING_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "writing wordcloud")
