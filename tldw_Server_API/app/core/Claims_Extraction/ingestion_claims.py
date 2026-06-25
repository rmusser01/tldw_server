"""
ingestion_claims.py - Ingestion-time claim (factual statement) extraction utilities.

Stage 2 MVP: Heuristic extraction of short factual sentences from chunks,
with storage in MediaDatabase.Claims. Optional, behind config flags.
"""

from __future__ import annotations

import contextlib
import hashlib
import sqlite3
import time
import uuid
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Claims_Extraction.alignment import align_claim
from tldw_Server_API.app.core.Claims_Extraction.budget_guard import (
    ClaimsJobBudget,
    ClaimsJobContext,
    estimate_claims_tokens,
)
from tldw_Server_API.app.core.Claims_Extraction.extractor_catalog import (
    LLM_PROVIDER_MODES,
    detect_claims_language,
    resolve_claims_extractor_mode,
)
from tldw_Server_API.app.core.Claims_Extraction.extractor_registry import (
    extract_heuristic_claims_texts,
    extract_ner_claims_texts,
    run_sync_claims_strategy,
)
from tldw_Server_API.app.core.Claims_Extraction.monitoring import (
    estimate_claims_cost,
    record_claims_alignment_event,
    record_claims_budget_exhausted,
    record_claims_fallback,
    record_claims_output_parse_event,
    record_claims_provider_request,
    record_claims_response_format_selection,
    record_claims_throttle,
    should_throttle_claims_provider,
)
from tldw_Server_API.app.core.Claims_Extraction.output_parser import (
    ClaimsOutputParseError,
    coerce_llm_response_text,
    extract_claim_texts,
    parse_claims_llm_output,
    resolve_claims_response_format,
)
from tldw_Server_API.app.core.Claims_Extraction.review_assignment import apply_review_rules
from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_alignment_config,
    resolve_claims_context_window_chars,
    resolve_claims_extraction_passes,
    resolve_claims_json_parse_mode,
    resolve_claims_llm_config,
)
from tldw_Server_API.app.core.config import settings as _settings
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    split_system_message,
)
from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt

_CLAIMS_COERCE_EXCEPTIONS = (TypeError, ValueError, OverflowError)

_CLAIMS_TEMPLATE_FORMAT_EXCEPTIONS = (
    KeyError,
    IndexError,
    TypeError,
    ValueError,
)

_CLAIMS_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
    TimeoutError,
    ConnectionError,
)

_CLAIMS_RESPONSE_PARSE_EXCEPTIONS = (
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    AttributeError,
)

_CLAIMS_STORE_EXCEPTIONS = _CLAIMS_NONCRITICAL_EXCEPTIONS + (sqlite3.Error,)

_INGESTION_CLAIMS_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": True,
}


def _normalize_claim_text(text: Any) -> str:
    """Normalize claim text for stable comparisons.

    Args:
        text: Raw claim text value. Any type is accepted and coerced to string.

    Returns:
        Lowercased string with surrounding whitespace removed and internal
        whitespace collapsed to single spaces.

    Behavior:
        Falsey inputs (including ``None``) normalize to ``""``.
    """
    return " ".join(str(text or "").strip().lower().split())


def _coerce_span(span_value: Any) -> tuple[int, int] | None:
    """Coerce span input into a validated ``(start, end)`` tuple.

    Args:
        span_value: Candidate span value. Accepted format is a 2-item list/tuple
            whose values are integer-coercible (for example, ``[10, "25"]``).

    Returns:
        A ``(start, end)`` tuple when valid; otherwise ``None``.

    Behavior:
        Returns ``None`` for unsupported shapes, coercion failures, negative
        starts, or non-positive lengths (``end <= start``).
    """
    if not isinstance(span_value, (list, tuple)) or len(span_value) != 2:
        return None
    try:
        start = int(span_value[0])
        end = int(span_value[1])
    except _CLAIMS_COERCE_EXCEPTIONS:
        return None
    if start < 0 or end <= start:
        return None
    return start, end


def _spans_overlap(first: tuple[int, int] | None, second: tuple[int, int] | None) -> bool:
    """Determine whether two spans overlap.

    Args:
        first: First optional ``(start, end)`` span.
        second: Second optional ``(start, end)`` span.

    Returns:
        ``True`` when spans overlap, or when either span is ``None``.

    Behavior:
        Concrete spans use half-open overlap semantics (``[start, end)``).
        Unknown spans (``None``) are treated as potentially overlapping.
    """
    if first is None or second is None:
        return True
    return first[0] < second[1] and second[0] < first[1]


def _dedupe_claim_candidates(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate claim candidates while preserving input order.

    Args:
        claims: Candidate claim objects containing ``claim_text`` and optional
            ``chunk_index``/``span`` fields.

    Returns:
        A filtered list of claim dicts with duplicates removed.

    Behavior:
        Two candidates are considered duplicates when they share the same
        ``chunk_index``, have the same normalized ``claim_text``, and their
        spans overlap per ``_spans_overlap``.
    """
    deduped: list[dict[str, Any]] = []
    for claim in claims:
        chunk_index = int(claim.get("chunk_index", 0))
        normalized = _normalize_claim_text(claim.get("claim_text"))
        span = _coerce_span(claim.get("span"))
        is_duplicate = False
        for existing in deduped:
            if int(existing.get("chunk_index", 0)) != chunk_index:
                continue
            if _normalize_claim_text(existing.get("claim_text")) != normalized:
                continue
            existing_span = _coerce_span(existing.get("span"))
            if _spans_overlap(existing_span, span):
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(claim)
    return deduped


def extract_claims_for_chunks(
    chunks: list[dict[str, Any]],
    *,
    extractor_mode: str = "heuristic",
    max_per_chunk: int = 3,
    language: str | None = None,
    budget: ClaimsJobBudget | None = None,
    job_context: ClaimsJobContext | None = None,
) -> list[dict[str, Any]]:
    """
    Extract a small set of candidate factual statements per chunk.

    Returns items with: chunk_index, claim_text.
    """
    claims: list[dict[str, Any]] = []
    resolved_mode = (extractor_mode or "heuristic").strip().lower()
    resolved_language = (language or "").strip().lower() if language else None
    if resolved_mode in {"auto", "detect"}:
        combined_text = " ".join(
            str((ch or {}).get("text") or (ch or {}).get("content") or "").strip()
            for ch in (chunks or [])[:5]
        )
        resolved_mode, resolved_language = resolve_claims_extractor_mode(resolved_mode, combined_text)

    parse_mode = resolve_claims_json_parse_mode(_settings, default_mode="lenient")
    alignment_mode, alignment_threshold = resolve_claims_alignment_config(
        _settings,
        default_mode="fuzzy",
        default_threshold=0.75,
    )
    context_window_chars = resolve_claims_context_window_chars(_settings, default=0)
    extraction_passes = resolve_claims_extraction_passes(_settings, default=1)
    current_mode = resolved_mode
    current_prompt_context = ""
    current_prompt_chunk = ""

    def _llm_extract_claim_texts(txt: str, max_items: int, _language_hint: str | None) -> list[str]:
        nonlocal current_mode
        nonlocal current_prompt_context
        nonlocal current_prompt_chunk
        cost_estimate = None
        default_provider, model_override, temperature = resolve_claims_llm_config(
            _settings,
            default_provider="openai",
            default_temperature=0.1,
        )
        provider = current_mode if current_mode in LLM_PROVIDER_MODES else default_provider
        provider_name = normalize_provider(provider)
        response_format = resolve_claims_response_format(
            provider,
            schema_name="ingestion_claims_extraction",
            json_schema=_INGESTION_CLAIMS_RESPONSE_SCHEMA,
        )
        record_claims_response_format_selection(
            provider=provider,
            model=model_override or "",
            mode="ingestion",
            response_format=response_format,
        )

        system = load_prompt("ingestion", "claims_extractor_system") or (
            "You extract specific, verifiable, decontextualized factual propositions. Output strict JSON."
        )
        base = load_prompt("ingestion", "claims_extractor_prompt") or (
            "Extract up to {max_claims} atomic factual propositions from CHUNK only. "
            "Never extract claims from CONTEXT. "
            "Each proposition should stand alone without surrounding context, be specific, and checkable. "
            "Return JSON: {{\"claims\":[{{\"text\": str}}]}}. Do not include explanations.\n\n"
            "CONTEXT (do not extract claims from this):\n{context}\n\n"
            "CHUNK (extract claims only from this):\n{answer}"
        )
        chunk_text = current_prompt_chunk or txt
        prompt_vars = {
            "max_claims": max_items,
            "answer": chunk_text,
            "chunk": chunk_text,
            "context": current_prompt_context,
        }
        try:
            prompt = base.format(**prompt_vars)
        except _CLAIMS_TEMPLATE_FORMAT_EXCEPTIONS:
            _tmpl = base.replace("{", "{{").replace("}", "}}")
            _tmpl = (
                _tmpl.replace("{{max_claims}}", "{max_claims}")
                .replace("{{answer}}", "{answer}")
                .replace("{{chunk}}", "{chunk}")
                .replace("{{context}}", "{context}")
            )
            prompt = _tmpl.format(**prompt_vars)

        messages = [{"role": "user", "content": prompt}]
        timeout_sec = 8.0
        try:
            timeout_sec = float(_settings.get("CLAIMS_LLM_TIMEOUT_SEC", 8.0))
        except _CLAIMS_COERCE_EXCEPTIONS:
            timeout_sec = 8.0

        def _call_provider() -> Any:
            override = globals().get("chat_api_call")
            if callable(override):
                return override(
                    api_endpoint=provider,
                    messages_payload=messages,
                    api_key=None,
                    temp=temperature,
                    system_message=system,
                    streaming=False,
                    model=model_override,
                    response_format=response_format,
                )

            adapter = get_registry().get_adapter(provider_name)
            if adapter is not None:
                app_config = ensure_app_config()
                resolved_model = model_override or resolve_provider_model(provider_name, app_config)
                if not resolved_model:
                    raise ChatConfigurationError(
                        provider=provider_name,
                        message="Model is required for provider.",
                    )
                system_message, cleaned_messages = split_system_message(
                    [{"role": "system", "content": system}] + messages if system else messages
                )
                request: dict[str, Any] = {
                    "messages": cleaned_messages,
                    "system_message": system_message,
                    "model": resolved_model,
                    "api_key": resolve_provider_api_key_from_config(provider_name, app_config),
                    "temperature": temperature,
                    "app_config": app_config,
                }
                if response_format is not None:
                    request["response_format"] = response_format
                return adapter.chat(request, timeout=timeout_sec)

            from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call as _cac  # type: ignore

            return _cac(
                api_endpoint=provider,
                messages_payload=messages,
                api_key=None,
                temp=temperature,
                system_message=system,
                streaming=False,
                model=model_override,
                response_format=response_format,
            )

        cost_estimate = estimate_claims_cost(
            provider=provider,
            model=model_override or "",
            text=prompt,
        )
        budget_ratio = budget.remaining_ratio() if budget is not None else None
        throttle, reason = should_throttle_claims_provider(
            provider=provider,
            model=model_override or "",
            budget_ratio=budget_ratio,
        )
        if throttle:
            record_claims_throttle(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                reason=reason or "throttle",
            )
            record_claims_fallback(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                reason=reason or "throttle",
            )
            return []
        if budget is not None:
            prompt_tokens = estimate_claims_tokens(prompt)
            if not budget.reserve(cost_usd=cost_estimate, tokens=prompt_tokens):
                record_claims_budget_exhausted(
                    provider=provider,
                    model=model_override or "",
                    mode="ingestion",
                    reason=budget.exhausted_reason or "budget",
                )
                record_claims_fallback(
                    provider=provider,
                    model=model_override or "",
                    mode="ingestion",
                    reason=budget.exhausted_reason or "budget",
                )
                return []

        try:
            import concurrent.futures as _futures

            start_time = time.time()
            _exec = _futures.ThreadPoolExecutor(max_workers=1)
            try:
                fut = _exec.submit(_call_provider)
                try:
                    resp = fut.result(timeout=timeout_sec)
                except _futures.TimeoutError:
                    with contextlib.suppress(_CLAIMS_NONCRITICAL_EXCEPTIONS):
                        fut.cancel()
                    _exec.shutdown(wait=False, cancel_futures=True)
                    _exec = None
                    record_claims_provider_request(
                        provider=provider,
                        model=model_override or "",
                        mode="ingestion",
                        latency_s=None,
                        error="timeout",
                        estimated_cost=cost_estimate,
                    )
                    record_claims_fallback(
                        provider=provider,
                        model=model_override or "",
                        mode="ingestion",
                        reason="timeout",
                    )
                    return []
            finally:
                if _exec is not None:
                    _exec.shutdown(wait=True)
            record_claims_provider_request(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                latency_s=time.time() - start_time,
                estimated_cost=cost_estimate,
            )
        except _CLAIMS_NONCRITICAL_EXCEPTIONS as exc:
            record_claims_provider_request(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                latency_s=None,
                error=str(exc),
                estimated_cost=cost_estimate,
            )
            record_claims_fallback(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                reason="provider_error",
            )
            logger.debug(f"LLM-based claim extraction failed ({current_mode}): {exc}")
            return []

        text = coerce_llm_response_text(resp)
        if budget is not None:
            budget.add_usage(tokens=estimate_claims_tokens(text))

        try:
            parsed = parse_claims_llm_output(
                text,
                parse_mode=parse_mode,
                strip_think_tags=True,
            )
            claim_texts = extract_claim_texts(
                parsed,
                wrapper_key="claims",
                parse_mode=parse_mode,
                max_claims=max_items,
            )
            if claim_texts:
                record_claims_output_parse_event(
                    provider=provider,
                    model=model_override or "",
                    mode="ingestion",
                    parse_mode=parse_mode,
                    outcome="success",
                )
                return claim_texts
            record_claims_output_parse_event(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                parse_mode=parse_mode,
                outcome="empty",
                reason="no_claims",
            )
            record_claims_fallback(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                reason="empty_claims",
            )
            return []
        except ClaimsOutputParseError as exc:
            record_claims_output_parse_event(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                parse_mode=parse_mode,
                outcome="error",
                reason=exc.__class__.__name__,
            )
            record_claims_fallback(
                provider=provider,
                model=model_override or "",
                mode="ingestion",
                reason="parse_error",
            )
            logger.debug(f"Failed to parse LLM claims JSON ({current_mode}): {exc}")
            return []

    strategy_map = {
        "heuristic": extract_heuristic_claims_texts,
        "ner": extract_ner_claims_texts,
        "aps": _llm_extract_claim_texts,
        "llm": _llm_extract_claim_texts,
    }
    llm_like_modes = {"llm", "aps", *LLM_PROVIDER_MODES}
    run_passes = extraction_passes if resolved_mode in llm_like_modes else 1
    include_context = context_window_chars > 0 and resolved_mode in llm_like_modes

    for pass_index in range(max(1, run_passes)):
        previous_tail = ""
        for ch in chunks or []:
            txt = (ch or {}).get("text") or (ch or {}).get("content") or ""
            meta = (ch or {}).get("metadata", {}) or {}
            idx = int(meta.get("chunk_index") or meta.get("index") or 0)
            lang_hint = resolved_language or detect_claims_language(txt)
            current_mode = resolved_mode
            current_prompt_chunk = txt
            current_prompt_context = previous_tail if include_context and previous_tail else ""

            dispatch = run_sync_claims_strategy(
                requested_mode=current_mode,
                text=txt,
                max_claims=max_per_chunk,
                strategy_map=strategy_map,
                fallback_mode="heuristic",
                language=lang_hint,
                catch_exceptions=_CLAIMS_NONCRITICAL_EXCEPTIONS,
            )

            for sent in dispatch.claim_texts:
                alignment_result = align_claim(
                    txt,
                    sent,
                    mode=alignment_mode,
                    threshold=alignment_threshold,
                )
                if include_context and (
                    alignment_result is None
                    or not getattr(alignment_result, "span", None)
                ):
                    continue
                record_claims_alignment_event(
                    context="ingestion_extract",
                    mode=alignment_mode,
                    result=alignment_result,
                )
                claims.append(
                    {
                        "chunk_index": idx,
                        "claim_text": sent,
                        "span": list(alignment_result.span) if alignment_result is not None else None,
                        "extractor_mode": dispatch.mode,
                        "extraction_pass": pass_index + 1,
                        "alignment_method": alignment_result.method if alignment_result is not None else None,
                        "alignment_score": alignment_result.score if alignment_result is not None else None,
                    }
                )
            if include_context:
                previous_tail = txt[-context_window_chars:]
    if run_passes > 1 or include_context:
        return _dedupe_claim_candidates(claims)
    return claims


def store_claims(
    db: Any,
    *,
    media_id: int,
    chunk_texts_by_index: dict[int, str],
    claims: list[dict[str, Any]],
    extractor: str = "heuristic",
    extractor_version: str = "v1",
) -> int:
    """
    Store extracted claims into Claims table via MediaDatabase.upsert_claims.
    Computes chunk_hash from the chunk text for linkage.
    """
    if not claims:
        return 0
    rows: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for c in claims:
        idx = int(c.get("chunk_index", 0))
        ctext = str(c.get("claim_text", ""))
        claim_uuid = str(c.get("uuid") or uuid.uuid4())
        c["uuid"] = claim_uuid
        chunk_txt = chunk_texts_by_index.get(idx, "")
        chash = hashlib.sha256(chunk_txt.encode()).hexdigest() if chunk_txt else hashlib.sha256(b"").hexdigest()
        span_start = None
        span_end = None
        span_value = c.get("span")
        if isinstance(span_value, (list, tuple)) and len(span_value) == 2:
            try:
                maybe_start = int(span_value[0])
                maybe_end = int(span_value[1])
                if maybe_start >= 0 and maybe_end >= maybe_start:
                    span_start = maybe_start
                    span_end = maybe_end
            except _CLAIMS_COERCE_EXCEPTIONS:
                span_start = None
                span_end = None
        rows.append({
            "media_id": int(media_id),
            "chunk_index": idx,
            "span_start": span_start,
            "span_end": span_end,
            "claim_text": ctext,
            "confidence": None,
            "extractor": extractor,
            "extractor_version": extractor_version,
            "chunk_hash": chash,
            "uuid": claim_uuid,
        })
    try:
        rows = apply_review_rules(db=db, claims=rows)
        for row in rows:
            if row.get("reviewer_id") is not None or row.get("review_group"):
                assignments.append(
                    {
                        "uuid": row.get("uuid"),
                        "reviewer_id": row.get("reviewer_id"),
                        "review_group": row.get("review_group"),
                    }
                )
        inserted = db.upsert_claims(rows)
        if assignments:
            from tldw_Server_API.app.core.Claims_Extraction.claims_notifications import (
                record_review_assignment_notifications,
            )
            owner_row = db.execute_query(
                "SELECT owner_user_id, client_id FROM Media WHERE id = ?",
                (int(media_id),),
            ).fetchone()
            owner_user_id = None
            client_id = None
            if owner_row:
                try:
                    owner_user_id = owner_row["owner_user_id"]
                    client_id = owner_row["client_id"]
                except _CLAIMS_RESPONSE_PARSE_EXCEPTIONS:
                    try:
                        owner_user_id = owner_row[0]
                    except _CLAIMS_RESPONSE_PARSE_EXCEPTIONS:
                        owner_user_id = None
                    try:
                        client_id = owner_row[1]
                    except _CLAIMS_RESPONSE_PARSE_EXCEPTIONS:
                        client_id = None
            if owner_user_id is None:
                owner_user_id = client_id or db.client_id
            notification_ids = record_review_assignment_notifications(
                db=db,
                owner_user_id=str(owner_user_id),
                assignments=assignments,
            )
            if notification_ids:
                from tldw_Server_API.app.core.Claims_Extraction import claims_jobs

                if claims_jobs.claims_jobs_enabled():
                    try:
                        claims_jobs.enqueue_claims_review_notification(
                            owner_user_id=str(owner_user_id),
                            notification_ids=notification_ids,
                        )
                    except _CLAIMS_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug("Failed to enqueue claims review assignment notification job: {}", exc)
                else:
                    from tldw_Server_API.app.core.Claims_Extraction.claims_notifications import (
                        dispatch_claim_review_notifications,
                    )

                    dispatch_claim_review_notifications(
                        db_path=str(db.db_path_str),
                        owner_user_id=str(owner_user_id),
                        notification_ids=notification_ids,
                    )
        return inserted
    except _CLAIMS_STORE_EXCEPTIONS as e:  # pragma: no cover
        logger.error(f"Failed to store claims for media_id={media_id}: {e}")
        return 0
