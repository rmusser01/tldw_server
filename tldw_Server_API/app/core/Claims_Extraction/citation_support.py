"""Bound complete generated text units to exact cited passages for semantic review."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
import threading
from collections.abc import Iterator, Mapping
from typing import Any

from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.Sync_Contracts.citation_support_types import MAX_CLAIM_VERIFICATION_BYTES

from .analyze_types import ClaimsAnalyzeCallable
from .citation_syntax import CITATION_RUN, extract_citation_ids
from .claims_engine import Claim, HybridClaimVerifier

LIMITS = {"max_units": 256, "max_text_chars": 100_000, "max_evidence_chars": 16_000,
          "concurrency": 4, "timeout_seconds": 120, "total_timeout_seconds": 120}
_PARAGRAPH = re.compile(r"\S[\s\S]*?(?=\n[ \t]*\n|\n[ \t]*(?:[-*+] |\d+[.)] )|\Z)")
_CITATIONS = CITATION_RUN
_ID = re.compile(r"(?:E\d+(?:\.\d+)+|P\d+|\d+)\Z")
_PROVENANCE = ("source_id", "media_id", "start_char", "end_char", "navigation_version", "page_spans")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit(text: str, start: int, end: int, ids: list[str]) -> dict[str, Any] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if not any(character.isalnum() for character in text[start:end]):
        return None
    return {"text": text[start:end], "start": start, "end": end, "evidence_ids": ids}


def _iter_units(text: str) -> Iterator[dict[str, Any]]:
    """Keep paragraph/bullet boundaries and literal IDs; never infer range interiors."""
    for paragraph in _PARAGRAPH.finditer(text):
        cursor = paragraph.start()
        for match in _CITATIONS.finditer(text, paragraph.start(), paragraph.end()):
            item = _unit(text, cursor, match.end(), extract_citation_ids(match[0]))
            if item:
                yield item
            cursor = match.end()
        tail = _unit(text, cursor, paragraph.end(), [])
        if tail:
            yield tail


def _units(text: str, max_units: int | None = None) -> list[dict[str, Any]]:
    """Bound parsed units while retaining the entire overflow tail as unchecked."""
    units = []
    for index, item in enumerate(_iter_units(text), 1):
        if max_units is not None and index > max_units:
            start = units[-1]["end"] if units else 0
            units.append({"id": f"claim-{index}", "text": text[start:], "start": start,
                          "end": len(text), "evidence_ids": []})
            break
        item["id"] = f"claim-{index}"
        units.append(item)
    return units


def _evidence_for_unit(unit: dict[str, Any], evidence: Mapping[str, Any]) -> tuple[list[Document], str, str | None]:
    documents = []
    contexts: dict[str, dict[str, list[int] | None]] = {}
    for citation_id in unit["evidence_ids"]:
        item = evidence.get(citation_id)
        if not _ID.fullmatch(citation_id) or not isinstance(item, Mapping):
            return [], "", "citation_not_resolved"
        text = item.get("text")
        if not isinstance(text, str) or not text.strip() or item.get("resolution_error"):
            return [], "", str(item.get("resolution_error") or "citation_not_resolved")
        documents.append(Document(id=citation_id, content=text, metadata={}, score=0.0))
        context = item.get("attribution_context")
        if isinstance(context, str) and context:
            start, end = item.get("start_char"), item.get("end_char")
            valid_range = type(start) is int and type(end) is int and 0 <= start < end and end - start == len(text)
            contexts.setdefault(context, {})[citation_id] = [start, end] if valid_range else None
    context_text = "\n\n".join(
        "CITED PASSAGE CONTEXT BINDINGS (use the following context only for these cited IDs; "
        "source ranges are [start, end), null means unavailable):\n"
        + json.dumps(bindings, ensure_ascii=False, separators=(",", ":")) + "\n" + context
        for context, bindings in contexts.items()
    )
    if sum(len(document.content) for document in documents) + len(context_text) > LIMITS["max_evidence_chars"]:
        return [], "", "cited_evidence_limit"
    return documents, context_text, None


def _strict_response(raw: Any) -> dict[str, Any]:
    """Require an explicit complete semantic judgment, not legacy text-only success."""
    if not isinstance(raw, dict):
        raise ValueError("judge_completion_unconfirmed")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("judge_response_invalid")
    if choices[0].get("finish_reason") != "stop":
        raise ValueError("judge_completion_unconfirmed")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or len(content) > 16_000:
        raise ValueError("judge_response_invalid")
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        raise ValueError("judge_response_invalid") from None
    if not isinstance(parsed, dict) or parsed.get("label") not in {"supported", "refuted", "nei"}:
        raise ValueError("judge_response_invalid")
    confidence = parsed.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("judge_response_invalid")
    if not isinstance(parsed.get("rationale"), str) or not parsed["rationale"].strip():
        raise ValueError("judge_response_invalid")
    return parsed


async def _verify_unit(
    unit: dict[str, Any], documents: list[Document], attribution_context: str, *,
    provider: str, model: str, analyze_fn: ClaimsAnalyzeCallable,
) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    abandoned = threading.Event()
    owned_coroutines: list[Any] = []

    def close_unstarted() -> None:
        for pending in owned_coroutines:
            if inspect.getcoroutinestate(pending) == inspect.CORO_CREATED:
                pending.close()

    def record_response(raw: Any) -> dict[str, Any]:
        try:
            observed["judgment"] = _strict_response(raw)
        except ValueError as exc:
            observed["error"] = str(exc)
            raise
        return raw

    async def await_response(pending: Any) -> dict[str, Any]:
        try:
            return record_response(await pending)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - redact arbitrary provider failures at this boundary.
            observed.setdefault("error", "judge_provider_failed")
            raise ValueError(observed["error"]) from None

    def strict_analyze(*args: Any, **kwargs: Any) -> Any:
        # ClaimsEngine invokes this wrapper in its bounded worker pool, so sync
        # adapters cannot block the event loop or bypass the provider deadline.
        try:
            raw = analyze_fn(*args, **kwargs)
            if not inspect.isawaitable(raw):
                return record_response(raw)
            if inspect.iscoroutine(raw):
                owned_coroutines.append(raw)
            wrapped = await_response(raw)
            owned_coroutines.append(wrapped)
            if abandoned.is_set():
                close_unstarted()
            return wrapped
        except Exception:  # noqa: BLE001 - redact arbitrary provider failures at this boundary.
            observed.setdefault("error", "judge_provider_failed")
            raise ValueError(observed["error"]) from None

    verifier = HybridClaimVerifier(strict_analyze, strict_citation_support=True,
        verification_provider=provider, verification_model=model,
        attribution_context=attribution_context, timeout_seconds=LIMITS["timeout_seconds"])
    try:
        async with asyncio.timeout(LIMITS["timeout_seconds"]):
            verification = await verifier.verify(
                Claim(id=unit["id"], text=unit["text"], requires_semantic_verification=True),
                query="Verify every proposition against its exact cited passages.", base_documents=documents,
                retrieve_fn=None, top_k=len(documents), mode="llm", doc_only_mode=True,
            )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return {**unit, "status": "error", "reason": "judge_timeout"}
    except Exception:  # noqa: BLE001 - failed verification is explicit, never supported.
        return {**unit, "status": "error", "reason": "judge_failed"}
    finally:
        # A sync adapter may return an awaitable after its caller's deadline.
        # Own both that raw coroutine and our validation wrapper, even if neither
        # has started, without waiting for the timed-out worker to finish.
        abandoned.set()
        close_unstarted()
    judgment = observed.get("judgment")
    if observed.get("error") or not judgment:
        return {**unit, "status": "error", "reason": observed.get("error", "semantic_judgment_unavailable")}
    status = "unsupported"
    if judgment["label"] == "supported" and judgment["confidence"] >= 0.7 and verification.label == "supported":
        status = "supported"
    elif judgment["label"] == "refuted" and judgment["confidence"] >= 0.7:
        status = "contradicted"
    return {**unit, "status": status, "reason": judgment["rationale"][:2048]}


def _receipt_evidence(evidence: Mapping[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    receipts = {}
    for citation_id in dict.fromkeys(value for row in findings for value in row["evidence_ids"]):
        item = evidence.get(citation_id)
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
            continue
        receipt = {key: item[key] for key in _PROVENANCE if key in item}
        receipt["text_sha256"] = _sha(item["text"])
        if isinstance(item.get("attribution_context"), str):
            receipt["attribution_context_sha256"] = _sha(item["attribution_context"])
        receipts[citation_id] = receipt
    return receipts


def _bounded_report(report: dict[str, Any], text: str) -> dict[str, Any]:
    """Never return apparent support whose complete provenance cannot be retained."""
    def fits(value: dict[str, Any]) -> bool:
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")) <= MAX_CLAIM_VERIFICATION_BYTES

    if fits(report):
        return report
    fallback = {**report, "verdict": "unverified", "evidence": {},
        "coverage": {"total": 1, "evaluated": 0, "supported": 0, "uncovered": 1, "complete": False},
        "findings": [{"id": "claim-1", "text": text, "start": 0, "end": len(text),
                      "evidence_ids": [], "status": "not_checked", "reason": "report_size_limit"}]}
    if fits(fallback):
        return fallback
    fallback["context"] = {"unavailable_reason": "report_size_limit"}
    if fits(fallback):
        return fallback
    raise ValueError("verification_report_size_limit")


async def verify_cited_text(
    text: str, evidence_by_id: Mapping[str, Any], *, provider: str, model: str,
    analyze_fn: ClaimsAnalyzeCallable | None = None, context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return portable findings without rewriting the draft or searching uncited sources."""
    if not isinstance(text, str):
        raise ValueError("verification_text_required")
    units = _units(text, LIMITS["max_units"]) if len(text) <= LIMITS["max_text_chars"] else [
        {"id": "claim-1", "text": text, "start": 0, "end": len(text), "evidence_ids": [], "limit": True}]
    semaphore = asyncio.Semaphore(LIMITS["concurrency"])

    async def check(index: int, unit: dict[str, Any]) -> dict[str, Any]:
        if unit.pop("limit", False) or index >= LIMITS["max_units"]:
            return {**unit, "status": "not_checked", "reason": "text_or_unit_limit"}
        if not unit["evidence_ids"]:
            return {**unit, "status": "uncited", "reason": "missing_citation"}
        if not any(character.isalnum() for character in _CITATIONS.sub("", unit["text"])):
            return {**unit, "status": "not_checked", "reason": "citation_without_claim"}
        documents, attribution_context, error = _evidence_for_unit(unit, evidence_by_id)
        if error:
            return {**unit, "status": "not_checked", "reason": error}
        if analyze_fn is None:
            return {**unit, "status": "not_checked", "reason": "judge_unavailable"}
        async with semaphore:
            return await _verify_unit(unit, documents, attribution_context,
                provider=provider, model=model, analyze_fn=analyze_fn)

    tasks = [asyncio.create_task(check(index, unit)) for index, unit in enumerate(units)]
    try:
        if tasks:
            await asyncio.wait(tasks, timeout=LIMITS["total_timeout_seconds"])
        completed = {index: task.result() for index, task in enumerate(tasks) if task.done()}
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    findings = [completed.get(index, {**unit, "status": "not_checked", "reason": "verification_deadline"})
        for index, unit in enumerate(units)]
    evaluated = sum(row["status"] in {"supported", "unsupported", "contradicted"} for row in findings)
    supported = sum(row["status"] == "supported" for row in findings)
    complete = bool(findings) and evaluated == len(findings)
    verdict = "supported" if complete and supported == len(findings) else "unverified"
    if any(row["status"] in {"unsupported", "contradicted", "uncited"} for row in findings):
        verdict = "needs_revision"
    report = {"version": 1, "text_sha256": _sha(text), "verdict": verdict, "provider": provider, "model": model,
        "coverage": {"total": len(findings), "evaluated": evaluated, "supported": supported,
            "uncovered": len(findings) - evaluated, "complete": complete},
        "findings": findings, "evidence": _receipt_evidence(evidence_by_id, findings),
        "limits": dict(LIMITS), "context": dict(context or {})}
    return _bounded_report(report, text)
