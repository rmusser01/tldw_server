"""Adapt exact section evidence and owned provider calls to strict Claims checks."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.Research_Workspace.section_attribution import render_attribution_context
from tldw_Server_API.app.core.Sync_Contracts.citation_support_types import validate_claim_verification


def section_evidence_map(section: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve saved evidence against the current, fingerprint-checked owned source."""
    evidence = {}
    source = snapshot["content"]
    for batch in section["batches"]:
        for item in batch.get("evidence", []):
            start, end = item.get("start_char"), item.get("end_char")
            if (
                type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source)
                or source[start:end] != item.get("quote")
            ):
                continue  # Missing bindings become explicit unknown-citation findings.
            evidence[item["id"]] = {
                "text": source[start:end], "source_id": section["source_id"], "media_id": section["media_id"],
                "start_char": start, "end_char": end, "navigation_version": section["navigation_version"],
                "page_spans": item.get("page_spans", []),
                "attribution_context": render_attribution_context(section.get("attribution_contexts", []), start, end),
            }
    return evidence


def batch_claim_text(batch: dict[str, Any]) -> str:
    """Keep every displayed extracted claim field and its actual passage bindings."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for item in batch.get("evidence", []):
        fields = tuple(str(item.get(key) or "") for key in ("claim", "author", "interpretation", "limitations"))
        groups.setdefault(fields, []).append(item["id"])
    lines = []
    for (claim, author, interpretation, limitations), ids in groups.items():
        text = claim
        for label, value in (("Attributed to", author), ("Interpretation", interpretation), ("Limitations", limitations)):
            if value:
                text += f" {label}: {value}."
        lines.append(text + " " + " ".join(f"[{item}]" for item in dict.fromkeys(ids)))
    return "\n\n".join(lines)


async def verify_section_batches(section: dict[str, Any], snapshot: dict[str, Any], verify: Any) -> dict[str, Any]:
    """Check completed extraction claims while retaining failed or unread portions."""
    evidence = section_evidence_map(section, snapshot)
    for index, batch in enumerate(section["batches"], 1):
        if batch["status"] != "completed":
            continue
        batch["verification_text"] = batch_claim_text(batch)
        own_evidence = {item["id"]: evidence[item["id"]] for item in batch.get("evidence", []) if item["id"] in evidence}
        await verify(batch, "claim_verification", batch["verification_text"], own_evidence,
                     f"{section['title']}, extracted claims in portion {index}")
    return evidence


def evidence_digest(evidence: dict[str, Any]) -> str:
    """Bind reuse to exact evidence and the citation-to-context prompt protocol."""
    serialized = json.dumps(evidence, sort_keys=True, ensure_ascii=False)
    if any(isinstance(item.get("attribution_context"), str) and item["attribution_context"] for item in evidence.values()):
        # Earlier prompts supplied these contexts without their citation bindings.
        serialized = "citation-attribution-bindings-v1\n" + serialized
    return hashlib.sha256(serialized.encode()).hexdigest()


def verification_is_current(
    report: Any, text: str, evidence: dict[str, Any], *, provider: str | None, model: str | None,
) -> bool:
    """Reuse unchanged terminal findings without promoting negative or uncovered results."""
    try:
        validate_claim_verification(report)
    except ValueError:
        return False
    if (
        not report["findings"] or report["provider"] != provider or report["model"] != model
        or report["text_sha256"] != hashlib.sha256(text.encode()).hexdigest()
        or report.get("evidence_sha256") != evidence_digest(evidence)
    ):
        return False
    previous_end = 0
    for finding in report["findings"]:
        start, end = finding["start"], finding["end"]
        if (end > len(text) or text[start:end] != finding["text"]
                or any(character.isalnum() for character in text[previous_end:start])):
            return False
        terminal = finding["status"] in {"supported", "unsupported", "contradicted", "uncited"}
        hard_limit = finding["status"] == "not_checked" and finding["reason"] in {
            "text_or_unit_limit", "cited_evidence_limit", "report_size_limit",
        }
        if not (terminal or hard_limit):
            return False
        previous_end = end
    return not any(character.isalnum() for character in text[previous_end:])


def unavailable_verification(text: str, *, provider: str | None, model: str | None) -> dict[str, Any]:
    """Retain a bounded, credential-free failure receipt instead of losing a draft."""
    return {
        "version": 1, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "verdict": "unverified",
        "provider": provider, "model": model,
        "coverage": {"total": 1, "evaluated": 0, "supported": 0, "uncovered": 1, "complete": False},
        "findings": [{"id": "C1", "text": text, "start": 0, "end": len(text), "evidence_ids": [],
                      "status": "error", "reason": "Citation support could not be checked. Retry verification."}],
        "evidence": {}, "limits": {}, "context": {},
    }


async def verify_section_text(
    text: str, evidence_by_id: dict[str, Any], *, settings: Any, provider_credentials: Any,
    chat_fn: Any, on_provider_success: Any = None, context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse the Job's already resolved credential and the accounted chat gateway."""
    from tldw_Server_API.app.core.Claims_Extraction.citation_support import verify_cited_text

    async def analyze(api_endpoint, input_data, prompt, api_key, system_message, temp=None, **kwargs):
        response = await chat_fn(
            api_endpoint=settings.provider, model=settings.model,
            messages_payload=[{"role": "user", "content": prompt or str(input_data)}],
            system_message=system_message, temp=0, max_tokens=settings.max_tokens, streaming=False,
            **{PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: provider_credentials},
            **({"response_format": {"type": "json_object"}} if settings.provider in {"openai", "deepseek"} else {}),
        )
        if on_provider_success is not None:
            await on_provider_success()
        return response

    return await verify_cited_text(
        text, evidence_by_id, provider=settings.provider, model=settings.model, analyze_fn=analyze, context=context,
    )


def render_claim_verification(report: Any) -> str:
    """Include visible support findings in ordinary Markdown and portable exports."""
    if not isinstance(report, dict):
        return "Citation support: not checked."
    label = {"supported": "supported by cited passages", "needs_revision": "needs review", "unverified": "not fully checked"}.get(
        report.get("verdict"), "not checked"
    )
    coverage = report.get("coverage") or {}
    lines = [f"**Citation support: {label}.** {coverage.get('evaluated', 0)} of {coverage.get('total', 0)} text units checked. "
             "Automated source-support assessment; separate from human review."]
    for finding in report.get("findings", []):
        if finding.get("status") == "supported":
            continue
        citations = " ".join(f"[{value}]" for value in finding.get("evidence_ids", []))
        lines.append(f"- {finding.get('status', 'not_checked')}: {finding.get('text', '')} {citations}\n"
                     f"  Reason: {finding.get('reason', 'Support has not been established.')}")
    return "\n\n".join(lines)
