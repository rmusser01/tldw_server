"""Deterministic and provider-backed synthesis for deep research artifacts."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,
    parse_structured_output,
)
from tldw_Server_API.app.core.Research.providers.synthesis import SynthesisProvider

from .models import (
    ResearchEvidenceNote,
    ResearchOutlineSection,
    ResearchPlan,
    ResearchSourceRecord,
    ResearchSynthesizedClaim,
    ResearchSynthesisResult,
)


def _stable_digest(*parts: str) -> str:
    joined = "::".join(part.strip().lower() for part in parts if part and part.strip())
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _title_for_focus_area(focus_area: str) -> str:
    return focus_area.replace("_", " ").strip().title()


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        candidate = str(item or "").strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _fallback_reason(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


_CONTRADICTION_MARKERS = (
    "contradict",
    "conflict",
    "disagree",
    "however",
    "in contrast",
    "opposes",
)


def _source_trust_profile(source: ResearchSourceRecord) -> dict[str, Any]:
    source_type = (source.source_type or "").strip().lower()
    provider = (source.provider or "").strip().lower()
    trust_tier = (source.trust_tier or "").strip().lower()

    if provider == "local_corpus" or trust_tier == "internal":
        trust_label = "local_corpus"
        snapshot_policy = "full_artifact"
    elif source_type in {"primary_document", "official_filing", "official_statement"}:
        trust_label = "primary_source"
        snapshot_policy = "metadata_only"
    elif source_type == "academic_paper" or provider in {"arxiv", "pubmed", "crossref"}:
        trust_label = "secondary_source"
        snapshot_policy = "metadata_only"
    elif source_type in {"web_result", "metadata_record"}:
        trust_label = "metadata_only"
        snapshot_policy = "metadata_only"
    else:
        trust_label = "external_source"
        snapshot_policy = "metadata_only"

    warnings: list[str] = []
    if snapshot_policy == "metadata_only":
        warnings.append("full_source_snapshot_unavailable")

    return {
        "source_id": source.source_id,
        "title": source.title,
        "provider": source.provider,
        "source_type": source.source_type,
        "trust_tier": source.trust_tier,
        "trust_label": trust_label,
        "snapshot_policy": snapshot_policy,
        "warnings": warnings,
    }


def _extract_contradictions(evidence_notes: list[ResearchEvidenceNote]) -> list[dict[str, Any]]:
    contradictions: list[dict[str, Any]] = []
    for note in evidence_notes:
        lowered = note.text.lower()
        marker = next((candidate for candidate in _CONTRADICTION_MARKERS if candidate in lowered), None)
        if marker is None:
            continue
        contradictions.append(
            {
                "note_id": note.note_id,
                "source_id": note.source_id,
                "focus_area": note.focus_area,
                "marker": marker,
                "text": note.text,
            }
        )
    return contradictions


class ResearchSynthesizer:
    """Build synthesis artifacts from normalized research evidence."""

    def __init__(self, *, synthesis_provider: Any | None = None) -> None:
        self._synthesis_provider = synthesis_provider or SynthesisProvider()

    async def synthesize(
        self,
        *,
        plan: ResearchPlan,
        source_registry: list[ResearchSourceRecord],
        evidence_notes: list[ResearchEvidenceNote],
        collection_summary: dict[str, object] | None = None,
        provider_config: dict[str, Any] | None = None,
        outline_seed: list[dict[str, str]] | None = None,
        approved_outline_locked: bool = False,
    ) -> ResearchSynthesisResult:
        deterministic = self._synthesize_deterministic(
            plan=plan,
            source_registry=source_registry,
            evidence_notes=evidence_notes,
            collection_summary=collection_summary,
            outline_seed=outline_seed,
        )
        if approved_outline_locked:
            return self._with_summary_mode(deterministic, mode="deterministic_outline_locked")
        synthesis_config = self._resolve_synthesis_config(provider_config)
        provider = str(synthesis_config.get("provider") or "").strip()
        model = str(synthesis_config.get("model") or "").strip()
        if not provider or not model:
            return self._with_summary_mode(deterministic, mode="deterministic")

        try:
            payload = await self._synthesis_provider.summarize(
                plan=plan,
                source_registry=source_registry,
                evidence_notes=evidence_notes,
                collection_summary=collection_summary,
                config=synthesis_config,
            )
            return self._build_provider_result(
                payload=payload,
                plan=plan,
                source_registry=source_registry,
                evidence_notes=evidence_notes,
            )
        except Exception as exc:
            return self._with_summary_mode(
                deterministic,
                mode="deterministic_fallback",
                fallback_reason=_fallback_reason(exc),
            )

    @staticmethod
    def _resolve_synthesis_config(provider_config: dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(provider_config, dict):
            nested = provider_config.get("synthesis")
            if isinstance(nested, dict):
                return dict(nested)
            if "provider" in provider_config or "model" in provider_config:
                return dict(provider_config)
        return {}

    def _finalize_result(
        self,
        *,
        plan: ResearchPlan,
        source_registry: list[ResearchSourceRecord],
        evidence_notes: list[ResearchEvidenceNote],
        outline_sections: list[ResearchOutlineSection],
        claims: list[ResearchSynthesizedClaim],
        report_markdown: str,
        unresolved_questions: list[str],
        summary: dict[str, Any],
    ) -> ResearchSynthesisResult:
        source_trust = [_source_trust_profile(source) for source in source_registry]
        trust_by_source_id = {entry["source_id"]: entry for entry in source_trust}
        contradictions = _extract_contradictions(evidence_notes)

        note_lookup: dict[tuple[str, str], list[ResearchEvidenceNote]] = defaultdict(list)
        fallback_note_lookup: dict[str, list[ResearchEvidenceNote]] = defaultdict(list)
        for note in evidence_notes:
            note_lookup[(note.source_id, note.focus_area)].append(note)
            fallback_note_lookup[note.source_id].append(note)

        verified_claims: list[ResearchSynthesizedClaim] = []
        unsupported_claims: list[dict[str, Any]] = []
        supported_claim_count = 0
        unsupported_claim_count = 0
        support_level_counts: dict[str, int] = defaultdict(int)
        trust_label_counts: dict[str, int] = defaultdict(int)

        for entry in source_trust:
            trust_label_counts[str(entry["trust_label"])] += 1

        for claim in claims:
            supporting_notes: list[ResearchEvidenceNote] = []
            for source_id in claim.source_ids:
                supporting_notes.extend(note_lookup.get((source_id, claim.focus_area), []))
            if not supporting_notes:
                for source_id in claim.source_ids:
                    supporting_notes.extend(fallback_note_lookup.get(source_id, []))

            supporting_note_ids = list(dict.fromkeys(note.note_id for note in supporting_notes))
            trust_labels = list(
                dict.fromkeys(
                    trust_by_source_id[source_id]["trust_label"]
                    for source_id in claim.source_ids
                    if source_id in trust_by_source_id
                )
            )
            snapshot_policies = list(
                dict.fromkeys(
                    trust_by_source_id[source_id]["snapshot_policy"]
                    for source_id in claim.source_ids
                    if source_id in trust_by_source_id
                )
            )
            warnings: list[str] = []
            if not supporting_note_ids:
                support_level = "unsupported"
                warnings.append("no_supporting_notes")
            elif any(label in {"local_corpus", "primary_source"} for label in trust_labels):
                support_level = "strong"
            elif len(set(claim.source_ids)) >= 2 or len(supporting_note_ids) >= 2:
                support_level = "strong"
            elif all(policy == "metadata_only" for policy in snapshot_policies) and snapshot_policies:
                support_level = "limited"
                warnings.append("metadata_only_support")
            else:
                support_level = "supported"

            verified_claim = ResearchSynthesizedClaim(
                claim_id=claim.claim_id,
                text=claim.text,
                focus_area=claim.focus_area,
                source_ids=list(claim.source_ids),
                citations=list(claim.citations),
                confidence=claim.confidence,
                support_level=support_level,
                supporting_note_ids=supporting_note_ids,
                trust_labels=trust_labels,
                snapshot_policies=snapshot_policies,
                warnings=warnings,
            )
            verified_claims.append(verified_claim)
            support_level_counts[support_level] += 1
            if support_level == "unsupported":
                unsupported_claim_count += 1
                unsupported_claims.append(
                    {
                        "claim_id": verified_claim.claim_id,
                        "text": verified_claim.text,
                        "focus_area": verified_claim.focus_area,
                        "reason": "no_supporting_notes",
                        "source_ids": list(verified_claim.source_ids),
                        "citations": list(verified_claim.citations),
                    }
                )
            else:
                supported_claim_count += 1

        verification_summary = {
            "query": plan.query,
            "claim_count": len(verified_claims),
            "supported_claim_count": supported_claim_count,
            "unsupported_claim_count": unsupported_claim_count,
            "contradiction_count": len(contradictions),
            "support_level_counts": dict(support_level_counts),
            "trust_label_counts": dict(trust_label_counts),
        }
        summary.update(
            {
                "verification_summary": verification_summary,
                "unsupported_claim_count": unsupported_claim_count,
                "contradiction_count": len(contradictions),
            }
        )
        return ResearchSynthesisResult(
            outline_sections=outline_sections,
            claims=verified_claims,
            report_markdown=report_markdown,
            unresolved_questions=unresolved_questions,
            synthesis_summary=summary,
            verification_summary=verification_summary,
            unsupported_claims=unsupported_claims,
            contradictions=contradictions,
            source_trust=source_trust,
        )

    def _build_provider_result(
        self,
        *,
        payload: Any,
        plan: ResearchPlan,
        source_registry: list[ResearchSourceRecord],
        evidence_notes: list[ResearchEvidenceNote],
    ) -> ResearchSynthesisResult:
        parsed = parse_structured_output(
            payload,
            options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
        )
        if not isinstance(parsed, dict):
            raise ValueError("synthesis payload must be a JSON object")

        source_index = {source.source_id: source for source in source_registry}
        note_index = {note.note_id: note for note in evidence_notes}
        unknown_source_ids: set[str] = set()
        unknown_note_ids: set[str] = set()

        outline_sections_payload = parsed.get("outline_sections")
        claims_payload = parsed.get("claims")
        report_sections_payload = parsed.get("report_sections")
        if not isinstance(outline_sections_payload, list):
            raise ValueError("missing outline_sections")
        if not isinstance(claims_payload, list):
            raise ValueError("missing claims")
        if not isinstance(report_sections_payload, list):
            raise ValueError("missing report_sections")

        outline_sections: list[ResearchOutlineSection] = []
        for item in outline_sections_payload:
            if not isinstance(item, dict):
                continue
            source_ids = _normalize_string_list(item.get("source_ids"))
            note_ids = _normalize_string_list(item.get("note_ids"))
            unknown_source_ids.update(source_id for source_id in source_ids if source_id not in source_index)
            unknown_note_ids.update(note_id for note_id in note_ids if note_id not in note_index)
            outline_sections.append(
                ResearchOutlineSection(
                    title=str(item.get("title") or _title_for_focus_area(str(item.get("focus_area") or ""))).strip(),
                    focus_area=str(item.get("focus_area") or "").strip(),
                    source_ids=source_ids,
                    note_ids=note_ids,
                )
            )

        claims: list[ResearchSynthesizedClaim] = []
        for item in claims_payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            focus_area = str(item.get("focus_area") or "").strip()
            source_ids = _normalize_string_list(item.get("source_ids"))
            citations_raw = item.get("citations") if isinstance(item.get("citations"), list) else []
            citations: list[dict[str, Any]] = []
            unknown_source_ids.update(source_id for source_id in source_ids if source_id not in source_index)
            for citation in citations_raw:
                if not isinstance(citation, dict):
                    continue
                source_id = str(citation.get("source_id") or "").strip()
                if not source_id:
                    continue
                citations.append({"source_id": source_id})
                if source_id not in source_index:
                    unknown_source_ids.add(source_id)
            claims.append(
                ResearchSynthesizedClaim(
                    claim_id=str(item.get("claim_id") or f"clm_{_stable_digest(text, *source_ids)[:12]}"),
                    text=text,
                    focus_area=focus_area,
                    source_ids=source_ids,
                    citations=citations,
                    confidence=float(item.get("confidence") or 0.0),
                )
            )

        if unknown_source_ids or unknown_note_ids:
            reason_parts: list[str] = []
            if unknown_source_ids:
                reason_parts.append(f"unknown source_id(s): {', '.join(sorted(unknown_source_ids))}")
            if unknown_note_ids:
                reason_parts.append(f"unknown note_id(s): {', '.join(sorted(unknown_note_ids))}")
            raise ValueError("; ".join(reason_parts))

        report_markdown = self._build_report_markdown(report_sections_payload)
        unresolved_questions = _normalize_string_list(parsed.get("unresolved_questions"))
        covered_focus_areas = [section.focus_area for section in outline_sections if section.focus_area]
        missing_focus_areas = [
            focus_area
            for focus_area in plan.focus_areas
            if focus_area not in covered_focus_areas
        ]
        for focus_area in missing_focus_areas:
            unresolved = f"missing evidence for focus area: {focus_area}"
            if unresolved not in unresolved_questions:
                unresolved_questions.append(unresolved)

        summary_payload = parsed.get("summary")
        if summary_payload is not None and not isinstance(summary_payload, dict):
            raise ValueError("synthesis_summary_must_be_object")
        summary = dict(summary_payload or {})
        summary.update(
            {
                "mode": "llm_backed",
                "query": plan.query,
                "focus_areas": list(plan.focus_areas),
                "section_count": len(outline_sections),
                "claim_count": len(claims),
                "source_count": len(source_registry),
                "unresolved_questions": list(unresolved_questions),
                "coverage": {
                    "covered_focus_areas": covered_focus_areas,
                    "missing_focus_areas": missing_focus_areas,
                },
            }
        )
        return self._finalize_result(
            plan=plan,
            source_registry=source_registry,
            evidence_notes=evidence_notes,
            outline_sections=outline_sections,
            claims=claims,
            report_markdown=report_markdown,
            unresolved_questions=unresolved_questions,
            summary=summary,
        )

    @staticmethod
    def _build_report_markdown(report_sections: list[dict[str, Any]]) -> str:
        lines = ["# Research Report"]
        for section in report_sections:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "").strip()
            markdown = str(section.get("markdown") or "").strip()
            if title:
                lines.extend(["", f"## {title}", ""])
            if markdown:
                lines.append(markdown)
        return "\n".join(lines).strip()

    def _with_summary_mode(
        self,
        result: ResearchSynthesisResult,
        *,
        mode: str,
        fallback_reason: str | None = None,
    ) -> ResearchSynthesisResult:
        summary = dict(result.synthesis_summary or {})
        summary["mode"] = mode
        if fallback_reason:
            summary["fallback_reason"] = fallback_reason
        return ResearchSynthesisResult(
            outline_sections=result.outline_sections,
            claims=result.claims,
            report_markdown=result.report_markdown,
            unresolved_questions=result.unresolved_questions,
            synthesis_summary=summary,
            verification_summary=result.verification_summary,
            unsupported_claims=result.unsupported_claims,
            contradictions=result.contradictions,
            source_trust=result.source_trust,
        )

    def _synthesize_deterministic(
        self,
        *,
        plan: ResearchPlan,
        source_registry: list[ResearchSourceRecord],
        evidence_notes: list[ResearchEvidenceNote],
        collection_summary: dict[str, object] | None = None,
        outline_seed: list[dict[str, str]] | None = None,
    ) -> ResearchSynthesisResult:
        source_index = {source.source_id: source for source in source_registry}
        notes_by_focus_area: dict[str, list[ResearchEvidenceNote]] = defaultdict(list)
        for note in evidence_notes:
            if note.source_id in source_index:
                notes_by_focus_area[note.focus_area].append(note)

        outline_sections: list[ResearchOutlineSection] = []
        claims: list[ResearchSynthesizedClaim] = []
        report_sections: list[str] = []
        unresolved_questions: list[str] = []
        covered_focus_areas: list[str] = []
        missing_focus_areas: list[str] = []

        initial_gaps = collection_summary.get("remaining_gaps", []) if isinstance(collection_summary, dict) else []
        for gap in initial_gaps:
            gap_text = str(gap).strip()
            if gap_text and gap_text not in unresolved_questions:
                unresolved_questions.append(gap_text)

        seed_sections = [
            {
                "title": str(section.get("title") or "").strip(),
                "focus_area": str(section.get("focus_area") or "").strip(),
            }
            for section in (outline_seed or [])
            if isinstance(section, dict)
            and str(section.get("title") or "").strip()
            and str(section.get("focus_area") or "").strip()
        ]
        ordered_focus_areas = [section["focus_area"] for section in seed_sections] or list(plan.focus_areas)
        seeded_titles = {
            section["focus_area"]: section["title"]
            for section in seed_sections
        }

        for focus_area in ordered_focus_areas:
            notes = notes_by_focus_area.get(focus_area, [])
            section_title = seeded_titles.get(focus_area, _title_for_focus_area(focus_area))
            if not notes:
                if seed_sections:
                    outline_sections.append(
                        ResearchOutlineSection(
                            title=section_title,
                            focus_area=focus_area,
                            source_ids=[],
                            note_ids=[],
                        )
                    )
                    report_sections.append(
                        "\n".join(
                            [
                                f"## {section_title}",
                                "",
                                "No collected evidence currently supports this section.",
                            ]
                        )
                    )
                missing_focus_areas.append(focus_area)
                unresolved = f"missing evidence for focus area: {focus_area}"
                if unresolved not in unresolved_questions:
                    unresolved_questions.append(unresolved)
                continue

            covered_focus_areas.append(focus_area)
            source_ids = list(dict.fromkeys(note.source_id for note in notes))
            note_ids = [note.note_id for note in notes]
            section = ResearchOutlineSection(
                title=section_title,
                focus_area=focus_area,
                source_ids=source_ids,
                note_ids=note_ids,
            )
            outline_sections.append(section)

            report_lines = [f"## {section.title}", ""]
            for note in notes:
                claim_id = f"clm_{_stable_digest(note.note_id, note.source_id)[:12]}"
                claims.append(
                    ResearchSynthesizedClaim(
                        claim_id=claim_id,
                        text=note.text,
                        focus_area=focus_area,
                        source_ids=[note.source_id],
                        citations=[{"source_id": note.source_id}],
                        confidence=float(note.confidence),
                    )
                )
                report_lines.append(f"- {note.text} [Sources: {note.source_id}]")
            report_sections.append("\n".join(report_lines))

        report_markdown = "\n\n".join(
            ["# Research Report", f"Question: {plan.query}"] + report_sections
        ).strip()

        synthesis_summary = {
            "query": plan.query,
            "focus_areas": list(plan.focus_areas),
            "section_count": len(outline_sections),
            "claim_count": len(claims),
            "source_count": len(source_registry),
            "unresolved_questions": list(unresolved_questions),
            "coverage": {
                "covered_focus_areas": covered_focus_areas,
                "missing_focus_areas": missing_focus_areas,
            },
        }

        return self._finalize_result(
            plan=plan,
            source_registry=source_registry,
            evidence_notes=evidence_notes,
            outline_sections=outline_sections,
            claims=claims,
            report_markdown=report_markdown,
            unresolved_questions=unresolved_questions,
            summary=synthesis_summary,
        )


__all__ = ["ResearchSynthesizer"]
