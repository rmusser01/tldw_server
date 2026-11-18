---
id: TASK-15504
title: Review large political PDF ingestion and evidence-based research UX
status: Done
labels:
- uat
- ux
- research
- pdf
references:
- Docs/Reviews/LARGE_POLITICAL_PDF_UAT_2026_11_15.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Walk through ingestion, document-scoped QA, investigation, evidence review and saved research for a nontechnical VIP using the 922-page Mandate for Leadership PDF. Record reproduced issues, code-confirmed risks, untested paths, and prioritized acceptance criteria before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record the runtime and document baseline and actual walkthrough outcomes
- [x] #2 Trace extraction, indexing, source scoping, citation verification, and research persistence with evidence
- [x] #3 Deliver prioritized findings and a practical VIP acceptance scenario without implementing product changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Scope confirmed: local workspace app; cited answers, research briefs/comparisons, claims/evidence, chapter and major-point summaries. Read-only product audit with disposable UAT data. Baseline 3451c811d0 on ci/pin-ubuntu-2404-playwright. ADR assessment: no new ADR required; this review proposes no durable architecture decision. Product implementation is out of scope for this pass.
Review artifacts: Docs/Reviews/LARGE_POLITICAL_PDF_UAT_2026_11_15.md and assets/2026-11-15-large-political-pdf-uat (five screenshots and sanitized JSON). Actual 922-page upload completed in 112 seconds; DeepSeek Flash exercised with two successful evidence-limited API generations. No complete browser answer or full-book summary passed. Eight prioritized findings distinguish browser/API reproduction, code-confirmed limits and untested risks. Independent ingestion and Research fact-checks incorporated.
Verification: 29 PDF/upload Python tests, 24 media-worker tests, 69 frontend query/provenance tests passed (122 total). Report links and JSON validated; artifact secret-pattern scan and diff whitespace check passed. Bandit not applicable: only Markdown, JSON and PNG deliverables, no product code. No screen-reader audit, full answer benchmark, semantic indexing, cross-device restore or successful answer-to-brief browser export. Disposable services stopped and temporary provider credential configuration scrubbed. ADR required: no; review changes no durable rule. Future work should respect ADR-003, ADR-007 and ADR-065.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the review, not product remediation. The PDF imports, but VIP self-service is blocked by summary coverage, missing retained original/page provenance, selected-source clarification and expensive incomplete retrieval. The report includes evidence, reproduction paths, a minimal connected VIP workflow, staged priorities and acceptance scenarios. No product changes or external publication.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
