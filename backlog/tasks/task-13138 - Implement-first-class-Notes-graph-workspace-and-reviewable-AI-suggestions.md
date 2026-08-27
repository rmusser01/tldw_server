---
id: TASK-13138
title: Implement first-class Notes graph workspace and reviewable AI suggestions
status: To Do
created_date: 2026-08-27 03:40
labels:
- notes
- knowledge-graph
- webui
- browser-extension
- llm
- jobs
priority: High
references:
- TASK-13134
- TASK-13135
- TASK-13136
- TASK-13137
documentation:
- Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md
updated_date: 2026-08-27 03:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add Graph as a first-class Notes view mode shared by the WebUI and browser extension, and add an on-demand, source-grounded suggestion workflow for the selected note. Suggestions use a whole-library Notes FTS shortlist plus one bounded configured LLM invocation to propose related-note links and tags. Suggestions remain provisional until explicitly accepted or rejected; acceptance uses the existing manual-link and tag mutation paths. This slice does not add embeddings, semantic edge types, background organization, library-wide themes, or saved layouts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Notes exposes Graph as a first-class view mode with a focused graph canvas, search, focus control, edge-type filters, layout/fit controls, and a responsive inspector rather than a routine modal.
- [ ] #2 The graph initially focuses on the selected or most-recent note, supports bounded interactive expansion, and offers all-notes mode only below the configured cap.
- [ ] #3 The inspector provides Details and Suggestions views, grounded evidence for both notes, Strong match/Possible match bands, accept/reject controls, provisional dashed edges, and tag suggestion chips.
- [ ] #4 Canvas and Relationships views provide equivalent access to graph relationships; keyboard, focus, non-color state, narrow-screen, overflow, and long-content behavior are covered.
- [ ] #5 Suggestion generation is exposed only beneath the existing Notes graph namespace and runs through Jobs with owner/dataset scoping, idempotent admission, max_retries=0, cancellation, bounded inputs/outputs, and no note content or credentials in Job payloads or logs.
- [ ] #6 FTS searches the active owner-scoped Notes library and excludes only the selected note, trash, and directly linked note pairs; shared tag/source membership does not exclude candidates.
- [ ] #7 One configured LLM invocation receives only a bounded allowlisted shortlist and tag catalog, treats notes as untrusted data, uses a strict output schema, and cannot introduce unknown note IDs, tools, provider settings, or unbounded fields.
- [ ] #8 Suggestion runs and provisional suggestions are durable, paginated, retention bounded, and keyed by content fingerprints; evidence is stored as fingerprint-bound canonical-text offsets and reconstructed on read rather than copied into suggestion records.
- [ ] #9 Relationship suggestions accept as ordinary undirected manual links with weight 1.0 and no model-selected semantic label/properties; tag suggestions use existing tag normalization and cap newly invented tags.
- [ ] #10 Accept/reject operations are idempotent and race safe. Acceptance uses compare-and-swap plus the existing mutation path and a bounded reconciliation lease; unchanged-version rejection suppresses the same pair/tag across model or prompt versions.
- [ ] #11 Accepting one tag does not stale sibling suggestions: title/body content fingerprints are independent of tag membership, and existing-tag acceptance resolves as an idempotent success.
- [ ] #12 Request-time validation uses HTTP conflict/validation/rate-limit/readiness responses; failures after 202 are represented by durable run status, stable error codes, and sanitized user guidance.
- [ ] #13 Generation validates the top-level response strictly, drops individually invalid or duplicate items, atomically persists the validated set, and records only aggregate validation counts; no invalid suggestion is exposed.
- [ ] #14 Current projection freshness is verified or the run reports degraded/unavailable discovery rather than claiming a complete current-library search.
- [ ] #15 A successful current-version run supersedes older pending suggestions while preserving current-version rejections; stale/obsolete records follow bounded retention and note/user deletion cascades.
- [ ] #16 Backend unit/integration/property tests, frontend component/contract/accessibility tests, Playwright desktop/mobile visual checks, and an offline suggestion-quality evaluation corpus cover the approved design.
- [ ] #17 Relevant Notes and API documentation is updated, touched code passes targeted tests and lint/type checks, and Bandit reports no new findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the approved design specification. Implement test-first in reviewable stages: persistence/state invariants; bounded shortlist and provider pipeline; Jobs/API contracts; acceptance/rejection/reconciliation; shared Notes Graph workspace; accessibility/responsive behavior; observability, retention, documentation, and quality evaluation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deferred work is tracked separately: TASK-13134 embeddings and semantic edges; TASK-13135 automatic background organization; TASK-13136 library-wide recurring themes; TASK-13137 saved graph views/layouts. This task remains review-first and on-demand.

Design approved in chat and written to Docs/superpowers/specs/2026-08-26-notes-second-brain-graph-suggestions-design.md. Self-review completed for incomplete markers, internal consistency, scope, asynchronous state transitions, privacy, cancellation, retention, and RBAC. The design-only change has no Bandit-applicable code scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
