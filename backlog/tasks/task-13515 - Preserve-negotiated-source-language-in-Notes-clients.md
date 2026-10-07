---
id: TASK-13515
title: Preserve negotiated source language in Notes clients
status: In Progress
created_date: 2026-10-07 03:07
dependencies:
- TASK-13513
- TASK-13514
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md
updated_date: 2026-10-07 05:10
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep language intent and exact source safe when clients and servers support different Notes versions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clients require explicit authenticated language capability and isolate or invalidate capability state on server/profile changes.
- [ ] #3 Old or unavailable peers leave actionable durable pending intent; a body-only acknowledgement never clears pending language.
- [ ] #4 Targeted compatibility tests and scoped static/security evidence cover publication rechecks and account isolation.
- [ ] #5 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
- [ ] #5 Clear Notes round trips and compatible private builder/receiver fixtures preserve exact source, language and lineage; unavailable private server transport retains ciphertext-backed local pending intent.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md (C); Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md (S), existing decisions
Reason: implement approved negotiated client contracts and private confidentiality/pending behavior without a new transport.
1. Read task4 brief, repo-local tasks/instructions/lessons and governing ADRs; trace current clear/private builders, receivers, outbox acknowledgements and authenticated Notes client capability paths.
2. Add failing real SQLite builder/receiver, compatibility/pending/restart and capability/profile controls before edits.
3. Preserve complete source/language using advertised compatible versions, old omission preservation and explicit null; keep private language inside ciphertext and reject clear leakage.
4. Retain note-scoped durable pending language/reason when complete/private support is unavailable; body-only acknowledgements cannot erase intent. Recheck authenticated capability before publication and isolate server/profile state.
5. Run named new and affected C/U selections plus scoped static/security checks; frontend typecheck for shared-client changes. Preserve inherited debt with immutable-base comparison; no full suite.
6. Document evidence/limitations in task notes and ADR pointers, self-review and commit explicit client files separately. Keep In Progress pending independent controller review.
Browser ownership/cache correction: U/src/services/tldw/domains/collections.ts overrides createNote, so update that actual owner and keep base method consistent. Use fresh authenticated capability probes on capability requests/publication instead of a new cache; same captured/pinned request scope for probe and publication, reject in-flight authority change before mutation/cleanup. Cover absent support, authenticated404, transport failure and profile changes. Cost: an additional authenticated read for publication and conservative pending state during capability transport failure.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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

## Implementation Notes

Implemented negotiated complete language snapshots, authenticated publication guards and exact omission/null preservation. Existing local mutation/outbox owners retain unsupported intent; scoped immutable origin proof prevents cross-account drain. Fresh probes and scoped mirror/current-authority injection are documented corrections under ADR194/065. Verification and limitations: task-4-report.md in the controller SDD directory. Status remains In Progress pending independent review.

Fix round1: remove duplicated base createNote implementation; existing collections mixin is the single owner through declaration merging/Object.assign. Browser25tests and final whole frontend typecheck pass. Explicit W ESLint config from U supports scoped syntax/rules with no introduced diagnostics against immutable fix base; W-only Next pages rule emits applicability notice. Inherited formatter debt unchanged. Task remains In Progress for independent review.
