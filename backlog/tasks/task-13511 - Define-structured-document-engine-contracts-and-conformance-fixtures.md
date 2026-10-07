---
id: TASK-13511
title: Define structured document engine contracts and conformance fixtures
status: Done
created_date: 2026-10-07 02:14
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-engines.md
updated_date: 2026-10-07 02:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Specify lossless local editing contracts and shared language examples before adopting parser engines.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Immutable request identities and typed result contracts enforce diagnostic ranges and truthful formatting admission.
- [x] #2 Versioned fixtures cover strict JSON, physical-line JSONL, YAML preservation and unsupported constructs.
- [x] #3 Both runtime fixture copies are byte-identical with SHA-256 recorded and focused contract tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/031-notes-capability-sync-domains.md (existing note transport); cross-app approved design ADR-194 is linked in the external specification
Reason: pure engine contracts implement the approved design; this task does not change persisted or versioned server payloads.
1. Read E1 in the approved engine plan.
2. Add shared fixtures and failing contract tests.
3. Implement pure models and request factories.
4. Run focused checks and fixture hash comparison; review and commit only this unit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented immutable wire contracts and request-aware result construction, rejecting invalid scalar ranges, incomplete/unqualified formatting, excerpt success, nonformatted replacements and mismatched check identities. Added 37 shared language fixtures and pure contract tests/factories. YAML unsupported fixtures remain grammar-valid with unsupported diagnostics, distinct from malformed syntax. Fixture SHA-256: fcf516e4c0bb8ff7a19748a4d5f7deb8a30ad0078ea0ce2af0e1e81af50fb0d7; copies verified byte-identical and hash pinned in both tests. 18 Vitest tests; scoped strict TypeScript check and Prettier pass. Bandit not applicable: touched production code is TypeScript. No parser or UI wiring. Self-review complete; independent review pending. Governing ADR: Docs/ADR/031-notes-capability-sync-domains.md; cross-app ADR-194; no new decision. Later engines/workers must use request-aware constructors for excerpt/source-bound guarantees.
Independent E1 review: spec compliant and task quality approved; no blocking findings. Existing frontend typecheck (bun run typecheck from apps/tldw-frontend) also passed with exit 0. Request-aware construction is a required downstream engine/worker integration boundary. Existing optional PyAudio and Node ExperimentalWarning noise is recorded as inherited environment debt.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
E1 implementation and targeted verification complete; awaiting independent review.
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
