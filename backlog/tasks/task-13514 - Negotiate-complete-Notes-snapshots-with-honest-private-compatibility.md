---
id: TASK-13514
title: Negotiate complete Notes snapshots with honest private compatibility
status: In Progress
created_date: 2026-10-07 02:38
dependencies:
- TASK-13513
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md
updated_date: 2026-10-07 04:51
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep source language and source text consistent across devices while older clients receive an explicit compatibility result instead of erasing metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strict v1 stays unchanged; newly negotiated clear notes.note versions advertise truthful supported and writable capabilities, while unavailable private transport is reported explicitly.
- [x] #2 Complete current snapshots preserve exact source and language through REST, all derived producers, durable envelopes and the real Notes materializer.
- [x] #3 Old snapshots preserve legacy language where safe and cannot downgrade a newer head; old consumers see an actionable unsupported-version result.
- [x] #4 Existing private ingress remains refused before append without exposing source or language in routing metadata; clients can retain encrypted metadata and pending intent when no negotiated peer exists.
- [x] #5 Language-only conflicts, omission/null, v1 to v2, tombstone restore, idempotent retries and two-client controls pass targeted Sync/API checks.
- [x] #6 Server ADR extension is linked and scoped static/security evidence is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md (existing extension of locked ADR031)
Reason: implement the approved newly negotiated complete clear/private Notes service contract.
1. Read F3 brief, task, relevant server instructions/ADRs and actual producers/registry/admission/materializer boundaries; recheck version allocation.
2. Add real failing strict version and two-client store tests for complete source/language, old-peer refusal and private confidentiality.
3. Add explicit clear/private version negotiation and one complete payload builder; preserve v1 fields/hash semantics and F2 atomic pre-append admission.
4. Route ordinary REST and all derived Notes producers through the builder; advertise truthful supported/writable capabilities.
5. Verify new and affected Sync/API selections, real stores/owned PostgreSQL where applicable, scoped static/Bandit and generated-client typecheck if contracts change.
6. Update existing ADR with allocated contracts/evidence, self-review and commit explicit F3 files; keep In Progress pending controller review.
Execution correction: captured S has no legacy encrypted notes domain/API ingress. Negotiate existing clear notes.note and qualify unchanged private refusal/unavailable capabilities. Do not add a transport; F4 retains local encrypted metadata and pending intent under approved unsupported-peer behavior.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented clear notes.note (2,2) complete source/language snapshots, locked v1, signer readiness, safe v1 preservation, actionable old-peer refusal, shared producer capture and atomic derived-task admission. Preserved unavailable private transport, durable conflict/retry behavior and qualified old Studio provenance. ADR065 documents boundaries. Final selection: 306 passed and one overly broad assertion; corrected task file 11 passed; healthy/race pair 2 passed. Real PostgreSQL/API focused 4 passed; earlier affected language file 32 passed. Ruff base1/current0 and Bandit0/0; generator and frontend whole typecheck passed. Warnings and inherited Black debt documented in task-3-report.md. Remains In Progress pending controller review; no full suite.
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
