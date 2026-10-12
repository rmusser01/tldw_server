---
id: TASK-13589
title: Negotiate complete Notes snapshots with honest private compatibility
status: Done
created_date: 2026-10-07 02:38
dependencies:
- TASK-13588
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md
updated_date: 2026-10-07 05:05
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
ADR path: Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md (existing extension of locked ADR031)
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
Implemented clear notes.note (2,2) complete source/language snapshots, locked v1, signer readiness, safe v1 preservation, actionable old-peer refusal, shared producer capture and atomic derived-task admission. Preserved unavailable private transport, durable conflict/retry behavior and qualified old Studio provenance. ADR070 documents boundaries. Final selection: 306 passed and one overly broad assertion; corrected task file 11 passed; healthy/race pair 2 passed. Real PostgreSQL/API focused 4 passed; earlier affected language file 32 passed. Ruff base1/current0 and Bandit0/0; generator and frontend whole typecheck passed. Warnings and inherited Black debt documented in task-3-report.md. Remains In Progress pending controller review; no full suite.
F3 review fix round1: runtime reader capabilities/stream negotiation now respect existing signer readiness while bounded post-scan checks refuse inaccessible v2 history with503 and no cursor advance. Omitted language binds the cached canonical row version when caller omits expected_version; explicit versions and create semantics retained. RED2 failed/4 passed; language GREEN33 passed; covering models/materializer/capture/batch/language278 passed; final key-loss/readiness/cached-row7 passed. Ruff and Bandit0/0 against dc463933; no formatter debt increase; generator/typecheck passed unchanged fingerprint. Exact commands, warnings and setup correction appended to task-3-report.md. In Progress pending scoped re-review.
Controller closure: fresh independent F3 task review found two reproduced Important gaps, both addressed in 06dc387cc42235a56d8b0817cb4c399fc0d39b8a and fresh scoped review approved. Signer-aware safe v1 reads and explicit inaccessible-v2 refusal, exact cached-row admission binding verified. Covering278/final focused7 passed; source/schema/PG evidence and inherited warning/format debt retained in report. Existing ADR070 extended; locked ADR031 untouched. Actual private transport remains unavailable with explicit refusal; encrypted client pending belongs to TASK13590. No full sweep or private-publication success claimed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Complete Notes (2,2) snapshots preserve exact source and language across current producers while v1 stays strict. Runtime negotiation preserves safe v1 without signing readiness and refuses inaccessible v2 before cursor advancement; omission-derived language is version-bound. Original and scoped fix reviews approved, targeted evidence and limitations recorded in ADR070/task notes.
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

## Renumbering provenance

Originally TASK-13514 in the reviewed structured-document candidate. The publication task is TASK-13589 because current dev assigns the former ID to unrelated work. This changes tracking identity only; acceptance criteria, status, original creation date and implementation evidence are retained. Original reviewed commits and local evidence preserve the historical ID.
