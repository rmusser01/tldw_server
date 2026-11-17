---
id: TASK-15192
title: Delete qualified native leaf messages without orphaning saved conversations
status: Done
labels:
- frontend
- backend
- chat
- ux
- bug
dependencies:
- TASK-13559
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair native saved-message Delete across full and compact ordinary/character chat: compose current raw/H1 authority, scoped numeric CAS and exact acknowledgement, then refresh the canonical selection without trimming rows before acknowledgement. Enforce an opt-in atomic leaf invariant at the shared backend mutation boundary, preserving existing legacy deletion semantics and explicit Sync limitations. Fix swallowed direct conflicts and misleading client success outcomes; keep all-ten/native/packaged acceptance separately tracked.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both chat interfaces admit only current owned native leaf targets, preserve drafts and original rows before acknowledgement, and retain existing local/legacy controls.
- [x] #2 Opt-in native leaf deletion checks descendants and numeric CAS atomically with parent-writing serialization; direct conflicts reach409, and unsupported Sync deletion is explicitly refused before mutation.
- [x] #3 Scoped client/proxy request, exact204 acknowledgement and guarded canonical refresh recover the deleted cursor at its verified parent or empty, suppress obsolete publication and distinguish uncertain or failed outcomes.
- [x] #4 Meaningful dual-backend/API and full/compact service/caller controls, both client types/static, touched Python Bandit and two independent reviews qualify this unit; remaining whole-program and Sync work remains tracked.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design investigation by the two authorized independent state/transport reviewers found native-to-local refusal, dropped request scope/cancellation, non-leaf parent tombstone orphaning H1, plain append/reparent races, swallowed direct CAS conflicts and nonuniform Sync-enrolled CAS. Governing local removeMessage already refuses hidden/late descendants; preserve legacy API default semantics with an explicit native leaf precondition. No cascade or reparent. ADR assessment: no new durable architecture decision; existing ADR049/050 and accepted D7 own native graph/authority rules, ADR059 owns canonical tracking. No new authority table, replay queue or dependency. Highest13559 verified across origin/dev/open PRs/both local backlogs; newestdev97ea9cd5fa included. Plan will specify controlled Sync refusal and separate unresolved Sync settlement/CAS work.
Qualified storage red: 14 failures/2 legacy positives on SQLite/PostgreSQL; API 3 causal failures/2 positives; transport 6 failures/25 existing negatives and positives (one local-returnResponse wire expectation corrected). Initial fixture history property and unapproved basetemp runs excluded. Independent source review found conversation→message→metadata/image ordering must cover actual PUT, standalone metadata/images and Sync tombstone writers; plan expanded accordingly. Destructive native Delete additionally needs atomic opaque message revision comparison because metadata/images can change without numeric CAS version advancement.
Independent reviews reproduced and repaired root defects: reserve the existing H1 writer revision for an owned Delete transition; bind dispatch to rendered native origin so retained confirmation after New cannot downgrade to unqualified legacy DELETE; preserve the Promise through the actual SidePanelBody and all other timeline parents; suppress obsolete post-ACK feedback after held mirror/refresh H1 replacement; require exact serialized boolean ok=true with numeric204; classify only own declared recovery codes. Qualified feedback RED shows four obsolete warning publications after valid parent replacement; both rendered-origin/revision controls reproduced actual failures; transport/feedback/compact wrapper RED eleven failures. Compact native-only fixture now clears its unrelated mocked historyId, matching actual selectors; exact qualified owner/workspace/rollback mirror tests are separate. Earlier invalid fixture failures are excluded from causal evidence. Backend transaction matrix136 passed, one skip; actual WebUI and extension types pass; touched Python production Bandit zero findings. Expanded real SQLite/Postgres HTTP opaque-CAS and actual PUT/Delete tests and connected legacy/Sync neighbors are running. Whole-program enhancements remain open under TASK-13458.
Verification harness reassessment: stopped after three formatter-guard discrepancies without repository writes. Printer equality preserved wrapping flags rather than behavior; switching to emitted AST revealed the remaining guard parsed .ts generic aliases in default TSX mode, producing bogus JSX/string chunks. Inspected before/after trees and existing transpileModule controls in persistence/composer/layout tests. Alternatives were source AST equivalence, emitted AST with the actual file type, or whole-project emit. Chose emitted AST with explicit fileName (actual .ts/.tsx) plus real compilers, preserving the safety gate and owned-region formatting. HTTP concurrency initial three failures were fixture serialization on one lifespan portal and absent metadata; independent request portals on the same official SQLite/Postgres DB/app/credentials qualified all four race cases. Opaque-only Sync refusal separately produced real500 instead of409 and now shares the pre-mutation native refusal. No invalid harness failure is claimed as product evidence.
Final bounded qualification:1477/1477 client40 suites;136 transaction positives/one existing skip;67 actual HTTP controls including four qualified real SQLite/Postgres PUT/Delete races;92 backend/legacy/Sync neighbors. Both actual client compilers,33 TS/six Python no-added static, touched production Bandit zero findings/errors, authored equivalent formatting and diff checks pass. Both authorized independent source reviewers CLEAR. Evidence /private/tmp/chat-native-leaf-delete-13560-{final-qualified-matrix,transactions-green,final-http,backend-neighbors,qualified-web-types,qualified-extension-types,qualified-eslint,qualified-ruff,format-final-qualified}.log and final-bandit.json. Master design updated; four stages complete, completed plan preserved /private/tmp/chat-native-leaf-delete-13560-completed-plan.md and owned plan retired. Preview compilation/token sync pass; emitted shared600.6KB exceeds unchanged600KB budget, so performance/native-browser acceptance remains open with parent13458. Legacy unqualified server Delete, enrolledSyncCAS and all-program/packaged/physical/AT/participant gates remain required, not silently completed.
Renumbered from TASK-13560 to TASK-15192 on 2026-10-31 (TASK-13960): TASK-13560 now names only "Data-URI redaction regex is quadratic on crafted input in the chat path" (backlog/tasks/task-13560 - Data-URI-redaction-regex-is-quadratic-on-crafted-input-in-the-chat-path.md). Before 2026-10-31 both tasks used TASK-13560; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native leaf Delete now composes exact rendered owner/source, raw numeric and opaque revisions, scoped204 ACK and guarded canonical parent/empty recovery in full/compact ordinary/character chat. Shared conversation-first storage and parent validation close hidden/late-child and metadata/image races; strict awaited confirmation and owned mirror handling suppress false/obsolete success. Existing legacy API compatibility remains; enrolledSyncnative preconditions refuse explicitly.1477 client/295 backend positives,one inherited skip, both types/static/Bandit and two source reviews qualify this bounded unit. Preview bundle budget and remaining legacy/all-program journeys remain open.
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
