---
id: TASK-13558
title: Regenerate native H1 replies against their existing owned admission
status: Done
labels:
- backend
- chat
- ux
- bug
dependencies:
- TASK-13556
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CM-01: saved native Regenerate is refused by H1 while an ordinary selected send always inserts user input. Reuse the existing completion regeneration marker, validate selected ancestry and the immutable saved user admission under the owner transaction, and settle an alternate assistant without duplicating the input or rewriting originals. Preserve legacy regeneration, scope/skills/Sync gates and ordinary consumed-selection semantics. Frontend composition and remaining legacy/edited input support continue separately under the all-ten program.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real owner-scoped streaming and non-streaming H1 regeneration preserve the original user and replies and append only an alternate assistant sharing the same admitted user.
- [x] #2 Fresh selected path, reply parent, canonical raw input and stored authority are validated before provider calls or writes; stale, missing, foreign, changed and conflicting requests fail with no effects.
- [x] #3 Dual-backend and real authenticated API regressions, legacy/consumed-selection compatibility, touched static/security checks and two independent reviews qualify this backend unit without claiming native UI or complete program acceptance.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; ADR049 and accepted D7 section 6.2 govern existing selection/admission authority, explicit regeneration and alternate replies. ADR059 governs tracking. Existing metadata.tldw_regenerate_from_message_id is reused with branch:true and a selected canonical assistant/direct user under the owner transaction. No original-row or authority rewrite, new endpoint or wire field. Missing legacy authority, edited admitted input, tool parents and Sync limitations remain program follow-ups. Highest ID 13557 was checked across dev, open PRs and both local backlogs; latest dev 97ea9cd5fa is included.

Causal logs: /private/tmp/chat-native-regenerate-13558-{api-red,db-red,capability-red,authority-red,status-red,parent-authority-red,parent-final-red}.log. Missing/malformed protected fields, JSON and interpretation kinds now have explicit refusals before writes; initial provider assertion corrections preserve exact existing text-part representation. A superseded blanket H1 regeneration refusal test now verifies failed-retry refusal and required authenticated binding before any storage.

Final verification: current API/context/new DB 135/135; shared SQLite/PostgreSQL admission/transactions 180 positive checks and one existing SQLite connection-lifetime skip. Broader 391 cases recorded 389 positives, the same skip and the superseded unit control; all 33 context cases pass in the final 135. Six authored Python files have two baseline/current static findings and zero introduced findings. Syntax and authored formatting pass. Bandit, excluding B101 for test assertions, has zero findings and zero scan errors. Both independent bounded source reviewers CLEAR the final candidate. Provider I/O is controlled; storage/authentication/routes are real. Logs: /private/tmp/chat-native-regenerate-13558-{final-qualified,parent-final-qualified,final-matrix,payload-qualified}.log and /private/tmp/bandit_chat_native_regenerate_13558_qualified.json.

All three bounded stages complete; master design updated. Only the owned plan retired to /private/tmp/chat-native-regenerate-13558-completed-plan.md. Frontend TASK-13559 and parent TASK-13458 remain In Progress, including complete native/extension/physical/AT/participant acceptance and remaining capability work. Preview chunk/fetch root remains unproved. No PR, push or merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native H1 Regenerate validates the chosen assistant/direct admitted user and exact original text/ordered images under the owner transaction, reuses protected immutable admission, and server-settles only an alternate assistant. Originals and ordinary/legacy contracts are preserved; malformed authority has explicit refusals. Final 135 API/context/new DB and 180 shared DB positives, one inherited SQLite skip, unchanged static baseline, clean Bandit and two bounded reviews qualify this backend unit. Shared client composition, legacy/edited/tool/Sync and complete native/extension/physical/AT/participant gates continue under TASK-13559/TASK-13458.
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
