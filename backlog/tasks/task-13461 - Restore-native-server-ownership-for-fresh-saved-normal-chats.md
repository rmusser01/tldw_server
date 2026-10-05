---
id: TASK-13461
title: Restore native server ownership for fresh saved normal chats
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Normal H1 first sends create a browser-only owner, preventing the established server autosave path. Restore server ownership before admission for fresh saved drafts when the connected server supports persisted chat; retain existing local owners, temporary chat, guarded native writes and recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh connected saved normal chat creates and loads a native owner before user admission and inference, and persists one user/assistant pair with inference save_to_db false.
- [x] #2 Focused regression suites pass and public repair is reviewed against latest dev without private hosted information.
- [x] #3 Absent native-draft opt-in preserves local behavior; failed or stale native creation/load cannot publish into another view/account or dispatch inference.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR check: ADR required: no. ADR-049 governs owner-validated continuation; this repairs the existing saved-chat contract rather than changing ownership policy. Plan stages: 1 reproduce exact first-send regression (in progress), 2 restore owner creation and receipt validation, 3 verify/review/submit upstream. Hosted rollout/evidence will be separately tracked in private repository. Human Change summary waived by owner.
Stage 1 complete: new mounted normal-mode test failed because createChat called zero times; focused first-send tests now pass. Stage 2 complete: shared normal owner establishment creates native draft only on explicit connected plain-draft opt-in, validates captured scope/cancellation and exact owner/view receipt before adoption. Cached hasChatSaveToDb was rejected as connectivity gate; current connection state supplies the opt-in. Stage 3 in progress: 84 normal/H1 tests and188 character/persona/overlay/extension tests pass; independent reviews running. Broader RAG baseline comparison in progress.
Review follow-ups: reject stale receipt epochs and rejected provisional leases; shared loader clears only its own cancelled provisional owner, preserving newer views and allowing retry. Dispatchable tool-bearing drafts keep their local path. Delayed reasoning content is persisted unchanged; measured elapsed milliseconds are explicitly client-only telemetry, not generic native message metadata. No backend/public schema or arbitrary-metadata capability added. Production frontend tsc passes with Node24/8GB; blanket shared-UI tsc hit default4GBheap and is not counted as passing. One RAG provider-routing test independently fails on untouched025627214c baseline; scoped run194pass/1excludedbaseline so far.
Verification:204 ownership/controller/service tests passed;296 sibling/saved-normal tests passed with1 pre-existing RAG provider assertion excluded (same failure reproduced on untouched origin/dev025627214c). Production frontend tsc passed with8GB heap. Plan:Docs/Plans/IMPLEMENTATION_PLAN_TASK_13461_fresh_native_chat_owner.md. Narrow cancelled-draft reset restores retry while retaining explicit reopen cancellation behavior. Generic-only diff, no production Python touched; Bandit not applicable to TypeScript. Independent final review pending.
Final independent review: no actionable findings remain after one-time owner lease guard before operation adapter clone, rejecting epoch-revoked retry before recapture/admission/dispatch.204 focused tests green; latest frontend production tsc green. Latest fetched dev remains025627214c. Human Change summary explicitly waived by requester. Generic PR preparation complete; no private code or deployment credentials in public diff.
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
