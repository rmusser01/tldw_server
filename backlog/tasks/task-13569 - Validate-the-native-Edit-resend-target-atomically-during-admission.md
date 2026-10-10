---
id: TASK-13569
title: Validate the native Edit-resend target atomically during admission
status: In Progress
labels:
- chat
- webui
- extension
- backend
- ux-audit
dependencies:
- TASK-13554
- TASK-13565
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protect Edit and resend against target-only changes between capture and admission. The before-message H1 selection excludes the edited user row; carry its original ID and opaque revision into the existing owner transaction and preserve immutable replay, the original graph, and the edited proposal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both admission paths validate the captured edited user target ID, live role, opaque revision and direct parent atomically before append or provider dispatch.
- [x] #2 Target-only edit, metadata/image/role drift and edit-Undo refuse with no new rows/provider call while preserving original graph and edited input.
- [x] #3 Unchanged target admits the intended edited sibling once; generic Branch, ordinary Send and protected Regenerate remain valid.
- [x] #4 Acknowledged existing-ID replay validates original immutable operation intent without retroactively rejecting a later target edit; conflicting intent is refused.
- [x] #5 Schema/docs/client capability qualify the added precondition; older servers refuse protected Edit and resend before mutation without an unguarded fallback.
- [ ] #6 Real database and authenticated ordinary/character HTTP regressions, clients, touched security/static and current native evidence meet the declared contract.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Two independent source reviews confirmed the target-only TOCTOU. Do not reopen13554 client freshness,13556 plain Save numeric PUT CAS,13558/59 atomic Regenerate or P1/P6 branch/freshness; those are separate completed operations. D7 section6.2, ADR049 selected-source ownership and ADR050 immutable recovery govern;059 tracking. Preserve ancestry-tolerant generic H1 validation. Meaningful RED must use real update_message after capture with retained ancestors/context unchanged and require409/no insert/provider; no empirical RED is claimed yet.

Design: retain the explicit edited target ID, opaque revision and direct parent as a strict optional version1 admission precondition carried only by protected Edit and resend. Both single-input and server-owned input-chain admission check it inside the existing owner transaction. Existing-ID replay compares immutable intent including the precondition before a fresh target check. Generic Send/Branch/Regenerate keep their behavior. Clients require exact schema capability before journal or POST and preserve the proposal on refusal. ADR assessment: existing049/050/059 apply; no new storage owner, table, recovery mechanism or durable architecture rule. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_atomic_edit_target_13569.md. Implementation begins after the current13549 consumer unit is verified and committed; In Progress records the authorized design work, not completion.
Stage1 now In Progress after Scene commit4932748894 and bounded current bundle. Both independent read-only assessments support strict optional3-field target precondition and existing owner-transaction validation; logical legacy predecessor remains independent of raw parent bound by opaque revision. Scope includes real metadata ABA/shared monotonic writer, immutable replay, exact primary operation capability and original echoed no-admission refusal through ordinary/Character/write tracking. All False metadata writers traced; PostgreSQL first-write placeholder must be controlled. Design/ADR assessment updated in Docs/Design/IMPLEMENTATION_PLAN_chat_atomic_edit_target_13569.md. Latest dev0c53973f17eede4074f4aa7894d0b42322340614 is included; no code/test result yet. ADR049/050/059 govern; no new durable architecture rule or dependency.
Implementation verified: connected backend261 passed/104 official PostgreSQL skips; client11 suites577 passed, plus final three changed suites263 passed; final legacy8 passed/8 skips. Both WebUI and extension types exit0;20 TS scopes introduce0 ESLint findings;10 Python scopes introduce0 Ruff findings. Bandit scans10 files:3 unchanged existing findings,0 new production/other test findings;63 added B101 results are required pytest assertions. Owned TS/Python formatting preserves executable AST. Both independent final source reviews are bounded CLEAR. Latest dev0c53973f17eede4074f4aa7894d0b42322340614 is included. Evidence prefix /private/tmp/chat-atomic-edit-target-13569-. Current native bundle/packaged/physical AT/participants remain unqualified, so AC6 and task completion stay open. Original three-field revision binds raw parent; reviewed logical predecessor remains separate. No implicit Branch action is granted for a changed Edit target. ADR049/050/059 govern; no new durable owner or dependency.
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
