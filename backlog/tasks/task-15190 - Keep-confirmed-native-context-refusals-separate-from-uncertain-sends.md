---
id: TASK-15190
title: Keep confirmed native context refusals separate from uncertain sends
status: Done
labels:
- chat
- ux
- history
- recovery
dependencies:
- TASK-13454
- TASK-13532
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native shared character dispatch retains every post-dispatch/pre-admission error as unknown, including the latest-dev exact unsupported saved-history-context HTTP409 response that precedes input append/provider calls. Repair shared classification and human notices under the existing local receipt model; preserve exact submitted input, confirmed creation state, all other uncertain responses and scope/view guards. Composer raw text remains covered by its existing independent retention. No backend capability, retry protocol, namespace or new dependency.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing latest-dev authenticated history-context refusal tests prove the exact response family precedes provider/input admission; every native transport/caller is traced.
- [x] #2 Failure-first shared sender/caller controls separate confirmed unsupported-context refusal from generic or malformed409, missing status, HTTP/SSE/provider/after-admission and changed-owner cases.
- [x] #3 Explain the confirmed refusal in existing full/compact/Quick error surfaces with no false saved or unknown outcome, while retaining stable error results and meaningful uncertain recovery.
- [x] #4 Focused regressions, both client types, no-added lint/owned formatting, two source reviews and canonical tracking pass; exact backend/native/fixture limits and Bandit assessment are recorded.
- [x] #5 Master and parent remain In Progress, completed owned plan is retired after evidence, and verified repair is committed locally without push/PR/dev merge.
- [x] #6 Reuse existing operation-scoped recovery to retain confirmed refused submitted input separately from uncertain turns and history; preserve creation acknowledgements, never resend automatically or infer rollback.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate pre-admission/context-refusal search found no equivalent; remote/openPR highest13530/local13532 makes13533 fresh. Fresh dev5ec8c7939f remains integrated83969a296b; previous0051665cd6. ADR assessment: ADR049/050 source/history lifetimes and ADR059 canonical tracking govern; no new durable API/recovery rule intended. Existing exact projection-rejection predicate and background status/details envelopes are analogous; trace actual streaming envelope before implementation. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_context_refusal_recovery_13533.md. Two existing authorized reviewers are independently checking transport/backend contract and state/creation/draft consequences.
Contract baseline: authenticated existing history API refusal/admission controls 8 passed, 21 deselected. Both numeric-409 transport envelopes preserve details.detail.status/code. Independent state review found host draft flush is unawaited: deleting dispatch recovery could delete the only acknowledged durable input. Revised minimal repair retains this exact input in existing local recovery with state refused, no outcome/admission/result IDs; known-refusal UI and Quick uncertainty filter reuse current primitives. No remote protocol or capability change; ADR-049/050/059 still govern, no new ADR. Creation stays bound until actual settlement.
Refinement after review: any observed stream chunk, including an empty content delta or role-only event, prevents confirmed-refusal classification. Human notice is gated by the qualified local domain Error, not matching raw Error.message. Full/compact forms may transform raw text before onSubmit; the turn journal proof preserves exact submitted intent, while raw composer text continues to rely on separate draft retention. AC revised from deleting receipt to retaining refused input.
Final evidence: frontend644/644 in11 suites; authenticated actual backend32/32 plus final formatted streaming/nonstreaming refusal6/6. Both final client type gates pass.14 authored TS/TSX files have zero added lint;15 owned frontend formatting checks and owned Python formatting pass. Bandit touched Python test113 baseline/current existing assertion findings, zero added. Both independent reviewers clear final source after reproduced empty/role event and raw-error-name collision repairs. Fresh dev5ec8c7939f still included. Master records exact declared transport/storage/account boundaries and submitted-vs-raw proof limits. Temporary completed plan retired after preserving evidence; local checkpoint only, no push/PR/merge.
Renumbered from TASK-13533 to TASK-15190 on 2026-10-31 (TASK-13960): TASK-13533 now names only "Restore production WebUI bundle budget after Knowledge changes" (backlog/tasks/task-13533 - Restore-production-WebUI-bundle-budget-after-Knowledge-changes.md). Before 2026-10-31 both tasks used TASK-13533; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The shared native character boundary distinguishes exact confirmed context refusals before any stream event from uncertain sends. Existing local refused recovery retains exact submitted input and bound creation; qualified human notices preserve original cause and accepted owner/view guards. Copy/Dismiss remains explicit, Quick excludes only refused from uncertainty, and no replay or history message is introduced. Final644 frontend checks,32 authenticated history API controls,6 formatted refusal cases, both client types, zero-added lint/Bandit and two independent reviews pass. Live latest-dev browser/packaged/AT and raw Form draft ACK remain parent gates; master and stages stay In Progress.
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
