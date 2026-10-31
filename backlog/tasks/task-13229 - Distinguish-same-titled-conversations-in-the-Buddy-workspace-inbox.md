---
id: TASK-13229
title: Distinguish same-titled conversations in the Buddy workspace inbox
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:16'
updated_date: '2026-09-09 07:14'
labels: []
dependencies: []
references:
  - TASK-13227
documentation:
  - Docs/Reviews/2026-09-09-buddy-v1-qualification.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13227 created two workspace conversations with the same generated title during send/retry. Buddy correctly stores separate IDs but renders identical selector and result labels, making the user's reply target hard to distinguish. Provide concise visible and accessible distinguishing context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user can distinguish same-titled conversations in both the workspace Buddy selector and result list before replying or acknowledging.
- [x] #2 Visible and accessible distinguishing context consistently identifies the selected conversation without replacing its saved title.
- [x] #3 Reply and acknowledgement remain bound to the explicit stable conversation ID; different-title labels remain concise.
- [x] #4 Buddy Management distinguishes duplicate conversation choices with the active locale while attachment writes retain the selected stable ID.
- [x] #5 Pending and queued read-aloud use current conversation labels without restarting active playback; touched frontend files satisfy configured zero-warning lint.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: presentation-only disambiguation, stable IDs and saved titles unchanged. Derive one shared label from loaded conversation summaries: ordinary titles remain concise; colliding titles gain creation-time context, with a stable ID fallback when timestamp context also collides or is absent. Put distinguishing context before long titles so it remains visible in narrow selectors. Reuse labels in selector, result rows, selected history/reply and read-aloud prefix; test same timestamps, missing metadata, long titles, exact reply and acknowledgement identities.

PR2934 Qodo follow-up: reuse shared localized conversation labels in Buddy Management, retaining exact attachment IDs; add failure-first regressions for duplicate/missing/colliding timestamps and locale changes. Stabilize speech label dependencies and consume current labels before playback without interrupting active speech. Test pending and queued speech updates; run focused UI tests and configured ESLint with zero warnings plus shared UI formatting. Preserve historical qualification artifacts; record new follow-up evidence separately. ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: bounded presentation and hook-dependency repair within existing stable ownership boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented shared duplicate-title context across picker/results/transcript/reply/status/speech, using localized created_at with stable-ID fallback for missing/null/equal dates; API Buddy summaries now expose nullable timestamps and dedicated frontend type. Saved title and all routing/ack IDs unchanged. Failure-first regressions cover long titles, null timestamps, equal/prefix-collision IDs and order stability. Final UI94passed and backend58passed1Postgresunavailable skip; focused changed-entrypoint typecheck0diagnostics; productionBandit0findings. Independent review's nullable-date finding fixed and re-reviewed. Existing ADR-005 applies; user/API docs and follow-up evidence updated. PR2934.
Reopened for PR2934 Qodo frontend review. Official MCP task_view hung and was stopped before mutation; repository AGENTS.md permits CLI fallback. Existing historical qualification evidence remains unchanged; follow-up work is tracked separately.

Review scope also includes bounded concrete type annotations in IndependentBuddyHost.test.tsx so the original frontend lint target set can reach zero warnings; no host production behavior changes.

PR2934 frontend follow-up implemented: BuddyManagementModal now reuses shared active-locale labels for duplicate-title choices while retaining exact option/attachment IDs. BuddyInteraction memoizes label inputs/callbacks, declares effect dependencies, and reads current labels after deferred authorization without interrupting active speech. Bounded touched-test annotations close historical lint warnings; removed an unsupported existing Testing Library exact option. Failure-first run: 5 failed / 20 passed (raw /private/tmp/tldw-task13229-qodo-ui-red.log); green focused run: 43 passed across management/interaction/host/label-helper tests (raw /private/tmp/tldw-task13229-qodo-ui-green.log). After the final type-only test correction, management rerun: 14 passed (/private/tmp/tldw-task13229-qodo-management-final.log). Configured ESLint reports 0 errors / 0 warnings for both components and all three touched tests (/private/tmp/tldw-task13229-qodo-ui-lint-configured-summary.log); shared UI formatter and whitespace checks pass. Focused typecheck has 0 diagnostics in five touched files and 15 dependency diagnostics outside them (/private/tmp/tldw-task13229-qodo-ui-typecheck-final.log); not a full project typecheck pass. ADR-005 applies; no new ADR or Python/Bandit work is required for this TypeScript-only follow-up. Historical evidence unchanged. Root owns combined qualification gate/build and final follow-up evidence; task remains In Progress until those finish.

Root final review: the seven-file Buddy UI gate passed 99 tests; five touched UI component/test files have zero configured lint warnings/errors and no direct TypeScript diagnostics (15 external dependency diagnostics remain). The final Chrome production build passed in 48.9 s with six source-hashed overlays, 11 verified manifest targets and 1,378-file ZIP integrity. Shared setup labels and pending/queued speech updates preserve exact target IDs and do not restart active playback. Separate review receipts are retained in Docs/Reviews/artifacts/buddy-pr2934-qodo; original captures remain historical. ADR-005 remains applicable; native/audio qualification stays open under TASK13227.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Same-titled conversations are distinct in Buddy Management, the workspace selector, result/status rows and speech. Localized timestamps precede titles; stable IDs handle missing/colliding timestamps. New deferred-read/queue checks preserve current labels without replaying active speech.99 focused UI tests and final Chrome build/manifest/ZIP checks pass; five touched component/test files have zero lint warnings/errors and no direct typecheck diagnostics. Evidence: Docs/Reviews/artifacts/buddy-pr2934-qodo. ADR-005 applies; no database schema change. Native and real-audio acceptance remains separate.
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
