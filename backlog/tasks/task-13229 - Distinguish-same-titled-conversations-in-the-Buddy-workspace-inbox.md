---
id: TASK-13229
title: Distinguish same-titled conversations in the Buddy workspace inbox
status: Done
created_date: 2026-09-09 05:16
priority: medium
references:
- TASK-13227
documentation:
- Docs/Reviews/2026-09-09-buddy-v1-qualification.md
assignee:
- '@codex'
updated_date: 2026-09-09 06:40
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: presentation-only disambiguation, stable IDs and saved titles unchanged. Derive one shared label from loaded conversation summaries: ordinary titles remain concise; colliding titles gain creation-time context, with a stable ID fallback when timestamp context also collides or is absent. Put distinguishing context before long titles so it remains visible in narrow selectors. Reuse labels in selector, result rows, selected history/reply and read-aloud prefix; test same timestamps, missing metadata, long titles, exact reply and acknowledgement identities.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented shared duplicate-title context across picker/results/transcript/reply/status/speech, using localized created_at with stable-ID fallback for missing/null/equal dates; API Buddy summaries now expose nullable timestamps and dedicated frontend type. Saved title and all routing/ack IDs unchanged. Failure-first regressions cover long titles, null timestamps, equal/prefix-collision IDs and order stability. Final UI94passed and backend58passed1Postgresunavailable skip; focused changed-entrypoint typecheck0diagnostics; productionBandit0findings. Independent review's nullable-date finding fixed and re-reviewed. Existing ADR005 applies; user/API docs and follow-up evidence updated. PR2934.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Duplicate conversation titles now show consistent creation-time context or stable identifiers across Buddy selection, results and replies. Saved names and target IDs are unchanged. Null timestamps and ID collisions have regressions; final UI/backend checks and independent re-review passed. ADR-005 and API/user/evidence docs updated; PR #2934.
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
