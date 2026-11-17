---
id: TASK-13709
title: Reassess Chat setup history and character roleplay UX on merged dev
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review latest remote dev after PR #3229 across shared WebUI and extension chat, with independent design and deterministic/browser assessments and a prioritized NN/G issue ledger. This task records the review; it does not authorize or implement new product changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pin the reviewed remote-dev revision and declare live/source coverage and limitations
- [x] #2 Assess first-time and power-user setup, loading conversations, and character roleplay across both clients
- [x] #3 Provide evidence, severity, concrete remedies, and all ten Nielsen heuristic scores
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reviewed revision: origin/dev 2e48dc91848887dcb74ebe39d6ff877fed4d64bd. ADR assessment: no new ADR required for a read-only UX assessment and report; no durable architecture rule changes. Independent assessments A and B remain isolated until design assessment A finishes. Source-verified production preview is temporary on port19770; stop PID46069 before final delivery. Preserve existing user conversations and unrelated worktrees.
Report: Docs/Reviews/CHAT_LATEST_DEV_UX_REASSESSMENT_13709_2026_10_15.md. Dual independent A/B review, one narrowed detector run, authenticated WebUI setup/history load+reload, rebuilt Chrome extension compact/full-page inspection. One P1 and five P2 findings; 25/40 heuristic health. Original9 conversations/28 messages deeply equal after review. Temporary preview PID46069 stopped; own tabs/contexts closed. Bandit N/A documentation-only. No new product changes or CI certification. Physical accessibility, generation, new card import/export and browser matrix remain unrun.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed latest-dev UX reassessment at 2e48dc91848887dcb74ebe39d6ff877fed4d64bd. Report documents remaining setup-recovery bug, confirmed extension starter-prompt loss, first-run hierarchy, roleplay setup order, preview semantics and compact tracked-history limits, with evidence and remedies. Review artifacts only; implementation remains future work.
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
