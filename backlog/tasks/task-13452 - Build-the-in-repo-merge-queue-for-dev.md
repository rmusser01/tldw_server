---
id: TASK-13452
title: Build the in-repo merge queue for dev
status: In Progress
labels:
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PRs into dev race each other: dev requires seven statuses on an up-to-date head, so every merge puts every other ready PR behind and all of them restart the required gates (65-120 minutes per cycle on PR #3155, which was pushed behind twice on 2026-10-04). Build a one-at-a-time queue so only the PR at the front is rebased and tested. Port of the tldw_chatbook queue (tldw_chatbook PR #2996), adapted to this repository's seven required contexts. Spec: Docs/superpowers/specs/2026-10-04-merge-queue-design.md. ADR-063.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Armed PRs into dev are processed one at a time in arming order; PRs behind the front are never rebased, dispatched or commented on
- [ ] #2 After a queue rebase all seven required contexts are produced on the new head, with change detection compared against dev's tip
- [ ] #3 A failed context is retried once and the PR is evicted with the reason on the second failure
- [ ] #4 The queue never arms auto-merge, merges or pushes, enforced by a guard test
- [ ] #5 With MERGE_QUEUE unset the queue does nothing; dry logs decisions without side effects
- [ ] #6 A required gate whose change detection failed reports red instead of being skipped
- [ ] #7 Existing CI contract tests stay green
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
