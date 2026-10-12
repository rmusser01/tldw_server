---
id: TASK-13611
title: Publish chat remediation draft PR against dev
status: In Progress
labels:
- chat
- review
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish the user-authorized accumulated remediation branch as a draft PR targeting remote dev. Inventory large shared UI/backend scope, record bounded verification and unresolved acceptance gates, validate canonical task records, and link the PR while remediation continues. No merge or ready-for-review claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The draft PR targets remote dev and includes the exact reviewed local branch without unrelated runtime artifacts.
- [ ] #2 Description follows repo template, accurately separates verified changes from open H2/UX/physical PostgreSQL gates, and leaves human Change summary pending.
- [ ] #3 Canonical task checks and normal owned-file hooks pass; PR URL, head/base identity and continuation scope are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment:no new decision; PR links existing049/050 and proposed069/070/071. User explicitly authorized PR againstremote dev. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_draft_pr_13611.md. Master13458 remainsInProgress.
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
