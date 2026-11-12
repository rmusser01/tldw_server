---
id: TASK-13914
title: Anthropic usage.iterations can bill other models
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the TASK-13878/13879/13880/13885 PR (plan Docs/superpowers/plans/2026-10-23-pricing.md). compute_costs prices the top-level Anthropic usage at the request model's rate. Usage iterations of type advisor_message or fallback_message carry their own model, which may be priced differently, so those calls are costed at the wrong rate. Price each iteration at its own model's rate, or document the error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A recorded Anthropic call whose usage.iterations names another model is costed per iteration at that model's rate, with a test, or the limitation is documented in Usage_Quotas.md
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
