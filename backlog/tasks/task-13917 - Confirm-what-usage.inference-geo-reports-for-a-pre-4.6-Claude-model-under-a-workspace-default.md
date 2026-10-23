---
id: TASK-13917
title: Confirm what usage.inference_geo reports for a pre-4.6 Claude model under a
  workspace default
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the TASK-13878/13879/13880/13885 PR (plan Docs/superpowers/plans/2026-10-23-pricing.md). compute_costs applies Anthropic's 1.1x US-only multiplier whenever usage.inference_geo is 'us', with no model gate. The pricing page scopes the multiplier to Claude 4.6 and later and says earlier models reject the parameter; the data-residency page says a workspace default_inference_geo applies when the parameter is omitted. What a pre-4.6 model reports in such a workspace is undocumented. If it reports 'us', that call is over-billed 1.1x; the alternative is a 4.6-and-later model gate in compute_costs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The behaviour is confirmed from Anthropic documentation or support and recorded, and compute_costs gates the multiplier by model if a pre-4.6 call can report 'us', with a test
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
