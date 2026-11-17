---
id: TASK-15554
title: Fix unavailable models and recovery in Media analysis
status: In Progress
created_date: 2026-11-17 05:03
references:
- https://github.com/rmusser01/tldw_server/pull/3491
updated_date: 2026-11-17 05:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use verified Ollama model inventory and suppress unavailable MLX runtime. Require a fresh catalog selection and give safe recovery guidance when Media analysis fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unavailable bundled Ollama and MLX models are excluded, while working models remain selectable.
- [ ] #2 The dialog waits for catalog verification and displays safe recovery guidance.
- [ ] #3 Backend and frontend regressions cover both behaviors.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reuse provider discovery, readiness, and chat error formatting. ADR-025 governs provider routing. No new ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
87 scoped backend and 132 scoped frontend tests passed for the initial change. Bandit found no issues in the edited backend sources. Added a regression for HTTP 402 usage-limit refusals in analysis: report the limit and skip the non-streaming retry.
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
