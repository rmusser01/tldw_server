---
id: TASK-2369
title: Design CATS API fuzzing harness
status: In Progress
labels:
- testing
- security
- api
documentation:
- Docs/superpowers/specs/2026-06-27-cats-api-fuzzing-harness-design.md
modified_files:
- Docs/superpowers/specs/2026-06-27-cats-api-fuzzing-harness-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved design spec for a broad-in-blocks CATS OpenAPI fuzzing harness for the tldw_server API. The spec should cover staged block architecture, isolation and egress controls, CATS command shape, reporting, triage, CI/nightly policy, and known OpenAPI validation issues discovered during exploration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents broad staged fuzzing blocks and risk metadata.
- [x] #2 Spec requires isolated runtime env, credential scrubbing, and external egress safeguards.
- [x] #3 Spec covers seeded state/refData needs for stateful blocks.
- [x] #4 Spec defines CATS failure policy, reporting artifacts, and CI/nightly/manual scope.
- [x] #5 Spec records the current vector store OpenAPI validation issue and rollout path.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-27: Wrote design spec at Docs/superpowers/specs/2026-06-27-cats-api-fuzzing-harness-design.md. Local self-review covered block architecture, egress controls, seeded state, failure policy, reporting, CI policy, and known CATS OpenAPI validation issue. Formal spec-review subagent was not dispatched because the available multi-agent tool requires explicit user request for subagents/delegation. Verification: git diff --check passed for the spec and task paths. Bandit skipped because this stage touched only Markdown/Backlog task documentation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
