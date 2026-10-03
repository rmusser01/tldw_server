---
id: TASK-13423
title: Design strict source-grounded answers before display
status: In Progress
assignee: []
created_date: '2026-10-03 00:13'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3090'
documentation:
  - Docs/Reviews/Claims_Integrity_Confirmation_2026_10_02.md
type: docs
ordinal: 15969
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Brainstorm the evidence-integrity workstream using confirmed server, WebUI, and extension findings from TASK-13422. Preserve accepted requirements, compare implementation approaches, review the design in sections, and prepare a written spec for user review. This task covers design and tracking only; it does not authorize product implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Accepted strict-source, supported-partial-answer, and verify-before-display policies are recorded with explicit failure semantics.
- [ ] #2 The shared verification approach and initial workflow scope are chosen through design review using existing Claims foundations.
- [ ] #3 Written design spec and ADR assessment are saved, self-reviewed, committed, and reviewed by the requester before implementation planning.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1 In Progress: requirements and approach selection. Stage 2 Not Started: sectioned design and scope review. Stage 3 Not Started: written spec, ADR assessment, self-review, commit, and requester review before writing-plans.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
User confirmed strict evidence as the default for source-based research; partially supported questions return supported facts with explicit gaps; verify before displaying the answer and show progress while generation/checks run. If no useful answer is supported, decline. Confirmed Claims verdict, corpus-scope, final-answer, streaming, and persistence gaps must be addressed by the design. Proposed architecture is not yet approved. ADR required: yes for the eventual durable verification/default and API or persistence contract; draft after approach/design review, reuse accepted ADR-007 and assess applicable ownership/governance rules. No product edits or tests are part of this design task yet.
<!-- SECTION:NOTES:END -->
