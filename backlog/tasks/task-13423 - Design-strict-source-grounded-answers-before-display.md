---
id: TASK-13423
title: Design strict source-grounded answers before display
status: In Progress
assignee: []
created_date: '2026-10-03 00:13'
updated_date: '2026-10-03 00:30'
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
Stage 1 Complete: context, accepted answer/display policies, and architecture approach chosen. Stage 2 In Progress: review architecture boundaries, rollout scope, verdicts, client flow, failures, and acceptance cases. Stage 3 Not Started: written spec and ADR assessment, self-review, commit, and requester review before writing-plans.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Accepted policies: strict evidence by default for source-based research; return supported facts with explicit gaps; decline when no useful answer is supported; verify before displaying answer text and show progress during generation/checks. On October 2, 2026 (America/Los_Angeles), user selected a shared server verification step while preserving existing generation, rather than moving Workspace generation into unified RAG. Detailed architecture, initial workflow scope, report/persistence, and failure semantics remain under sectioned design review. Reuse core Claims foundations after correcting confirmed verdict/scope/final-answer gaps; no generic final-answer verification HTTP contract currently exists. ADR required: yes for the eventual durable verification/default and API or persistence contract; draft with the written spec and assess accepted ADR-007 plus applicable ownership/governance rules. This is design/tracking only; no product edits, installs, or implementation tests yet. Written spec review and implementation-plan review remain separate prerequisites.
<!-- SECTION:NOTES:END -->
