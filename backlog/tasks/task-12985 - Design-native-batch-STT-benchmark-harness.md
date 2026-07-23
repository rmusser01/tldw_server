---
id: TASK-12985
title: Design native batch STT benchmark harness
status: In Progress
assignee: []
created_date: '2026-07-23 06:09'
updated_date: '2026-07-23 06:34'
labels:
  - stt
  - benchmark
  - design
dependencies: []
references:
  - 'https://github.com/pipecat-ai/stt-benchmark'
documentation:
  - Docs/superpowers/specs/2026-07-22-native-batch-stt-benchmark-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the approved design for a standalone batch-only STT benchmark that invokes tldw_server native provider adapters, uses deterministic strict and normalized WER/CER, and supports regression and comparison profiles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design captures architecture, dataset contract, scoring, timing, reporting, recovery, and regression policy
- [ ] #2 Design records the review of pipecat-ai/stt-benchmark and the decisions to avoid Pipecat and LLM judges
- [ ] #3 Approved design is self-reviewed, linked from this task, and committed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write the approved design to Docs/superpowers/specs/2026-07-22-native-batch-stt-benchmark-design.md; self-review for placeholders, contradictions, ambiguity, and scope; record verification and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design written and self-reviewed. No TODO/TBD placeholders remain. Architecture, dataset/scoring contracts, timing definitions, recovery, privacy, and regression policy are internally consistent. Upstream review is pinned to pipecat-ai/stt-benchmark commit 66f2cbf8. Verification: git diff --check passed. Bandit not applicable because only Markdown and Backlog records changed.

Independent design review completed. Initial review found resume timing, cross-model compatibility, retry reduction, scoring determinism, comparison fairness, duration validation, and privacy gaps; all were resolved. Focused re-review confirmed no remaining Critical or Important issues and returned Ready to commit: Yes.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
