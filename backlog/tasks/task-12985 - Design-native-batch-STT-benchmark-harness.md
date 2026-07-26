---
id: TASK-12985
title: Design native batch STT benchmark harness
status: Done
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
- [x] #1 Design captures architecture, dataset contract, scoring, timing, reporting, recovery, and regression policy
- [x] #2 Design records the review of pipecat-ai/stt-benchmark and the decisions to avoid Pipecat and LLM judges
- [x] #3 Approved design is self-reviewed, linked from this task, and committed
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

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved native batch STT benchmark design committed as a09ee18033. The spec documents the pinned Pipecat benchmark review, standalone native-adapter architecture, hybrid English-first dataset contract, deterministic strict/normalized WER and CER, cold/warm timing, fair comparison modes, resumable schema-versioned artifacts, privacy and network-egress safeguards, regression policy, migration from existing golden tests, and verification strategy. Verification: placeholder scan and git diff --check passed; independent review returned Ready to commit with no remaining Critical or Important issues. No code changed, so tests and Bandit were not applicable. No known blockers.
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
