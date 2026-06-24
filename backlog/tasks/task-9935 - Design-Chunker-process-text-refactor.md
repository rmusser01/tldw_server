---
id: TASK-9935
title: Design Chunker process_text refactor
status: Done
created_date: 2026-06-24 04:49
labels:
- chunking
- refactor
- design
priority: High
updated_date: 2026-06-24 21:51
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a behavior-preserving refactor of Chunker.process_text into smaller internal process_text components. Scope is a written design/spec only; implementation planning follows after user review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design captures architecture, data flow, components, error handling, testing, and staged implementation approach
- [x] #2 Design explicitly preserves public Chunker.process_text behavior and return shape
- [x] #3 User review gate is recorded before implementation planning
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use brainstorming workflow to write Docs/superpowers/specs/2026-06-24-chunker-process-text-refactor-design.md, self-review it for ambiguity and scope issues, then commit the spec and Backlog task on a separate design branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote design spec at Docs/superpowers/specs/2026-06-24-chunker-process-text-refactor-design.md. Self-review completed: placeholder scan clean; tightened pipeline constructor guidance and telemetry extraction constraints; scope remains design-only and implementation is gated on user review.
Addressed spec review findings: clarified pipeline-owned input validation, documented non-circular LLM/telemetry dependency handling, made the process-only method option exclusion set explicit, and noted the dependency on the Chunking hardening regression tests/base branch.
Addressed final spec review findings: added a shared noncritical exception policy module requirement, required representative output-equivalence tests before production logic moves, and clarified that each extraction stage must be wired into the active Chunker.process_text path before the next stage.
Addressed additional spec review findings: corrected normal dispatch to call chunk_text, documented pre-validation process metric behavior, made enable_frontmatter_parsing raw bool coercion explicit, and added required tests for normal dispatch and LLM override cleanup on exceptions.
Addressed follow-up spec review finding: clarified that process_text output normalization is path-specific and that normal chunk_text fallback objects must continue to become str(obj) with empty metadata unless a targeted behavior-change test requires otherwise.
Addressed follow-up testing-spec review finding: made invalid-input metrics counter coverage mandatory by requiring a monkeypatched telemetry hook, removing the previous optional/practicality wording.
Addressed LLM context ownership review finding: required _LLM_UNSET and llm_override_scope to live in a shared non-circular Chunking module outside process_text, and documented that general Chunker LLM helpers must not import from the internal process_text package.
User approved the reviewed design spec and authorized continuing to implementation planning.
Implementation planning continued under TASK-9936 after the user approved the reviewed design spec. The plan file is Docs/superpowers/plans/2026-06-24-chunker-process-text-refactor.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and reviewed the Chunker.process_text refactor design spec, then created a separate implementation-plan task and plan document for execution. No production code changed in this design task.
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
