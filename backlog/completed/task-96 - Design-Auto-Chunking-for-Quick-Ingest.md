---
id: TASK-96
title: Design Auto Chunking for Quick Ingest
status: Done
assignee:
  - codex
created_date: '2026-05-06 16:13'
updated_date: '2026-05-07 04:45'
labels:
  - design
  - chunking
  - quick-ingest
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved design spec for novice-facing Auto Chunking in tldw_server. The design should capture the existing chunking/template inventory, the approved Quick Ingest UX where Chunking defaults to Auto with a Manual escape hatch, deterministic media-aware planner behavior, explicit AI-assist opt-in, API compatibility rules, and the backend wiring gaps found during discovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design spec is written under Docs/superpowers/specs with the approved Auto Chunking product contract and constraints.
- [x] #2 The spec documents existing backend/frontend chunking capabilities and the current integration gaps relevant to Auto Chunking.
- [x] #3 The spec defines API fields, Quick Ingest behavior, media-aware planner behavior, AI-assist fallback behavior, and metadata/rationale output.
- [x] #4 The spec includes testing and rollout considerations without implementing code changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a design-only spec at Docs/superpowers/specs/2026-05-06-auto-chunking-design.md using the user-approved brainstorming decisions.
2. Ground the spec in discovered repo evidence: Chunking module capabilities, media request schemas, media-add/job parsing gap, PDF-specific template support, Quick Ingest UI/service behavior, and query-time Agentic Chunking distinction.
3. Keep the spec implementation-ready but avoid code changes: product contract, API/UI fields, deterministic planner behavior by media type, AI-assist opt-in/fallback, storage/metadata outputs, testing, rollout, and open risks.
4. Verify the spec file exists and contains the approved sections; run lightweight diff/status checks. No Bandit is required for documentation-only changes, but record that skip.
5. Update TASK-96 with verification notes and acceptance criteria status after the spec is written.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-06: Wrote Docs/superpowers/specs/2026-05-06-auto-chunking-design.md as a documentation-only spec. It captures the approved Quick Ingest Auto/Manual model, deterministic planner behavior, explicit AI-assist opt-in and fallback, API fields, metadata output, existing repo inventory, integration gaps, rollout, and testing considerations. Bandit is not applicable at this step because only documentation and Backlog task files were changed.

2026-05-06: Staged the design spec and TASK-96 file, then attempted `git commit -m "docs: design auto chunking for quick ingest"`. Commit failed because the existing worktree has unrelated unresolved merge conflicts in Docs/superpowers/plans/2026-05-03-native-codegraph-foundation-implementation-plan.md, Docs/superpowers/plans/2026-05-03-worker-lifecycle-deprecated-code-removal-implementation-plan.md, and backlog/tasks/task-16 - Implement-native-CodeGraph-foundation-slice.md. I did not touch or resolve those unrelated conflicts.

2026-05-06: Independent design review found planning risks and patched the spec: structure_aware/schema handling, async job result exposure via chunking_plan, Auto-vs-Manual precedence for saved advanced settings, both Quick Ingest submission paths, and V1 derived views as metadata only.

2026-05-06: Second spec-review pass approved the updated spec with no blocking issues. Applied its advisory wording change so rollout says Mode defaults to Auto when Chunking is enabled, avoiding any implication that the global Chunking toggle default changes.

2026-05-07: Finalized the parent Auto Chunking design task after verifying on origin/dev that TASK-96.1 through TASK-96.9 are all Done. This parent closeout only updates Backlog bookkeeping; no product code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Auto Chunking design and implementation series completed. The design spec established the novice-facing Quick Ingest contract: enabling Chunking uses Auto by default, Manual exposes advanced settings, deterministic media-aware planning is the default, and LLM boundary assistance is explicit opt-in with deterministic fallback metadata.

Follow-on subtasks TASK-96.1 through TASK-96.9 delivered the implementation plan, backend request parsing and resolver wiring, deterministic planner, Quick Ingest payload/UI controls, AI-assist fallback documentation, real boundary assistant adapter, and PR review fixes. Verification was recorded on each subtask; this parent task is a design/backlog finalization update with no code changes, so Bandit is not applicable for this final bookkeeping change.
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
