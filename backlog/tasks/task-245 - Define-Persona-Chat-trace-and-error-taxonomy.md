---
id: TASK-245
title: Define Persona Chat trace and error taxonomy
status: Done
assignee:
  - Codex
created_date: '2026-05-10 21:03'
updated_date: '2026-05-10 21:16'
labels:
  - persona
  - chat
  - stage-2
  - evaluations
  - planning
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
  - 'https://github.com/rmusser01/tldw_server/issues/1546'
  - 'https://github.com/rmusser01/tldw_server/pull/1545'
  - 'https://github.com/rmusser01/tldw_server/issues/635'
  - 'https://github.com/rmusser01/tldw_server/pull/1551'
  - 'https://github.com/rmusser01/tldw_server/issues/1552'
documentation:
  - Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md
  - Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define Slice 1 for Stage 2 Persona Chat quality work from GitHub issue #1546. Produce a repo-grounded trace/error taxonomy artifact for ordinary persona-backed chat in the Buddy/Persona system. The artifact must document whether real traces are available or synthetic fixtures are the right first pass, define human-readable failure labels with trigger conditions and expected evidence, map labels to deterministic fixture checks, optional future judge candidates, or human-only review, and identify the minimum deterministic fixture set for the next PR-sized implementation slice. Keep Buddy/Live renderer, VN/CYOA, external benchmark adoption, and LLM-as-judge implementation out of scope. Tracker ownership stays explicit: #1510 remains Buddy/Live reliability, while VN/CYOA work remains tracked under #1391.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current ordinary persona-backed chat contracts, prompt assembly, memory mode, exemplar, restore, telemetry, and existing tests are rechecked from source.
- [x] #2 The artifact reviews at least 20 representative persona-chat cases or explicitly documents why synthetic fixtures are the right first pass.
- [x] #3 Each failure label includes trigger conditions, expected evidence, and classification as deterministic, judge-candidate, or human-only.
- [x] #4 The artifact identifies exact backend/frontend surfaces needed for deterministic fixture coverage.
- [x] #5 The artifact names the next deterministic fixture PR/task so it can be opened directly from this work.
- [x] #6 Docs verification, git diff hygiene, and Bandit skip rationale are recorded for this docs-only slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read the merged Stage 2 definition artifact and issue #1546.
2. Inspect source/tests for persona-backed chat identity, prompt assembly, memory modes, exemplar use, restore behavior, and telemetry labels.
3. Create a durable taxonomy artifact with representative/synthetic case rationale, failure labels, evidence requirements, classification, and minimum deterministic fixture set.
4. Update Backlog notes and final summary, run documentation verification and git diff hygiene, then package for PR review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md. Rechecked ordinary persona-backed chat identity, conversation response fields, frontend create/reuse/restore paths, memory mode UI, runtime exemplar guidance, prompt assembly, retrieval, memory writeback, telemetry, backend tests, frontend tests, and dialogue-tree robustness reports from source. The artifact documents why synthetic fixtures are the right first pass, defines 20 representative ordinary Persona Chat fixture cases, maps failure labels to deterministic/judge-candidate/human-review handling, identifies backend/frontend deterministic fixture surfaces, and names the next PR/task as Stage 2: Add deterministic Persona Chat quality fixtures.

Verification: taxonomy doc marker scan returned no matches and git diff --check passed. Runtime tests were not run because this slice changes docs/Backlog only. Bandit is not applicable because no Python files changed.

GitHub packaging: opened PR #1551 for the trace/error taxonomy artifact, opened follow-up issue #1552 for deterministic Persona Chat quality fixtures, and linked both from parent tracker #1546.

PR #1551 review-fix pass started: verify and address unresolved taxonomy comments for case-id casing, fixture label mismatch, undefined PC-UX-001 label, chat.py evidence line, and tracker-boundary wording.

Review edits applied: canonicalized fixture case_id casing to PC-CASE-###, aligned the fixture example labels with PC-CASE-001, updated the prompt-reveal evidence line to chat.py:3375, defined PC-UX-001, and restated #1510/#1391 tracker boundaries in the taxonomy and task.

Review verification: git diff --check passed; rg found no stale lowercase pc-case ids or chat.py:3387 evidence reference; label consistency script reported missing= with no undefined labels. Bandit remains skipped because this review pass changed docs and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined the Persona Chat trace/error taxonomy for #1546 with synthetic-fixture rationale, 20 representative cases, failure labels, deterministic fixture surfaces, and the next deterministic fixture task.

Packaged the taxonomy artifact in PR #1551 and opened follow-up issue #1552 for deterministic fixture coverage.

PR #1551 review follow-up fixed the taxonomy comments by canonicalizing case-id casing, aligning example labels, defining PC-UX-001, correcting the chat.py evidence line, and restating #1510/#1391 tracker boundaries.
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
