---
id: TASK-13356
title: Persist VN generation recipe snapshots and replay them on Retry
status: In Progress
assignee: []
created_date: '2026-09-25 16:11'
updated_date: '2026-09-25 19:58'
labels:
  - vn-assets
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
  - 'https://github.com/rmusser01/tldw_server/pull/3015'
documentation:
  - Docs/superpowers/specs/2026-09-25-vn-generation-recipe-snapshots-design.md
  - Docs/API-related/VN_ASSET_PACKS_API.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Issue #2021 follow-up after PR #2954. Freeze the authored generation recipe when a batch is accepted, resolve worker-specific execution settings once, and make slot Retry reproduce the failed recipe while Regenerate uses current settings. Preserve existing packs and jobs; explicitly handle legacy batches without snapshots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Queued generation continues to use its accepted recipe after pack, slot, character, or world-book edits.
- [x] #2 A failed slot Retry uses the failed recipe; deliberate regeneration uses current settings.
- [x] #3 Worker execution records effective backend and model without persisting credentials or local secret values.
- [x] #4 Tests cover drift, retry/restart, duplicate delivery, legacy batches, and API behavior; docs state the contract.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #3015 against dev is ready for review with the requester-authored Change summary. Recipe snapshots, Retry provenance, local-model drift protection, and replay-safe fanout are implemented. Qodo three correctness bugs and CodeRabbit fanout completion bug were reproduced and fixed: latest-batch slot guards preserve sibling failures; batch counters/status update atomically; generation status reports per-slot failed-source recipe availability; synchronous generation endpoints run in FastAPI threadpool. Review follow-up also added docstrings, reflowed added long lines, classified VN tests, used the CharacterStore update API in a generation test, and added contextual recipe diagnostics without changing stable API error codes. Verification after final changes: full VN suite 305 passed; frontend VN 37 passed; typecheck, OpenAPI drift, scoped Ruff, Bandit (0 findings), and diff check passed. Black --check reports pre-existing whole-file formatting drift; no broad reformat. Authenticated browser QA unavailable in isolated checkout. GitHub Actions remain queued. Remaining #2021 work: crash-after-file-registration exactly-once recovery and mutable local model file contents.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented versioned VN batch recipes and one-time execution choices. Retry reproduces the recorded failed slot recipe; Regenerate captures current settings. Legacy batches cannot claim a faithful Retry. Worker fanout replay and duplicate delivery preserve terminal state, and implicit local model paths are guarded by digest/mode without being stored in batch or item metadata. Remaining #2021 work includes exactly-once recovery after file persistence and mutable local model contents.
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
