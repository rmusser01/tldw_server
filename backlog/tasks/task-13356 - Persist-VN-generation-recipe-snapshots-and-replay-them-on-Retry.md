---
id: TASK-13356
title: Persist VN generation recipe snapshots and replay them on Retry
status: In Progress
assignee: []
created_date: '2026-09-25 16:11'
updated_date: '2026-09-25 17:28'
labels:
  - vn-assets
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
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
Implemented in codex/vn-recipe-snapshots. Verification: VN backend 289 passed; frontend VN monitor/workbench 18 passed; frontend typecheck passed; OpenAPI fingerprint check passed; scoped Ruff has no new findings (three existing BLE001/UP035 warnings); Bandit touched production files 0 findings; git diff --check passed. Browser QA not run: isolated checkout has no authenticated VN pack and worker-backed data. Crash-after-file-registration exactly-once recovery and mutable local model bytes remain separate #2021 work.

Independent review found and fixed three gaps: replay of transient fanout-failed parent jobs, slot-aware WebUI source binding, and implicit local model path/mode drift detection without persisting the configured path. Added a parent/child completion-race guard. Latest checks: VN backend 292 passed; frontend VN components 19 passed; typecheck, OpenAPI drift, scoped Ruff excluding three established baseline warnings, git diff --check, and Bandit (0 findings) passed. Authenticated browser QA remains unverified.

Final pre-PR verification: 297 VN backend tests passed; 20 focused frontend tests passed; frontend typecheck, OpenAPI drift check, scoped Ruff excluding three verified pre-existing warnings, git diff --check passed; Bandit 0 findings. Authenticated browser QA not available in isolated checkout. Two read-only independent reviews were addressed, including path redaction, per-slot failure provenance, transient parent/child race handling.
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
