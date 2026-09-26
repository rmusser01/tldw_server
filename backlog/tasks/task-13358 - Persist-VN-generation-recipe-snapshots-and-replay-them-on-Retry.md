---
id: TASK-13358
title: Persist VN generation recipe snapshots and replay them on Retry
status: In Progress
assignee: []
created_date: 2026-09-25 16:11
updated_date: 2026-09-26 21:54
labels:
- vn-assets
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/2021
- https://github.com/rmusser01/tldw_server/pull/3015
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #3015 against dev is ready for review with the requester-authored Change summary. Recipe snapshots, Retry provenance, local-model drift protection, and replay-safe fanout are implemented. Qodo three correctness bugs and CodeRabbit fanout completion bug were reproduced and fixed: latest-batch slot guards preserve sibling failures; batch counters/status update atomically; generation status reports per-slot failed-source recipe availability; synchronous generation endpoints run in FastAPI threadpool. Review follow-up also added docstrings, reflowed added long lines, classified VN tests, used the CharacterStore update API in a generation test, and added contextual recipe diagnostics without changing stable API error codes. Verification after final changes: full VN suite 305 passed; frontend VN 37 passed; typecheck, OpenAPI drift, scoped Ruff, Bandit (0 findings), and diff check passed. Black --check reports pre-existing whole-file formatting drift; no broad reformat. Authenticated browser QA unavailable in isolated checkout. GitHub Actions remain queued. Remaining #2021 work: crash-after-file-registration exactly-once recovery and mutable local model file contents.

Renumbered from the VN branch-local TASK-13356 to TASK-13358 with explicit requester approval on 2026-09-26 UTC, resolving the collision with the unrelated ADR task from dev. Only the VN task ID/file and its VN spec/plan/PR references changed; ADR tracking remains untouched. Latest PR #3015 review follow-up is bc1b775699: rejected parent enqueue writes owned-slot failure atomically, retryable variant attempts remain active, final slot/batch failures commit together, and lost-response fanout recovery preserves persisted images. Seven new regressions pass. Final 312-test VN run: 308 passes, zero assertion failures, four disk-full setup errors; all four passed on isolated rerun. Scoped Ruff, OpenAPI drift, diff checks passed; Bandit returned zero findings. Tracking-only renumber needs no additional runtime tests or Bandit scan. ADR check: no new ADR required for review correctness fixes; Docs/ADR/003-jobs-vs-scheduler-default.md continues to govern Jobs ownership. Task remains In Progress until PR review/CI and merge finish.

Rebased PR #3015 onto origin/dev 59bd584503 on 2026-09-26 UTC at requester direction; no conflicts and git range-diff confirms all five PR patches unchanged. Fresh rebased-tree verification: 312 VN backend tests passed (zero errors or failures), 37 frontend VN tests passed, frontend typecheck passed, OpenAPI drift check passed, scoped Ruff passed with only the previously documented BLE001/UP035 exclusions, Bandit returned zero findings/errors, and diff check passed. Typecheck initially could not resolve Playwright from shared test helpers because the isolated worktree lacked the apps-level dependency link; using the existing installation resolved that environment issue without tracked config or lockfile changes. Qodo and CodeRabbit had no unresolved findings before this push; re-review and current-head CI are required before merge. The human-owned Change summary remains unchanged. dev rules permit merge commits and require backend-required, security-required, coverage-required, frontend-required, e2e-required, container-build-check, and frontend-license-policy/trusted/dev.

2026-09-26 11:31 UTC: backend-required on 440803c2a5 failed only at OpenAPI drift (expected schema_count 3210, actual 3209; current CI hash f9cc19147feb9dc2bc438d19f4b7d5c8a2c9946596f715a47bff7fa69b4ddfe1). Investigation found local Pydantic 2.11.7 versus declared/CI 2.13.5; preparing a temporary dependency overlay to reproduce and compare schemas without altering the shared environment. dev advanced to a2826f103f through unrelated VZ test/docs work. Clean worktree and remote-head ownership confirmed at 440803c2a5; scoped rebase and verification remain required before merge.

CI failure reproduced exactly using a temporary dependency overlay: Pydantic 2.13.5/pydantic-core 2.46.5, pydantic-settings 2.15.0, Starlette 1.7.0. Full schema comparison showed identical paths/VN schemas; two identical OscePatientContext input/output definitions are merged and three references updated. Refreshed fingerprint to CI hash f9cc19147feb9dc2bc438d19f4b7d5c8a2c9946596f715a47bff7fa69b4ddfe1 and regenerated ignored frontend API types. Rebased onto dev a2826f103f without conflicts; six original patches unchanged in range-diff. Fresh verification on rebased tree: 312 VN backend tests passed using the CI-aligned overlay, 37 frontend VN tests passed, frontend typecheck passed, OpenAPI drift passed, scoped Ruff passed with previously documented BLE001/UP035 exclusions, Bandit zero findings/errors, diff check passed. Temporary UI dependency link removed after verification. No application code, shared environment, dependency requirements, or unrelated work changed. Awaiting new-head bot reviews and all required CI checks before merge; human Change summary preserved verbatim.
2026-09-26 21:38 UTC: dev advanced to f5fa1f3a41855aa02871d8b76d0ec0cebbaf9e07 through unrelated Sync blob-upload expiry work. Confirmed a clean owned worktree and remote PR head 9bfeb497d849184e3c774ea83f3763ab260d0307 before rebasing. Rebase completed without conflicts; git range-diff confirms all seven prior PR patches unchanged. Fresh rebased-tree verification passed: 312 VN backend tests (207.61s) using the CI-aligned temporary overlay, 37 frontend VN tests, frontend typecheck, OpenAPI drift, scoped Ruff with documented BLE001/UP035 exclusions, Bandit zero findings/errors, and diff check. Temporary UI dependency link removed. No runtime code, shared dependencies, or unrelated work changed. Human Change summary remains verbatim. Current-head re-review and all required CI checks are still required after the protected push; Stage 4 remains In Progress.
Current-head Qodo re-review on 689bbe20eb reported malformed stored Retry recipes returning 500 rather than the documented conflict (issuecomment-5850128510). Code inspection confirms authored slot fields are indexed after top-level-only validation, and execution JSON/slot entries are decoded without the worker's error mapping. Scoped follow-up: reproduce malformed snapshots through the Retry API, validate consumed authored/execution slot fields with existing Pydantic, share execution-recipe parsing with the worker, and map invalid snapshots to the existing stable recipe-invalid codes. Keep legacy absence unavailable, preserve valid snapshots verbatim, and create no retry batch or job on rejection. Rerun full VN tests, scoped Ruff/Bandit/OpenAPI checks, then obtain current-head re-review.
Malformed-snapshot regressions reproduced all 18 cases before the fix: raw 500/400 errors, incorrect conflict codes, and incorrectly accepted 202 retries. After strict consumed-field validation and shared execution parsing, all 18 passed. The first broader run exposed seven existing lazy-depth/fanout tests because lazy-depth recipes legitimately record zero variants; corrected the validator to match the existing slot schema (ge=0), retaining exact seed-count validation. A new unit regression observed that zero-variant rejection before the correction. Focused final run: 81 passed, including six recipe unit cases preserving lazy-depth slots, unknown metadata, and legacy absence. Fresh Ruff, compilation, OpenAPI drift, and Bandit (zero findings/errors) passed. Full 336-test VN verification is running before commit/push; no checks are bypassed.
Final malformed-snapshot follow-up verification: all 336 VN backend tests passed (198.45s), including 18 new API regressions and six recipe unit cases. Authored slot fields and execution slot fields are strictly validated with Pydantic without rewriting snapshots; seed count must match variant count, and valid zero-variant lazy-depth slots remain accepted. Retry and worker execution parsing share stable invalid/unavailable error mapping. Rejected snapshots create no batch or job. Fresh scoped Ruff with documented pre-existing exclusions, compileall, OpenAPI drift, diff checks, and Bandit zero findings/errors passed. The 37 frontend VN tests and typecheck passed earlier in this same dev rebase; this follow-up changes no frontend or public schemas. dev remains f5fa1f3a41855aa02871d8b76d0ec0cebbaf9e07 and remote ownership remains 689bbe20ebc8ad00d6864b1061ddc20dfe190f93. Current-head reviews and CI remain mandatory after publishing the scoped fix.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
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
