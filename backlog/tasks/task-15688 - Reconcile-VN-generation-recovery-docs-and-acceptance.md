---
id: TASK-15688
title: Reconcile VN generation recovery docs and acceptance
status: Done
labels:
- vn-assets
- jobs
- documentation
references:
- https://github.com/rmusser01/tldw_server/issues/2021
- Docs/API-related/VN_ASSET_PACKS_API.md
- Docs/Design/VN_PENDING_COMMAND_RECOVERY.md
- Docs/ADR/003-jobs-vs-scheduler-default.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the bounded generation-recovery closeout in GitHub #2021. Reconcile reload instructions and worker recovery with merged behavior, rerun deterministic checks, and record deployment-specific acceptance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The VN API guide documents tab-scoped same-request recovery, preflight limits, recipe reuse and durable worker replay.
- [x] #2 The relevant generation, recovery, and browser checks run from current dev with reproducible fixture behavior and recorded results.
- [x] #3 Operator guidance identifies the live worker, provider, database and Redis checks that apply to a deployment.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base commit 9dc8ad084e. ADR required: no. ADR-003 governs the existing use of Jobs for user-visible work. The guide and acceptance fixtures document the merged behavior.
Acceptance reconciliation for #2021. The API guide records owner-scoped advisory preflight and monitor guidance, targeted Retry with the stored source recipe, same-tab replay of unresolved commands, and durable Jobs and variant recovery. The published guide mirrors the source.
Verified on remote dev base 9dc8ad084e with 328 backend VN generation and recovery tests, 369 VN frontend tests, three Chromium recovery scenarios, and 34 published-doc refresh tests. Scoped ESLint, frontend typecheck, strict MkDocs build, and git diff --check passed. Bandit is inapplicable because this change touches Markdown and TypeScript test fixtures.
The automated checks use test workers and mocked providers or API responses. #2021 remains open for qualification on a target deployment with actual queues, databases, storage, image provider/model, and GPU runtime where used. Verify Redis on deployments that depend on it. Local Redis binaries are available, but no live VN generation rollout was run.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled the API and published guides with shipped recipe, worker and browser recovery. Restored the browser and storage-failure fixtures, verified the automated acceptance evidence, and documented the remaining deployment checks for #2021.
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
