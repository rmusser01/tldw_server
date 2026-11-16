---
id: TASK-15550
title: Show PDF extraction and analysis progress in Quick Ingest
status: In Progress
created_date: 2026-11-15 15:43
labels:
- media-ingestion
- webui
- bug
updated_date: 2026-11-15 16:09
references:
- https://github.com/rmusser01/tldw_server/pull/3477
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bring the tested PDF progress and cancellation changes to public dev. Use the existing job status and cancellation APIs, retain self-hosted and extension behavior, and cover the worker and browser regressions. ADR assessment: no new ADR because this reuses the existing Jobs and Quick Ingest architecture; ADR-006 governs touched Python security checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PDF jobs report real extraction, analysis chunks, and save stages through existing job status.
- [x] #2 Cancel reconciles the server response and prevents remaining PDF work before persistence.
- [x] #3 Public dev regression tests pass and a PR records the staging benchmark and its limitations.
- [ ] #4 Required checks pass after the dev rebase and PR regression fixes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Port against public dev 94a566393c. Reused existing Jobs and direct WebUI tracking. Preserved current worker shutdown and detached embedding credential context. Focused tests passed: 52 backend and 452 shared frontend. Corresponding staging run used a 12,695,915-byte, 44-page PDF. Browser cancellation during analysis switched to Cancelled in 0.66 s and saved no media. Median click-to-saved-result was 21.8 s without analysis and 156.9 s with DeepSeek Flash across two sequential runs per mode. One analysis repeat hit the provider output limit on five chunks. No new ADR. Existing Jobs architecture and ADR-006 apply.
Public dev PR opened. Compared Bandit scans of touched backend files against the public dev baseline. No new findings and one inherited warning. CI and human review follow the normal public PR workflow.
Rebased PR #3477 onto current dev and resolved the task ID collision with the chat-history task.
CI follow-up preserves cancellation save admission and PDF progress while keeping public call signatures and function-size ceilings. The SQLite cancellation race hook and shifted Jobs baseline were corrected. The Playground guard passes after rebasing. The audio URL test reproduces on clean dev with the CI allowlist.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added real PDF progress and server-acknowledged cancellation for tracked remote jobs. Preserved the public worker's shutdown watcher and detached embedding credentials boundary. Added focused worker and shared UI regression coverage. Public PR #3477 targets dev.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
