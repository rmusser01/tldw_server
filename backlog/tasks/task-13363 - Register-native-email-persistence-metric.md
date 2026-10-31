---
id: TASK-13363
title: Register native email persistence metric
status: Done
assignee: []
created_date: '2026-09-25 18:56'
updated_date: '2026-09-25 19:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The full live SQLite email upload succeeds but metrics_manager logs Metric email_native_persist_total not registered. Add the missing metric definition with the labels used by persistence, test recording, and verify no missing-metric warning in the synthetic upload path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 email_native_persist_total is registered with the emitted labels and can record a successful persistence event
- [x] #2 Focused regression tests and live synthetic upload do not produce an unregistered-metric warning
- [x] #3 Metrics documentation and release evidence reflect the validated behavior
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: persistence emits email_native_persist_total with path_kind/outcome but the built-in MetricsRegistry ingestion definitions omit it. Regression test will call the real persistence helper and inspect the real registry sample.

TDD red: missing registry sample caused KeyError sum. Added counter definition with path_kind/outcome, then 8 focused ingestion/metrics tests passed. Full Uvicorn SQLite synthetic upload passed again with zero outbound/model attempts and no unregistered-metric warning. Ruff clean; Bandit 0 findings/0 errors; git diff --check clean. An unrelated first rerun imported another checkout via inherited PYTHONPATH and failed before server startup; rerunning with explicit worktree PYTHONPATH resolved it. Logs /tmp/email_metric_pytest_13363_final.log and /tmp/email_live_sqlite_probe_13363_metric.log.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Registered email_native_persist_total with emitted path_kind/outcome labels. Added a behavior regression test proving persistence events are recorded. The live synthetic SQLite upload now runs without the missing-metric warning; release evidence updated.
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
