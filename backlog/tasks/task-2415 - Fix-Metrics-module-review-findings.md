---
id: TASK-2415
title: Fix Metrics module review findings
status: Done
assignee: []
created_date: '2026-06-23 18:19'
updated_date: '2026-06-24 03:47'
labels:
  - metrics
  - security
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the Metrics module review findings from 2026-06-23. Scope: remove raw user_id labels from exported metrics, keep latest gauge series independent of sample buffer eviction, make duplicate metric registration type-safe/idempotent, reject negative counter increments, and remove the unused logger_config module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Public metrics exports do not expose raw user_id label names or values.
- [x] #2 Gauge exports preserve latest values for active label sets even when the rolling sample buffer is small.
- [x] #3 Duplicate metric registration is idempotent for compatible definitions and rejected for incompatible definitions without corrupting later recordings.
- [x] #4 Counters cannot be decremented through negative increments.
- [x] #5 Unused logger_config Metrics helper is removed or decommissioned.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-23-metrics-module-review-fixes-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started implementation. Added focused plan for Metrics review findings.

Implemented Metrics registry fixes: user_id labels normalize to hashed user_hash values, gauge latest values are retained outside the rolling sample buffer, duplicate metric registration is compatibility-aware, typed helpers reject incompatible metric operations, negative counter increments are rejected, and unused logger_config references were removed. Verification: source .venv/bin/activate && python -m pytest -q --confcutdir=tldw_Server_API/tests/Metrics tldw_Server_API/tests/Metrics/test_metrics_label_normalization.py tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py tldw_Server_API/tests/Metrics/test_audio_stt_metrics.py -> 25 passed, 2 warnings. Bandit: python -m bandit -r tldw_Server_API/app/core/Metrics -f json -o /tmp/bandit_metrics_module_fixes.json -> 0 results. Full repository test suite was not run; scope was the Metrics module review findings and unrelated repo-wide pytest setup imports heavy Research/RAG dependencies.

PR review follow-up on 2026-06-24: rebased branch onto origin/dev, added docstrings and return annotations to new Metrics tests, rejected duplicate registrations with conflicting descriptions, and made label-name aggregation helpers treat user_id as the user_hash alias. Verification after review fixes: Metrics pytest focused suite -> 28 passed, 2 warnings; Bandit on tldw_Server_API/app/core/Metrics -> 0 results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Metrics module review findings. Public metric label normalization now hashes user identifiers into user_hash, gauge exports use durable latest-value state instead of the rolling sample buffer, duplicate registration is idempotent only for compatible definitions, incompatible typed helper/bridge recordings are rejected, counters reject negative increments, and the unused logger_config module plus stale references were removed. Focused Metrics tests pass and Bandit found no issues in the touched Metrics code.

PR review comments addressed: new tests now include docstrings/return annotations, duplicate metric compatibility includes description, and label aggregation lookup helpers apply the user_id -> user_hash alias.
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
