---
id: TASK-13254
title: Distinguish offline fixtures from live Email M2 lag evidence
status: Done
assignee: []
created_date: '2026-09-13 19:04'
updated_date: '2026-09-13 19:08'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct misleading staging SLO success claims emitted by the Email M2 metrics checker for offline Prometheus fixtures. Preserve thresholds, metrics and exit statuses while exposing evidence source in text and JSON.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Offline snapshot and delta reports identify fixture evidence and make no staging validation claim
- [x] #2 Live endpoint snapshot and sampled-window evidence remain distinguishable
- [x] #3 Real CLI regression tests cover passing and failing thresholds without network or Gmail access
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Trace input modes and reproduce false staging claim; write failing CLI tests using synthetic snapshots and mocked transport only for live-path coverage; minimally label evidence source in text and JSON; run focused tests, Ruff and Bandit, record verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed Helper_Scripts/checks/email_m2_gate_validation.py and new tldw_Server_API/tests/Helper_Scripts/test_email_m2_gate_validation.py. Bounded correction with module documentation; no broader runbook edits. TDD red: seven tests failed against original checker (unconditional staging claim and missing evidence fields). Green: 13 passed, 4 existing environment warnings, combining seven new cases with six dual-read parity tests. Exact float fixture expectation was corrected to pytest.approx for 56.99999999999999 p95. Offline tests invoke real subprocess CLI; live branches exercise main with only external fetch and sleep replaced, no sockets/Gmail. Ruff test file clean; checker has exactly 33 baseline findings and 33 current findings, no additions (verified against git HEAD). Bandit production alone clean; combined production/tests clean with B101 excluded for pytest assertions, narrowly justified subprocess nosec annotations. Reports: /tmp/bandit_email_m2_checker.json and /tmp/bandit_email_m2_all.json. git diff --check clean. Verification command: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Helper_Scripts/test_email_m2_gate_validation.py tldw_Server_API/tests/Helper_Scripts/test_email_search_dual_read_parity.py -q --tb=short --basetemp=/tmp/email-m2-verification. No commits or staging; coordinating parent integrates. Read-only benchmark assessment sent to parent: serial per-message writes, seeded RNG but wall-clock dates, no true ingestion throughput measurement, no benchmark run performed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Email M2 reports now include evidence_source (offline_fixture or live_endpoint) and evidence_note in text/JSON alongside existing snapshot/delta mode. Fixture success only attests threshold checks and explicitly leaves staging evidence unverified. Live endpoint reports require deployment provenance confirmation. Thresholds, useful metrics, passed boolean and process exit status retain their original meaning. Seven focused regressions and six adjacent tests pass; no new lint or security findings.
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
