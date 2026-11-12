---
id: TASK-13819
title: Benchmark workflow SQLite cache safety and improve WebSocket diagnostics
status: Done
labels:
- backend
- workflows
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean up retired task-owned worktrees, measure the merged reusable SQLite cache-disable change against a controlled cached baseline on latest dev, improve safe workflow WebSocket failure diagnostics, independently review a new PR and merge after all checks and human rationale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retired task worktrees are archived with needed local evidence preserved
- [x] #2 Reproducible real-database benchmark reports latency throughput and correctness with environment and methodology
- [x] #3 WebSocket failures expose stable safe reason and close-code diagnostics with regression coverage
- [x] #4 Connected tests security checks independent exact-head review and human Change summary complete before merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline origin/dev696d7fdeced92f23b72ae416aeddc6ea9fd5f945 contains merged PR3252. Reuse clean attached shared-ui-coverage worktree. Primary checkout and other sessions remain untouched. Cleanup limited to task-attached retired chat audit/remediation worktrees and owned generated artifacts; preserve ignored runtime data/keys privately before managed archival. ADR required:no, anticipated benchmark and local observability change preserve storage/auth policy; ADR059 governs task editing. Stages:1 cleanup and baseline;2 benchmark and safe diagnostics regressions;3 validation review and integration. Existing user authorization covers this bounded follow-up; no new storage/concurrency design planned.
Cleanup complete: archived both retired attached chat worktrees using managed recovery; preserved ignored DB/key/audit evidence privately at /private/tmp/ws13819-cleanup-recovery; stopped task-owned Python server59482 on19767 after cwd verification; removed63 untracked Next build directories. Benchmark smoke4/4 passes. Found deterministic resource bug: after borrowing all3 pooled connections, returning them leaves pool empty; regression fails on latestdev. Repair only the empty-list return guard; compare both cache settings with identical pool behavior. Diagnostics red6/6 demonstrate absent reasons/logs and raw exception-text logging; separate preaccept lookup regression added. No architecture/policy change.
Focused final13/13 pass with exit0. Native macOS initial post-test exit134 traced through DiagnosticReports to onnxruntime_pybind11_state.so Microsoft Events telemetry background thread; benchmark-only6/6 exits0. Private local pytest plugin now calls existing ONNX telemetry opt-out; no app production policy changed. Retired worktree archive tool requests returned queued and managed attachment payloads changed, but checkout directories remain; verify/reconcile archival before final cleanup acceptance. Benchmark cached controls also reproduced malformed-row IndexError; explicit error accounting regression now passes. Expected auth denials retain WARNING; internal lookup/stream errors use ERROR. No token/run/exception text enters new endpoint logs.
Cleanup verified complete: native archive requests stayed queued with checkout directories intact, so fallback preserved full non-ignored state in refs/codex/snapshots and removed only the two inspected retired worktrees with ordinary git worktree remove (no force). Ignored runtime evidence remains in private recovery tarballs; manifest records snapshot hashes and old heads. No primary checkout, shared venv, other sessions or shared node_modules target changed. Connected172/172 tests pass and process exits0; source-only formatting follows. Benchmark timings will be collected after test processes exit.
Completed paired macOS/Python 3.12.13/SQLite 3.50.4 and Linux arm64 container/Python 3.12.14/SQLite 3.46.1 runs, each 476,000 operations per cache setting; uncached all correct and all pools retained. Serial costs about 2–5us macOS and 6–15us Linux; concurrent cached controls invalid on both. Raw trial JSON and method/limits report in Docs/Performance. Initial independent code review found no actionable issues; final report/exact-head follow-up pending.
Final report/raw JSON independently recomputed and reviewed with no actionable findings. Final scoped Ruff, Black for new files, scope-predicate exact baseline and production Bandit pass (zero findings/errors). Test Bandit excludes assertion B101 and narrowly documents fixed-local-CLI subprocess and synthetic leak-detection credential sentinels; zero remaining findings. Final focused suite rerun pending exit. New PR and requester Change summary remain required merge gates.
Final focused24/24 pass with exit0 (including benchmark6 and full extras18); connected172/172 also exit0. Pre-commit all selected files passes; production Bandit zero findings/errors, test scan zero remaining after assertions exclusion and narrow fixed-local CLI/synthetic sentinel annotations. Report and safe diagnostics guide independently reviewed without findings. Integration stage remains open pending new PR, human-written rationale, exact-head hosted checks, latest-dev merge.
PR https://github.com/rmusser01/tldw_server/pull/3254 targets dev. Independent exact code/report head54799cb9e6bfeab53dd653fa6169638318eea890 approved with no findings; all production fingerprints match benchmark reports. Requester supplied the PR-specific human Change summary directly in this task; copied verbatim into PR body. All local deliverables/acceptance checks complete. Hosted checks and latest-dev serial merge remain operational gates; do not merge before all pass. The implementation plan is retired after local delivery; integration progress is tracked in PR3254.
Operational integration complete: PR3254 merged into dev at0566130cbed2670efca6bbfa3c309031d2d3bc3b after262 successful checks,34 expected skips and1 neutral, all7 required statuses green. Reviewed head d436d8a0b299dac1b071b9ce8ccf2ca8801a0da2 is ancestor of remote dev; merged DB/tool hashes match benchmark report. Closure recorded during the immediately following TASK13820 benchmark work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered reproducible paired macOS/Linux SQLite cache benchmark and raw reports, repaired borrowed connection return after pool exhaustion, and added safe fixed WebSocket failure diagnostics with operator guidance. Both platforms completed476000 correct uncached operations per setting; concurrent cached controls invalid. Focused24 and connected172 tests exit0; scoped lint, security, predicate baseline and precommit pass. Independent exact-code/report review found no issues. Requester-owned PR3254 rationale recorded; hosted CI and merge must still complete before operational closure.
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
