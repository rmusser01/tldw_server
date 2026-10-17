---
id: TASK-13820
title: Benchmark workflow WebSocket latency under concurrent load
status: In Progress
labels:
- backend
- workflows
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measure real-TCP workflow WebSocket setup, snapshot and persisted-event latency on latest dev under concurrent clients; repair demonstrated root causes, independently review, and merge a new PR after all checks and requester rationale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproducible real-network benchmark reports latency percentiles correctness failures and environment for concurrent clients
- [x] #2 Any demonstrated responsiveness or lifecycle bugs have failing-first regression coverage and root-cause repairs
- [x] #3 Raw baseline and fixed measurements and limits are documented without production credentials or data
- [ ] #4 Connected tests lint Bandit independent exact-head review requester summary and hosted checks pass before merge
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline origin/dev0566130cbed2670efca6bbfa3c309031d2d3bc3b includes merged PR3254. Reuse clean task-attached isolated shared-ui-coverage worktree on codex/workflows-ws-load-20261016; primary checkout untouched. Bounded scope: production app over loopback Uvicorn, real JWT auth, disposable DBs, clients1/16/64, setup/snapshot/event latencies plus correctness and event-loop responsiveness. No auth or statement-cache policy change. ADR required:no for benchmark and local responsiveness repair under existing storage/auth rules; assess again if evidence requires architectural changes. ADR059 governs task editing.
Harness red4 missing-tool tests, reviewed smoke5/5 exit0 (real TCP, zero/three history, poisoned test environment, parent-exit, args). Reproduced pending-read red2/2: get_run and get_events block unrelated HTTP on ASGI loop; original real DB reads retained. Initial pilot3trials/case,5rounds all correct; concurrent64 setup exposed >1s maximum loop stalls. Pilot is not final performance evidence. Independent review found inherited pytest flags and timeout-orphan risks; added observed server runtime facts, exclusive empty env, explicit false flags, expected-parent watcher and test process-group cleanup. New mode regression failed before observation; parent-exit regression timed out before fix; now smoke5/5 clean. Official native matched ABBA runs use30 rounds across multiple poll periods,6 cases at clients1/16/64 and history0/500. Linux runtime image lacks Git; do not add dependencies solely for timing; Linux real-TCP smoke/regressions run in hosted CI.
Official baseline-A completed valid with observed production test_mode=false and explicit_pytest_runtime=false, all auth controls403, no delivery errors. Applied standard Starlette run_in_threadpool only to workflows_ws get_run/get_events; both pending-read regressions now pass2/2 after baseline failed2/2. Read-only independent production review found no actionable issue; shared bounded threadpool may queue above40 readers. Connected workflow/API/auth/storage/harness tests running before fixed timings. Refetched origin/dev still0566130cbed2670efca6bbfa3c309031d2d3bc3b; MERGE_QUEUE unset.
First ABBA timing set delivered58320 live events but showed mixed setup regressions. Separate untimed watchdog stack sampling proved >1s loop stalls in Uvicorn ASGI exception logging/Loguru traceback local repr after normal peer departures: workflows_ws re-raised WebSocketDisconnect. Three failing-first endpoint tests1000/1001/1006 confirm undesired propagation. Repair returns after existing safe INFO diagnostic. Strengthen harness to record ERROR exception classes and reject unhandled server failures; earlier ABBA is exploratory, not final validated comparison. Recollect matched final measurements after both fixes; retain original data as diagnostic evidence.
Final connected suite182/182 passed exit0 in174.23s after both production fixes. Final scoped Ruff/Black clean; Bandit production+helper and tests excluding assertion B101:0 findings/errors. Production logging guard blocked initial Loguru observer (caught by baseline full-run negative control); replaced with standard non-suppressing uvicorn.error filter, preserving production sinks/formatters. Real baseline1client control exits1 validfalse with WebSocketDisconnect1 despite correct delivery; reviewer approved filter. Final frozen ABBA collection restarted with this verified observer. Do not use earlier observer-blocked or exploratory artifacts as final results.
Second matched ABBA: baselineA/B each486 WebSocketDisconnect ASGI errors (validfalse), repaired2-fix variants eachzero errors with exact14580 live events/run. High-history setup/drain still regressed, so separate repaired-source watchdog confirmed main loop in Uvicorn permessage_deflate compression during history send bursts; await send_json need not suspend. New real DB/JWT/Starlette immediate-transport fairness regression fails old handler (probe only executes after500events). Independent reviewer approved minimal per-event asyncio.sleep(0) after cursor advancement. Apply third repair without changing frames, page size or1s polling. Final matched order reuses preceding baselineB as leadingcontrol, then new fixedA/B and fresh baselineC; same6a05ffb8 harness/runtime/options, no overlapping heavy workloads. Earlier intermediate variants excluded from final claims.
Final same-harness/runtime ABBA raw results and report published in Docs/Performance/WORKFLOWS_WEBSOCKET_LOAD_2026_10_16_TASK_13820.md plus four workflows_websocket_load_13820_macos_*.json. Leading baseline reuses preceding same-source error-aware baselineB, then final fixedA/fixedB/freshbaseline; untimed diagnostics/tests between runs, never overlapping. Baseline972 spurious ASGI exceptions total; fixedzero. All58320 live+486000 history frames exact. Emptyhist64 mediantrialP95 handshake1766.50→232.29ms; hist500x64 replay942.34→3170.31ms, maxloop558.72→135.35ms but loopP99 3.56→72.24ms. Material replay/P99 costs prominent; no general latency/SLO claim. Independent reviewer recalculated every median/max/range/count/hash and recommends ship after final gates. Final code lint/Black clean; Bandit0findings/errors for production/helper andtests(B101assert excluded). Final183 connected rerun pending. Plan Docs/Design/IMPLEMENTATION_PLAN_workflows_ws_load_13820.md; no newADR under unchanged storage/auth/cache/frame policy; ADR059 governs tracking. Human Change summary and all hosted checks remain mandatory before merge.
Final connected183/183 passed exit0 in131.48s on the exact final route/harness source, after all matched timings. Native source identity assertions passed; no benchmark overlapped tests. Independent review has no actionable findings. Commit/PR preparation underway; requester summary and hosted integration checks pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Benchmarked the production ASGI workflow WebSocket path with real JWT/SQLite over loopback under1/16/64 clients and0/500-event histories. Fixed blocking database reads, normal disconnects escalating into ASGI errors, and history bursts monopolizing the loop. Added red-green coverage, isolated real-network harness with server-error validation, four source-fingerprinted raw reports and explicit replay/P99 cost disclosure. All183 connected tests and scoped lint/format/security checks pass; independent review verified code and every reported result. Merge remains conditional on requester-owned Change summary and all hosted checks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
