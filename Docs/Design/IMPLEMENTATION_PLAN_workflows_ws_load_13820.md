# Workflow WebSocket load benchmark — TASK-13820

Baseline: origin/dev0566130cbed2670efca6bbfa3c309031d2d3bc3b.
ADR check: no new ADR for a benchmark or local responsiveness repair; preserve auth, owner checks, frames and caching policy. ADR-059 governs tracking.

## Stage 1: Baseline and measurements
**Goal**: Measure production ASGI WebSocket path over real loopback TCP with isolated data.
**Success Criteria**: Reproducible setup/snapshot/event percentiles and correctness at1/16/64 clients, runtime/source identity, lifecycle cleanup and clear limits.
**Tests**: Harness validation/smoke behavior; baseline private app run.
**Status**: Complete

## Stage 2: Evidence-driven repair
**Goal**: Repair only reproduced root causes and compare matched workloads. Follow-up: remove concurrent per-row SQLite transfers with one native JSON page, preserving the original row/decode path for legacy payloads, unavailable JSON functions and oversized aggregates. Replace per-frame bare yields with a loop-shared 1ms work budget and a timer checkpoint, retaining ordered individual frames. No global read admission limit or schema shortcut. Follow-up profiling identified repeated metric-label normalization per frame; remove only the redundant second normalization inside MetricsRegistry.record, retaining first-pass validation and raw-label lookup/export normalization. Add normalization-once and stored-label regression coverage plus existing collision/privacy/bridge/alias verification. Disposable workload ABBA improved replay/P99 medians; final full-matrix evidence remains required.

**Scope and ADR assessment**: Shared get_events callers include REST, webhooks, investigations, voice workflows and WebSockets; PostgreSQL remains on its current path. Compatibility tests cover stored payload types, filters/limits/order, missing JSON1, oversized values and genuine database errors. Scheduler tests cover cumulative work, cancellation, separate loops and cleanup. These are local retrieval/scheduling implementation details under existing persistence/auth/frame rules, so no new ADR is needed; ADR-059 governs tracking. Performance must be established against a freshly matched baseline before this repair is considered complete.
**Success Criteria**: Red/green regressions preserve auth, ownership, event delivery and safe diagnostics; paired before/after results.
**Tests**: Connected workflow/stream/auth suites and benchmark correctness/error cases.
**Status**: Complete — measured candidate136070c23e passes310 connected tests, scoped lint/format/Bandit and independent code/data review. Full six-case interrupted/resumed ABBA has72 valid trials, exact delivery and zero errors in both variants. At64 clients/history500 replayP95 856.82→802.14ms and loopP99 6.34→2.65ms; both orders improve. Report retains setup/single-client costs, desktop activity and resumed gap. Earlier rejected prototypes remain excluded.

## Stage 3: Review and integration
**Goal**: Deliver a reviewed new PR against latest dev.
**Success Criteria**: Scoped lint/Bandit, independent exact-head review, requester Change summary and hosted checks before merge.
**Tests**: All hosted checks; latest-dev ancestry and merged source verification.
**Status**: In Progress

Integration rebase: all checks passed on ff733d7976, but dev advanced before merge. Rebased unchanged patches onto e1f7770953; independent review confirms all four measured source hashes and the lifespan-disabled benchmark path are unchanged. Retain the timing evidence; rerun connected tests and require fresh exact-head review/hosted checks before merge.

CI follow-up: isolate the chat refund test exception-state injection to the endpoint module binding, preserving refund/cleanup assertions and Python/framework exception handling. Verify failing-first with current CI framework versions, then adjacent runtime lifecycle tests and the simplified endpoint suite. No production change or benchmark rerun.

Latest integration: clean8-patch rebase onto dev7c94a2efb1 after upstream sandbox teardown/test-quality changes; exact independent review and four measured hashes unchanged. CI-version chat suite217 passed/1 skipped and9 lifecycle cases passed; new quality ratchet and fresh hosted checks remain integration gates.

Final freshness refresh:7411 hosted CI266success/36expectedskip and all7requiredsuccess; dev advanced to29fda0d860. Rebase preserves9patches and4measuredhashes; independent review confirms upstream changes outside measured path. Refresh private source and verify310connected+9chatcleanup plus quality/shardguards, final exactheadreview and fresh hosted checks before merge.

Freshness refresh after3978: all266 hosted checks and all7requiredstatuses passed, but dev advanced to b5b572c01d. Recovered disk-full partial checkout with rebase abort; preserved12 verified upstream-only remnants outside the worktree. Clean rebase retains all10patches and4measuredhashes. Independent review confirms upstream quota middleware bypasses WebSockets and audio/media/frontend changes are outside measured path. No benchmark rerun; refreshed connected/cleanup tests, quality/shard guards, final metadata review and fresh hosted checks remain integration gates.

Latest integration verification:309connected cases passed initially; the harness startup failure came from a stale tracked-source inventory in the disposable private fixture. Refreshed its inventory for55upstreamfiles, then all5harnesscontracts passed36.33s. No repository source change. Chat cleanup9passed23.79s; quality ratchet1passed22.51s; shardguard851patterns5123files0newuncovered. Final metadata review/push and fresh hosted checks remain.

Hosted CI follow-up: adminprovider recovery test reused its synthetic50ms timeout, making released-capacity verification depend on fast thread scheduling. Controlled60ms worker delay reproduces CIhealthy0failure; test-only restore1s health deadline before recovery preserves initial50ms/headroom/admission/cleanup assertions. Same controlledscenario green; fulladmin78passed, scopedquality/security checks and independent delta review clean. No runtime or benchmark change. Fresh hosted checks remain mandatory.

Audio CI follow-up: heartbeat sanitization fixture waited before upstream now-starts heartbeat; move its wait to later quota check, preserving injection/response/log assertions. Import-resilience tests restored module cache but leaked parent-package bindings; restore both exact states across all3tests and avoid later monkeypatchundo erasing restoredmodules. Seededapp-loaded order reproduces slotsetupfailure;4cache/binding regressions red then green, seeded8passed andbroaderaudio65passed. No production/harness/measurement change; independentdelta review and scoped lint/format/Bandit clean. Fresh hosted checks remain integration gate.

E2E CI follow-up: pytest9.1 log capture iterates a live logger registry and races background registration; deterministic probe reproduces teardown RuntimeError on9.1.1 and passes9.0.3. Temporarily bound dev pytest and e2e explicit install to>=9.0.3,<9.1, keeping capture enabled. No application/benchmark change or new architecture decision; connected verification and independent review remain required.

Pytest workaround verification: deterministic capture probe red9.1.1/green9.0.3; connected310passed133.05s and e2e contracts2passed5.82s under isolated9.0.3. TOML/YAML/specifier consistency/diff checks and direct quality guard pass. Independent actual-delta review clean. Bandit not applicable to dependency/YAML/docs-only delta; Python scope unchanged. Require fresh hosted e2e and all broader checks on pushed head before merge.

Freshness refresh after43aae: all266 hosted checks passed; dev advanced to a6719bf24f. Clean14patch rebase retains every patch and all4measuredhashes. Upstream Workflow execution/retry and chat/TTS/audio worker changes are outside the measured DB-row WebSocket replay path; independent scope review clean. Refreshed140upstreamfiles and private tracked inventory. No timing rerun; refreshed connected/retry/quota/chatcleanup/admin checks and final quality/review/hosted gates required.

Latest a671 verification:342connected cases passed131.89s, including32newWorkflow quota/retry cases; chat cancellation/refund/cleanup14passed7.44s, adminruntime34passed2.44s under pytest9.0.3. Quality guard and shard coverage852patterns5128files0newuncovered pass. Report provenance refreshed; all4measuredhashes unchanged. Metadata-only final delta; fresh exact-head review/push/hosted checks remain integration gates.

Safe-regex CI follow-up: exact million-character sample with a$ creates a million end-anchor candidates; controlledCPUcontention reproduces regex_timeout on real fastpath with unchanged100msbound. Test-data-only replace prefix with999999b+finala, preserving exactsize/realengine/match/span/securitybudget. Samecontrolledred→green and fullsafe-regex verification required; no runtime or timing change.

Safe-regex fixture verification: samecontrolledcontention redregex_timeout→green1passed0.73s; fullfile204passed2.60s. Ruff/touchedfunctionBlack/Bandit clean; fullfileformatdrift and13securityfixturefindings identicalbefore/after. Qualityguard passes; independentone-line review clean. Realengine/inputsize/match/span/100mslimit retained. No production/benchmark change; freshfinalheadreview/push/hostedchecks required.

Admin storage CI follow-up: stale numericFD assertion fails when closed descriptor is reused; deterministicclose→unrelatedopen reproduces. Test-only capture realstreams.closed/successfulrawclosecalls and scope injectedOS/tempfilehelpers to servicebindings; preserve all5failurepoints/oldJSON/tempcleanup. No productionchange; samecontrolledcase/fullshard/quality/security/review required.

Admin storage verification: deterministicreuse red1failed→green1passed0.60s; full63-casehostedselection passed12.34s. Independentdelta clean; Ruff/touchedfunctionBlack/Bandit0findings/errors andqualityguard pass. FullfileBlackdrift preexisting; no bulkformat. All5failurepoints/publication/tempcleanup/realclosure retained. No production/harness/timingchange; finalheadreview/push/freshCI remain.
