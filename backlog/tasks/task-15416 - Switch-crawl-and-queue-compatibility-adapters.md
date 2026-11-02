---
id: TASK-15416
title: Switch crawl and queue compatibility adapters
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 9 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15407. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 9 behavior and interfaces match the approved spec and plan.
- [ ] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 9. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Tasks1-8 independently accepted. Begin Task9 one-time legacy/enhanced adapter cutover. Mandatory preflight: preserve explicit parent_scraper/_execute_job callback and submitter context (baseline-notes.md); verify synchronous interrupted crawl retains actual-work and checkpoint ownership beyond logical return then releases after physical exit. Do not use bare asyncio.run as drain proof. Preserve trusted-local resume paths, signatures/exports/progress snapshots, filters/cookies/dedup/extra fields and established transport failure lists. Resolve wrapper size debt by decomposition/cutover, not ceiling bump. No service/central Jobs migration or dual execution.
Task9 preflight at BASE 1080d0f114: NEEDS_CONTEXT before production/test edits. Canonical orchestration/crawl.py execute_crawl always constructs article.start_page and has no live per-page result transform. Enhanced extraction currently calls _apply_dedup per page; converting/deduplicating only after CrawlReport would leave canonical partial-success accounting and checkpoint articles inconsistent. Requested controller ruling for smallest private nonserialized article-result transform applied before engine collection, preserving WorkHandle cancellation/drain and no lower-to-owner imports. Full inspection/status report: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-9-report.md. Status remains In Progress; no implementation, tests or commits yet.
Resumed under regenerated Task9 execution ruling: private post-canonical transform through execute_crawl/start_page authorized. TDD: initial adapter tests 5 failed (missing transform/conversion and queue export identities), then 5 passed; queue shim ownership/limiter tests 2 failed and 5 passed before cutover, then adapter plus existing call-context scope 9 passed (30 warnings). Implementing canonical adapter composition next; retain In Progress pending independent gate. No plan/ledger edits.
Task9 additional NEEDS_CONTEXT: enhanced HTTP acquisition calls CookieManager.get_cookies(url) by exact netloc; canonical execute_crawl accepts only one custom_cookies list. Requested private trusted cookie-manager snapshot in existing credentials, selected per target before explicit/route cookie precedence; seed-only flattening risks dropping child-host cookies or broadening credential transmission. Paused before broadening. Partial code: approved postcanonical transform, to_legacy_crawl, canonical queue shim (parent/replaced callback, rate wait), legacy adapters and drain-aware sync bridge. Sync held-actual-work test RED checkpoint_in_use after release; GREEN passed with lock retained before release and available afterward. Enhanced crawl/batch cutover and fixture migration not complete; no commits. Current scoped compile and Bandit pass; Ruff/format pending cleanup; global size ratchet 1 failed/12 passed with wrapper Task9 debt plus known later-task HTTP/service failures. Details and exact commands in task-9-report.md.
Second ruling read and capability audit completed: rate waits remain in accepted private queue overrides; batch method/parent/replaced callback path retained; canonical routing/policy/header preparation; exact-netloc cookie snapshot selected before each admitted page/discovery request; approved postcanonical dedup transform; progress snapshots remain separate from canonical report finalization. No further interface expansion identified. Cookie tests RED 2 failed then GREEN 2 passed. Enhanced parent-stop/partial-progress tests RED 2 failed then GREEN 2 passed. Legacy held-work sync writer-release regression remains green. Migrating old characterization transport/extraction boundaries to canonical dependencies; remaining quality and full covering run pending.
Cutover complete in working tree; final covering/quality evidence pending. Added repeated interrupted-sync proof (3 simultaneous held actual-work invocations): writer locks remain exclusive before release; after release bridge/default executor threads retire, loops close, no pending tasks or retained callback/secret graphs. No new extraction executor lifecycle. Native instance stop now waits canonical queue-owned tasks before closing its browser. Broad run found cutover-boundary migrations and legacy history/transport projection preservation; fixed via canonical history shim and placeholder-free legacy transport failure conversion, preserving meaningful extras. Stable recursive telemetry moved to canonical composition; headers now retain explicit-agent > custom-UA > route-UA article precedence with normalized job byte charging. Task9 wrapper640>638 removed and its stale baseline entry deleted without raises. Task remains In Progress pending independent gate; no controller plan/ledger staged.
Final verification: required Task1/9 plus recursive/config/queue compatibility scope 153 passed, 317 warnings, no skips/deselections; final full Web_Scraping+WebScraping run 3452 passed, 28 skipped, 3 deferred failures (documented arXiv consumer denial assertions x2 and import-inventory artifact drift), 7513 warnings. All Task9 failures from the first covering run resolved. Ruff check/format (20 files), compileall, Bandit touched production (0 findings/errors/nosec/skips), and diff-check pass. Touched-production max function 300 AST lines; obsolete recursive-wrapper baseline removed. Global size run 12 passed/1 failed only four shared HTTP and one service ceilings assigned later tasks; untouched versus BASE1080d0f114. Full report task-9-report.md records TDD, capability/lifetime audit, changed paths, warnings/skips, and rollback caveat. Retain In Progress pending independent gate. Standalone commit prepared; no plan/ledger changes.
Committed rollback boundary 8d0e2cff46 with configured pre-commit checks passing (invocation-local hook; no shared hook installation or dependency installs). Final self-review found native _fetch_html curl offload still used unobserved asyncio.to_thread: held-work RED proved live/payload lease dropped before thread exit. Changed only that offload to existing owned_to_thread and added native callback regression; focused curl/header/module covering tests and Bandit/compile/Ruff rechecked. Follow-up commit retains lease through actual exit. Main full covering evidence remains 3452 passed/3 deferred failures/28 skipped; all Task9 compatibility tests green. Keep In Progress for independent gate; full command/output details in task-9-report.md.
Review fix round 1/3: Important Findings I1-I4 verified. Controller requested independent I1/I3/I4 fixes with focused TDD and separate commits while I2 browser per-dispatch interception/proxy-attestation binding awaits exact ruling. Task remains In Progress; no claim all findings closed. Original fix review BASE f7f4e449e0.
I1 finite sync ownership implemented with default separate capacity4, immediate queue_overloaded rejection before thread/loop dispatch, lease through loop/default-executor retirement, validated idle-only capacity seam and PID-safe inherited-lease handling. RED 4 failed/30 deselected; GREEN focused lifetime/startup/PID 10 passed/24 deselected; covering adapter+legacy sync+environment 41 passed/95 warnings. Scoped Ruff/format/compile/Bandit and diff-check pass, production AST max110. I2 ruling now received; all findings not yet closed. Remain In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
