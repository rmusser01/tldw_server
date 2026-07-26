# TASK-12971 Subagent-Driven Development Progress

## 2026-07-14 — Plan and architecture review

- Completed prerequisite: TASK-12971; TASK-12968.2 remains blocked on TASK-12968.7.
- Worktree: `.worktrees/research-source-catalog-deep-research` on `codex/research-source-catalog-deep-research-design`.
- Approved design: `Docs/Design/2026-07-13-research-source-coverage-shared-discovery-design.md`.
- Implementation plan: `Docs/superpowers/plans/2026-07-14-connected-peer-verified-http-hop-implementation-plan.md`.
- Read-only architecture review rejected reuse of `afetch_json`, cached HTTPX/aiohttp clients, the separate certificate-pinning socket, and the synchronous MCP docs transport as the security boundary.
- Selected approach: HTTPcore 1.x public async streaming plus a validated-address network backend, one fresh HTTP/1.1 pool per hop, no redirects/retries/proxy/session state, and incremental bounded decompression.
- Adversarial plan review found that HTTPcore's internal per-event h11 limit did not bound aggregate informational responses. The plan now requires an independent raw-stream gate that cumulatively caps all 1xx/final header bytes and raw post-header wire bytes before HTTPcore receives them.
- The same review added whole-hop deadline/offloaded-DNS requirements, environment-independent certifi TLS context construction, request target/header-count/framing limits, peer-port verification, IPv6 Host coverage, and bounded decompressor finalization.
- Simplicity review removed redundant accepted-response counters, a redundant response-extension peer lookup, a real TLS certificate fixture, and multiple ambient-state harnesses. HTTPS stays covered with deterministic HTTPcore fake streams plus one real HTTP smoke test.
- Corrected a Backlog dependency cycle: TASK-12971 supplies the stable primitive/import/test paths; TASK-12968.2 proves later gateway consumption after this prerequisite completes.
- Stage 1 used a RED-first contract suite, then added the isolated request/DNS boundary without modifying legacy HTTP clients.

## 2026-07-14 — Stage 1 contracts, dependencies, and DNS policy

- RED: the focused test command stopped during collection because `tldw_Server_API.app.core.Security.http_hop` did not exist.
- GREEN: 115 tests passed across the hop contract, dependency-floor, egress, absent-env, and global-env suites.
- Static checks: Ruff, Black, compileall, and `git diff --check` passed on the touched Stage 1 scope.
- Added immutable bounded request/response/limit contracts, sanitized typed errors, strict ASCII IDNA and origin-target validation, and complete-set public-address validation.
- Added a public bounded raw DNS wrapper while retaining `_resolve_host_ips()` list compatibility for existing callers.
- Added direct `httpcore[asyncio]>=1.0.9,<2` and `certifi>=2024.2.2` dependency floors.
- Two adversarial reviews found and drove fixes for IPv6 site-local/scoped answers, malformed percent escapes/backslashes, hidden exception contexts, a potentially unbounded heartbeat wait, and a misplaced whole-hop-deadline stage criterion.
- Final independent re-review: clean, with no remaining actionable Stage 1 findings.

## Stage ledger

| Stage | Status | RED evidence | GREEN evidence | Review |
| --- | --- | --- | --- | --- |
| 1. Contracts/dependency/DNS | Complete | Import/collection failure: missing `Security.http_hop` | 115 focused/adjacent tests passed; static checks passed | Two review/fix rounds; final clean verdict |
| 2. Validated peer transport | Complete | 32/32 failed: missing `_execute_http_hop` | 127 contract/transport tests and 71 compatibility tests passed; static checks passed | Two review/fix rounds; final clean verdicts |
| 3. Bounded response streaming | Complete | Initial suite: 27 failed, 10 passed | 179 contract/transport/streaming and 259 focused/compatibility tests passed; static/security checks passed | Security/spec/simplification reviews clean after fixes |
| 4. Isolation/concurrency/finalization | Complete | Standalone logging regression failed with `httpcore` at `NOTSET`; strengthened child-logger/strict-level case also failed the first guard | 263 focused/compatibility tests passed; static/security checks passed | Initial test review fixed four findings; final security and simplification re-reviews clean |

## 2026-07-14 — Stage 2 validated-peer transport

- RED: all 32 initial deterministic HTTPcore fake-backend cases failed because `_execute_http_hop` did not exist.
- GREEN: 127 contract/transport tests passed after additions from self-review and independent review; 71 legacy MCP docs-fetcher, HTTP-client, and egress compatibility tests also passed.
- Added a one-use pinned backend that dials only the first validated address, verifies exact peer IP and port before writes, and blocks HTTPcore's independent connection-reassignment path from creating a second dial.
- Preserved the route host in HTTPcore's origin for SNI and emitted exact DNS/IPv4/bracketed-IPv6 Host authorities on default and alternate ports.
- Added a fresh certifi-only TLS context, required the approved SNI and cached SSL-object evidence, reverified the post-TLS peer, forced HTTP/1.1, rejected `101` upgrades, and kept redirects as ordinary responses.
- Added deterministic cleanup/cancellation/timeout coverage for TCP, TLS, and response reads; transport errors remain sanitized and retry-free.
- Review fixes added controlled `Accept-Encoding: identity`, TLS timeout mapping, SSL evidence validation, raw-HTTP SSL metadata masking, expanded/scoped peer cases, explicit request framing, and a one-use dial guard.
- Stage 3 intentionally owns authoritative aggregate header/raw-wire/decompressed/parser counters and the public one-argument entrypoint; Stage 2's execution seam remains module-private until those invariants are installed.
- Final independent spec and security re-reviews: clean, with no remaining actionable Stage 2 findings.

## 2026-07-14 — Stage 3 bounded response streaming

- RED: the initial 37-case streaming suite produced 27 failures and 10 passes before authoritative response limits and the public entrypoint existed. Later review regressions were also demonstrated red before each fix.
- GREEN: 179 contract/transport/streaming tests and the complete 259-test focused/legacy compatibility set passed after the final review fixes. Black, focused Ruff, compileall, Python 3.10 grammar parsing, Bandit, and `git diff --check` passed on the Stage 3 scope.
- Added an independent plaintext response guard ahead of h11 that cumulatively bounds every informational/final header block and all post-header wire bytes, including chunk framing and trailers.
- Added encoded `Content-Length` preflight, strict framing/status checks, identity/gzip/zlib-deflate streaming, bounded decompressor input/final drain, decompressed/parser ceilings, stable sanitized transport/protocol/timeout errors, and a whole-hop deadline covering DNS through response close.
- Published the exact one-argument `request_http_hop(request)` API while retaining only a module-private deterministic resolver/backend seam for tests.
- Raised HTTPcore's central logging floor to INFO because its DEBUG paths emit complete response headers and raw protocol exception bytes; regression coverage proves successful `Set-Cookie` values and malformed-body secrets are not traced.
- Review-driven fixes covered unsafe `Decompress.flush()` assumptions, `Transfer-Encoding` plus `Content-Length`, out-of-range statuses, `204`/`205` framing, HTTP/1.0 transfer encoding, HTTPcore wire-log leakage, `HEAD` precedence and coalesced trailing bytes for chunked `205` responses, deterministic decoder fixtures, actual stream-close assertions, and an optimized-Python-safe TLS invariant.
- Full-file Ruff on legacy `app/main.py` still reports 10 unrelated baseline findings; the six-line logger-floor change is regression-tested and compile-checked, and the focused Stage 3 files are Ruff-clean.

## 2026-07-14 — Stage 4 isolation, concurrency, and integration

- Added a real loopback `asyncio.start_server()` smoke test through the one-argument public API. It resolves the approved DNS hostname to the test-only loopback address, proves the actual dial, Host header, explicit Authorization, connected-peer evidence, and returned `302`, and observes exactly one request with no redirect follow.
- Added one combined hostile-environment HTTPS test covering proxy variables, `NO_PROXY`, temporary HOME/`.netrc`, ambient auth/cookie values, CA/keylog/client-certificate variables, certifi trust roots, explicit route Authorization, direct validated-IP dialing, and preserved route SNI.
- Added a barrier-coordinated concurrent success/failure test. An oversized second response closes with typed `response_too_large` while the first request retains its own stream, credential, three-byte wire counter, result, and cleanup.
- The three isolation/smoke cases initially passed against the completed Stage 1-3 architecture. A separate standalone logging regression failed red because the public primitive depended on `app.main` to raise HTTPcore's logging floor. The first one-line guard then failed a strengthened regression because it lowered a stricter operator level and did not override an explicitly DEBUG `httpcore.http11` child logger. The final guard raises only insecure effective levels and leaves WARNING/ERROR policies intact.
- GREEN: the complete focused and legacy compatibility suite collected 263 tests and passed all 263. Ruff, Black, compileall, Python 3.10 grammar parsing, Python 3.12 AST parsing, Bandit (zero findings across 1,513 production LOC), and `git diff --check` passed.
- Runtime matrix limits: no `python3.10` executable is installed locally. Python 3.12.11 is installed, but its environment lacks pytest/httpcore, so the focused suite ran only under the project Python 3.11.13 virtual environment. No dependencies were installed solely for this compatibility check.
- Final adversarial reviews found no remaining security, correctness, handoff, or simplification issues after the logger-level, SNI, concurrent-failure, task-cleanup, test-marker, and OpenAlex prerequisite corrections.

## 2026-07-14 — TASK-12968.7 OpenAlex credential evidence hardening

- Task 1 complete (implementation range `7a09dcf186..9a19be77dc`; independent spec/quality review clean with no Critical, Important, or Minor findings).
- RED: the authoritative Node inventory suite reported 17 passed / 1 failed because the frozen OpenAlex reason did not state that a free API key is still a credential.
- GREEN: the OpenAlex row now records substantive authentication, API-overview, and dated 2026-02-24 pricing evidence while remaining `credentialed_out_of_scope` with an `api_key` route. Derived rows/ledger digests changed; all other frozen counts and digests remained stable.
- TASK-12968.2's contract is explicitly V2-only: OpenAlex is typed unavailable/skipped with zero executable attempts, physical reservations, or gateway calls. V1 remains unchanged, and no secret-reference interface or credentialed branch was added.
- Fresh controller verification: Node inventory suite 18/18; schema plus legacy-selection pytest 22/22; authoritative validator `errors=[]`, exact report equality, 191 mapped / 35 credentialed; `git diff --check` passed. Five pytest warnings are existing environment/config warnings.
- Bandit not applicable because no production Python code changed. No live OpenAlex request was made; authenticated enablement remains deferred.
- TASK-12968.7 finalized Done with 5/5 acceptance criteria and 6/6 DoD items checked; final-summary markers normalized under the approved narrow repair.

## 2026-07-14 — TASK-12968.2 discovery execution foundation

- TASK-12968.2 moved to In Progress after confirming TASK-12968.1, TASK-12971, and TASK-12968.7 are Done; the official plan is linked from Backlog.
- Execution remains isolated to `.worktrees/research-source-catalog-deep-research`; the dirty root worktree is untouched.
- Corrected the stale OpenAlex note: V2 is typed unavailable/skipped with zero attempts, allowances, reservations, or gateway calls; no secret-reference interface or positive credentialed branch is in scope; V1 stays unchanged.
- Preflight review froze actual V1 behavior for Task 1: default selection is the three open-research-graph sources, all-eight execution requires explicit selection, aggregation follows catalog priority, and malformed direct-adapter output fails the whole source.
- Architecture review separated planner allowances from runtime reservations/debits, made the executor the sole gateway dispatch owner, bound dispatch capabilities to one planned attempt/route/policy/allowance, added pre-dispatch and pre-commit revocation checks, and expanded static boundary coverage to all V2 modules.
- Split the original oversized adapter stage into executor/journal, five JSON adapters, arXiv, PubMed, and registry/network-boundary tasks. Task 1 is In Progress; no runtime code has been edited yet.
- Task 1 RED: the focused execution-contract suite failed exactly because `research-discovery-legacy-execution-v1.json` was absent before the fixture and broker identity assertions were added.
- Task 1 now characterizes the real default catalog → `ResearchSourceRouter` → `ResearchDiscoveryService.search` path with all eight recording adapters, an injected no-I/O OA resolver, a default-resolver tripwire, and owner-scoped temporary `ResearchSessionsDB` factories; V1 production code remains unchanged.
- Frozen coverage includes omitted/empty/explicit/category/union selection, priority ordering despite reverse completion, provider arguments/counts, ordered statuses/warnings, partial and all failure, valid empty, whole-source malformed-payload failure, every stable public-schema field, the equivalent persisted snapshot/request/effective configuration, and hard-coded broker source/evidence identities.
- Independent-review hardening was demonstrated RED against the old partial golden, then added exact discovery-ID/snapshot equality, timezone-aware snapshot retention and broker retrieval bounds, and 10-second event guards with search-task cleanup. The official Backlog CLI consolidated Task 1 notes; its repeated empty Final Summary markers received the approved narrow marker-only repair.
- Task 1 GREEN after review hardening: focused suite 16/16; full catalog/router/service/selection/identity/broker matrix 127/127; compileall, Ruff, Black, and `git diff --check` passed. Five warnings are existing environment/deprecation warnings. Tests-only Bandit (B101 excluded because assertions are the test contract) reported zero findings and zero errors.
- Task 1 / Stage 1 is Complete; later V2 tasks remain Not Started.
- Independent Task 1 re-review of `c845c0a9af..803b6d199a` is CLEAN with no Critical, Important, or Minor findings. Task 2 / Stage 2 is now In Progress.
- Task 2 RED began with three expected collection errors for the absent V2 modules. Review-driven regressions later demonstrated attribution-only coalescing duplication, shallow tuple validation, incomplete page/redirect/retry dispatch accounting, nonzero result allowance for skipped-only plans, eager package import I/O, missing legacy lazy-submodule attributes, and zero-page attempt undercounting before each fix.
- Task 2 GREEN: 57 focused contract/registry/planner tests, the exact 97-test plan matrix, 107 impacted package/import/endpoint tests, and the complete 574-test Research suite passed. The complete Research suite preceded the final lazy-submodule edge hardening; all subsequent focused, exact, and impacted suites passed after it.
- Task 2 static/security gates passed: compileall, Ruff, Black, Python 3.10 AST parsing, and diff hygiene were clean; Bandit reported zero findings and zero errors across 1,487 production lines.
- Two independent review rounds drove fail-closed coalescing, deep immutable typing, complete physical budgets, true normal-import purity with V1-compatible lazy exports, and final legacy-submodule/zero-page hardening. The same reviewer returned CLEAN after the final fixes. Task 2 / Stage 2 is Complete.
- External controller review of `803b6d199a..c5f45ffff3` reopened Task 2 / Stage 2 as In Progress for predicate canonicalization, deeper frozen-plan integrity, separate logical-attempt and physical dispatch-group identity, adapter metadata, truthful global returned-result allowance, transitive lazy-submodule compatibility, and stronger static/read-I/O purity regressions. Overall TASK-12968.2 remains In Progress.
- Task 2 external-review fixes separate stable per-target logical attempt IDs from target-independent physical dispatch-group IDs, freeze adapter identity/version and route limits, derive aggregate allowances at the plan boundary, and canonicalize effective predicate values before equality, hashing, serialization, or matching.
- External-review fix GREEN on the settled diff: 64/64 focused contract/planner tests, 115/115 exact plan tests, 200/200 impacted package/jobs/endpoint tests, and 594/594 complete Research tests passed. Compileall, Ruff, Black, Python 3.10 AST parsing, and diff hygiene passed; Bandit found zero issues/errors across 1,161 touched production lines.
- Internal read-only review closed static no-I/O alias escapes and corrected Task 1/Task 2 plan status tracking. Task 2 / Stage 2 and overall TASK-12968.2 remain In Progress pending external controller re-review.
- The controller re-review left one Minor test-boundary gap: the pure-module scanner did not explicitly name the delivered `Security.http_hop` gateway facade. The three synthetic import forms failed 3/3 before the scanner rule was added, then passed together with the existing static and subprocess purity guards.
- Minor-fix verification is clean: 78/78 contracts/planner/registry tests passed; compileall, Ruff, Black, and `git diff --check` passed. Bandit is not applicable because this round changes tests and tracking records only. Task 2 / Stage 2 remains In Progress pending controller re-review.
- Final external Task 2 re-review is CLEAN with no Critical, Important, or Minor findings. Fresh controller verification passed 118/118 exact Task 2 plus legacy identity/broker tests; fresh Bandit reported zero findings/errors across 1,561 production LOC; diff/worktree hygiene passed. Backlog AC #2 is complete, and Task 2 / Stage 2 is Complete.
- Task 3 RED: the new gateway suite stopped during collection because `Research.discovery.gateway` did not exist. TASK-12971's contract, transport, and streaming preflight passed 183/183 before production code was added.
- Task 3 added a credential-free `dispatch_once(...)` facade that recomputes and binds canonical route policy, validates exact intent shape and limits, performs a final fail-closed active-policy check, constructs one normalized empty-header/body request, and awaits the injected/public one-hop primitive exactly once.
- Redirects remain typed single-hop responses; cancellation propagates; bounded bodies and allowlisted `content-type` metadata are returned with public/derived resolved-address, connected-peer, header-byte, wire-byte, decoded-byte, Host/SNI, ceiling, and elapsed evidence. Stable errors discard query, body, path, secret, and provider exception detail.
- Task 3 self-review RED exposed missing full public hop ceilings and a malformed injected-header tuple escaping as raw `ValueError` (2 failed, 12 passed); both regressions are GREEN at 14/14 after the response boundary was hardened.
- Task 3 final verification passed 286/286 gateway, TASK-12971, contract, planner, registry, and legacy tests. Compileall, Ruff, Black, Python 3.10 AST parsing, and `git diff --check` passed; Bandit reported zero findings and zero errors across 253 gateway production LOC.
- Task 3 / Stage 3 is Complete; overall TASK-12968.2 remains In Progress for Tasks 4-9.

## 2026-07-20 — TASK-12976 conservative frontend licensing cutoff

- Execution worktree: `.worktrees/frontend-licensing-cutoff` on `codex/frontend-licensing-cutoff`.
- Approved plan: `Docs/superpowers/plans/2026-07-20-conservative-frontend-licensing-cutoff-implementation-plan.md`.
- Branch baseline after Backlog ID repair: `4ac5e2a1ebe8dedc7cf3cdc263157d2126f93929`.
- Re-verified public refs: `main` `7a23be3202e360f2d8e7cfe208e13ba406cf0507`; `dev` `29acaca8c781213e27b12066372df13855e2e7a6`; draft PR #2727 head `60ce244fb6a65a79489b3f77299340afa501be24`.
- Baseline verification: 19/19 focused workflow, release, Docker, and OpenAPI cache tests passed; two existing environment/config warnings were emitted.
- Task 1: complete (commits fee1783..26ed722, review clean)
- Task 2: complete (commits e835efb..624896d, review clean)
- Task 3: complete (commits 4501e10..99fdd18, review clean)

## 2026-07-20 — TASK-12977 base-controlled frontend license gate

- Task 1: complete (commits 7a23be3..e66028e, review clean; isolated `main` bootstrap worktree ready; baseline 1/1 passed). Backlog MCP hung and the CLI serializer corrupted empty final-summary markers, so the canonical task record was preserved and the execution evidence remains in the SDD report.
- Task 2: complete (commits e66028e..6dd2d64, review clean; RED missing-module evidence captured; GREEN 16/16; Ruff, Black, Bandit, and diff checks clean).
- Task 3: complete (commits 6dd2d64..66f4aea, final review clean; 27/27 focused tests, actionlint 1.7.12, Ruff, Black, Bandit, and diff checks clean). Robert approved the scoped marker-preserving TASK-12977 edit after Backlog MCP/CLI failures; AC #3 and plan Stages 1–3 are recorded complete.
- Task 3 hardening: complete through `dc9c6146fc`. Review identified and fixed cross-base status reuse with branch-qualified `/main` and `/dev` contexts plus `edited`; exact workflow/job surfaces are contract-locked. Robert then approved exact-commit authorization after review clarified that static required statuses are SHA/base-scoped rather than PR-author-scoped. The design records identical-SHA transferability and fail-safe denial of service, and the rollout now observes real source-bound `/main` and `/dev` results before either ruleset changes. Focused and complete independent reviews are clean with no Critical, Important, or Minor findings. Fresh controller verification: 29/29 tests, actionlint 1.7.12, Ruff, Black, Bandit, marker integrity, stale scans, full diff check, and clean worktree all passed. Draft bootstrap PR is approved to open; merge/activation remain gated on Robert's human-written Change summary and live source evidence.
- Task 4 technical rollout and Task 5 local reconciliation are complete. PR #2753 merged to `main`; PR #2754 proved `/main` and was closed unmerged; PR #2755 proved `/dev`; rulesets `5653432` and `19362594` now require their matching source-bound contexts from App `15368` without bypass actors. The rejected PR-controlled/newline gate was removed, the trusted files match merged `main`, and `frontend-required.yml` matches `origin/dev`. Fresh verification passed 40/40 plus actionlint 1.7.12, Ruff, Black, Bandit, deterministic cases, evidence assertions, and diff checks; independent security and corrected-plan reviews were CLEAN. PR #2753's missing human-written Change summary remains recorded policy noncompliance. TASK-12977 awaits the committed reconciliation head's live `/dev` result before closure.
- TASK-12977 complete. Reconciled code commit `f7c635d34749663fcb52a5ee93561d8013bad022` passed `frontend-license-policy/trusted/dev` in run `29813192487` / job `88578513698` from App `15368`. TASK-12976 Task 4 is complete and execution continues with image/publication isolation and final cutoff verification; PR #2755 stays draft and PR #2727 remains held.

## 2026-07-21 — TASK-12976 final licensing cutoff

- Task 5 complete in `7a4967d3c0`: the production API Dockerfile excludes all protected roots and bundles the legal corpus; `publish-ghcr-main` is backend-only while frontend images remain build-checked. TDD reproduced both policy failures, 20/20 focused tests passed, and independent review was CLEAN.
- Final verification passed 62/62 targeted Python tests, 2/2 protected About tests, Ruff, Black, pinned actionlint 1.7.12, stale-language and whitespace checks, and Bandit with zero findings/errors across 2,554 LOC.
- The production API image built successfully as `sha256:1e44c831aef0790cf7b6a392df1991efaac27be7c1abba24fc011221b9a2b1ed`; runtime assertions proved all four protected roots absent and the required legal files present.
- Full-branch review caught an overbroad nested-notice phrase and stale PR #2727 evidence. The strengthened tests failed 2/2 before correction; all four notices now mirror the root categories and preserve Markdown as GPL, the evidence records current head `e8bcc4c8b705df50a5f7e6299335ba8001ff4811`, 10/10 policy tests pass, and final re-review is CLEAN.
- TASK-12976 is technically complete and PR #2755 remains the draft cutoff PR into `dev`; PR #2727 remains open, draft, and blocked behind it. The required human-authored PR #2755 Change summary is still empty, so the PR is not merge-ready. Counsel terms, custom grants, frontend CLA, completed Countdown grants, and protected artifact publishing remain deferred.

## 2026-07-25 — TASK-12987 Moderation PolicyEvaluator structural extraction

- Execution worktree: `.worktrees/moderation-policy-evaluator-design` on `codex/moderation-policy-evaluator-refactor`.
- Fresh branch is based on current `origin/dev`; the stale implementation branch is preserved separately.
- The former local `TASK-12986` execution ID collided with an unrelated canonical task merged into `dev`, so implementation is tracked under `TASK-12987`.

| Task | Status | TDD evidence | Review |
| --- | --- | --- | --- |
| 1. Decision and dispatch characterization | Complete | Characterization baseline 44/44; new oracle first/final runs 30/30; Black, Ruff, and diff checks clean; test-only Bandit clean with B101 excluded | Approved; no findings after cumulative-provenance clarification |
| 2. Scan, redaction, and limits characterization | Complete | Baseline 30/30; expanded and strengthened runs 67/67; Black, Ruff, test-only Bandit, and diff checks clean | Approved after stronger value/identity immutability snapshots |
| 3. EvaluationLimits and decision evaluator | Complete | RED: absent module collection error; GREEN: strengthened direct 37/37 and combined 104/104; compile, Black, Ruff, production/test Bandit, and diff checks clean | Approved after descriptor, lossless-limit, placeholder, and immutability anchors |
| 4. Direct evaluator redaction | In progress | Pending | Pending |
| 5. ModerationService delegation | Pending | Pending | Pending |
| 6. Real-service caller regressions | Pending | Pending | Pending |
| 7. Final verification and scope audit | Pending | Pending | Pending |
