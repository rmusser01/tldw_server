# TASK-12971 Subagent-Driven Development Progress

## 2026-07-14 — Plan and architecture review

- Active task: TASK-12971, blocking TASK-12968.2.
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
| 2. Validated peer transport | Not Started | Pending | Pending | Pending |
| 3. Bounded response streaming | Not Started | Pending | Pending | Pending |
| 4. Isolation/concurrency/finalization | Not Started | Pending | Pending | Pending |
