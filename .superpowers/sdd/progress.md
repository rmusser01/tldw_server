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
| 2. Validated peer transport | Complete | 32/32 failed: missing `_execute_http_hop` | 127 contract/transport tests and 71 compatibility tests passed; static checks passed | Two review/fix rounds; final clean verdicts |
| 3. Bounded response streaming | Not Started | Pending | Pending | Pending |
| 4. Isolation/concurrency/finalization | Not Started | Pending | Pending | Pending |

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
