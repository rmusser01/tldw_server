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
