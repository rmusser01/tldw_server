---
id: TASK-12971
title: Add a connected-peer-verified streaming HTTP hop primitive
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-14 16:55'
labels:
  - security
  - http-client
  - egress
  - ssrf
  - research
dependencies:
  - TASK-12968.1
references:
  - TASK-12968
  - TASK-12968.1
  - TASK-12968.2
  - TASK-2338
documentation:
  - Docs/Design/2026-07-13-research-source-coverage-shared-discovery-design.md
  - >-
    Docs/superpowers/plans/2026-07-14-connected-peer-verified-http-hop-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add one reusable outbound HTTP hop primitive that resolves and validates the complete DNS answer set, connects only to a validated address while preserving Host/SNI, verifies the connected peer, disables automatic redirect/retry loops, ignores ambient proxy/netrc/cookie/credential state, and streams through separate wire and decompressed-byte ceilings. This closes the security gap between the existing URL/DNS policy and TASK-12968.2's shared discovery gateway; the discovery gateway must orchestrate each redirect or retry as a separately reserved dispatch rather than hiding it inside this primitive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A one-hop async API accepts only an already-normalized route policy/request and performs exactly one physical request without automatic redirects or retries.
- [ ] #2 The transport connects to a policy-validated address, preserves the approved Host header and TLS SNI, obtains connected-peer metadata, and fails closed when the peer cannot be verified against the validated set.
- [ ] #3 HTTP and HTTPS tests cover private and mixed DNS answers, DNS rebinding, alternate ports, malformed IP forms, SNI/Host behavior, absent peer metadata, and redirect destinations without following them.
- [ ] #4 The primitive disables trust_env and proves that environment proxies, .netrc, ambient cookies, client certificates, and injected authorization cannot escape the explicit request contract.
- [ ] #5 Wire bytes, decompressed bytes, headers, time, and parser input are bounded while streaming; oversized or compressed-bomb responses fail before full materialization and return sanitized bounded errors.
- [ ] #6 Existing http-client callers remain compatible; focused concurrency/security tests and Bandit pass; the delivered public import and focused-test paths are inserted into TASK-12968.2's blocked plan, which must later prove gateway consumption instead of wrapping afetch_json after this prerequisite completes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-14-connected-peer-verified-http-hop-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-14: Began implementation in the isolated worktree after TASK-12968.1 completion. Two read-only architecture reviews rejected the legacy cached HTTP clients, afetch_json, separate certificate-pinning socket, aiohttp cookie/session defaults, and the synchronous MCP docs transport as the TASK-12971 security boundary. Selected a dedicated Security/http_hop.py using HTTPcore 1.x's public custom async network-backend interface, a fresh HTTP/1.1 pool per hop, retries=0, validated-address dialing, connected-peer verification, and bounded incremental decompression.

Adversarial plan review found that HTTPcore's internal h11 limit is per incomplete event and does not bound repeated informational responses. The reviewed plan now adds an independent raw-stream aggregate 1xx/final-header and body-wire gate before HTTPcore parsing, offloaded DNS under a whole-hop deadline, environment-independent certifi TLS context construction, explicit request framing/size gates, peer-port verification, and bounded decompressor finalization. Simplicity review removed redundant counters/rechecks and excess integration harnesses. AC #6 was clarified to remove a dependency cycle: TASK-12971 publishes stable import/test paths, and dependent TASK-12968.2 later proves consumption. TDD production edits have not started yet.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-07-14 Stage 1 complete: added immutable normalized request/limit/error contracts, strict canonical host/target/header validation, complete-set DNS/IP denial (including mapped, NAT64, 6to4, Teredo, site-local, scoped, private, reserved, and malformed answers), an off-event-loop bounded resolver adapter with legacy compatibility, and direct HTTPcore/certifi dependency floors. Verification: 115 focused and adjacent tests passed; Ruff, Black, compileall, and git diff --check passed. Two adversarial review rounds were resolved and the final re-review reported no actionable findings. Stage 2 validated-peer transport remains in progress.

2026-07-14 Stage 2 complete: added a one-use HTTPcore backend/pool that dials only the first validated literal address, verifies peer IP and port before HTTP bytes and again after TLS, preserves the route Host/SNI, uses a fresh certifi-only TLS context with cached SSL evidence, forces HTTP/1.1, blocks UDS/retry/second-dial paths, rejects 101 upgrades, and returns redirects without following them. Verification: 127 contract/transport tests and 71 legacy compatibility tests passed; Ruff, Black, compileall, and git diff --check passed. Two independent review/fix rounds ended clean. Authoritative raw/header/decompression counters and the public one-argument entrypoint remain Stage 3 work.

2026-07-14 Stage 3 complete: added an aggregate plaintext header and wire-byte guard ahead of h11, encoded-length preflight, bounded identity/gzip/zlib-deflate decoding, decompressed and parser-input ceilings, stable sanitized failure mapping, one whole-hop deadline, and the public one-argument request_http_hop API. Central HTTPcore logging is floored at INFO to prevent response headers and hostile protocol bytes from leaking through DEBUG traces. Review fixes covered unsafe decompressor finalization, framing and status edge cases, RFC-correct 205 and HEAD behavior including coalesced trailing bytes, deterministic decoder fixtures, actual failure-stream closure assertions, and an explicit optimized-Python-safe TLS invariant. Verification: 179 contract/transport/streaming tests and 259 complete focused/legacy compatibility tests passed; focused Ruff, Black, compileall, Python 3.10 grammar parsing, Bandit, and git diff --check passed. Independent security, spec/correctness, and simplification re-reviews ended clean. Full-file Ruff on legacy app/main.py retains 10 unrelated baseline findings; the six-line HTTPcore logger-floor delta is regression-tested and compile-checked.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
