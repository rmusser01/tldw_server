---
id: TASK-12971
title: Add a connected-peer-verified streaming HTTP hop primitive
status: To Do
labels:
- security
- http-client
- egress
- ssrf
- research
priority: High
references:
- TASK-12968
- TASK-12968.1
- TASK-12968.2
- TASK-2338
documentation:
- Docs/Design/2026-07-13-research-source-coverage-shared-discovery-design.md
dependencies:
- TASK-12968.1
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
- [ ] #6 Existing http-client callers remain compatible unless explicitly migrated, focused concurrency/security tests and Bandit pass, and TASK-12968.2 consumes this primitive rather than wrapping afetch_json.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
