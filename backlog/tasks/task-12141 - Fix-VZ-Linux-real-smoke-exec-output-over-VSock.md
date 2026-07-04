---
id: TASK-12141
title: Fix VZ Linux real smoke exec output over VSock
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 17:55'
labels:
  - sandbox
  - vz-linux
  - real-smoke
  - bugfix
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real Apple Virtualization.framework smoke reaches guest readiness but exec output propagation is unreliable: direct helper exec can time out and the host smoke can complete with empty stdout. Investigate and fix the guest/helper VSock request-response path with minimal code and tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Guest VSock client preserves host bytes buffered immediately after ready ACK
- [x] #2 Regression test covers ready ACK plus exec request in one host write
- [x] #3 Real host VZ smoke passes with rebuilt guest bundle
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Observed on local Apple Silicon host with signed macos-vz-helper and Debian arm64 bundle.', 'Helper daemon bundle smoke passed. Real smoke selected tests: 2 passed, 1 failed on missing stdout token.', 'Manual helper probe reached guest readiness then direct exec_guest timed out at guest_transport_timeout.', 'Leading hypothesis: guest VSock client primes handshake/ready with bufio.Reader, then ServeStream reads from raw conn; bytes buffered after ready can be lost from ServeStream.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Root cause confirmed with failing test: primeConnection used a bufio.Reader for handshake/ready, then Run passed the raw conn into ServeStream. If the helper's ready ACK and first exec request arrived together, the exec bytes could remain buffered in the discarded reader.

Fix: return the priming bufio.Reader from primeConnection and pass it to ServeStream so buffered exec bytes remain visible.

Verification: GOCACHE=/private/tmp/tldw-go-build-cache go test ./internal/guest; GOCACHE=/private/tmp/tldw-go-build-cache go test ./... in tools/tldw-agent; rebuilt Debian arm64 bundle; real host smoke final_exit_code=0 at /private/tmp/tvz-e2e.4iJ3wt/evidence/host-smoke-evidence.json.

Bandit: not applicable to touched production code; this change is Go-only under tools/tldw-agent.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed VZ Linux guest VSock exec delivery by preserving the buffered reader used during handshake/readiness when entering ServeStream. Added a regression test for ready ACK plus exec request arriving in one host write, then verified unit tests and real Apple VZ smoke against a rebuilt Debian arm64 bundle.
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
