---
id: TASK-13134
title: Fix VZ Linux real smoke exec output over VSock
status: Done
assignee: []
created_date: '2026-08-27 02:10'
updated_date: '2026-08-27 02:11'
labels:
  - sandbox
  - vz-linux
  - real-smoke
  - bugfix
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real Apple Virtualization.framework smoke reached guest readiness but exec output propagation was unreliable because the guest VSock client discarded bytes buffered after the ready ACK. Fix the guest/helper request-response path and keep regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Guest VSock client preserves host bytes buffered immediately after ready ACK
- [x] #2 Regression test covers ready ACK plus exec request in one host write
- [x] #3 Regression test waits for and validates the exec response before closing the helper-side pipe
- [x] #4 Real host VZ smoke passes with rebuilt guest bundle
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: primeConnection used a bufio.Reader for handshake/ready, then Run passed the raw connection into ServeStream. If the helper's ready ACK and first exec request arrived together, the exec bytes could remain buffered in the discarded reader. Fix: return the priming bufio.Reader from primeConnection and pass it to ServeStream. PR review follow-ups: the regression test now waits for and validates the exec response, and uses a 5s read deadline to avoid CI-load flakiness while still failing genuine hangs. Verification: GOCACHE=/private/tmp/tldw-go-build-cache go test ./internal/guest; GOCACHE=/private/tmp/tldw-go-build-cache go test ./... in tools/tldw-agent; git diff --check; rebuilt Debian arm64 bundle; real host smoke final_exit_code=0 at /private/tmp/tvz-e2e.4iJ3wt/evidence/host-smoke-evidence.json. Bandit N/A: Go-only production change.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed VZ Linux guest VSock exec delivery by preserving the buffered reader used during handshake/readiness when entering ServeStream. Added and hardened a regression test for ready ACK plus exec request arriving in one host write, then verified unit tests and real Apple VZ smoke against a rebuilt Debian arm64 bundle.
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
