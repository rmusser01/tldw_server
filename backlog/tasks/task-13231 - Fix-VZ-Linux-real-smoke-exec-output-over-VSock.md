---
id: TASK-13231
title: Fix VZ Linux real smoke exec output over VSock
status: Done
assignee: []
created_date: '2026-08-27 02:10'
updated_date: '2026-09-10 00:52'
labels:
  - sandbox
  - vz-linux
  - real-smoke
  - bugfix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2628'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review PR 2628 against latest dev; rebase its three commits; verify the reader regression and full Go suite; correct only this PR task identity; push and assess fresh CI and review state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: primeConnection used a bufio.Reader for handshake/ready, then Run passed the raw connection into ServeStream. If the helper's ready ACK and first exec request arrived together, the exec bytes could remain buffered in the discarded reader. Fix: return the priming bufio.Reader from primeConnection and pass it to ServeStream. PR review follow-ups: the regression test now waits for and validates the exec response, and uses a 5s read deadline to avoid CI-load flakiness while still failing genuine hangs. Verification: GOCACHE=/private/tmp/tldw-go-build-cache go test ./internal/guest; GOCACHE=/private/tmp/tldw-go-build-cache go test ./... in tools/tldw-agent; git diff --check; rebuilt Debian arm64 bundle; real host smoke final_exit_code=0 at /private/tmp/tvz-e2e.4iJ3wt/evidence/host-smoke-evidence.json. Bandit N/A: Go-only production change.

2026-09-10 PR review and rebase: rebased the three PR commits unchanged onto origin/dev 40345571a2cfc8b3a8893545836097d27e4ee86c. Current dev still discards the handshake buffered reader, so the fix remains applicable. Independent code review found no actionable Go issues. Both prior GitHub review threads (early pipe closure and one-second deadline) are resolved. User explicitly approved renaming this record from TASK-13134 to TASK-13231 because TASK-13134 collides with existing dev records and the Backlog tools do not support ID renames; other task records are untouched. Fresh verification in tools/tldw-agent: go test ./... passed before rebase; go test -race -count=1 ./... and go vet ./... passed after rebase; the buffered-exec regression passed 20 consecutive runs. A temporary Go overlay reverting only the reader handoff caused that regression to fail after five seconds, proving it detects the original bug. CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build of cmd/tldw-agent-guest passed. gofmt and git diff --check passed. Bandit ran from the project venv on the touched guest directory and found no Python files (zero LOC); it provides no Go security coverage. The original real Apple VZ smoke evidence above is historical and was not rerun during this review. Fresh GitHub checks on rebased commit 742801866d currently have no failures; final task correction will require checks on its new head.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserved the buffered VSock reader from handshake/readiness through command serving, with a deterministic regression test that verifies the exec response. Rebased onto current dev and confirmed the fix is still needed. Full Go race tests, go vet, repeated regression checks, an intentionally failing reader-handoff mutation, and the Linux arm64 guest build validate the change. Renamed the historical task from TASK-13134 to unique TASK-13231 with explicit user approval. Technical review has no remaining Go findings; merging remains subject to checks passing on the final PR head and the repository human-owned Change summary gate.
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
