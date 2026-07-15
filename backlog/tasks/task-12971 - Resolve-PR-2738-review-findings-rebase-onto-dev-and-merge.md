---
id: TASK-12971
title: 'Resolve PR #2738 review findings, rebase onto dev, and merge'
status: In Progress
labels:
- playlist-ingest
- code-review
- pr-2738
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address every actionable inline/top-level review finding on PR #2738, verify the remediations, rebase the branch onto the latest origin/dev, monitor/fix required CI, and merge once all repository merge gates (including the human-written Change summary) are satisfied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All actionable review comments are fixed or dispositioned with evidence and inline replies.
- [ ] #2 All review threads are resolved.
- [ ] #3 Targeted and regression tests, lint/type/compile checks, Bandit, and diff checks pass after the final rebase.
- [ ] #4 Branch is rebased onto latest origin/dev and pushed with lease.
- [ ] #5 Required CI is green and PR is mergeable.
- [ ] #6 Human requester has supplied the required human-written Change summary.
- [ ] #7 PR #2738 is merged into dev.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review audit found seven unresolved inline threads (one Gemini portability finding plus six Qodo findings). Implemented RED/GREEN remediations: moved playlist SQL store under DB_Management with a legacy import alias; centralized all playlist domain exceptions; replaced dynamic identifier interpolation with fixed query maps; made docs-info represent effective sidecar worker availability; added drain admission guards to preflight/run/retry; added endpoint docstrings; made plan commands portable; linked the Windows-only skip to TASK-12971. Also corrected a stale integration helper to send the required replay-safe client_request_id and processing_options. Verification: focused RED reproduced all issues; focused GREEN 6 passed; corrected workflow integration 9 passed; fresh full backend matrix 611 passed with RUN_JOBS=1 including PostgreSQL migrations, quotas, RLS, and store parity; scoped Ruff, compileall, and git diff --check pass; Bandit reports 0 findings/errors over 12,737 production lines. Independent final review returned zero Critical, Important, or Minor findings. Rebase/push/thread resolution/CI/merge remain.
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
