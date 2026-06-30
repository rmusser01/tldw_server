---
id: TASK-218
title: Address PR 1495 ACP productionization review feedback
status: Done
assignee: []
created_date: '2026-05-10 04:20'
updated_date: '2026-05-10 04:42'
labels:
  - acp
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1495'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable reviewer feedback on PR #1495 for the ACP productionization workstream while preserving the existing ACP architecture and avoiding unrelated refactors. Focus on reviewer-identified correctness, security, documentation, lint, and test gaps across the ACP API, orchestration, scheduling, frontend Agent Tasks/Registry UI, readiness checks, and Backlog task markdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All actionable unresolved PR #1495 review threads are either fixed or answered with code-grounded rationale.
- [x] #2 Agent Tasks hosted-mode requests do not send local ACP API-key or bearer credentials through the hosted transport.
- [x] #3 Agent task diagnostics inspection cancels or ignores stale requests when a new inspect starts or the modal closes.
- [x] #4 ACP readiness treats an empty agent list as unavailable instead of inheriting overall health.
- [x] #5 ACP schedule misfire grace validation preserves explicit zero and rejects negative values with tests.
- [x] #6 ACP audit, orchestration artifact, workspace access, completion signal, and failure-recording feedback is handled with tests where behavior changes.
- [x] #7 Backlog task markdown lint issues and duplicate Definition of Done entries identified by review are cleaned up.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved PR #1495 review feedback across hosted-mode Agent Tasks auth, task diagnostics cancellation, ACP readiness empty-agent status, schedule misfire grace validation, audit readback/admin scope access, orchestration artifact summaries, workspace 403 path disclosure, completion/review marker validation, stable failure reason storage, Agent Registry stale health clearing, docstrings, and Backlog markdown cleanup. Verification: compileall for touched backend production modules; focused pytest for ACP schedules, ACP endpoints, orchestration API, and workspace helper tests; focused Vitest for Agent Tasks hosted transport and readiness; Bandit on touched backend production files; git diff --check.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed actionable PR #1495 review feedback with backend and frontend fixes plus regression coverage. Preserved ACP architecture while tightening security-sensitive request headers, stale diagnostics handling, audit persistence, scheduling validation, failure reason stability, and review marker parsing. Cleaned reviewer-flagged Backlog markdown issues and recorded verification.
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
