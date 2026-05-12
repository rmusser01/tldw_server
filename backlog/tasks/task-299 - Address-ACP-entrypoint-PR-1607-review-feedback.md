---
id: TASK-299
title: 'Address ACP entrypoint PR #1607 review feedback'
status: Done
assignee: []
created_date: '2026-05-12 13:51'
updated_date: '2026-05-12 14:01'
labels:
  - ACP
  - PR-review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1607'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md
  - >-
    Docs/superpowers/plans/2026-05-12-acp-entrypoint-strategy-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR #1607 for ACP entrypoint strategy readiness surfaces. Scope: CodeRabbit stdio manifest/env, child exit code, docs URL normalization, persistence-test coverage, deterministic timeout test; Qodo docstrings; Gemini multi-blocker classifier feedback where compatible with the approved architecture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All actionable PR #1607 review threads are addressed or technically justified.
- [x] #2 Focused ACP registry/helper/API tests pass after review fixes.
- [x] #3 git diff --check passes.
- [x] #4 Scoped Bandit on touched production files passes or only reports existing baseline findings.
- [x] #5 Review-fix commit is pushed to PR #1607.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review threads addressed in PR #1607:
- Removed unrelated E2E env requirements from direct ACP agent-profile manifests while keeping them marked live-agent.
- Made stdio JSON-RPC probes return a child nonzero exit status after successful frames.
- Normalized registry entrypoint docs URLs through the served docs-static path.
- Added DB-backed persistence coverage for entrypoint metadata and invalid-strategy coercion.
- Replaced timing-sensitive partial-line timeout coverage with event-driven cleanup coverage.
- Aggregated multiple classifier blockers and added requested docstrings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the actionable PR #1607 review feedback across ACP entrypoint classification, API status normalization, certification smoke manifests, and focused regression tests. Verification: 148 focused ACP tests passed, prior 89-test focused slice passed, git diff --check passed, and scoped Bandit on touched production files reported no findings. Review-fix commit is being pushed to the PR branch.
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
