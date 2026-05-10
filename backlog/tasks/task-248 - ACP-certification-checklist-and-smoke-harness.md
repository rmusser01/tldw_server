---
id: TASK-248
title: ACP certification checklist and smoke harness
status: Done
assignee: []
created_date: '2026-05-10 21:32'
labels:
  - ACP
  - compatibility
  - certification
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1539'
  - 'https://github.com/rmusser01/tldw_server/pull/1554'
  - 'https://github.com/rmusser01/tldw_server/pull/1555'
documentation:
  - Docs/Development/ACP_Compatibility_Matrix.md
  - Docs/Development/Agent_Client_Protocol.md
  - Docs/Development/ACP_Production_Readiness.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement #1539 PR 2: make downstream-agent compatibility claims reproducible with documented certification checklists and a focused smoke harness that reuses existing ACP backend, orchestration, frontend, and Go runner verification where possible. Keep scope to certification workflow, docs, and helper automation; do not build installer or marketplace behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A contributor can certify an agent by following a checklist.
- [x] #2 Certification does not require undocumented local state.
- [x] #3 Stub/smoke-tested and live-E2E-tested levels are clearly different.
- [x] #4 Minimum checks cover session start, prompt, structured completion, artifacts, diagnostics, cancel/close, review loop, workspace env, MCP server injection, and sandbox behavior where applicable.
- [x] #5 Focused smoke harness or command manifest reuses existing ACP test suites and runner verification rather than inventing parallel checks.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

Implemented certification workflow for #1539 PR2 in worktree
`.worktrees/acp-certification-smoke-harness`.

Added `Docs/Development/ACP_Certification_Checklist.md` with `stub-smoke`,
`live-e2e`, sandbox, evidence record, and matrix update checklists.

Added `Helper_Scripts/Testing-related/acp_certification_smoke.py` plus pytest
coverage in
`tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`.

Linked the checklist/harness from `ACP_Compatibility_Matrix.md`,
`Agent_Client_Protocol.md`, and `ACP_Production_Readiness.md`.

Verification: `python -m pytest
tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q`
passed; CLI JSON manifest emitted for `stub-smoke`; `py_compile` passed;
`git diff --check` passed; Bandit JSON at
`/tmp/bandit_acp_certification_smoke.json` has zero results.

Draft PR created: https://github.com/rmusser01/tldw_server/pull/1555.

## Final Summary

Added a reproducible ACP downstream-agent certification workflow for #1539 PR2.
The new checklist distinguishes stub-smoke, live-E2E, and sandbox evidence; the
helper emits/runs static command manifests that reuse existing backend ACP
suites, mocked browser setup/run/diagnose coverage, and Go runner verification;
and ACP protocol/readiness/matrix docs now link to the workflow. Verified with
focused pytest, CLI JSON output, py_compile, git diff --check, and Bandit on the
new helper with zero findings.
