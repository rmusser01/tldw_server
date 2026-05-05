---
id: TASK-48
title: 'Address PR #1270 CodeGraph context/impact review follow-ups'
status: In Progress
assignee:
  - '@Codex'
created_date: '2026-05-05 00:44'
updated_date: '2026-05-05 00:45'
labels:
  - codegraph
  - mcp
  - pr-review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1270'
  - TASK-46
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the current PR #1270 review comments for the native CodeGraph context/impact slice: bounded impact relationship query materialization, include_code=null default behavior, test docstrings, and the human-authored attestation request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Qodo SQL relationship materialization comment is addressed with a regression test and implementation bound.
- [x] #2 Qodo include_code=null default comment is addressed with a regression test and implementation fix.
- [x] #3 Qodo docstring comment is addressed for new PR test coverage or explicitly pushed back with repo-grounded rationale.
- [ ] #4 CodeRabbit human-authored attestation comment is responded to without falsely claiming human authorship.
- [x] #5 Focused CodeGraph/MCP tests, Ruff, Bandit on touched production scope, and git diff --check are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED regression coverage for bounded relationship row selection and codegraph.context include_code=null defaulting.
2. Implement SQL LIMIT/max_rows in the CodeGraph repository impact traversal and normalize include_code=None to the documented default true in the MCP module.
3. Add concise docstrings to new PR test modules/functions touched by this slice.
4. Run focused CodeGraph/MCP tests, Ruff, Bandit on touched production code, and git diff --check.
5. Reply to and resolve implementable PR review threads; respond to the human-authored attestation request without generating a false human attestation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED verification: targeted regression tests failed before implementation because impact traversal did not pass a max_rows selector bound and codegraph.context include_code=null returned no snippets.

Implemented SQL LIMIT/max_rows for impact relationship selection, normalized codegraph.context include_code=null to the default true, and added concise docstrings to new CodeGraph test modules/helpers.

Verification so far: targeted regression tests passed; focused CodeGraph/MCP pytest suite passed with 97 passed and 5 warnings; Ruff touched scopes passed; Bandit /tmp/bandit_codegraph_context_impact.json reported zero findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
