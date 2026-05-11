---
id: TASK-244.9
title: >-
  Implement Backlog.md Python clone documents milestones and Definition of Done
  parity
status: Done
assignee:
  - codex
created_date: '2026-05-11 00:21'
labels: []
dependencies:
  - TASK-244.8
references:
  - 'https://github.com/MrLesk/Backlog.md'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-10-backlog-md-python-compatibility-clone-design.md
  - >-
    Docs/superpowers/plans/2026-05-10-backlog-md-python-compatibility-clone-implementation-plan.md
parent_task_id: TASK-244
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 8 from the Backlog.md Python compatibility clone implementation plan. Add document, milestone, and Definition of Done default parity on top of the safe mutation core, using the same validation-before-write, path-containment, atomic-write, CLI, and pure MCP safety boundaries established by TASK-244.8.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Nested document create, list/search, view, and update work under backlog/docs while preserving omitted metadata
- [x] #2 Document path validation rejects absolute paths and parent traversal before writes
- [x] #3 Milestone add/list/rename/remove/archive operate on milestone files and task references as requested
- [x] #4 Definition of Done defaults can be read and replaced in config
- [x] #5 New task creation inherits project DoD defaults unless disabled and task-specific DoD additions do not mutate defaults
- [x] #6 CLI and pure MCP adapters expose document, milestone, and DoD default operations without shell execution
- [x] #7 Focused docs/milestones/DoD tests, security regression tests, Bandit, diff checks, and two-stage review are completed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing document tests for nested create, list/search, view by ID/path, update preserving omitted metadata, and path traversal rejection.
2. Write failing milestone tests for add/list/rename/remove/archive and optional task reference updates.
3. Write failing Definition of Done tests for config defaults get/upsert, task creation inheritance unless disabled, and task-specific additions that do not mutate project defaults.
4. Run focused tests to verify RED failures against the current implementation.
5. Implement document and milestone services using existing path containment and atomic write patterns; implement DoD defaults config writes; extend safe task creation for DoD inheritance controls and task-specific DoD additions.
6. Add CLI and pure MCP adapters for document, milestone, and DoD operations.
7. Run focused docs/milestones/DoD tests, security regression tests, full backlog-py tests, Bandit, diff checks, and two-stage review before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `DocumentService` for docs-relative nested Markdown documents under `backlog/docs`, including path/id lookup, recursive list/search, metadata-preserving updates, omitted body-source preservation, symlink-read escape rejection, path traversal rejection, and atomic writes.
- Added `MilestoneService` for active and archived milestone files plus optional task frontmatter updates on rename/remove, case-insensitive task-reference matching, symlink-read escape rejection, and best-effort rollback for failed multi-file task-reference updates.
- Added config-backed Definition of Done default get/upsert helpers and extended task creation to inherit defaults unless disabled, with task-specific additions appended without mutating project config.
- Exposed document, milestone, and DoD default operations through Click CLI commands, package-level pure MCP exports, helper functions, and workflow resource text.
- RED command: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_documents.py tools/backlog-py/tests/test_milestones.py tools/backlog-py/tests/test_definition_of_done.py -v`; result: expected collection failures for missing `backlog_py.core.documents`, `backlog_py.core.milestones`, and MCP DoD functions.
- Review RED command after two-stage review findings: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_documents.py tools/backlog-py/tests/test_milestones.py tools/backlog-py/tests/test_definition_of_done.py tools/backlog-py/tests/test_mcp_resources.py -v`; result: expected failures for symlinked read escapes, milestone rollback, stale DoD config reuse, CLI clear defaults, and MCP package/resource exposure.
- GREEN focused command after review fixes: same expanded focused pytest command; result: `34 passed, 2 warnings`.
- Full backlog-py command: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests -v`; result: `88 passed, 2 warnings`.
- Security regression command: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_security_paths.py -v`; result: `4 passed, 2 warnings`.
- Bandit command: `source .venv/bin/activate && python -m bandit -r tools/backlog-py/src/backlog_py -f json -o /tmp/bandit_backlog_py_task8_final2.json`; result: zero findings.
- Diff check command: `git diff --check`; result: clean.
- Two-stage review completed. Spec and code-quality reviewers found case-sensitive milestone ref updates, stale MCP exposure text/exports, symlinked doc/milestone read escapes, milestone split-state risk on task-write failure, stale DoD defaults for long-lived callers, CLI inability to clear defaults, and body-source preservation risk; all were addressed with regression coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented document, milestone, and Definition of Done default parity for the Backlog.md Python clone. The slice now includes safe nested docs, active/archive milestone operations, DoD config defaults, CLI and pure MCP exposure, review-driven safety fixes, and passing pytest/Bandit/diff verification.
<!-- SECTION:FINAL_SUMMARY:END -->
