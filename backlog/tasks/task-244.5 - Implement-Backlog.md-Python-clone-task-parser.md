---
id: TASK-244.5
title: Implement Backlog.md Python clone task parser
status: Done
assignee:
  - codex
created_date: '2026-05-10 22:57'
updated_date: '2026-05-10 23:11'
labels: []
dependencies:
  - TASK-244.4
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
Implement Task 4 from the Backlog.md Python compatibility clone implementation plan. Add a conservative, loss-conscious task Markdown parser and no-op renderer that preserve frontmatter, owned sections, checklist raw lines, and unknown content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fixture task includes frontmatter, unknown metadata, owned sections, acceptance criteria, implementation notes, final summary, and Definition of Done markers
- [x] #2 Parser preserves unknown frontmatter and owned sections
- [x] #3 No-op parse/render round trip is exact for the fixture task
- [x] #4 Parser raises a structured error for unterminated owned sections
- [x] #5 Verification and Bandit results are recorded before completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write the representative fixture task and parser tests first, then verify the missing parser failure.
2. Implement conservative frontmatter splitting, section marker detection, checklist parsing, and no-op rendering.
3. Preserve raw content for unrecognized body text and checklist lines.
4. Run focused parser tests, the accumulated focused suite, Bandit on touched source, and diff checks.
5. Run spec-compliance and code-quality review before finalizing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Red test captured: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_task_parser.py -v` failed with `ModuleNotFoundError: No module named 'backlog_py.markdown'`.
- Implemented a conservative parser that splits frontmatter only at a starting `---`, preserves raw source/frontmatter/body, detects owned section markers, parses checklist items while retaining raw lines, and raises `TaskMarkdownParseError` for unterminated owned sections.
- Verification passed: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_task_parser.py -v` reported 5 passed.
- Verification passed: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_inventory.py tools/backlog-py/tests/test_oracle_manifest.py tools/backlog-py/tests/test_project_discovery.py tools/backlog-py/tests/test_task_parser.py -v` reported 18 passed.
- Bandit passed: `source .venv/bin/activate && python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task4.json`; output JSON has zero severity findings in totals.
- Whitespace check passed: `git diff --check`.
- Known skip: Task 4 plan Step 6 remains unchecked because the controller explicitly instructed not to commit.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Controller verification 2026-05-10:
- Reproduced package-directory parser fixture failure from tools/backlog-py: repo-root-relative fixture path caused FileNotFoundError in 3 parser tests.
- Fixed fixture path to resolve relative to test_task_parser.py.
- Added a red regression test for CRLF frontmatter opening markers; it failed because _split_frontmatter only accepted ---\n.
- Updated frontmatter splitting to accept LF, CRLF, and EOF marker opening lines using splitlines(keepends=True).
- Re-ran parser tests from repo root: 6 passed.
- Re-ran parser tests from tools/backlog-py: 6 passed.
- Re-ran accumulated focused suite: inventory + oracle + project + parser -> 19 passed.
- Re-ran Bandit: python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task4.json -> exit 0 with results: [].
- Re-ran git diff --check -> exit 0.

Controller test hardening 2026-05-10:
- Expanded parser preservation test to assert implementation notes, final summary, and DoD checklist visibility.
- Broke the long fixture path into a cwd-independent Path(__file__)-relative expression.
- Re-ran parser tests from repo root: 6 passed.
- Re-ran parser tests from tools/backlog-py: 6 passed.
- Re-ran accumulated focused suite: inventory + oracle + project + parser -> 19 passed.
- Re-ran Bandit and git diff --check after test hardening; both exit 0 and Bandit results remain empty.

Review closeout 2026-05-10:
- Spec-compliance review approved with no missing Task 4 requirements or extra scope.
- Code-quality review approved with no blockers. Deferred non-blocking parser hardening: decide how unknown uppercase marker sections should be represented before mutation support, and define duplicate section handling before editing sections.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the loss-conscious Backlog.md task parser slice. Added a representative fixture task, parser models, conservative frontmatter splitting, owned section and checklist marker detection, raw checklist preservation, exact no-op rendering from raw source, and structured unterminated-section errors. Latest verification: parser tests passed 6/6 from both repo root and package directory, the accumulated focused suite passed 19/19, Bandit reported no findings, diff checks passed, and both spec/code-quality reviews approved.
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
