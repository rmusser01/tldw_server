---
id: TASK-244.3
title: Implement Backlog.md Python clone pinned oracle manifest
status: Done
assignee:
  - codex
created_date: '2026-05-10 21:55'
updated_date: '2026-05-10 22:06'
labels: []
dependencies:
  - TASK-244.2
references:
  - 'https://github.com/MrLesk/Backlog.md'
  - 'https://www.npmjs.com/package/backlog.md'
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
Implement Task 2 from the Backlog.md Python compatibility clone implementation plan. Add a pinned oracle fixture manifest and loader that records the upstream Backlog.md version/source metadata and marks agent-critical fixtures without invoking Node/Bun during normal runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 tools/backlog-py includes an oracle manifest fixture pinned to Backlog.md 1.44.0 metadata
- [x] #2 Oracle manifest loader parses version/source/hash metadata and fixture entries into typed Python models
- [x] #3 Manifest tests are written red-first and pass after implementation
- [x] #4 README documents the pinned oracle fixture policy and normal runtime remains Node/Bun-free
- [x] #5 Verification and Bandit results are recorded before completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write focused tests for the pinned manifest loader and confirm the missing module failure.
2. Add the oracle package, manifest fixture, and YAML loader using yaml.safe_load.
3. Update the README with the oracle policy and no-runtime-Node/Bun constraint.
4. Run focused tests, Bandit on the touched package source, and diff checks.
5. Run spec-compliance and code-quality review before finalizing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- 2026-05-10: Red-first oracle manifest test run captured the expected missing package failure: `ModuleNotFoundError: No module named 'backlog_py.oracle'`.
- 2026-05-10: Added `backlog_py.oracle` package, pinned `manifest.yml` for `backlog.md@1.44.0`, and typed dataclass loader using `yaml.safe_load`.
- 2026-05-10: Updated `tools/backlog-py/README.md` with the pinned oracle fixture policy and the normal-runtime Node/Bun-free constraint.
- 2026-05-10 verification: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_oracle_manifest.py -v` passed: 2 passed, 2 warnings.
- 2026-05-10 verification: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_inventory.py tools/backlog-py/tests/test_oracle_manifest.py -v` passed: 4 passed, 2 warnings.
- 2026-05-10 verification: `source .venv/bin/activate && python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task2.json` passed with 0 results and 0 errors.
- 2026-05-10 verification: `git diff --check` passed with no output.
- Known skip: no commit created per user instruction.

Controller verification 2026-05-10:
- Re-ran focused oracle tests: python -m pytest tools/backlog-py/tests/test_oracle_manifest.py -v -> 2 passed.
- Re-ran Task 1+2 focused tests: python -m pytest tools/backlog-py/tests/test_inventory.py tools/backlog-py/tests/test_oracle_manifest.py -v -> 4 passed.
- Performed controlled red check by temporarily renaming mcp:workflow-overview in the manifest; test_manifest_marks_agent_critical_fixtures failed, then passed again after restoration.
- Re-ran Bandit: python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task2.json -> exit 0 with results: [].
- Re-ran git diff --check -> exit 0.

Code-quality review follow-up 2026-05-10:
- Reproduced package-cwd failure with source ../../.venv/bin/activate && python -m pytest tests/test_oracle_manifest.py -q from tools/backlog-py: FileNotFoundError for tools/backlog-py/tests/fixtures/oracle/manifest.yml.
- Fixed root cause by resolving MANIFEST_PATH relative to test_oracle_manifest.py instead of repository cwd.
- Verified package-cwd test command now passes: 2 passed.
- Re-ran root focused tests after the fix: inventory + oracle tests -> 4 passed.
- Re-ran Bandit and git diff --check after the fix; both exit 0.
- Non-blocking reviewer suggestions kept for later hardening: stricter boolean/scalar manifest validation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the pinned oracle manifest slice for the Backlog.md Python compatibility clone. The new oracle package loads a Backlog.md 1.44.0 manifest fixture with upstream source metadata and agent-critical fixture entries, the README documents the fixture policy and Node/Bun-free runtime constraint, and focused tests now pass from both the repository root and the package directory. Verification included controlled red checks, focused pytest, Bandit with no findings, diff checks, and spec/code-quality review approvals.
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
