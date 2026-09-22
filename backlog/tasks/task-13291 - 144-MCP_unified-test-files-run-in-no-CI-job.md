---
id: TASK-13291
title: 144 MCP_unified test files run in no CI job
status: To Do
assignee: []
created_date: '2026-09-22 04:34'
updated_date: '2026-09-22 15:29'
labels:
  - ci
  - mcp
  - testing
dependencies: []
references:
  - '.github/workflows/ci.yml:1759'
  - 'pyproject.toml:638'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_Server_API/app/core/MCP_unified/tests` holds **144 test files / 90,042 LOC** and is declared in `pyproject.toml:638` `testpaths`. No required CI gate executes it.

Verified:
- The six contractual gates in `Docs/Development/CI_REQUIRED_GATES.md` run explicit paths under `tldw_Server_API/tests/` only. The `platform-mcp-core` shard runs `tldw_Server_API/tests/MCP` + `tldw_Server_API/tests/MCP_unified` (`ci.yml:1759-1760`) — a **different tree** with a confusingly similar name.
- The only workflow naming in-app files is `.github/workflows/mcp-unified-rc.yml`, which is (a) not a required gate, (b) `paths:`-filtered, and (c) lists ~10 of the 144 files.

Consequence already realized: `test_filesystem_glob_marks_file_size_unavailable` was added 2026-06-03 in `5009fc8b95` and has been **red ever since** (`OSError: metadata unavailable`, unguarded `is_symlink()` at `filesystem_module.py:1778`). A red test survived ~3.5 months on the main line because no job could see it.

The placement is partly deliberate: `apps/mcp-unified/` has no `tests/` directory, and 35 of these files import the standalone `mcp_unified` package rather than `core.MCP_unified` — so this tree is that package's de-facto test home, parked where it can import both. What is not deliberate is that the ~105 files testing `core.MCP_unified` inherited a location no server CI job looks at. **The fix is a shard change, not a tree move.**

Sequencing matters: the tree has never been gated, so an unknown number beyond the one known red test may fail. Run it locally and triage first; do not gate blind.

Found by the comprehensive core-module review; independently verified by the orchestrator (file count, LOC, testpaths entry, and the workflow path audit).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The in-app tree is run locally and every current failure is triaged and recorded before any gating change
- [ ] #2 The known red test (filesystem glob / unguarded is_symlink at filesystem_module.py:1778) is fixed or explicitly quarantined with a reason
- [ ] #3 tldw_Server_API/app/core/MCP_unified/tests is added to the platform-mcp-core shard (ci.yml:1758-1761 and its four siblings) or another required gate
- [ ] #4 A green run of that gate is recorded with its observed output
- [ ] #5 The naming collision between tests/MCP_unified and app/core/MCP_unified/tests is documented so the next reader does not assume the gate covers both
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TRIAGE (2026-09-22), per this task's own requirement not to gate blind.

Collection is healthy: 3,351 tests collect from tldw_Server_API/app/core/MCP_unified/tests in 6.4s.

But the tree CANNOT currently be run as a single pytest session. It aborts partway with a native error:
  libc++abi: terminating due to uncaught exception of type std::__1::system_error: recursive_mutex lock failed: Invalid argument
No summary line is produced. Run file-by-file it is fine -- test_filesystem_module.py alone gives 103 passed plus the one known red test -- so this is a cross-test interaction, not a single bad test. Reproduced on macOS/py3.12; unknown on the Linux runners.

Second blocker, independent: test_runtime_package_boundary.py shells out to 'python -m build' and fails with "Backend 'setuptools.build_meta' is not available". That is environment-dependent and would need the build backend present on the runner, or those tests marked and excluded.

CONSEQUENCE FOR THIS TASK: adding the tree to the platform-mcp-core shard is NOT the one-line change the original description assumed. The gating path is:
  1. Reproduce the native abort on a Linux runner (it may be macOS-only).
  2. If it reproduces, shard the tree so no single session runs all 3,351, or isolate the interacting tests.
  3. Decide on test_runtime_package_boundary.py: install the build backend on the runner, or mark those tests and exclude them.
  4. Fix or quarantine test_filesystem_glob_marks_file_size_unavailable (red since 2026-06-03, finding mcp-unified-5, unguarded is_symlink at filesystem_module.py:1778).
  5. Only then add the paths to the shard.

Related evidence: the shard-coverage guard reports baseline=130 test files already grandfathered as unshared repo-wide, so this tree is the largest instance of a standing problem rather than a one-off.
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
