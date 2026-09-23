---
id: TASK-13291
title: 144 MCP_unified test files run in no CI job
status: To Do
assignee: []
created_date: '2026-09-22 04:34'
updated_date: '2026-09-23 15:05'
labels:
  - ci
  - mcp
  - testing
dependencies:
  - TASK-13356
  - TASK-13357
  - TASK-13358
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
TRIAGE DONE (2026-09-23), which this task required before gating. Two fixes shipped; the
gate itself is blocked on 13 remaining failures, now all filed.

BASELINE, app/core/MCP_unified/tests on dev 91e8bbf: 14 failed, 8 errors, 3324 passed,
5 skipped. After the two fixes below: 13 failed, 0 errors, 3325 passed, 13 skipped.

FIX 1 -- 8 of the 22 were not regressions. build_standalone_distributions runs
`python -m build --no-isolation` under PIP_NO_INDEX=1, so every build-system requirement
must be installed WITH dist-info metadata. A venv where setuptools is importable but
unregistered makes the build fail with "Backend 'setuptools.build_meta' is not
available", and every dependent test ERRORed. test_runtime_package_boundary already had a
guard but called it at one site only -- its module-scoped fixture built without it, and
test_gateway_protocol_artifact_consumer had no guard at all. The check now lives in the
shared helper both build through. Verified conditional: skips when metadata is absent,
does not skip when present, so CI still runs these.

FIX 2 -- test_filesystem_glob_marks_file_size_unavailable, the test this task cites as
proof of the gap (red since 5009fc8b95, 2026-06-03), is green. The size block already
tolerated OSError, but candidate.is_symlink() above it was bare; on 3.12 that routes
through Path.stat(follow_symlinks=False), the same call the size block guards. One
unreadable entry aborted the whole fs.glob. Probed: removing the guard turns it red again.

TWO SECURITY/LICENSING FINDINGS, both invisible because of exactly this gap:
- TASK-13357 (high, supply-chain): mcp-unified-publish.yml's publish-pypi job has a
  `github.event_name == 'push'` branch requiring neither target == 'pypi' nor the
  confirm_publish typed string the manual path requires, the pypi environment has
  protection_rules: [] (no reviewers), and publish-testpypi has no push branch. So a
  merge to main that bumps apps/mcp-unified/pyproject.toml publishes to production PyPI
  with no confirmation and without staging to TestPyPI.
  test_mcp_unified_publish_workflow_is_manual_and_gated exists to prevent this and is red.
- TASK-13356 (high, licensing): apps/mcp-unified/LICENSE ships full GPL-3.0 text. The
  project is GPL-2.0 (9a34da262a) and the root LICENSE became a licensing-boundary
  document in da0ec87d7d (TASK-12976), so the shipped file matches neither.

REMAINING 13, all filed:
- 5 in test_runtime_package_boundary: 1 -> TASK-13356, 4 -> TASK-13357.
- 8 others -> TASK-13358, with a per-test triage table. Flagged first:
  test_flashcards_export_rejects_cross_workspace_card_in_apkg_path fails with
  KeyError 'rows' before reaching its isolation check, so that cross-workspace property
  is currently unverified in either direction.

THE SHARD CHANGE IS NOT IN THIS PR. Gating the tree requires the 13 resolved, and two of
them are owner decisions (what licence the published artifact carries; whether a
version-bump push is meant to publish). Adding the shard now would mean either a red
required gate or 13 xfails -- and batch-xfailing them would reproduce the invisibility
this task exists to end. Blocked on TASK-13356, TASK-13357, TASK-13358.
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
