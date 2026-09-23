---
id: TASK-13291
title: 144 MCP_unified test files run in no CI job
status: In Progress
assignee: []
created_date: '2026-09-22 04:34'
updated_date: '2026-09-23 23:10'
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
- [x] #1 The in-app tree is run locally and every current failure is triaged and recorded before any gating change
- [x] #2 The known red test (filesystem glob / unguarded is_symlink at filesystem_module.py:1778) is fixed or explicitly quarantined with a reason
- [x] #3 tldw_Server_API/app/core/MCP_unified/tests is added to the platform-mcp-core shard (ci.yml:1758-1761 and its four siblings) or another required gate
- [ ] #4 A green run of that gate is recorded with its observed output
- [x] #5 The naming collision between tests/MCP_unified and app/core/MCP_unified/tests is documented so the next reader does not assume the gate covers both
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE. Two parts.

1) FIXED THE RED TEST. filesystem_module.py:1778 candidate.is_symlink() was unguarded, 15 lines above the correctly guarded stat(follow_symlinks=False) at :1797. Path.is_symlink() lstats and re-raises EACCES/EIO/ESTALE, so one unreadable entry aborted the ENTIRE fs.glob walk instead of degrading that entry. Now wrapped in except OSError with a debug log, falling back to the directory entry kind - which is exactly the shape test_filesystem_glob_marks_file_size_unavailable asserts. Red since 2026-06-03 (5009fc8b95) across 28 commits to that file; now green.

2) WIRED THE TREE INTO CI AS ITS OWN SHARD, not appended to platform-mcp-core. Verified why: running the in-app tree in the same process as tests/MCP_unified makes test_rag_module::test_rag_module_jsonrpc_tools_call_smoke and test_gateway_protocol_stdio::test_default_adapter_preserves_globals_and_closes_only_duplicated_fds fail through cross-test state pollution, while both pass in isolation (77 passed). That is the singleton/lifecycle isolation class from the 2026-07-04 audit. A separate shard avoids it AND leaves the currently-green platform-mcp-core byte-identical - the ci.yml diff is 110 insertions, 0 deletions.

VERIFICATION: extracted the exact paths value from the parsed YAML and ran the real shard command locally. platform-mcp-inapp => exit 0, 2948 passed, 0 failures, 2m14s. All 5 duplicated matrix copies updated identically (the matrix is duplicated 5x in ci.yml - a separate problem).

QUARANTINE: 10 files were ALREADY red when the tree was wired in (21 entries total) and carry --ignore with a comment saying the list may only shrink. 13 of those 21 are the two distribution-building files, which build sdists and shell out and do not belong in a unit shard anyway. The other 8 are genuine product/test drift accumulated while the tree was unwatched - e.g. refresh_token() missing a required positional argument, a changed result shape (KeyError: rows). Follow-up task filed to drain the quarantine.

2026-09-23 reconciliation: AC1 met (prior note records the local run and triage: 21 red entries in 10 files, categorized; quarantine since fully drained by TASK-13343, 9c21dc31bf/64c253c0c3). AC2 met (filesystem_module.py:1777-1789 wraps candidate.is_symlink() in except OSError; test_filesystem_glob_marks_file_size_unavailable passes today). AC3 met (own shard platform-mcp-inapp added beside platform-mcp-core in all 5 matrix copies of ci.yml, e.g. :1771; runs the whole tree with no --ignore). Caveat: like platform-mcp-core, the full-suite shards feed 'Full Suite (Ubuntu / Python 3.12)', not one of the six rulesets checks in CI_REQUIRED_GATES.md (backend-required only runs tests/unit). AC5 met (ci.yml comment at :1761-1770 explains the in-app tree vs tests/MCP_unified and why they are separate shards; not mirrored in Docs/). AC4 NOT met: only a local run of the shard command is recorded (2948 passed, and that was with the since-removed quarantine); commits 7c348a05ae/9c21dc31bf are not on any remote branch, so no CI run of platform-mcp-inapp exists yet. Bandit on filesystem_module.py: no issues.

2026-09-23 (post-quarantine-drain) local re-run of the exact CI shard command (DATABASE_URL=sqlite:///./Databases/users.db, TEST_DATABASE_URL/POSTGRES_TEST_DB unset, -p pytest_asyncio.plugin -m 'not jobs and not e2e', PYTEST_DISABLE_PLUGIN_AUTOLOAD=1) on tldw_Server_API/app/core/MCP_unified/tests at ea1cbc6941+: exit 0, 3343 passed, 14 skipped, 0 failed, wall 179s serial; slowest single test 2.9s. Sizing: ~3 min locally vs the 60-minute shard timeout, so one shard is ample even at a 5-10x CI slowdown; no split needed. check_shard_coverage.py: OK (new_uncovered=0). AC4 still open: it needs a real CI run, which needs this branch pushed (commits are local-only; gh run list for fix/core-module-review-batch-1 returns nothing). Not pushed by this agent - pushing is the owner's call. Once pushed, record the 'Full Suite shard (Ubuntu / Python 3.12 / platform-mcp-inapp)' result here and close.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
