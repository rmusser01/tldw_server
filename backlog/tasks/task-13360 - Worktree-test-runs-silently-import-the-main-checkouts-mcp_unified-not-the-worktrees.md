---
id: TASK-13360
title: >-
  Worktree test runs silently import the main checkout's mcp_unified, not the
  worktree's
status: To Do
assignee: []
created_date: '2026-09-23 16:04'
updated_date: '2026-09-23 16:06'
labels:
  - tooling
  - testing
  - dx
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Editable installs pin absolute paths into the main checkout, so tests run from a worktree import the wrong source. Detail in implementation notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A worktree test run either imports the worktree's own source, or fails loudly instead of silently using another checkout's
- [ ] #2 The guard covers every first-party editable install, not just mcp_unified
- [ ] #3 Worktree setup guidance records that re-running pip install -e from a worktree corrupts the shared venv for every other checkout
- [ ] #4 Verified by editing a file under apps/mcp-unified/src in a worktree and confirming the test run observes the edit or fails
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two editable installs pin absolute paths into the MAIN checkout, so any test run from a
worktree imports the main checkout's source rather than the worktree's own.

Verified 2026-09-23 from the worktree .claude/worktrees/core-review-fixes:

  $ cat .venv/lib/python3.12/site-packages/__editable__.mcp_unified-0.2.1.pth
  <repo-root>/apps/mcp-unified/src

  $ cat .venv/lib/python3.12/site-packages/__editable__.tldw_profile_core-0.1.0.pth
  <repo-root>/packages/tldw_profile_core/src

  $ python -c 'import mcp_unified; print(mcp_unified.__file__)'
  <repo-root>/apps/mcp-unified/src/mcp_unified/__init__.py

The worktree has its own apps/mcp-unified/src/mcp_unified/, and it is not the one imported.

EFFECT: editing apps/mcp-unified/src/ or packages/tldw_profile_core/src/ in a worktree and
running its tests exercises UNMODIFIED code from the main checkout. Silent in both
directions -- a fix appears not to work, or worse, a test appears to pass against code that
was never changed. A worktree run also picks up whatever uncommitted state the main
checkout happens to have, so results are not reproducible.

SCOPE: 36 test files under app/core/MCP_unified/tests import mcp_unified directly
(TASK-13291 counted 35). Everything importing tldw_profile_core is affected the same way.
Invisible unless someone checks __file__.

OPTIONS, none free:
1. Per-worktree venv. Correct and fully isolating; costs disk and setup per worktree.
2. Re-run pip install -e from inside the worktree. Fixes the path but BREAKS the main
   checkout and every other worktree, since the venv is shared -- a footgun, not a fix.
3. Make the shared venv's path entries relative or resolved at import time. Not supported
   by the editable-install mechanism.
4. A conftest guard that fails loudly when a first-party package resolves outside the
   current rootdir. Does not fix it, but converts a silent wrong-source run into an
   immediate, legible error. Cheapest real mitigation.

Option 4 plus documenting option 1 is probably the answer; option 2 must be called out as
unsafe wherever worktrees are documented.

Found while triaging TASK-13358: two of those tests turned out to be environment-dependent,
which prompted checking where the imports actually resolve.
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
