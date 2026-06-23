## Stage 1: Confirm Conflict Scope
**Goal**: Verify the current `dev` to `main` PR conflict surface.
**Success Criteria**: `origin/main` into `origin/dev` merge preview identifies the conflicted files and no unrelated worktree changes are present.
**Tests**: `git status --short --branch`; `git merge-tree --write-tree origin/dev origin/main`.
**Status**: Complete

## Stage 2: Resolve With Dev Precedence
**Goal**: Merge `origin/main` into the PR head while preserving `dev` for overlapping conflicts.
**Success Criteria**: The merge completes with `dev` content retained for conflicted `README.md` hunks and no unresolved paths remain.
**Tests**: `git merge origin/main -X ours`; `git status --short`.
**Status**: Complete

## Stage 3: Verify And Push
**Goal**: Confirm the merge commit is clean enough to update PR #1982.
**Success Criteria**: Conflict markers are absent, whitespace checks pass, and GitHub reports PR #1982 no longer dirty after pushing to `dev`.
**Tests**: `rg '<<<<<<<|=======|>>>>>>>' README.md`; `git diff --check`; `gh pr view 1982 --json mergeStateStatus,statusCheckRollup`.
**Status**: Complete

## Stage 4: Repair Current PR Check Failure
**Goal**: Address the MCP Unified Internal RC failure observed after conflict resolution.
**Success Criteria**: The standalone `mcp-unified` wheel includes all source packages required by gateway imports, including policy grants.
**Tests**: targeted MCP package-boundary pytest; `PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python make mcp-unified-rc`.
**Status**: Complete
