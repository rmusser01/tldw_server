# Claims Stage2C Publication

Tracking: TASK-15465. User authorized rebase, verification and PR creation;
merging is not authorized. Work only in the existing isolated Claims worktree.
ADR check: no new durable rule; ADR094 (resource admission/publication),
ADR003/058/082 govern. SDK ADR095 remains Proposed. Renumbering preserves
accepted rationale and other owners' task/ADR records.

## Stage 1: Integrate Current Dev
**Goal**: Rebase the completed branch and repair private namespace collisions.
**Success Criteria**: Current fetched dev is an ancestor; both resource and
enrolled-workflow guards retained, unique canonical tasks/ADRs.
**Tests**: Conflict inspection, task/ADR audit, compile and ID guard.
**Status**: Complete

## Stage 2: Fresh Qualification
**Goal**: Qualify Claims/Jobs and affected storage/API/config paths on the rebase.
**Success Criteria**: Official backend fixtures, scoped regressions, full-file
configured lint comparison, Bandit and applicable hooks recorded honestly.
**Tests**: Claims/Jobs; publication/snapshot/existing-only Media; Watchlists;
Config/router; Backlog; Chroma; full repository attempt with default gates.
**Status**: In Progress

## Stage 3: Publish The PR
**Goal**: Push the verified feature branch and create a PR against dev.
**Success Criteria**: Exact published head matches; no merge, gate bypass,
fabricated human Change summary or unrelated main-checkout change.
**Tests**: GitHub PR head/base/state verification, clean local worktree.
**Status**: Not Started
