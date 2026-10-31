# Knowledge Workstream Burndown Implementation Plan

> For agentic workers: execute assigned independent tasks under their Backlog IDs; parent owns integration, native UI, tracking and review.

**Goal:** Complete the remaining Knowledge workflow fixes and record actual validation coverage; correct TASK13514 to reflect merged PR3205.
**Architecture:** Reuse existing SQLite transactions, WebClipper/Media/Workspace APIs, Notes provenance and package test harnesses. Preserve original evidence and versions. No new source database, crawler, dependency or background refresh.
**Spec:** Docs/Design/2026-10-06-knowledge-followup-source-context.md; additional explicit web-capture policy requires a reviewed design.
**ADR check:** ADR020 governs content backends; ADR042 governs browser retrieval; ADR065 governs independent Notes provenance. SQLite/test repairs need no new ADR. The web-capture/refresh/version decision needs a new ADR after owner review.

## Stage 1: Baseline and tracking
**Goal:** Current dev isolation and independently tracked remaining units.
**Success Criteria:** dev26ae4fd679; primary checkout preserved; TASK13530/13530.1/13530.2 and existing13512/13514.1 linked.
**Tests:** Read current tasks/reports; red concurrent Sync trace; bounded frontend baseline.
**Status:** Complete

## Stage 2: Restore transaction and frontend contracts
**Goal:** Atomic Sync initialization, current frontend fixtures and one compatible extension React runtime.
**Success Criteria:** create_tables preserves caller transaction; strict malformed catalogs remain rejected; all32 reproduced Notes failures and runner errors are repaired without weakening production authority guards; WXT uses one React/ReactDOM18.3.1 pair satisfying existing OpenUI peers.
**Tests:** Existing test_sync_v2_store.py concurrent regression plus rollback/SQLite script cases; Stage5/20/26 and Notes locale contracts; relevant Notes tests/lint; actual bundle/runtime identity, existing peer contracts and CDP capture/save/scoped Ask. Final evidence: 866 Notes tests,224 QA tests,5 extension runtime/asset tests,types/build and cited local-model CDP answer.
**Status:** Complete

## Stage 3: Explicit external-web capture and refresh
**Goal:** Review and implement a capture/version policy through existing APIs.
**Success Criteria:** Explicit accepted capture/refresh; prior evidence retained; exact extraction/version identity; public retrieval security and owner/stale fences preserved.
**Tests:** Failed/restricted capture, cancelled preview, stale owner/workspace, duplicate retry, changed page/new capture and pinned historical preview.
**Status:** Not Started

## Stage 4: Native, assistive and participant validation
**Goal:** Complete actual available interactions and distinguish external coverage gaps.
**Success Criteria:** Native context menu→draft→saved Note→scoped Ask; spoken VoiceOver and Safari flows; actual mobile and novice/power-user sessions when participants/devices are available.
**Tests:** Real controls and API readback through CDP; spoken labels/focus/live updates; existing ten-task study protocol with observed human outcomes. CDP canonical Note/scoped Ask passed. Native-panel chat streaming and capture-origin wording remain qualified follow-ups.
**Status:** In Progress

## Stage 5: Review and accurate closeout
**Goal:** Verified integrated changes and truthful task/report status.
**Success Criteria:** Appropriate tests/type/build/lint/Bandit; independent review; TASK13514 marked merged/Done with proof; remaining external qualifications retain open tasks.
**Tests:** Canonical Backlog format; diff/hook checks; targeted affected suites and exact resulting artifacts.
**Status:** In Progress

Current release report: Docs/Reviews/KNOWLEDGE_BURNDOWN_2026_10_07.md. Stage3 awaits concrete design approval; Stage4 retains native-menu/spoken/device/participant gaps. Stage5 is in progress for verified fix publication and accurate tracking.
