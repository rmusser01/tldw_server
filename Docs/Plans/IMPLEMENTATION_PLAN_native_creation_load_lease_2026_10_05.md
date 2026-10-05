# Native first-create load lease — TASK-13454.4

Baseline d3479250c6; fresh dev025627214c unchanged. Reuse the sender, draft binding and H1 preparation/publication guards. No new owner, store, API or retry policy.

## Stage 1: Reproduce held-load publication
**Goal**: real H1 loading regression through a mounted existing sender caller.
**Success Criteria**: settings/tools/Stop/New changes expose the current stale capture or loading owner; a valid first load remains the control.
**Tests**: held capture with actual H1/model/tool/draft stores and stated protocol/storage doubles.
**Status**: Complete

## Stage 2: Fence preparation and qualify retirement
**Goal**: full independent preparation lease reaches H1 publication, then ends on accepted bind.
**Success Criteria**: no late rows/binding/dispatch; no revocation after an accepted saved load; newer H1 remains untouched.
**Tests**: drift, cancellation, New, valid bind and newer navigation regressions.
**Status**: Complete

## Stage 3: Verify and locally commit
**Goal**: bounded delivery with current task and review evidence.
**Success Criteria**: relevant suites/types/extension compile/no added lint/diff/two independent reviews.
**Tests**: Bandit N/A to TypeScript/docs-only production. Unknown creation, explicit greeting UI/storage/provider/Sync and native/packaged/manual evidence remain later program gates.
**Status**: Complete

Final: red8/11 actual-H1 cases then317passing tests across8suites; no existing cancellation/no-auto assertion relaxed. WebUI types, extension entrypoint compile, zero added lint3files, clean diff, both independent reviews. Bandit N/A. Memory/protocol/config IO fixtures; no live account/browser-pointer/native/provider/creation rollback claim. Already-started bookmark IO may finish without publication. Durable unknown create and explicit greeting/Sync remain open. Own .3 plan archivedd3479250c6 and removed.
