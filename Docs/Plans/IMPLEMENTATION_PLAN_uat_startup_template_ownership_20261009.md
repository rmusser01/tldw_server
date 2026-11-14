# Saved startup template account isolation

Associated task: TASK-13260.278.18.83.45. The root task recorded this source-backed follow-up before edits. Existing Prompt privacy repair and real cross-account leak motivated inspection; saved bundles copy private Prompt content and numeric Studio identity without ownership, and apply continues after awaited identity changes.

ADR assessment: No new ADR. Reuse existing captured Prompt owner identity and canonical client ownership. ADR-055 governs backend-neutral Studio persistence; no backend architecture change.

## Stage 1: Reproduce stored-copy and delayed-apply failures
**Goal**: Cover private bundle parsing/visibility, preserved local legacy rows, owner stamping, retained unrelated storage, and retirement after awaited identity application.
**Success Criteria**: New tests fail on current production for demonstrated ownership boundaries; existing local mapping behavior retained.
**Tests**: New pure and rendered hook tests using actual bundle codec and hook.
**Status**: Complete

## Stage 2: Minimal ownership repair
**Goal**: Store opaque owner on new authenticated copies; hide ambiguous unowned Prompt-linked legacy bundles without deleting them; scope preview/edit/delete/apply; fence late apply on principal/config change. Preserve local IDs and numeric Studio IDs.
**Success Criteria**: New failures pass, original affected bundle/role-play tests pass in UI and WebUI, touched lint and syntax clear.
**Tests**: New causal tests plus existing bundle mapping, integration, role-play helper/hook and rendered role-play consumers.
**Status**: Complete

## Stage 3: Review and live UAT
**Goal**: Independent actual source review, combined build, then real owner-switch and valid same-owner live recovery.
**Success Criteria**: Private copies never appear/apply under another account; valid shared local and same-owner behavior remains. Broad UAT acceptance remains open until actually exercised.
**Tests**: Actual browser workflow on reviewed combined candidate.
**Status**: In Progress

## Scoped evidence and current limits

- Initial real codec/hook baseline: 7 privacy failures and 1 valid control. First fixed scope: 24 passes across 5 files. Added ownership controls plus all existing rendered Form role-play consumers: 43 passes across 2 files. These are split UI evidence, not a combined 59-case run.
- First affected WebUI check: 59 passes across 6 files, natural exit 0 in 9.392 seconds outer.
- Stronger role-play foreign-copy preservation assertion then reproduced one actual failure: stored owners [alice] instead of [alice,bob]. Minimal repair uses all stored templates for the role-play upsert. Final genuinely affected 45-case/3-file hook and rendered Form scope: UI 45 passes (22.731 seconds outer), WebUI 45 passes (21.326 seconds outer), natural exit 0. Unchanged codec/helper cases were excluded from that final replay.
- Four touched TypeScript modules parse without errors. Normal ESLint: bundle codec 0/0; changed hook retains 6 identical inherited warnings with no errors; parent Form retains 68 identical inherited warnings with no errors; new tests 0/0. The existing Next pages-directory advisory is retained. TypeScript-only unit has no applicable Bandit scope. Combined configured semantic checks remain pending.
- Actual Prompt inventory parent binding and local UUID/numeric Studio mapping remain. Parent consumes the separately reviewed localOnlyReady authority contract; configured unknown/rejected ownership remains closed.
- Independent source review, combined build and actual same-owner/account-switch UAT are open. No runtime acceptance, publication or merge is claimed.

## Independent follow-up resolution

- The alleged principal/config late Drawer presentation race was withdrawn after tracing the actual module-level account watcher: clearSession increments restoreRevision, already checked by both captured assistant and Drawer destination guards. No Drawer production change was made. Three real Form/Drawer held-scene controls retain successor state on principal/authority changes and permit the equivalent-config continuation. Two initial fixture setup failures (missing actual PRESETS, then missing AntD Tabs export) were not causal defect evidence; the standard partial mocks corrected those dependencies.
- A distinct real shared-local save bug reproduced 2 failures with 2 authenticated-link controls passing: both Startup and Role-play saves became hidden after storage reload. The minimal correction stores known shared local selections as unlinked literal/system-template copies; configured authenticated UUID/Studio links remain, and ambiguous legacy links are still hidden and retained.
- Final genuinely affected configured UI and WebUI scopes each passed 83 cases across four files, natural exit 0 (17.012 and 21.789 seconds outer). Original rendered test bodies remain identical by TypeScript syntax-tree comparison (22 original declared bodies); three new Drawer cases use the actual Drawer with the actual parent guard. No original destination/rollback/no-write assertion was replaced.
- Current source and new tests parse without errors. Exact project ESLint rules applied from the apps base: hook 0 errors/6 identical inherited warnings, new owner test 0/0, rendered Form test 0 errors/25 identical inherited warnings. The first narrower lint base ignored these files and is not qualification. Existing Next pages discovery advisory and fixture chat-settings diagnostics are retained. TypeScript-only delta has no applicable Bandit scope.
- Root source is frozen for final peer review. Combined semantic types, build, live account/local recovery and broad UAT acceptance remain open.
