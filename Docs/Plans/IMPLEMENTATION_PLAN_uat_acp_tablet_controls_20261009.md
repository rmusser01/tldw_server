# ACP tablet session and tools controls

Associated task: TASK-13260.278.18.83.45. Root recorded the distinct 858×981 live failure and before-edit scope. The healthy empty chat had no usable Create control: the nonmobile layout starts at 768px, while header toggles and sidebars stayed CSS-hidden below 1024px and toggle handlers only opened drawers on mobile.

## Design and ADR assessment

ADR required: no. Reuse the existing useDesktop hook, which matches the CSS lg breakpoint exactly at 1024px, and existing session/tools drawers. [ADR-016](../ADR/016-acp-session-and-orchestration-persistence.md) remains governing; no runner, persistence, ownership, workspace, connection, permission or API contract changes. The header exposes translated show/hide labels and pressed state for the currently applicable pane/drawer. Mobile keeps its tabs and hidden toggles; recovery keeps its hidden toggles. Desktop retains persisted pane preferences. Root approved adding only useDesktop exports to three affected parent test mocks; the prior C04 selection production and all 13 test bodies/assertions remain unchanged.

## Stage 1: Causal mounted regression
**Goal**: Reproduce tablet pane-toggle failure before production edits.
**Success Criteria**: At the actual 858px viewport query, clicking the real header's session control fails to open the existing native Drawer/Create form.
**Tests**: Real width-aware matchMedia boundary, production media hooks, parent/header/drawers/SessionPanel/Form/Tools/query/store/REST; external fetch/config/storage/WebSocket and unrelated chat/workspace/permission panels only.
**Status**: Complete

## Stage 2: Minimal responsive correction
**Goal**: Route non-desktop pane toggles to existing drawers and expose accessible nonmobile controls.
**Success Criteria**: New tablet and 767/768/1023/1024 boundary cases plus genuinely affected parent/header/recovery tests qualify in normal UI/Web configs; preserved assertions and no new static findings.
**Tests**: New responsive module; connection, health-availability, launch-agent-selection, Header and Recovery modules. Exclude unchanged standalone modal/SessionPanel controls. Node20.19.5/Vitest4.0.18, 8GB heap, 600s outer/300s case bounds; max three issue attempts.
**Status**: Complete

## Stage 3: Freeze and independent review
**Goal**: Self-review complete hunks/new bodies and hand off stable hashes for peer/root review.
**Success Criteria**: Scoped baseline/current lint and TypeScript syntax; Bandit N/A for TypeScript. Root owns final global types/build/live 858px acceptance. Retain exact pre-unit bytes only in memory; no proof files or runtime/parent-task/Git/provider actions.
**Tests**: Peer source review; root actual responsive UI acceptance. JSDom media-hook and native control coverage cannot establish browser CSS layout or actual runner/auth behavior.
**Status**: In Progress

## Attempt ledger

1. Causal red on unchanged production: selected observed-width case failed naturally (one failed / six explicitly filtered), exit 1, Vitest 3.17s / outer 3.452s. Clicking the actual session control updated the pane preference but no Sessions Drawer opened. The failure is at the intended handler/Drawer boundary, not an accessible-label or initial visibility fixture. JSDom does not apply the shipped responsive Tailwind styles; the root's actual live DOM establishes the additional CSS-hidden symptom. Exact pre-unit bytes and scoped lint remain in memory.
2. First affected UI green invocation: 40 passing / one failing / 41 total, natural exit 1, Vitest 19.85s / outer 20.066s. The failing observed-width case reached the real Drawer and correct translated/pressed state, then could not find a Create dialog by its accessible name. Installed @rc-component/util/rc-util useId intentionally returns the same test-id in NODE_ENV=test; the nested Drawer and Modal title linkage collides. Two existing nested-modal test patterns locate the actual dialog by its unique form field. Apply that bounded locator correction, retain native title/visibility/Form/Cancel/close/noPOST assertions and all 40 passes; do not alter framework IDs or production.

Connected caller finding recorded before its repair: the tablet Tools drawer child onHide also writes rightPaneOpen(false), changing a saved desktop preference. Root confirmed this is within the original preservation contract. Add a real in-panel Hide regression before removing only that write. Keep the desktop Tools onHide and the existing native Drawer X behavior unchanged. This distinct caller issue gets its own causal invocation; final bounded qualification covers only the failed nested-modal fixture plus the new caller, then first full Web scope.

3. Connected Tools caller causal red on the otherwise corrected production: the selected real in-panel Hide case failed naturally (one failed / seven explicitly filtered), exit 1, Vitest 2.41s / outer 2.656s. The Drawer closed, but the actual storage boundary received [acp-playground-right-pane, false]. Remove only the tablet child preference write; retain the desktop onHide behavior. This is a production preservation failure, separate from the nested-modal fixture above.

4. Final bounded qualification: UI selected only the earlier failed observed-width case and new Tools Hide case. Both passed naturally, exit 0, Vitest 3.62s / outer 3.865s; six explicitly filtered. Retaining the preceding 40 passes yields 42 distinct UI passes across invocations, not one combined 42-case green run. The first Web run covered all six affected modules, 42/42 passed naturally with no skips, exit 0, Vitest 29.43s / outer 29.813s. Native Antd CSS parser diagnostics remain visible in JSDom; no assertion or production workaround was introduced.

## Frozen qualification scope

Two production paths: ACPPlayground/index.tsx and ACPPlaygroundHeader.tsx. Three existing parent tests add only the required useDesktop mock export: connection, health-availability and launch-agent-selection; exact pre-unit/current comparison confirms every other byte and all existing assertions remain unchanged. The complete new tablet-controls module has eight cases: effective Drawer/Create/Cancel at 858px, 768/1023 boundaries, native Tools Drawer X, actual Tools in-panel Hide, desktop persisted toggles at 1024, mobile tabs/Create at 767 and unavailable recovery. The real media hooks execute through a width-aware matchMedia boundary; actual Header/Drawer/Tabs/SessionPanel/Create form/Tools/query/store/REST are retained. External config/storage/fetch/WebSocket and unrelated panels are doubled. No session creation POST occurs in the new controls.

Scoped baseline ESLint on five paths and current ESLint on six paths both have zero errors/warnings. TypeScript syntax parsing passed for all six changed TSX files. Bandit is not applicable to the TypeScript-only scope. Self-review read every current production hunk and all eight new bodies; desktop child onHide, mobile/recovery toggle hiding, session readiness/permissions/workspace and prior C04 selection production are unchanged. Exact retained pre-unit bytes and unified deltas are held only in memory.

Production/test source freeze confirmed 2026-10-09 05:31:31 UTC. Independent peer review and root global types/build/live responsive acceptance remain open; configured JSDom controls do not establish responsive browser CSS, actual authentication or runner/session execution. No runtime, API, browser, profile, provider, parent-task or Git action was taken by this unit.

## Independent transition finding and scoped continuation

Peer review substantiated a distinct P2 after the first freeze: opening a tablet Drawer and resizing to desktop leaves its local open flag true. Installed Antd forwards className to the Drawer section while the mask/root/portal scroll lock remain governed by open/animatedVisible, so lg:hidden alone cannot retire the active overlay. Root recorded the separate before-edit task note and approved closing only local drawer flags when isMobile || isDesktop; return to tablet must remain closed and saved desktop preferences must not change. Root's completed pre-transition build/types are explicitly prior-source evidence, not qualification of this correction.

Before production edits, add two real Sessions/Tools transition cases. The test browser matchMedia boundary will deliver actual change callbacks to the unchanged production media hooks for 858→1024→858; retain all eight fixed-width bodies/assertions and their prior qualification. Test native Drawer closure/mask retirement, accurate header state, no storage write and no POST. This distinct transition issue has its own max-three-attempt causal ledger. Earlier source/test freeze is superseded only by the explicitly owned transition test/effect changes; no runner, API or preference architecture change is authorized.

Transition attempt 1: selected Sessions/Tools cases both failed naturally, exit 1, Vitest 4.86s / outer 5.167s; eight fixed-width cases explicitly filtered. Each real production media hook entered desktop state (header showed persisted open pane), but its actual Drawer root retained ant-drawer-open after the change event. Both failed at the intended retirement assertion, after native open Drawer/mask preconditions passed. Apply only the approved local-state closing effect; no preference writes, media hook changes or CSS workaround.

Transition attempt 2: normal configured UI current tablet module 10/10 passed naturally, exit 0, Vitest 8.25s / outer 8.441s. Sequential Web current module 10/10 passed naturally, exit 0, Vitest 11.35s / outer 11.627s. Both have no skips. The eight fixed-width bodies are byte-identical to the prior freeze; shared matchMedia fixture now additionally delivers real change subscriptions, so the current whole module was genuinely affected. Retain the prior 34 other affected parent/Header/Recovery/C04 cases without replay: 44 distinct qualified cases per client across invocations, not a combined 44-case run. Current scoped six-path ESLint remains zero errors/warnings and all six TSX syntax checks pass. Bandit N/A.

Final transition source/test freeze confirmed 2026-10-09 05:39:53 UTC: only the approved seven-line parent effect and test media-boundary/new two bodies differ from the first freeze. Exact memory comparison confirms all original eight test bodies and other production bytes unchanged. Peer narrow re-review, root fresh global types/build and actual 858px/resize acceptance remain pending; no earlier build is claimed for this correction.
