---
id: TASK-13271
title: Consolidate the three chat shortcuts surfaces and correct the documented binding
status: Done
assignee: []
created_date: 2026-09-19 14:47
updated_date: 2026-09-19 14:47
labels:
- chat
- webui
- consistency
- ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three controls whose names are variants of "shortcuts" open three different panels: a left-rail control opens the page navigator, a top-bar control opens the keyboard reference, and a chat toolbar control opens a third chat-specific panel. The chat panel documents Shift+/ as "Open keyboard shortcuts"; pressing Shift+/ actually opens the page navigator. The keyboard reference lists its own binding as a third value. Verified live by probing each control by accessible name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One concept has one name; navigation is not labelled as shortcuts.
- [x] #2 One keyboard-shortcuts surface exists, on one binding.
- [x] #3 Every advertised binding performs the action it documents.
- [x] #4 The chat-specific bindings appear as a section of the single surface.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-09: User authorized all identified chat issues and enhancements. Investigation during TASK-13270 found normal WebLayout mounts legacy KeyboardShortcutsModal while PageHelpModalHost and PageHelpModal also listen to the same shortcuts event, opening two dialogs. Consolidate under existing demand-loaded PageHelpModalHost, preserve explicit shortcuts/tutorial intent, remove the competing page-navigator ? binding and chat-specific help panel, label navigation by its purpose, and document the effective configured chat bindings in the unified surface. Test real events/keyboard/rendered dialogs, close/focus/draft and warm/cold opening. ADR required: no; presentation ownership and existing optional chunk behavior only; ADR049/050 conversation ownership unchanged, ADR059 tracking. Paired plan Docs/Plans/IMPLEMENTATION_PLAN_chat_tour_host_13270.md.
Visible shared role=dialog and role=alertdialog guards now govern both question-mark opening and Escape dismissal; actual Joyride and real stacked Command Palette controls qualify ownership. Hidden retained ancestor visibility reuses the existing helper. One existing global Help host owns tutorial/shortcut intent and configured binding documentation; page navigation has purpose-specific labels, full and compact scopes exclude unavailable commands. Both independent source reviews clear current ownership and target lifecycle delta; final lazy resource-load review/build/native gates still pending.
Independent source follow-up confirms remaining capability mismatches: header-hidden/independently hidden-sidebar full shells still register Sidebar/Quick shortcuts without their UI hosts, palette Switch Model targets absent shell settings; compact wide Pro forced docked sidebar cannot honor collapse, although overlay toggle works. Keep AC3 open until handlers/palette/docs share actual availability and Pro collapse is resolved. Normal CmdK works after waiting for real lazy dialog. This is distinct from unified reference ownership, which is qualified; do not mark whole shortcut task Done from the current matrix.
Unified ownership/reference core is verified with13270 and committed incrementally;AC3 stays open for host capability mismatch and compact docked Pro collapse. Own continuation plan Docs/Plans/IMPLEMENTATION_PLAN_chat_shortcut_capabilities_13271.md; source reviewers checking media hydration/mobile host edges before implementation.
Both independent source assessments clear the bounded capability design and identify cold mobile Drawer/initial media-state edges. Choose existing AntD Drawer forceRender={quickChatOpen} only on invocation to make its existing Quick Helper host reachable; use !hideHeader&&!hideSidebar consistently for Sidebar/Quick handlers, palette and Help, and !hideHeader for shell Model. Compact retains existing sidebarOpen as sole visibility, opens only on entering wide Pro, closes on leaving, respects explicit Collapse, and Find always opens it. Initialize media state from actual matchMedia to avoid narrow Pro being temporarily docked. No eager helper render, dependency, persistent setting or new owner rule.
Causal red:full/shared/Help6 failures with57 positives; compact10 failures with215 positives, plus isolated actual keyboard-collapse counterexample. Production reuses one full shell predicate across rendering/registration/palette/Help and Sidebar event, demand-renders cold mobile helper on invocation, and compact sidebarOpen honors collapse/default layout transitions/Find with actual initial media detection. Shared legacy open-event behavior retained. No owner/guard relaxed. Green qualification pending.
Independent final source review found creation recovery Find ignored by available Web legacy sidebar and hidden by Focus shell. Add causal legacy event and explicit Focus reveal regressions. Keep shortcut capabilities tied to actual hosts; reuse existing layout preference and existing route-scoped sidebar event. Source reviewers assessing bounded reveal lifecycle before production edits.
Native rebuilt dp25EEz-S-cRHCWz48i4r: header-hidden Help omits unavailable bindings, actual settings Save configures disposable account while preserving exact input, desktop CtrlB reveals actual shared Find, explicit existing event exits Focus/reveals Find with exact draft. Earlier hidden keyboard probe awaited inner Exit Focus instead of actual outer shell disappearance; Quick opened before hide, so invalidate that negative proof and repeat against real host visibility. Source trace found generic shortcut ref publishes current configuration in passive effect; add causal committed-layout disabled/rebound tests and publish via existing React layout effect. No source-owner guard or external storage change.
Final native Naxfx_HU63FLUMyL-8UST actual shared compact debug route exposed two real Search commands dialogs on MetaK: compact owns CommandPaletteHost while surrounding WebLayout also owns one despite _app hiding shell header/sidebar. Do not select the first dialog as workaround. Reuse existing exact compact debug route boundary to omit only redundant parent palette; preserve full Focus palette lifetime and compact own host. Add connected real host/rendered dialog causal regression before production repair; packaged extension remains separate qualification. Earlier native selector/report mistakes provide no passing receipt.
Second independent review of the same compact boundary found parent full New and child compact New both register document handlers. The same compact-owned predicate must disable the parent New registration as well as omit its palette; preserve normal full Focus New. Add an actual generic-hook sibling counterexample and positive full hidden-shell control, not an I/O-only mock.
Final native compact Find opens search but its focus is not stable: after two frames the palette close setTimeout restores the old header control. This is a real failure receipt, not passing Find qualification. Root CommandPalette unconditional delayed restore overrides a current search destination. Add connected real-host causal action-focus plus ordered timer test; restore opener only when removed palette left BODY active, retain existing Escape/stack positives. Broader source/user-focus owned return13568 remains separately required.
Expanded final palette qualification found 3 stale MCP Hub fixture selectors still asking for old Type a command or search placeholder; actual current labelled textbox is Search commands (already in HEAD). Preserve MCP/theme/shared-language assertions and target actual accessible role/name. The failed84-suite run is not credited as passing; rerun after fixture maintenance. No production search behavior or fixture assertion disabled.
Final qualification: 84 suites / 2,478 tests pass; both client compilers pass; 16 TypeScript scopes add no static findings. Owned formatting is runtime-equivalent, and both independent source reviewers clear the bounded changes. Build ytoMDnFW1ltWydegFVTvU has all 798 referenced assets, matching shared tokens and unchanged budgets (595.7 KB shared; 791.9 KB route). Native controls qualify compact Pro Collapse, keyboard, one palette, stable Find, narrow overlay Close and cold narrow defaults. Full Focus qualifies disabled actual hosts and no latent Quick action after the existing writer-event reveal. The real mobile helper opens from a closed React host with a ready exact draft; browser/chunk caches are warm. Original Casual mode and 1280x900 viewport are restored. The entire 25-row canonical response is unchanged, with no Send or generation. Failed selectors, earlier unstable Find and unawaited probes do not count. The broader old cockpit fixture remains TASK-13549; source/user-focus ownership remains TASK-13568. Physical AT, packaged extension, full persona journeys and participant acceptance remain open. Bandit is not applicable to TypeScript-only authored source/tests. Latest dev cd5160201cb32e05d254accdd8ac1377e1d164ad remains integrated. Master design and closure ledger are updated; completed plan is archived at /private/tmp/IMPLEMENTATION_PLAN_chat_shortcut_capabilities_13271_completed.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
One effective shortcut reference now matches actual hosts. Shared keyboard actions use committed configuration; compact Pro honors Collapse and Find; Web compact owns one New/palette. Palette closing preserves the chosen search destination and cancels stale return timers. Final connected/native/build checks qualified with exact drafts and canonical messages preserved; whole remediation remains open.
<!-- SECTION:FINAL_SUMMARY:END -->
