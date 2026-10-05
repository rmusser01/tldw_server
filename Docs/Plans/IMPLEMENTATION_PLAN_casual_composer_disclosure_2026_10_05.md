# Casual composer disclosure — TASK-13455.4

Baseline: `242f7aa268`; latest `origin/dev c2260945ee` freshly verified. Approved design: `Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md`, Casual and Pro section. This bounded Stage 2 child changes shared full-page presentation and existing editor entry points, not settings/persistence/native ownership. Compact sidepanel continuity remains Stage 4.

Keep the message, model, Attach, Send and Customize chat prominent. Reuse the existing explicit Casual advanced disclosure preference and existing controls, with task groups Research & context, Character & your role, and Response settings. Show active customization/warnings and active dictation/voice/Stop/queue/recovery when relevant outside the disclosure. The Casual active identity opens the existing setup without clearing it. Retain direct Pro controls and every moved capability. Preserve draft/settings on layout switches; provide visible trigger/fallback focus, one-layer Escape behavior and existing nested dirty/Apply guards. Ensure the full-page narrow/cockpit shell does not hide Casual primary controls. Experimental composer variants retain the same slot/state authority. Actual English and flattened packaged message entries must agree.

## Stage 1: Reproduce Casual hierarchy and focus failures
**Goal**: meaningful mounted real-translation/settings/UI tests for the current crowded layout and hidden entry points.
**Success Criteria**: primary/disclosed controls, active identity non-mutation, persisted explicit disclosure, narrow/collapsed behavior, mode/draft preservation and focus gaps fail before implementation.
**Tests**: actual toolbar, English/AntD/settings paths with domain editor/IO doubles; actual shell integration where practical; retain existing capability and Pro assertions.
**Status**: Complete

## Stage 2: Implement task-based disclosure and active summaries
**Goal**: concise primary workflow with all existing secondary editors and clear active state.
**Success Criteria**: every moved action remains reachable in its task group, Pro direct capability retained, active effects visible, no settings/source/draft mutation by presentation changes, full-page mobile/cockpit Casual launcher reachable.
**Tests**: mounted Casual/Pro, narrow/collapsed, context/identity/tools/formatting and active speech controls; relevant shell/roleplay integration.
**Status**: Complete

## Stage 3: Verify keyboard layers and compatibility
**Goal**: predictable Escape and focus with current child popovers/drawers and retained settings.
**Success Criteria**: close returns to visible Customize or current initiating control, nested layers close one at a time, mode changes cannot strand focus or steal another control, dirty/Apply safeguards unchanged.
**Tests**: actual UI/keyboard and held preference reads, nested editor lifecycle, mode switch/draft tests, shared English parity, frontend typecheck and no added touched lint.
**Status**: Complete

## Stage 4: Review, record and commit
**Goal**: incremental verified delivery with both authorized independent reviews.
**Success Criteria**: no remaining substantiated Critical/Important finding, task/plan/diff checked, coverage limits explicit, only previous own plan removed after archival at `242f7aa268`.
**Tests**: bounded relevant regression and security validation; Bandit N/A for TypeScript/docs-only production scope. Native geometry, VoiceOver, live journeys and whole Stage 2/program remain open.
**Status**: Complete

Red-first evidence: `/private/tmp/casual-composer-disclosure-red.log`, 15 failed / 4 passed before implementation; missing task disclosure/model reachability and destructive Casual identity entry point reproduced.

Integration red cases reproduced actual outer-arrow/no-op, embedded-context Escape, unmounted starter listener, preset selection semantics and popup parent bubbling (6 failed / 80 passed). Fixed with retained hidden existing selectors, actual open notifications, Pro-only outer arrow, separate true overlay state, selected-state semantics and child-owned Escape. Preference read/write failure checks exposed a permanently gated stock hook; the narrow adapter preserves the same old key and provides session-only operation without an unread-preference write.

Popover check reassessment: three assertion attempts exposed jsdom's unfinished AntD leave animation rather than a still-open state (`ant-zoom-big-leave-active`). Reviewed actual AntD motion token, existing popup lifecycle and cached-node behavior; alternatives were driving animation-end, asserting only the controlled state, or using AntD's supported reduced-motion configuration. The keyboard-layer test uses `ConfigProvider` motion=false for a deterministic actual widget lifecycle; native/default animation and geometry remain Stage 4/5 gates. The held watch test initially deadlocked by intercepting the shim's own set-before-get old-value read; it now holds only the initial get and verifies the newer watch value wins. No production animation override or mocked popup was introduced.

## Verification and review — 2026-10-05

- Final affected regression: **13 suites / 268 tests passed** (`/private/tmp/casual-composer-disclosure-verified.log`); final selector suite after test-only lint correction: **41 passed** (`/private/tmp/casual-composer-selector-return-settled.log`). Frontend typecheck passed and touched-scope lint has no added findings relative to `242f7aa268`. Extension entrypoint compile passed earlier; this is not a packaged build or native journey claim. Eight English source entries match the flattened packaged playground namespace. Diff whitespace and exact task normalization checked.
- Independent reviews reproduced watched-hidden focus, actual portalled-picker return, unrelated task focus, and the explicit cockpit selector's second unchecked animation frame. Focus repair qualifies current owned focus, defers to ordinary picker return, focuses the visible Customize control when that return is hidden, and retires picker returns on reopen/unmount. Explicit-selector regressions were three red before implementation. The first green rerun's reopen assertion wrongly expected body focus; it now allows the new picker's legitimate search autofocus, and the settled full suite passes. The final test-only no-unused-expression lint finding was corrected before completion.
- Both authorized independent reviewers report no remaining substantiated Critical/Important finding. The actual toolbar, shell, AssistantSelect, translations, storage shim, preset semantics and AntD popup paths are exercised with domain transport/settings/store doubles as recorded above. The draft harness proves toolbar non-mutation, not full draft durability. VoiceOver, default-motion/native 320/390/200% geometry, real backend/provider journeys and full extension packaging remain later acceptance gates.
- The existing import-sort Prettier plugin crashed on the new test's AST. Existing shared formatting options without that plugin formatted the focused files; no formatter/config/dependency was introduced to the repository. Bandit N/A: TypeScript/docs-only production scope. Previous own character-session plan removed after archival at `242f7aa268`.
- Fresh fetch discovered `origin/dev 025627214c` (cached-dialog settings preservation) after these checks. Commit this verified unit then update the isolated branch and verify affected settings/composer paths before continuing. The whole Stage 2/program is open: combined discovery/offline mirrors, durable native contracts, roleplay setup/greeting, extension continuity and accessibility/live journeys remain.
