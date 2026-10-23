---
id: TASK-13275
title: Progressive disclosure for the chat composer toolbar
status: Done
assignee: []
created_date: 2026-09-19 14:47
updated_date: 2026-09-19 14:47
labels:
- chat
- webui
- density
- ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured 63 visible interactive controls on the default chat screen before any conversation exists, of which the composer toolbar alone contributes a row of 13. The first-run task is one control: type a sentence and press Enter. Three nested disclosure layers already exist inside the composer, so a user hunting one control may open several independent disclosures. The prior audit measured 93 controls, so density is improving and the remaining reduction is layout-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The default first-run composer presents the message field, model, attach and the slash hint.
- [x] #2 Remaining tools are reachable behind one labelled disclosure.
- [x] #3 A returning user keeps whatever disclosure state they last chose.
- [x] #4 Toolbar groups carry a visible separator, not only a screen-reader group label.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-23: User requested an Impeccable redesign/re-imagining of the /chat composer. Inspecting the current shared Playground composer, toolbar, optional V1/V3/V5 shells, and responsive behavior. Design direction and user approval pending; no implementation edits yet.
Incumbent inspected at desktop 1440x1000 and mobile 390x844 using the local Next.js app in its existing offline test-bypass mode; no chat sends or backend mutations. Evidence: .playwright-mcp/composer-before-desktop.png and composer-before-mobile.png (local runtime artifacts). Findings: a separate options chevron plus Advanced controls disclosure; nested visual shells; persistent empty token/status chips; mobile cockpit suppresses the toolbar including the model selector while retaining Improve prompt, delivery and Buddy controls. Proposed direction pending user confirmation: one integrated editing surface; persistent model, attachments, labelled Tools and send; conditional active-context strip; one grouped Tools tray; preserve draft, IME/slash/mention, queue/stop, persistence, voice and warning behavior. Implementation scope: shared PlaygroundForm, ComposerTextarea, ComposerToolbar/overflow, relevant locales and existing unit/mobile/keyboard regression tests; inspect optional V1/V3/V5 shell compatibility. ADR assessment: no new ADR expected for local composer presentation and disclosure changes using existing state and actions; no durable architecture rule proposed.
User selected the recommended calmer writing-space direction. Implementing in managed worktree /Users/macbook-dev/.codex/worktrees/chat-composer-calm/tldw_server2 on codex/chat-composer-calm. Baseline: ComposerToolbar and ComposerTextarea focused Vitest suites passed, 37 tests. ADR required: no; local presentation/disclosure changes preserve shared WebUI/extension state, actions, APIs, storage and dependencies.
Implemented the calmer composer in the managed worktree. Verification: 71 focused Vitest tests passed across 8 files; TypeScript completed with exit 0; scoped ESLint has 0 errors and no new findings versus HEAD (existing warnings 79 -> 77); Impeccable detector returned []; Bandit completed with no findings (touched implementation is TypeScript, no Python scanned). Eight browser checks passed, including persisted disclosure and V1/V3/V5 delivery controls. Initial screenshot review found mobile focus mode hides the theme toggle, so the fixture is being corrected to assert the actual theme before final light/dark capture. Fresh code review scored all three context/action/audio-focus findings resolved with red/green regression tests.
Final verification: 71 focused Vitest tests passed; all 8 Playwright checks passed after asserting actual light/dark themes at 1440x1000 and 390x844, with composer-scoped axe, Escape/draft behavior, preference reload, and optional V1/V3/V5 shell checks. TypeScript exit 0; git diff --check passed; Backlog canonical check passed. Fresh code-review verdict: all 3 findings resolved. Fresh Impeccable finish review (fallback rubric in a clean subagent): ship, no material fixes. Documenter checked all 8 captures, scoped components and incumbent tokens/metadata; existing documentation accurately describes the result, no edits. Design: Docs/Design/chat-composer-calm.md. No global theme/token change. Temporary preview aliases removed before commit. Live provider and microphone hardware flows and the full backend suite were not run.
User follow-up: remove the dedicated command-hint/persistence footer row. Keep / commands · @ mentions beside attachment/Tools, and Saved/Temporary beside model/delivery within the existing action bar. Preserve responsive wrapping and privacy behavior; verify actual row geometry at desktop/mobile. This is a bounded layout correction; no architecture change or ADR required.
Compact-row follow-up complete: command hints moved beside attachment/Tools; Saved/Temporary moved beside model/delivery. Removed the separate metadata footer, so the desktop offline composer has three rows. Mobile hints fit beside Tools with the redundant icon hidden at small widths, preserving touch targets and the existing two action rows. Browser regression reproduced the original separate-row problem (54px center spread), then verified shared vertical row bounds in dark/light desktop/mobile. Final: 29 existing toolbar tests pass; all 8 browser tests pass; TypeScript exit 0; ESLint 0 errors and the one existing any warning; Impeccable layout detector []; Bandit no findings (TypeScript scope, no Python scanned); git diff --check passed. Updated Docs/Design/chat-composer-calm.md. One optional-shell check timed out during local HMR in the earlier run; all optional shells passed in the final stable run. Temporary preview aliases removed. Live backend/provider and microphone flows were not rerun for this layout-only follow-up.
User follow-up: move dictation/voice actions into the existing controls row; regroup Tools content into intentional task rows, including adjacent Prompt and System prompts. Preserve mounted audio controls and focus behavior. Planned rows: sources together; Modes/MCP together; Prompt/System prompts/assistance together; character selection then Buddy/role-play setup; response settings/presets together, output formatting/OpenUI together, remaining tools/compare together. Bounded presentation change; no new ADR required.
Tools grouping/audio follow-up complete: moved the stable dictation/voice container into the model/delivery row, preserving focus and active-stop access when Tools closes. Mobile dictation uses the microphone icon with its existing accessible name. Sources share a row; Modes/MCP share a row; Prompt, System prompts and prompt assistance share a row with a visual separator; character selection and Buddy/role-play use intentional rows; settings/generation/compare are together and formatting/OpenUI/output tools are together. No action callbacks or feature capability rules changed. Browser red check reproduced the separate audio row; an initial post-edit run served the stale toolbar until the local preview restarted. Final stable run: all 8 Playwright checks pass, including actual row geometry for audio and related Tools actions in light/dark desktop/mobile, axe, draft/Escape, reload persistence, and optional V1/V3/V5 shells. 31 component/integration tests pass, including audio start/stop focus; TypeScript exit 0; scoped ESLint 0 errors with the one existing any warning; Impeccable layout detector []; Bandit no findings (TypeScript scope); git diff --check passed. Viewed final desktop dark and mobile light captures in one bounded review; saved all 8 captures under .impeccable/review/regrouped-tools. Temporary preview aliases removed. Live provider/microphone hardware and full backend tests were not run for this presentation-only follow-up.
Published as draft PR https://github.com/rmusser01/tldw_server/pull/3309 from clean branch codex/chat-composer-pr. The branch isolates only the three reviewed composer UI commits from common UI baseline 590e7e5794; unrelated RPG ancestry is excluded. Fresh isolated checkout: 63 focused tests passed and TypeScript noEmit passed using existing dependencies/generated API types. Newer dev changes conflict with the composer, so reconciliation and the requester-authored Change summary are required before readiness/merge. Active header workspace and original branches were left unchanged.
2026-10-23 follow-up: user requested rebasing composer PR #3309 onto latest dev and replacing Qodo with a fresh manual subagent review. Preserve the accepted composer layout and newer dev draft/preparation/persistence fixes. Perform integration, focused checks/manual review, then update the PR. ADR required: no; integration only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The /chat composer now has one calm writing surface and a remembered Tools tray. Command hints, persistence and audio controls share existing action rows. Tools actions are grouped by task, with Prompt and System prompts adjacent, people controls together, generation settings together and output tools together. Preserved drafts, context, privacy locks, starter events, active audio stopping/focus and optional shells. Latest follow-up: 31 component/integration tests and all 8 browser checks pass; TypeScript passes; scoped lint has no new findings.
<!-- SECTION:FINAL_SUMMARY:END -->
