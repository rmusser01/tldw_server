---
id: TASK-12135
title: Implement Chat Workspace inspector and status runtime state
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-10-04 04:02'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2033'
  - 'https://github.com/rmusser01/tldw_server/issues/1239'
  - 'https://github.com/rmusser01/tldw_server/pull/3159'
documentation:
  - >-
    Docs/superpowers/specs/2026-07-13-chat-workspace-hydration-offline-follow-up-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #2033 for /chat-workspace: replace placeholder inspector/status rail behavior with accurate runtime and workspace state, degraded/offline recovery copy, hydration/send-disabled safety, and focused component/browser coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chat Workspace remains non-ready and chat sends stay disabled while the workspace store is not hydrated, even when a workspace ID is present.
- [x] #2 Live-backend browser coverage verifies connected-to-offline rail transitions and suppresses stale streaming state.
- [x] #3 Live idle Chat Workspace rails show Select a model rather than Ready with no selected model and qualified workspace/history.
- [x] #4 Typed/staged sending is disabled without a selected model, dispatches zero RAG/chat inference, and retains drafts/staging; actual model selection restores readiness without an automatic send.
- [x] #5 Owning red/green regressions, production TypeScript/lint/build and exact-source no-mock desktop/mobile Chrome UAT verify the bounded model-readiness follow-up without degrading existing status precedence or optional-persona behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-chat-workspace-status-rails-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #2600 review remediation after rebase on latest origin/dev: added explicit hasModelSelected runtime state, made WorkspaceStatusStrip and InspectorRail require workspaceReady, verified selectedModelLabel is passed into InspectorRail, and strengthened stop-generation smoke coverage so delayed stream output does not land after abort. Verification: ChatWorkspace Vitest folder passed 8 files/74 tests; chat-workspace live-backend Playwright smoke passed 4/4; Stage 5 Chat Workspace release gate passed 1/1; git diff --check passed; UI package tsc --noEmit still fails on unrelated baseline outside Chat Workspace, and captured-log grep found no ChatWorkspace or rail errors; Bandit not applicable because no Python source changed.

2026-07-13 PR #2600 follow-up: addressing review findings for real storeHydrated readiness and the missing offline browser transition. Work will use TDD on the existing PR branch.

2026-07-13 PR #2600 follow-up complete. TDD evidence: the new non-empty-ID/storeHydrated=false page test failed before the readiness fix, then passed after ChatWorkspacePage derived one readiness boolean from hydration plus normalized identity. Added live browser coverage that starts streaming, transitions the real connection store to unreachable, and verifies both rails show server recovery while suppressing stale streaming state. Fresh verification: Chat Workspace Vitest 8 files/75 tests passed; frontend tsc --noEmit passed; live-backend Playwright 5/5 passed; focused ESLint exited 0 with only the pre-existing PERSONA_ID warning; git diff --check passed. Bandit is not applicable because the touched implementation and tests are TypeScript/TSX only.

2026-10-03 workstream re-check: GitHub2033 remains open. Prior hydration/offline work is retained as completed historical scope. New inherited idle no-model readiness gap is verified on latestdev4c4f197f: runtime classifier omits hasModelSelected and WorkspaceChatPanel sendDisabled omits model availability. Reopen this matching workstream task for a bounded follow-up; closedTASK12580 addressed different /chat aggregate-provider behavior. New task creation failed in CLI with Maximum call stack size exceeded; official MCP explicit-ID attempt timed out300s and verified noTASK13421.1.8 file was created. No manual task file edits or further blind create retries. Plan/source edits for this unit will be separate from PR3071 checkpoint correction and follow its qualification. Preserve offline/demo/bypass, hydration/history/error/streaming precedence, optional persona, staged sources/drafts, and no automatic sends; require TDD and no-mock Chrome raw-CDP desktop/mobile validation.

2026-10-03 requester directly approves the bounded missing-model design and recurring parent-PR follow-up. Execute TASK12135 separately on codex/chat-workspace-model-readiness-20261003 from published qualified4ec6b976d7 atop currentdevd7997bc205; PR3071 remains unchanged. Reuse existing normalized useSelectedModel value and shared runtime classifier/send guard; preserve higher-priority offline/demo/bypass/workspace/history/recovery/streaming/sending/error states, optional persona, staged context/drafts, explicit Auto/server routing and no automatic sends. TDD owning rails/panel, production types/lint/build, actual Chrome raw CDP desktop/mobile; no UAT mocks/interception/injected state. Current parent API backend remains unchanged and archived frontend remains separately source-qualified. Implementation proceeds only after this approved tracking checkpoint.

2026-10-03 model-readiness TDD: frozen regression RED5 expected failures/129passes; GREEN134passes in3 owning suites, full Chat Workspace273passes/14suites. Full production frontend tsc --noEmit exits0; actual repo-root scoped ESLint analyzes7files with0errors/0warnings/0ignored. Initial frontend-cwd lint ignores all7 files and is not qualification. Configured8file pre-commit checks pass; UI detector reports no findings. No Python source changes: Bandit is not applicable to this TS-only unit. Minimal4production-file fix forwards required hasModelSelected to shared classifier, retains status precedence, disables typed/context buttons and handler, and avoids duplicate Select-a-model pills; Auto and optional persona remain usable. Browser controls investigation first used wrong /models route (actual404, exact own poller7468 stopped; tab remains), then failed a stale prior-route conversation precondition before navigation. Both remain failed, not UAT passes; corrected source route /settings/model inspection and full native RED/GREEN are still required. No inference send has been performed by this unit. Separate parent3071 publishedhead remains4ec6b976d7 and is not modified by this branch; recurring30minute authorized follow-up is ACTIVE. New criteria and finalization remain unchecked pending native source qualification and PR review.

2026-10-03 final bounded model follow-up qualification: separate commit db29d8a19c rebased cleanly onto parent PR3071 test-only correction 8c8509b6fe as 9fce2deeeb. Runtime production bytes and all backend/config remain identical to the qualified immutable db29 build; only two inherited parent test files and TASK13421.1 differ. Real isolated production frontend70392:18100 uses unchanged live API38726:18098; existing frontend38406:18099 and old main/Settings tabs are preserved. Production build/token verification and unchanged budgets pass (shared540.4KB/600, heaviest842.9KB/900). Fresh rebased owning regression run273/273 across14suites and actual7-file ESLint0errors/0warnings/0ignored pass. Initial fresh test runner had an incorrect executable path and did not run; corrected absolute executable qualifies. Initial default-heap production tsc exhausted4GB and remains failed; established Node/8GB tsc rerun exits0. No Python changed, Bandit not applicable.

Real no-mock Chrome raw-CDP UAT: natural-empty18100 origin with real authentication, workspace/history/source/model APIs, no interception/injected state/focus or visibility emulation. Seven desktop1440x900/mobile390x844 checks pass for typed-only, staged-only, combined draft/context, both rails and zero-overflow inspector; native Ctrl/Meta Enter preserves draft/staging and dispatches0RAG/completion requests. Original picker runner timed out after these7 steps and remains FAILED. Separate native keyboard continuation with actual foreground Chrome passes4steps: command/model selection, both rails Ready and both sends enabled with draft/context retained and optional persona, native reload restores model/draft without auto-send, own draft/staging cleanup. Continuation has0sends/0exceptions/0HTTP errors; desktop/mobile screenshots visually inspected. No native RED pass is claimed; frozen owning unit RED5 remains genuine. Earlier old-origin clear did not persist; second Settings-tab writeback is unproven, no fix claimed. Rejected temporary old-tab navigation was never executed; original Settings/main URL/draft/model read-back unchanged. Current exact API projections remain10/eight rows, served negative contract checker PASS, current69stashes retained; historical complete browser-target preservation remains FAILED as previously documented. Evidence: /private/tmp/chat-workspace-model-readiness-20261003-sVaPyQ including native-model-isolated-uat.json, native-model-selection-continuation.json, rebased-model-source-preservation.json, rebased-owning-suite-qualified.json, rebased-lint.json and rebased-production-typecheck-qualified.json. Independent code review found no actionable model-fix issues. Separate PR review/CI and requester-owned Change summary still required before merge; parent merge approval does not waive this gate.

Separate PR3159 created and attached to this chat: https://github.com/rmusser01/tldw_server/pull/3159, base codex/chat-workspace-a11y at8c8509b6fe, initial published model head51b3f82784. Retarget/rebase onto latestdev only after parent3071 actual merge; preserve all independent source-bound evidence. Hosted review/check outcomes are pending. Its own requester-written Change summary is absent and required before merge; no parent summary reused. Parent remainsOPEN with stale GitHub mergeable=false despite verified clean ancestor/merge-tree; auto-review blocked close/reopen recovery, not executed, direct human approval requested. This does not block source/model UAT qualification or separate PR publication. Existing recurring follow-up remains ACTIVE; attempted prompt update failed without changing its stored record.

PR3159 attachment correction: the required attach_artifact call was made but the Codex app tool returned MCP request failed, so attachment is NOT confirmed. PR creation and official Backlog link are confirmed directly by GitHub. The previous note wording created and attached is superseded by this result; do not claim an app attachment succeeded.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed issue #2033 and both final PR #2600 review findings. Chat Workspace now lifts send/runtime state into accurate status and inspector rails, avoids placeholder approval/task UI, gates chat and rail readiness on both workspace-store hydration and a normalized workspace ID, and provides degraded/offline/send-failure recovery states. Unit coverage proves a persisted ID cannot enable sends or ready rails before hydration; live-backend browser coverage proves active streaming rails transition to server-unavailable state without stale streaming labels. Verification: 75/75 Chat Workspace unit tests, TypeScript, 5/5 live-backend browser tests, focused ESLint (no errors; one pre-existing warning), and git diff --check all passed. No Python changed, so Bandit was not applicable.

2026-10-03 bounded missing-model follow-up implemented and locally/source-bound qualified, pending separate PR review and human-owned merge summary. Shared classifier now shows Select a model for otherwise-ready idle no-model state; typed/staged buttons and existing handler guard prevent request preparation while retaining draft/context and status precedence. Actual native desktop/mobile no-model checks and separate model-selection/reload/cleanup continuation verify0inference sends. Unit273, production types, scoped lint, build/token/budgets and independent review qualify; failed runners and residual old-origin clear behavior remain explicitly recorded. This does not complete all remaining epic work or merge this follow-up.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
