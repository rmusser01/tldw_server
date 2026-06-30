---
id: TASK-457
title: Implement Character Chat Phase 7 model usability and send gating
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-21 08:01'
labels:
  - character-chat
  - roleplay
  - frontend
  - implementation
dependencies: []
references:
  - TASK-456
  - TASK-455
  - TASK-426
  - TASK-454
documentation:
  - Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
  - >-
    Docs/superpowers/plans/2026-05-20-character-chat-phase7-model-usability-send-gating-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 7 from the Character Chat first-class PRD and implementation plan: model usability contract, readiness truth, status-surface alignment, SEND gating, provider/model failure recovery, and real-backend verification. Scope must stay on /chat Character Chat role-play readiness and send behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Model usability helper classifies loading, no_server, no_selection, no_models, selected_missing, provider_unconfigured, model_unavailable, degraded, and ready states with focused unit coverage.
- [x] #2 Character Chat readiness panel, status strip, runtime inspector, composition preview, model selector copy, and SEND action consume one shared model-usability result and do not show positive health copy for unusable models.
- [x] #3 Character selected plus no usable model blocks or converts SEND into a setup action without invoking submit, without calling /complete-v2, and without losing draft/character/session state.
- [x] #4 Provider/model setup failures show actionable model/provider recovery copy instead of generic retry-only guidance when the failure is configuration-specific.
- [x] #5 Real-backend Playwright verification covers no-provider/send-gating without simulated frontend responses; successful-send is verified only through a real callable provider or explicitly marked blocked by environment.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Execution started with superpowers:subagent-driven-development and superpowers:test-driven-development. Controller ran baseline before production edits: bunx vitest run ../packages/ui/src/utils/__tests__/chat-model-availability.test.ts ../packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx --reporter=verbose passed with 3 files / 39 tests. Two Task 1 workers were shut down after no edits/progress reports; controller implemented Task 1 locally under TDD. Red test: focused utility test failed with 11 failures because buildChatModelUsability was not a function. Green tests: focused utility test passed with 33 tests; baseline trio passed with 3 files / 50 tests. Task 1 added the pure model usability contract/helper and unit coverage only; UI wiring remains pending.

Task 1 final: addressed spec-review gaps for provider aliases, colon-bearing local model IDs, and provider-qualified duplicate matching. Verification: focused utility suite passed with 41/41 tests; baseline trio passed with 3 files / 58 tests; git diff --check clean. Review gates: Task 1 spec review approved and code-quality review approved for HEAD ebda82474. Task 1 remains UI-unwired by design; Task 2 will map Character Chat readiness to the usability contract.

Task 2 local TDD: added failing readiness and panel tests for models-loading, selected-model-missing, provider-unconfigured, model-unavailable, no-models-available copy, ready, and send-disabled ordering. Implemented buildCharacterChatReadiness on top of buildChatModelUsability and added precise blocker copy preserving character/draft context. Verification: focused readiness/panel suite passed with 2 files / 50 tests; baseline trio passed with 3 files / 63 tests; git diff --check clean. AC #2 is not checked yet because status strip, runtime inspector, composition preview, model selector, and SEND controls are later tasks.

Task 3 local TDD: added red tests for explicit model usability in the status strip and composition preview. Implemented Character Chat model-usability propagation through Playground.tsx, PlaygroundStatusStrip, PlaygroundRuntimeInspector, and composition preview so provider-unconfigured/model-unavailable/loading states no longer show Ready/Healthy copy on those surfaces. Verification: status strip focused suite passed with 1 file / 17 tests; composition preview suite passed with 1 file / 8 tests; combined Task 3 suite passed with 2 files / 25 tests; baseline trio passed with 3 files / 67 tests; git diff --check clean. TypeScript compile was attempted with bunx tsc --noEmit --pretty false and failed only on existing unrelated baseline errors outside the touched files.

Task 3 review fixes: added regression coverage for degraded model usability with canSend=false, explicit no_server status-strip copy, and runtime streaming priority over model-usability blockers. Implemented canSend propagation from buildChatModelUsability into Playground, status strip, runtime inspector, and composition preview. Verification: focused Task 3 suite passed with 3 files / 51 tests; broader character readiness/model surface suite passed with 5 files / 101 tests; git diff --check clean. bunx tsc --noEmit --pretty false still fails only on existing unrelated baseline TypeScript debt outside touched files.

Task 3 review-fix follow-up: addressed reviewer findings for no_selection status-strip copy, legacy modelUnavailable degraded-positive copy, and runtime inspector model-catalog loading. Added regressions for all three. Verification: targeted status-strip/runtime suite passed with 2 files / 45 tests; broader character readiness/model surface suite passed with 5 files / 104 tests; git diff --check clean. bunx tsc --noEmit --pretty false still fails only on existing unrelated baseline TypeScript debt outside touched files.

Task 3 final quality follow-up: added getMatchingCharacterChatModelUsabilityCopy so Playground only reuses readiness copy when the readiness blocker matches the model-usability blocker. This prevents no-character copy from appearing as model-readiness detail. Verification: utility suite passed with 1 file / 46 tests; broader character readiness/model surface suite passed with 5 files / 105 tests; git diff --check clean. bunx tsc --noEmit --pretty false still fails only on existing unrelated baseline TypeScript debt outside touched files.

Task 3 legacy fallback hardening: status strip and composition preview now ignore legacy modelUnavailableMessage/modelUnavailableDetail unless modelUnavailable is true, and Playground only passes those fallback values for the chat-model unavailable path. Added regressions for no-character/no-model fallback leakage. Verification: affected status/composition suite passed with 2 files / 33 tests; broader character readiness/model surface suite passed with 5 files / 107 tests; git diff --check clean. bunx tsc --noEmit --pretty false still fails only on existing unrelated baseline TypeScript debt outside touched files.

Task 3 final status-strip fallback hardening: modelUnavailableReason now also ignores legacy modelUnavailableMessage unless modelUnavailable is true. Added a provider_unconfigured stale-fallback regression. Verification: status-strip focused suite passed with 1 file / 24 tests; broader character readiness/model surface suite passed with 5 files / 108 tests; git diff --check clean. bunx tsc --noEmit --pretty false still fails only on existing unrelated baseline TypeScript debt outside touched files.

Task 4 local TDD: added PlaygroundSendControl character-gating coverage, then implemented PlaygroundSendBlocker so blocked Character Chat primary SEND becomes a button action with blocker copy and never calls onSubmitForm. Threaded the blocker from Playground readiness into PlaygroundForm and replaced composer SEND entry points (primary send control, form submit, Enter-to-send, voice submit ref, and ChatComposer v1/v3/v5 onSend) with one guarded handler that opens the setup action while preserving the draft. Added integration coverage proving blocked character-chat SEND invokes setup, does not call onSubmit, preserves the draft text, and retains character-context copy. Verification: focused composer suite passed with 2 files / 4 tests; broader Phase 7 focused suite passed with 7 files / 112 tests; git diff --check clean. bunx tsc --noEmit --pretty false still fails only on existing unrelated baseline TypeScript debt outside touched files. Bandit not applicable for this Task 4 slice because touched code is TypeScript/TSX and docs only.

Task 5 local TDD: added failing dropdown and form-level tests for Character Chat model-usability selector copy. Implemented optional model-usability label/title/warning props on ChatModelSelectorDropdown, suppressed the generic positive connection-health badge when a model-usability override is active, and passed Character Chat model-usability label/title from Playground through PlaygroundForm. The selector now keeps provider/model identity visible while showing copy such as Provider setup needed or Checking model readiness, without Healthy/Ready copy for blocked/loading usability. Verification: selector/form suite passed with 2 files / 6 tests; broader Phase 7 focused suite passed with 8 files / 115 tests; git diff --check clean. bunx tsc --noEmit --pretty false still fails only on existing unrelated baseline TypeScript debt outside touched files. Bandit not applicable for this Task 5 slice because touched code is TypeScript/TSX and docs only.

Task 6 local TDD: added provider/model recovery coverage for Character Chat stream failures and the existing Playground error banner. Implemented conservative structured failure classification so provider_not_configured/missing API key and model-not-callable failures encode an open-model-settings recovery payload while arbitrary 503s remain retry/transient. The banner now opens model settings for local recovery payloads instead of defaulting to Health & diagnostics. Updated the starter integration test for current dev's collapsed Explore chat modes entrypoint after rebasing. Verification: focused Task 6 suite passed with 2 files / 10 tests; broader Phase 7 focused suite passed with 10 files / 125 tests; git diff --check clean. bunx tsc --noEmit --pretty false still fails only on existing unrelated baseline TypeScript debt outside touched files. Bandit not applicable for this Task 6 slice because touched implementation/test files are TypeScript/TSX/docs/backlog only.

Task 7 real-backend verification: added `apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts`, which creates a real character through the FastAPI backend, selects a real backend-advertised blocked model, observes network calls without simulating successful chat responses, verifies Character Chat setup surfaces align, preserves draft/character state, and asserts no `/api/v1/chats/*/complete-v2` request is made while blocked. The spec also contains opt-in real provider-failure and successful-send scenarios that skip unless the environment provides explicit real models for those checks. Real-browser verification exposed that the domain `models-audio` client mixin dropped backend model readiness flags before caching, so Task 7 preserves `is_configured`, `provider_is_configured`, and `catalog_only`, adds a regression in `tldw-api-client.models-normalization.test.ts`, and bumps `TldwModels` cache schema to 6 to evict stale readiness caches. Verification: backend health `status: ok`; focused Vitest suite passed with 6 files / 110 tests; real-backend Playwright suite passed with 1 passed / 2 skipped. The two skips were intentional environment skips: no explicit forced provider-failure model, and no non-local/non-custom callable model to prove successful-send without simulation.

Task 8 final verification: focused unit/component suite passed with 7 files / 94 tests. `bunx tsc --noEmit --pretty false` still fails on inherited baseline TypeScript debt in Media read-along, Evaluations embeddings recipe config, Workspace StudioPane, keyboard shortcut config, persona live control, and tier-4 admin llamacpp E2E fixtures; a touched-scope E2E response narrowing issue was fixed and no TypeScript errors remain in touched files. Real-backend Playwright verification was rerun against the real FastAPI backend at `http://127.0.0.1:8000`; backend health returned `status: ok`, and the Phase 7 journey suite passed with 1 passed / 2 skipped. The active no-provider/send-gating scenario used a real backend-created character and real model metadata, and the provider-failure/successful-send scenarios skipped for the same environment reasons as Task 7. Bandit skipped because no Python files were touched by this frontend/docs slice.

Rebase closeout: rebased `codex/character-chat-post-phase6-prd` onto current `origin/dev` at `027bfeb52`. Fresh focused suite passed with 7 files / 94 tests. Fresh `bunx tsc --noEmit --pretty false` still fails only outside the Phase 7 touched scope; current dev adds inherited Watchlists RunDetailDrawer TypeScript errors alongside the previously recorded baseline files. Real-backend Playwright was rerun after starting the backend outside the sandbox so it could bind `127.0.0.1:8000`; backend health returned `status: ok`, and the Phase 7 journey suite passed with 1 passed / 2 skipped.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 7 is complete for the scoped Character Chat model-usability/send-gating slice and has been rebased onto current `origin/dev`. The WebUI now uses one model-usability contract across readiness, status, runtime/composition, model selector, and composer SEND behavior; blocked Character Chat sends become setup/recovery actions without losing draft or character state and without calling `/complete-v2`. Real-backend E2E verification covers the blocked no-provider/catalog-only path without simulated successful responses; provider-failure and successful-send checks remain opt-in/skipped unless a suitable real model is configured. Final verification recorded the inherited TypeScript baseline separately from touched files and confirmed there are no touched-scope TypeScript errors after the Task 8 E2E narrowing fix.
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
