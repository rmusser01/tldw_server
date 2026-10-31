# Agentic Qodo frontend findings 2/3/4/7/10/11/14/15

This report records the initial frontend patch. The subsequent independent review found additional media mutation/persistence gaps and a setup resume gap; the parent ledger and media-security follow-up record their corrections. In particular, the initial setup inspection at the end of this report did not exercise the resumed wizard and was superseded by the failing client regression.

Outcome: all eight validated findings fixed in the shared candidate worktree
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`.
Investigated HEAD `254165e7b5`, branch `codex/release-main-0.1.43`.
Tracking: TASK-13263.1 (parent owns task/release documentation and publication).
No commits or pushes by this agent.

## Per-finding dispositions

2. **Fixed: missing chat loading cleanup.** `useServerChatLoader` now clears isLoading while its owned load still controls the selected ID, before clearing serverChatId on 404 or invalid scoped metadata. Existing finally guard remains, so an older completion cannot clear a different chat's loading state. Regression uses a real-mutating setServerChatId mock and reproduces true-last/loading-stuck before fix.

3. **Fixed: same-server principal media boundary.** `useMediaRequestLifetime` now uses the existing `watchChatAccountChanges` boundary, in addition to connection disconnect/server changes. This covers principal/credential events, config-updated authorityChanged, browser storage and extension storage authority replacement. Abort is synchronous and permanent for a mounted lifetime; ordinary renders do not revive it. Retirement callbacks remove this search lifetime's TanStack query data, clear private search metadata/totals, and clear navigation selected/detail/content/error/pending references. `ViewMediaPage` remounts its entire content generation at the same boundary, clearing other view-owned state and allowing the next account to load. Unmount also purges that lifetime's query prefix rather than retaining private results until cache GC.

   Broken invariant: private results from account A must neither remain visible nor populate account B after a same-server transition. Previously, the principal event did not abort requests or remount content. The shared watcher is the narrow native boundary because TldwApiClient config changes may use only config-updated, not the logout event. Existing authority comparisons keep ordinary same-owner config refreshes valid.

   Red/green checks cover displayed + cached private data with a held old response on principal, credential and config transitions; browser StorageEvent is an alternate delivery path; selected detail is cleared and its late replacement rejected; actual ViewMediaPage loads Bob and removes Alice on same-server principal/config events; ordinary connected/background checks and authorityChanged:false preserve valid current requests. A separate final bypass pass found prior-unmounted query entries still retained until GC; the added unmount cache assertion failed before the cleanup and passes now. Per-useId deletion preserves unrelated queries.

   Security review: applied codex-security fix-finding, independently traced source/event/caches/callers and reviewed affected conditions. Agent `/root/agentic_security_boundaries` supplied a separate pre-patch audit confirming the config-event alias and page-owned-state boundary. Parent owns final independent candidate review (fresh reviewer dispatch unavailable). No auth/server authorization behavior weakened.

4. **Fixed: mirrored role names.** `server-chat-mirror` falls back to You/User role, System/System role, or Assistant/assistant role, matching the existing chat mapper's naming convention. Explicit supplied names remain. Tests persist unnamed user and assistant messages; unnamed user previously stored Assistant.

7. **Fixed: OSCE focus lifecycle.** Removed both untracked timeout paths. Save-error focus runs in a React effect after its alert commits; the Modal afterOpenChange handler focuses directly after its transition. The pre-existing open-effect timeout remains tracked and cleaned up. A late rejected reveal after unmount formerly scheduled a new timeout; now it schedules none. Existing real-Modal accessible confirmation/error-focus tests continue passing.

10. **Fixed: OSCE strict null checks.** Replaced loose comparisons in elapsed-time and pending-note branches with explicit strict null/undefined alternatives, including the inverse flush check. Both absent forms and valid falsey values retain their semantics. Invalid-time controls and pending-save/flush tests pass.

11. **Fixed: PromptSelect focus lifecycle.** Recipe application now sets the existing focus-request ref and lets the editor-open/recipe-mode commit effect focus the restored textarea. Removed the untracked timeout. Strengthened all three existing recipe override/undo scenarios to assert focus immediately after applying, without waiting on a deferred timer; they failed before the fix and pass now.

14. **Fixed: translated recommended duration.** Added option:quiz.osceRecommendedMinutes and the English locale value with {{minutes}} interpolation. The test supplies a translated marker and validates the calculated eight-minute value, failing on the previous literal JSX.

15. **Fixed: Flashcard strict session checks.** Explicit strict null/undefined guards preserve omission of absent session IDs and avoid closing a nonexistent session. Added real-hook/mutation/HTTP controls for both null and missing response IDs; existing owned-session/inactive-session/authority tests pass unchanged.

## Changed scope

Production:
- apps/packages/ui/src/hooks/chat/useServerChatLoader.ts
- apps/packages/ui/src/db/dexie/server-chat-mirror.ts
- apps/packages/ui/src/components/Review/hooks/useMediaSearch.ts
- apps/packages/ui/src/components/Review/hooks/useMediaNavigationState.ts
- apps/packages/ui/src/components/Review/ViewMediaPage.tsx
- apps/packages/ui/src/components/Quiz/osce/OscePracticePanel.tsx
- apps/packages/ui/src/components/Common/PromptSelect.tsx
- apps/packages/ui/src/components/Flashcards/hooks/useFlashcardReviewRun.ts
- apps/packages/ui/src/assets/locale/en/option.json

Tests: the eight corresponding suites in the command below. ViewMediaPage's existing client mock additionally supplies initialize, because the new real principal-event test legitimately invokes useHomeMilestoneScope's initialization path. No tests disabled or assertions weakened; no timeout increases.

## Final verification

From `apps/tldw-frontend`:

```sh
./node_modules/.bin/vitest run --root ../packages/ui --config ../packages/ui/vitest.config.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx src/db/dexie/__tests__/server-chat-mirror.test.ts src/components/Review/hooks/__tests__/useMediaSearch.outage.test.tsx src/components/Review/__tests__/useMediaNavigationState.permalink-hydration.test.tsx src/components/Review/__tests__/ViewMediaPage.connection.test.tsx src/components/Quiz/osce/__tests__/OscePracticePanel.test.tsx src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx src/components/Flashcards/hooks/__tests__/useFlashcardReviewRun.test.tsx
```

**241 tests passed, 8 files passed, 18.19s**, no unhandled errors. Log `/tmp/qodo-agentic-front-final.log`.

`./node_modules/.bin/tsc --noEmit`: exit0, empty diagnostics log `/tmp/qodo-agentic-front-tsc.log`.

ESLint actual touched TS/TSX files with `apps/tldw-frontend/eslint.config.mjs`: existing104 diagnostics/current104, **zero new diagnostics** vs exact HEAD. Comparison normalizes embedded source line numbers in diagnostic text only, preserving rule/message/count; artifact `/tmp/qodo-agentic-frontend-lint.json`, reproducible script `/tmp/check-agentic-frontend-lint.py`. The repository has existing lint debt; whole touched scope is not falsely reported warning-free. Initial added test-any warning corrected.

`git diff --check`: clean.

Required Bandit attempted from server venv over eight touched production TS/TSX paths, artifact `/tmp/qodo-agentic-frontend-bandit.json`. Bandit is a Python analyzer and cannot parse these TypeScript files; its zero findings are not a security validation claim. TypeScript, ESLint, behavior regressions and the explicit boundary review are the relevant evidence.

Red artifacts:
- `/tmp/qodo-agentic-front-core-red.log`: five original regression failures (missing loading, role label, principal/credential cache retirement, navigation detail).
- `/tmp/qodo-agentic-front-focus-red.log`: five focus/translation failures, including three parameterized recipe scenarios.
- `/tmp/qodo-agentic-media-unmount-red.log`: private prior-lifetime data retained after unmount.

Remaining limits: no full repo/frontend suite or remote CI rerun by this agent. Parent owns publication and final independent candidate review.

## Additional parent-requested setup compatibility inspection

No public absolute-filesystem model restoration assertions were found in UnifiedSetupWizard tests. The model-handoff tests do rely on safe non-filesystem identifiers such as org/gemma:Q4 and provider-qualified model IDs; FirstChatStep uses a basename local.gguf. UnifiedSetupWizard providerSelectionFromState returns null if both public model fields are redacted, allowing reselection rather than requiring private path recovery. No setup UI source edits made.
