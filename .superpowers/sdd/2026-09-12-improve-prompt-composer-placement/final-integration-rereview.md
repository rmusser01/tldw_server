# Final integration scoped re-review — TASK-12984.2 / TASK-12984.3

## Verdict

**APPROVED. All four prior Important findings are addressed; no residual or new Critical/Important finding was established in the scoped fix wave.**

Reviewed range `151b532e61..9396471166b9a92c58fc4427d89ff66c8ab9f885` on `codex/single-text-prompt-recipes`. The implementation remained at that HEAD throughout verification. This is the single requested scoped re-review, not a new implementation wave or rebase. Read the complete `final-integration-fix-report.md` and prior `final-integration-review.md`, retaining the previously read design, task records, progress ledgers, reports, and approvals as context.

## 1. Active PUT ownership and settlement — closed

- `recipe-persistence-uncertainty.ts:206-210` now reserves the direct operation with an exact-ID/owner UUIDv4 provisional receipt. The existing registry's provisional guard blocks reconciliation and other reservations; Pull can no longer interpret an active direct write as reconcilable scoped uncertainty.
- `tldw/request-core.ts:224-237,523-532` carries the authoritative receipt on both success and error. It remains transport-owned metadata, not a field trusted from the server response body. The extension background still creates its own authoritative receipt.
- `api-send.ts:158-162` no longer acknowledges extension delivery before the sync consumer commits locally. `prompt-sync.ts:712-721` validates the receipt against local ID, expected owner, and dispatched actual owner. Its `finally` at lines 794-800 acknowledges only after local settlement.
- Every asynchronous failure return inside that protected region is awaited. This matters: `return failure(...)` would run `finally` before the durable-error write settles. Both success and error compensation now retain the provisional guard until their local work finishes. Failed or missing ACKs remain fail-closed; ACK removes only its exact provisional handoff, not scoped/unknown evidence.

The unchanged original real-chain probe now exits 0: while the first PUT is held, authority is `unknown_owner`, same-owner Pull returns false/error, second push returns false, and PUT count stays **1**. The final reviewer also independently reran the six maintained direct/extension interleavings: transport held, local success commit held, and local durable-error settlement held all pass. The tests exercise the real sync/Studio/apiSend/request-core/authority chain, substituting platform/storage/network boundaries. They assert actual PUT count, blocked Pull/retry, retained authority, and correct eventual synced or scoped-error state—not merely helper call counts.

Preservation evidence inspected and freshly exercised includes:

- Registry exact receipt/old receipt/unknown/Forget/restart cases (`recipe-persistence-registry.test.ts:5,29,79,94`).
- Background forged ACK rejection, cross-surface ownership, restart with durable error, and provisional-before-fetch behavior (`background.recipe-persistence-owner.test.ts:105,183,418,440`).
- Different/null owner and unknown quarantine rejection, failed reconciliation release and failed durable compensation, and durable-only restart recovery (`prompt-sync.uncertainty.test.ts:497,526,547,657,699,740`).
- Lost extension response across owner changes and reopen, no direct replay, and local-only confirmed Forget (`recipe-persistence-owner.contract.test.ts:1009`; `recipe-persistence-owner.surface-bridge.test.tsx:824`).
- Builder ACK request/reply loss and compound response/marker/durable failures (`PromptRecipeBuilder.dispatch.test.tsx:336,369,423`).

No new owner derivation, credential routing, server-side idempotency claim, page-local extension authority, automatic ambiguous retry, or recovery-scope expansion was introduced.

## 2. System recipe followed by improvement — closed

`PromptSelect.tsx:476-482` clears `recipeUndo` synchronously at the shared entry to both Improve now and Review changes, before taking the improvement entry snapshot or starting the operation. The rest of the system adapter and its exact override/template handling is unchanged.

The new real-component matrix covers both modes × success/pending-cancel/failure × undefined/empty/custom original overrides (**18 cases**). It reflects the owner's accepted compiled override before improvement, asserts that only the compiled system draft enters the request, and checks immediate invalidation while the request remains held. Late success after pending Cancel does not revive either Undo. Review success does not mutate the override before Apply. Successful improvement creates exactly one newer Undo that restores the compiled recipe, never the pre-recipe original; cancel/failure leave the compiled recipe intact. Template selection, quick prompt, and library persistence remain untouched.

The separately expanded recipe-only Undo test covers undefined, empty, and whitespace/Unicode custom overrides, proving exact raw restoration and selected template identity. These exact raw values are distinguished from their effective editor display, which can inherit the template. Reset after the improvement scenarios still resolves the selected template. Existing model/provider/context/backend/template invalidation and focus cases remain in the passing full system-modal suite.

The original unchanged in-memory stale-Undo probe also now passes (**1 passed, 88 unselected**); its prior destructive stale button no longer exists. No repository test/source was changed to run that probe.

## 3. Capabilities authentication and rate limiting — closed

`prompts.py:743-749` restores `Depends(get_auth_principal)` before the capability rate dependency while retaining the same principal parameter for capability authorization metadata. The helper and rate-limit policy are unchanged.

The entire existing `test_prompt_improvement_api.py` now passes (**67 tests**), including the previously failing catalog test: request 1 is 200, request 2 is 429. Real unauthenticated rejection and admin/user/permission capability metadata cases remain covered; no test expectation or authentication policy was weakened.

An independent read-only counter wrapped the existing rate test's principal dependency and capability policy: across two requests, authentication resolved exactly twice (once per request); both first-response authorization flags received the identical authenticated principal object; request 2 got its own principal and was rejected by the real limiter before entering endpoint policy. Thus the duplicated dependency declaration uses FastAPI request caching rather than duplicating authentication or changing the authorization principal. Adjacent auth-principal resolver and auth-dependency hardening modules also pass (**55 tests**).

## 4. Maintained shell wiring and approved placement — closed

The refreshed shell file passes all **8 tests** as part of the seven-file Track A/state/diff/localization gate. It now asserts the approved send clusters rather than the removed toolbar-owned topology:

- One Playground action immediately before external Send, outside ComposerToolbar.
- One shared Sidepanel action supplied to the mutually exclusive legacy/pro controls and v1/v3/v5 slots; v5 uses its Send slot and v1/v3 use the shared bottom controls.
- Existing form/draft revision, mutation owner, exact saved attempt, active model/provider, backend key, authorization revision, conversation/draft context, sending/streaming state, narrow surface, and return-focus wiring.
- Exact submit/queue completion, no duplicate toolbar/ControlRow action, unchanged Sidepanel entry routing, model-recovery destination, and explicit Quick Chat exclusion.

These remain intentionally source-contract assertions, complemented by the passing rendered action/menu/control/toolbar/system tests. The rewrite removes obsolete syntax assumptions without deleting ownership or route proof; it adds explicit authorization/context and per-variant checks. No tests were skipped or disabled in the maintained full gates.

Production placement is unchanged in this fix range. The only component production diff is the one-line system Undo invalidation. Playground/Sidepanel mounting, shared composer action, menu/feedback portal, overlay positioning/interception/listener cleanup, and toolbar/control layout files have no production diff. Existing overlay browser approvals therefore remain relevant prior evidence, not newly rerun browser results.

## Fresh verification

All maintained commands below completed with exit 0 on the reviewed HEAD. The independent persistence reviewer ran the 23-file gate and original transport probe; the final reviewer inspected its evidence and reran the six new held-operation cases. All other gates and the original system probe were run by the final reviewer.

| Gate | Fresh result |
| --- | --- |
| Persistence/transport/owner/recovery/surface/caller matrix | 23 files / 1,666 passed, shuffle seed 12984 |
| Composer/menu/system/recipe/editor/builder matrix | 10 files / 360 passed |
| Track A controller/panel/diff/localization plus shell | 7 files / 69 passed |
| Prompt improvement/capabilities API module | 67 passed, 8 warnings |
| Adjacent auth-principal resolver and dependency hardening | 55 passed, 4 warnings |
| Direct/extension active PUT targeted rerun | 6 passed, 27 unselected |
| Original real-chain active PUT/Pull probe | Exit 0, one PUT |
| Original system stale-Undo probe | 1 passed, 88 unselected |
| Existing real limiter plus authentication/identity counter | Pass: 200 then 429, one auth resolution/request |
| Extension `bun run compile` | Exit 0 |
| Bandit on changed Python production endpoint | 0 findings, 0 errors, 0 skips; 1,765 LOC |
| Scoped diff check and placement no-diff check | Exit 0 |

The 62-test builder-dispatch suite within the component gate plus the 23-file persistence gate reconstructs the fix report's **24 files / 1,728 tests** without double-counting a separate run. This review reports its own explicit API/auth total (**122**) rather than inheriting the report's unexplained 114-test selection. The four requested correction gates are all covered directly, with broader adjacent tests retained.

### Exact commands

From `apps/packages/ui`, using the installed repository-pinned Vitest 4.0.18:

```sh
./node_modules/.bin/vitest run src/services/__tests__/recipe-persistence-owner-contract.test.ts src/services/__tests__/recipe-request-snapshot.test.ts src/services/__tests__/recipe-persistence-registry.test.ts src/services/__tests__/recipe-persistence-authority.test.ts src/services/__tests__/recipe-persistence-owner.contract.test.ts src/services/__tests__/request-core.persistence-scope.test.ts src/services/__tests__/api-send.test.ts src/services/__tests__/prompt-sync.structured-prompts.test.ts src/services/__tests__/prompt-sync.auto-sync.test.ts src/services/__tests__/prompt-sync.uncertainty.test.ts src/services/__tests__/structured-prompt-transport.test.ts src/services/__tests__/prompt-studio.recipe-policy.test.ts src/db/dexie/__tests__/firefox-prompt-write-order.test.ts src/db/dexie/__tests__/prompt-rollback.test.ts src/entries/__tests__/background.recipe-persistence-owner.test.ts src/services/__tests__/recipe-persistence-uncertainty.test.ts src/services/tldw/__tests__/request-core.quickstart.test.ts src/services/tldw/__tests__/request-core.hosted.test.ts src/services/__tests__/tldw-auth.refresh-rotation.test.ts src/entries/__tests__/background.effective-auth.test.ts src/hooks/__tests__/useRecipePersistenceOwner.test.tsx src/services/__tests__/recipe-persistence-owner.surface-bridge.test.tsx src/components/Option/Prompt/__tests__/prompt-sync.owner-callers.test.tsx --sequence.shuffle --sequence.seed=12984 --reporter=dot

../../tldw-frontend/node_modules/.bin/vitest run src/components/Common/PromptAssist/__tests__/PromptAssistMenu.test.tsx src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelComposerControlArea.prompt-assist.test.tsx src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx src/components/Common/PromptAssist/recipes/__tests__/recipe-editor-state.test.ts src/components/Common/PromptAssist/recipes/__tests__/built-in-recipes.test.ts --maxWorkers=1 --no-file-parallelism --reporter=dot

../../tldw-frontend/node_modules/.bin/vitest run src/components/Common/PromptAssist/__tests__/PromptAssistPanel.test.tsx src/components/Common/PromptAssist/__tests__/prompt-assist-state.test.ts src/components/Common/PromptAssist/__tests__/PromptDiff.bounded.test.tsx src/components/Common/PromptAssist/__tests__/PromptReviewSurface.test.tsx src/components/Common/PromptAssist/__tests__/PromptAssistLocalization.test.tsx src/components/Common/PromptAssist/__tests__/usePromptAssist.test.tsx src/components/Chat/composer/__tests__/PromptAssistComposerAction.shell-wiring.test.ts --maxWorkers=1 --no-file-parallelism --reporter=dot

../../tldw-frontend/node_modules/.bin/vitest run src/services/__tests__/recipe-persistence-owner.contract.test.ts -t 'same-owner Pull cannot release an existing PUT' --maxWorkers=1 --no-file-parallelism --reporter=verbose

../../tldw-frontend/node_modules/.bin/vitest run src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx --config /tmp/prompt-recipe-final-review.vitest.config.mts -t 'final integration probe' --maxWorkers=1 --no-file-parallelism --reporter=verbose
```

From the assigned worktree root:

```sh
sh /tmp/recipe-active-write-pull-probe.sh
source .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompt_improvement_api.py
source .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py
source .venv/bin/activate && python -m bandit tldw_Server_API/app/api/v1/endpoints/prompts.py -f json -o /tmp/bandit_final_integration_rereview.json
git diff --check 151b532e61..9396471166
git diff --exit-code 151b532e61..9396471166 -- apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx apps/packages/ui/src/components/Common/PromptAssist/PromptAssistMenu.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx apps/packages/ui/src/components/Sidepanel/Chat/form.tsx apps/packages/ui/src/components/Sidepanel/Chat/SidepanelComposerControlArea.tsx
git status --short --untracked-files=all
```

Extension compile ran `bun run compile` from `apps/extension`. Main-review outputs are in `/tmp/final_integration_rereview_{components,adjacent,backend,auth_adjacent,active_put,original_undo_probe,extension_compile,bandit}.log`. The persistence gate and original transport probe outputs were captured directly by tools. Complete original probe sources remain preserved in the prior review's appendices; no temporary diagnostic is needed to interpret the verdict.

The auth counter ran the existing limiter test in `pytest.MonkeyPatch.context()`, temporarily wrapping `tests.TestClient` to count its already-overridden `get_auth_principal`, and wrapping `prompts._is_prompt_persistence_authorized` to record object identity. After the test's own 200/429 assertions, it asserted two principal resolutions, two endpoint policy checks both identical to the first principal, and distinct principals between requests. No router/source/test file was edited.

## Limits, safety, and handoff

This scoped approval closes the four findings from the cumulative final review; it does not claim a new full browser/build/typecheck/PostgreSQL run. Full browser suites remain the controller's stated post-rebase gate. Prior full-Web build app-shell size-budget failure and unrelated full-frontend typecheck/legacy formatting debt are not relabeled as passing. Prior inclusive Python Bandit had 23 unchanged LOW B311 jitter findings; this fresh changed-endpoint scan is genuinely clean. Existing Node localStorage and Python deprecation/cleanup warnings remain non-failing. Live-backend browser and manual screen-reader UAT were not performed here.

The assigned worktree was clean before review and after every test/probe. Only this review artifact is committed. No production, test, Backlog, dependency, lockfile, user data, or placement file was edited; no rebase or PR mutation was performed. The main checkout still has no tracked changes and exactly its two pre-existing untracked TTS files:

- `Docs/superpowers/specs/2026-09-10-tts-gateway-metadata-preferences-design.md`
- `backlog/tasks/task-13140.1 - Design-TTS-gateway-voice-discovery-pricing-and-preference-history.md`

The controller owns the requested rebase and subsequent integration gates. Repository human-written PR Change summary policy is unchanged. Verification-before-completion guided the fresh evidence checks; explicitly authorized parallel review provided an independent persistence/recovery inspection.
