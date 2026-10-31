# Candidate security/regression review

Reviewed the current working-tree changes in `ViewMediaPage.tsx`, `useMediaSearch.ts`, `useMediaNavigationState.ts`, and `setup.py`, plus their direct callers, request/auth helpers, and changed tests. Read repository AGENTS.md and SECURITY.md; no narrower security policy applies. Findings below are source-traced; reproduction steps are proposed checks, not executed tests. No repository files were changed.

## 1. [P1] In-flight bulk mutations survive account retirement and continue under the new account

**Status:** Surviving boundary bypass in a direct caller, predating this patch; the added remount does not repair it.

**Locations:** `apps/packages/ui/src/components/Review/hooks/useMediaSelection.ts:391-405,476-497`; changed boundary at `apps/packages/ui/src/components/Review/ViewMediaPage.tsx:121-130` and keyed child at line 312.

`MediaPageContent` also owns `useMediaSelection`, whose bulk tag/delete handlers iterate selected IDs and await each `bgRequest`. Neither loop captures a media lifetime nor checks it after an await, and none of these requests carries an abort signal. React unmounting the old child does not cancel an async function. After switching account while one request is pending, the next iteration therefore issues its old selected ID using the current account configuration. `request-core.ts:223` calls `runtime.getConfig()` for each request, and `background-proxy.ts:839-862` resolves current storage when no explicit snapshot is supplied. On the per-user media databases, integer IDs can overlap, so this can delete a new account's unrelated item or copy the previous account's keywords into it. Aborting search/detail alone does not close this boundary.

**How to verify:** Mount the real selection hook with media IDs 1 and 2 selected. Start bulk delete or bulk keyword addition, defer the first mocked `bgRequest`, switch the configured principal and dispatch the account-boundary event/unmount, then resolve the first request. Assert that no second request occurs. The present loop issues DELETE/PUT for ID 2. An integration variant gives both accounts distinct items with ID 2 and checks which account receives the second mutation.

## 2. [P2] Account remount reloads private collection names from unscoped storage

**Status:** Surviving private-state/cache isolation gap in a direct caller, predating this patch.

**Locations:** `apps/packages/ui/src/components/Review/hooks/useMediaSelection.ts:17,112,125-127`; disclosure sink `apps/packages/ui/src/components/Review/ViewMediaPage.tsx:1474-1478`; changed account-remount boundary at lines 121-130.

The new boundary discards React state and the lifetime-specific search query, but collections are read again from the shared `media:collections:v1` storage key. It has no server or account discriminator, and there is no retirement cleanup for it. Collection names and item counts are rendered directly in the collection selector for the replacement account, even when that account's media listing is empty. Favorites similarly reload from `media:favorites` and can mark unrelated same-numbered items. This means a collection such as a private project/client name remains visible to the next signed-in user despite the search-results test passing.

The same persistence pattern exists for the media-type cache: `useMediaSearch.ts:916-935` repopulates the cleared list from a global `reviewMediaTypesCache` immediately after the remount. That is lower sensitivity than collection names but confirms that resetting the hook state alone does not retire persisted media metadata.

**How to verify:** As account A, create a collection with a distinctive name and favorite one media item. Switch to B on the same server without clearing browser storage; have B's media API return an empty listing. Open the collection filter and observe A's collection name/count. For a component test, seed the storage adapter, render `ViewMediaPage`, dispatch the real account-boundary event, resolve the new listing empty, and assert that the distinctive name is absent; it currently remains.

## 3. [P2] Local-model setup resume loses the model even for configured API-key administrators

**Status:** Regression introduced by the new anonymous projection.

**Locations:** `tldw_Server_API/app/api/v1/endpoints/setup.py:1339-1347`; direct caller `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts:84-89`; consumers `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx:97-117,756-759`.

The state endpoint now preserves filesystem-shaped model IDs only for an authenticated `system.configure` principal. However, the shipped `getFirstRunState()` always requests `noAuth: true`; `request-core.ts:299` treats that as an instruction to skip credential headers. Consequently, a manual API-key setup session still receives the anonymous projection even if its configured key belongs to the administrator. On reopening/reloading setup after selecting a local model such as `/opt/models/local.gguf`, both saved model fields are missing. `providerSelectionFromState()` returns null, while `stepFromState()` can still advance to `first_chat` from the completed step list; rendering that step explicitly requires a non-null `providerSelection`. The first-chat body (including its back/edit actions) is therefore absent. Existing new endpoint tests stub principal resolution and do not exercise this real caller contract.

**How to verify:** Use manual API-key single-user setup with a path-shaped llama.cpp model; finish through MCP tools, then reload before first chat. Inspect GET `/first-run/state` to confirm the key is omitted, the saved default model is absent, and the first-chat controls are missing. A deterministic wizard test can supply a resumed state with completed prerequisite steps and `providers.default_provider` but no `default_model`, then assert that the user has a usable first-chat or provider-reselection path. Preserve anonymous redaction while supplying authenticated restoration when credentials exist and an explicit reselection path when they do not.


## Verified disposition after the independent review

The three findings above describe the reviewed intermediate patch, not outstanding defects in the final candidate. Parent source review traced the corrected direct callers and reran their owning suites.

1. Selection operations capture owner and abort signal before dispatch; pre/post-await guards stop sequential DELETE/PUT, note lookup→delete, late quota/progress publication and retained undo after retirement. Already accepted server operations cannot be rolled back by client cancellation; no subsequent old operation is allowed to dispatch for the replacement account.
2. Favorites/collections use owner-stamped storage envelopes keyed by the existing verified server/account scope. Unknown owners cannot read/write; same-owner reload remains supported. Legacy unowned values stay stored without being adopted by the next account. The shared media-type cache is no longer consumed or populated.
3. Setup state uses available credentials and falls back to anonymous progress only for401. A redacted resume with no model returns to provider selection, avoiding a blank first-chat panel. Backend anonymous path redaction and administrator restore remain intact. Three regressions failed before correction, then105 setup-client/wizard/transport tests passed.

Package-owned frontend, quota controls and the media follow-up evidence are recorded in the parent ledger. No new auth permissions or weaker server checks were introduced. No live account data or VM operations were used during validation.
