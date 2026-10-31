# Recipe Persistence Owner Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Task 8's divergent configured and transport scopes with one recipe-specific expected/actual owner contract shared by the real WebUI and extension surfaces, synchronization, uncertainty recovery, and the request that is actually dispatched.

**Architecture:** A pure owner module canonicalizes the effective transport base, closed authentication source, organization, and stable principal into an opaque `ownerId` plus a separate authorization revision. One authoritative request resolver produces the effective request snapshot and owner; UI surfaces receive only the sanitized owner view, while v2 Save/Update sends `expectedOwnerId` and request-core compares it with the actual snapshot before fetch. Uncertainty is owned by an app-level registry in direct WebUI and the MV3 background in the extension, with scoped markers separated from exact-ID unknown-owner quarantine.

**Tech Stack:** React 18, TypeScript, WXT browser messaging, existing tldw request transport, TanStack Query, Dexie prompt storage, Vitest 4/Testing Library, existing locale tooling, Playwright in the later parity plan.

**Spec:** `Docs/superpowers/specs/2026-07-22-chat-prompt-improvement-recipes-design.md` section 10.6

**Backlog task:** `TASK-12984.2.1`

**Parent implementation:** `Docs/superpowers/plans/2026-08-01-single-text-structured-recipes.md`; this recovery plan replaces only the unapproved ownership/recovery portion of Task 8. Original Tasks 9 and 10 remain blocked until this plan receives independent approval.

## Global Constraints

- Do not modify `buildChatSurfaceScopeKey`, its serialization, or any global chat/session/service-prompt scope consumer.
- Do not move, resize, restyle, or otherwise change the composer Improve prompt control; the layout defect remains a separate post-Track-B task.
- Keep `single_text_recipe_v2.supported=false` until the original Task 10 release gate.
- Keep schema-v1 create, update, pull, sync, and prompt-library behavior compatible. Only schema-v2 create/update requires a trustworthy owner precondition.
- Raw API keys, bearer tokens, cookies, CSRF values, refresh tokens, request headers, and request snapshots must never leave their authoritative context or enter owner IDs, query keys, logs, errors, metrics, browser messages, or synchronized records.
- Use domain-separated SHA-256 for API-key fingerprints, credential revisions, and opaque owner IDs; do not use the legacy FNV chat-surface fingerprint for recipe ownership.
- `RecipePersistenceAuthSource` is exactly `manual_api_key | runtime_api_key | manual_bearer | cookie_session`.
- Runtime API-key override wins when eligible; otherwise cookie transport wins; otherwise configured single-user API key; otherwise configured multi-user bearer. Unsupported or inconsistent combinations have no owner.
- Cookie-session ownership requires an authoritative authenticated user ID. Never infer it from an unused bearer token, stored user hint, username, or email.
- Same-subject bearer rotation preserves `ownerId` but changes `authorizationRevision`.
- UI code receives only `{ ownerId, authorizationRevision }`; effective request snapshots and credential material remain inside direct request-core or the extension background.
- A v2 mutation with missing or mismatched expected/actual owner fails before fetch. Apply, preview, editing, and v1 persistence remain available as specified.
- Once expected and actual owner match, the authoritative request context marks the exact local ID under that owner immediately before fetch. A lost extension response therefore leaves a lock in the background even when the page cannot send a follow-up message. Only a known no-mutation result or completed matching-owner reconciliation clears it.
- Server response bodies cannot supply or override dispatch ownership metadata.
- Unknown-owner quarantine blocks Save/Update for the exact local ID across every owner and is cleared only by explicit confirmed Forget or authoritative process termination; ordinary reconciliation and deletion do not silently clear it.
- Scoped uncertainty is visible and clearable only under the matching `ownerId` plus exact local ID.
- Extension sidepanel and pop-out share owner resolution and uncertainty through the background. Closing one surface does not release another surface's lock.
- Captured unsafe extension mutations never fall back to a second direct request after timeout or ambiguous messaging failure.
- No new runtime dependency or lockfile change. Server-side mutation idempotency remains deferred.
- Update `TASK-12984.2.1` through Backlog MCP/CLI after each task. TypeScript-only tasks have no Bandit target; retain the cumulative Track B Python Bandit gate for original Task 10.

## File Structure

- Create `apps/packages/ui/src/services/recipe-persistence-owner.ts`: pure canonical owner/auth-revision types and hashing; no browser or storage access.
- Create `apps/packages/ui/src/services/tldw/recipe-request-snapshot.ts`: authoritative transport/auth selection and sanitized owner-view projection used by request-core and owner-resolution entry points.
- Modify `apps/packages/ui/src/services/tldw/request-core.ts`: consume one snapshot, enforce expected owner before v2 fetch/retry, and return transport-owned metadata.
- Modify `apps/packages/ui/src/services/api-send.ts`: carry the discriminated recipe-persistence policy/result and forbid ambiguous unsafe extension replay.
- Modify `apps/packages/ui/src/services/prompt-studio.ts`: thread capture/require policy without changing ordinary callers.
- Replace `apps/packages/ui/src/services/recipe-persistence-uncertainty.ts`: thin context-aware client facade over direct singleton or extension-background registry.
- Create `apps/packages/ui/src/services/recipe-persistence-registry.ts`: pure registry class for scoped markers and unknown-owner quarantine.
- Create `apps/packages/ui/src/hooks/useRecipePersistenceOwner.ts`: owner-resolution lifecycle for both real chat surfaces.
- Modify `apps/packages/ui/src/entries/background.ts`: authoritative extension owner-resolution and registry message handlers.
- Modify `apps/packages/ui/src/services/background-proxy.ts` only if the direct WebUI authority needs an existing config/session adapter; do not create a parallel transport.
- Modify `apps/packages/ui/src/services/prompt-sync.ts`: expected-owner input, discriminated dispatch result propagation, and matching-owner cleanup.
- Modify `apps/packages/ui/src/components/Common/PromptAssist/recipes/PromptRecipeBuilder.tsx`: consume authoritative owner view and classify known rejection, scoped uncertainty, or unknown quarantine.
- Modify `apps/packages/ui/src/components/Common/PromptAssist/recipes/SingleFieldRecipeEditor.tsx`: expose explicit unknown-owner recovery without weakening Apply.
- Modify `apps/packages/ui/src/components/Common/PromptSelect.tsx` and `apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx`: use the shared owner hook and exact owner/revision capability keys.
- Modify `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`, `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`, `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`, and `apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx` only to forward the sanitized owner view if the hook cannot live entirely inside the two adapters; no DOM/class/style change is allowed.
- Modify English source/generated locale files only for the confirmed unknown-owner recovery action and explanation.

---

## Stage 1: Canonical Owner and Transport Contract

**Goal:** Make owner identity and dispatch metadata explicit, collision-resistant, and independent from global chat/session scope.

**Success Criteria:** Pure tests cover every auth source, normalized transport base, same-sub refresh, credential change, and absent cookie principal; the request transport compares expected and actual owners from the same snapshot before fetch.

**Tests:** Owner unit tests, request-core transport tests, api-send extension policy tests, existing request/refresh/quickstart/hosted suites.

**Status:** Not Started

### Task 1: Define the recipe-only owner identity

**Files:**

- Create: `apps/packages/ui/src/services/recipe-persistence-owner.ts`
- Create: `apps/packages/ui/src/services/__tests__/recipe-persistence-owner-contract.test.ts`
- Modify: `apps/packages/ui/src/services/recipe-persistence-uncertainty.ts` only to import the new types; do not change registry behavior yet.

**Interfaces:**

- Consumes: `sha256`, `bytesToHex`, and `utf8ToBytes` from the already installed `@noble/hashes`; validated effective transport/auth material from Task 2.
- Produces:

```ts
export type RecipePersistenceAuthSource =
  | "manual_api_key"
  | "runtime_api_key"
  | "manual_bearer"
  | "cookie_session"

export type RecipePersistenceOwnerMaterial = Readonly<{
  effectiveBase: string
  authMode: "single-user" | "multi-user"
  authSource: RecipePersistenceAuthSource
  orgId: string | null
  principalKind: "user" | "api_key"
  principal: string
}>

export type RecipePersistenceOwnerView = Readonly<{
  ownerId: string
  authorizationRevision: string
}>

export function deriveRecipePersistenceOwner(
  material: RecipePersistenceOwnerMaterial,
  credentialRevisionMaterial: string
): RecipePersistenceOwnerView
```

- `principal` is an authoritative user ID for user owners and raw API-key material only inside this pure authority-boundary call. The function hashes key material immediately and never returns it.
- `credentialRevisionMaterial` is the selected API key or bearer token for manual/runtime credentials. Cookie authority supplies a non-secret session revision when available or a fresh per-resolution nonce; cookie values themselves are never inputs.

- [ ] **Step 1: Write failing owner-identity tests**

Create table-driven tests proving:

```ts
it.each([
  ["auth source", { authSource: "manual_api_key" }, { authSource: "runtime_api_key" }],
  ["organization", { orgId: "1" }, { orgId: "2" }],
  ["effective base", { effectiveBase: "https://a.test" }, { effectiveBase: "https://b.test" }],
  ["principal", { principal: "alice" }, { principal: "bob" }]
])("separates %s", (_label, left, right) => {
  expect(owner({ ...base, ...left }).ownerId).not.toBe(owner({ ...base, ...right }).ownerId)
})

it("keeps owner stable but rotates authorization on same-subject token refresh", () => {
  expect(owner(userMaterial, "token-a").ownerId).toBe(owner(userMaterial, "token-b").ownerId)
  expect(owner(userMaterial, "token-a").authorizationRevision)
    .not.toBe(owner(userMaterial, "token-b").authorizationRevision)
})
```

Also assert normalized default ports and trailing slashes compare equal, active deployment base paths compare distinctly, output contains neither raw credentials nor raw custom endpoint text, malformed bases fail, and no function imports or calls `buildChatSurfaceScopeKey`.

- [ ] **Step 2: Run the owner tests and confirm RED**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run src/services/__tests__/recipe-persistence-owner-contract.test.ts --reporter=dot
```

Expected: module-not-found/test failures only; existing controls pass.

- [ ] **Step 3: Implement canonical serialization and domain-separated hashing**

Use a fixed versioned tuple, not object-key iteration:

```ts
const canonical = JSON.stringify([
  "recipe-persistence-owner-v1",
  normalizeEffectiveBase(material.effectiveBase),
  material.authMode,
  material.authSource,
  material.orgId ?? null,
  material.principalKind,
  material.principalKind === "api_key"
    ? digest("recipe-api-key-principal-v1", material.principal)
    : material.principal
])
```

Return `ownerId` and `authorizationRevision` as separate domain-separated digests. Validate the closed enums and non-empty principal before hashing. Export no raw/canonical material.

- [ ] **Step 4: Run owner tests and scoped static checks, then confirm GREEN**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run src/services/__tests__/recipe-persistence-owner-contract.test.ts --reporter=dot
./node_modules/.bin/eslint src/services/recipe-persistence-owner.ts src/services/__tests__/recipe-persistence-owner-contract.test.ts
./node_modules/.bin/prettier --check src/services/recipe-persistence-owner.ts src/services/__tests__/recipe-persistence-owner-contract.test.ts
```

- [ ] **Step 5: Update Backlog and commit Task 1**

```bash
backlog task edit TASK-12984.2.1 --append-notes "Task 1: canonical recipe-only owner identity and credential revision complete; record RED/GREEN and exact test counts." --plain
git add apps/packages/ui/src/services/recipe-persistence-owner.ts apps/packages/ui/src/services/__tests__/recipe-persistence-owner-contract.test.ts apps/packages/ui/src/services/recipe-persistence-uncertainty.ts "backlog/tasks/task-12984.2.1 - Implement-recipe-persistence-owner-contract.md"
git commit -m "feat(prompts): define recipe persistence owner identity (TASK-12984.2.1)"
```

### Task 2: Bind expected and actual owner to one request snapshot

**Files:**

- Create: `apps/packages/ui/src/services/tldw/recipe-request-snapshot.ts`
- Create: `apps/packages/ui/src/services/__tests__/recipe-request-snapshot.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/request-core.ts`
- Modify: `apps/packages/ui/src/services/api-send.ts`
- Modify: `apps/packages/ui/src/services/prompt-studio.ts`
- Modify: `apps/packages/ui/src/services/__tests__/request-core.persistence-scope.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/api-send.test.ts`
- Modify: existing quickstart, hosted, refresh, and background-effective-auth tests when their typed metadata changes.

**Interfaces:**

- Consumes: `deriveRecipePersistenceOwner` from Task 1; existing `resolveBrowserRequestTransport`, cookie-session predicate, runtime API-key override, and an authoritative authenticated-principal callback for every multi-user source (`manual_bearer` and `cookie_session`).
- Produces:

```ts
export type RecipePersistenceRequestPolicy =
  | Readonly<{ mode: "capture" }>
  | Readonly<{
      mode: "require"
      expectedOwnerId: string
      localId: string
    }>

export type RecipePersistenceDispatch =
  | Readonly<{ state: "not_dispatched"; actualOwnerId: null }>
  | Readonly<{ state: "dispatched"; actualOwnerId: string | null }>
  | Readonly<{ state: "unknown"; actualOwnerId: null }>

export type RecipeOwnerResolution = Readonly<{
  view: RecipePersistenceOwnerView
  snapshot: Readonly<{
    url: string
    effectiveBase: string
    headers: Readonly<Record<string, string>>
    credentials?: RequestCredentials
  }>
}>

export type RecipeDispatchAuthority = Readonly<{
  markDispatched(localId: string, ownerId: string): void | Promise<void>
}>
```

`ApiSendPayload` gains `recipePersistence?: RecipePersistenceRequestPolicy` and `ApiSendResponse` gains `recipePersistence?: RecipePersistenceDispatch`. This is local client metadata: `expectedOwnerId`, `localId`, and all dispatch metadata are stripped before HTTP serialization and never appear in a server header or body. Remove the two overlapping capture/require booleans after all callers migrate.

- [ ] **Step 1: Add failing effective-snapshot and dispatch-state tests**

Cover manual API key, runtime override precedence, manual bearer with authoritative user, manual bearer without authoritative user, cookie session with authoritative user, cookie session without user, quickstart same-origin, advanced URL normalization, active deployment base path, absolute/no-auth rejection, and unknown auth combinations. Assert `view` contains no URL, header, or credential fields.

Add request-core tests proving:

```ts
expect(fetchFn).not.toHaveBeenCalled()
expect(result.recipePersistence).toEqual({
  state: "not_dispatched",
  actualOwnerId: null
})
```

for missing/mismatched expected owner. Add a deferred-config test that changes owner before resolution, a same-subject refresh that performs exactly two fetches under one owner, and changed source/base/org/principal refresh tests that perform no retry.

Also inject a fake `RecipeDispatchAuthority` and assert `markDispatched(localId, actualOwnerId)` completes exactly once immediately before `fetchFn`. If marking rejects, assert no fetch. A raw success does not clear the marker because only sync can confirm local reconciliation.

- [ ] **Step 2: Add failing extension fallback/coalescing tests**

Verify an unsafe captured mutation whose background message times out returns `{state:"unknown", actualOwnerId:null}` and never calls direct `tldwRequest`. Verify captured and ordinary GET requests do not share a coalesced promise. Verify response-body fields named `recipePersistence` cannot spoof transport metadata.

- [ ] **Step 3: Run focused transport tests and confirm RED**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/recipe-request-snapshot.test.ts \
  src/services/__tests__/request-core.persistence-scope.test.ts \
  src/services/__tests__/api-send.test.ts \
  src/services/tldw/__tests__/request-core.quickstart.test.ts \
  src/services/tldw/__tests__/request-core.hosted.test.ts \
  src/services/__tests__/tldw-auth.refresh-rotation.test.ts --reporter=dot
```

- [ ] **Step 4: Extract one authoritative snapshot resolver**

Move transport-base normalization and effective auth-source selection behind `resolveRecipeRequestSnapshot`. For captured requests, request-core must use the returned URL, headers, credentials mode, and owner rather than rebuilding any of them. Ordinary non-recipe requests retain their current path.

The resolver accepts the authenticated multi-user principal as an explicit authority input; it does not fetch a user or read component state. The authority obtains that principal from the current-user endpoint for both manual bearer and cookie-session transport. Reject placeholder keys and bearer/cookie credentials without an authoritative principal as no owner; never trust an unverified token claim or component-stored user hint.

- [ ] **Step 5: Enforce expected owner and discriminated metadata**

For `mode:"require"`, validate expected owner, validate the exact local ID, and complete `markDispatched(localId, actualOwnerId)` before setting `state:"dispatched"` or calling `fetchFn`. Overlay dispatch metadata after parsing the response so server JSON cannot replace it. On refresh, build a second immutable snapshot and compare owner/base before retry; do not create a second marker for the same local operation.

For `mode:"capture"`, allow v1/pull dispatch when owner is null but never use null to clear uncertainty. Preserve every existing caller that omits `recipePersistence`.

- [ ] **Step 6: Migrate Prompt Studio options and remove old boolean policy**

Use optional trailing options:

```ts
createPrompt(payload, { recipePersistence: { mode: "require", expectedOwnerId, localId } })
updatePrompt(id, payload, { recipePersistence: { mode: "require", expectedOwnerId, localId } })
getPrompt(id, { recipePersistence: { mode: "capture" } })
```

Do not change callback arity for callers that omit options.

- [ ] **Step 7: Run focused and adjacent transport suites to GREEN**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/recipe-persistence-owner-contract.test.ts \
  src/services/__tests__/recipe-request-snapshot.test.ts \
  src/services/__tests__/request-core.persistence-scope.test.ts \
  src/services/__tests__/api-send.test.ts \
  src/services/tldw/__tests__/request-core.quickstart.test.ts \
  src/services/tldw/__tests__/request-core.hosted.test.ts \
  src/services/__tests__/tldw-auth.refresh-rotation.test.ts \
  src/entries/__tests__/background.effective-auth.test.ts --reporter=dot
bun run --cwd ../../extension compile
```

- [ ] **Step 8: Update Backlog and commit Task 2**

```bash
backlog task edit TASK-12984.2.1 --append-notes "Task 2: expected/actual owner bound to immutable request snapshot; record RED/GREEN, compatibility, and extension replay evidence." --plain
git add apps/packages/ui/src/services/tldw/recipe-request-snapshot.ts apps/packages/ui/src/services/tldw/request-core.ts apps/packages/ui/src/services/api-send.ts apps/packages/ui/src/services/prompt-studio.ts apps/packages/ui/src/services/__tests__/recipe-request-snapshot.test.ts apps/packages/ui/src/services/__tests__/request-core.persistence-scope.test.ts apps/packages/ui/src/services/__tests__/api-send.test.ts apps/packages/ui/src/services/__tests__/tldw-auth.refresh-rotation.test.ts apps/packages/ui/src/services/tldw/__tests__/request-core.quickstart.test.ts apps/packages/ui/src/services/tldw/__tests__/request-core.hosted.test.ts apps/packages/ui/src/entries/__tests__/background.effective-auth.test.ts "backlog/tasks/task-12984.2.1 - Implement-recipe-persistence-owner-contract.md"
git commit -m "fix(prompts): bind recipe writes to request owner (TASK-12984.2.1)"
```

---

## Stage 2: Shared Authority and Recovery State

**Goal:** Make both real surfaces observe one authoritative owner and one uncertainty registry per application authority.

**Success Criteria:** Direct WebUI system/composer share a singleton; extension sidepanel/pop-out share background state; scoped and unknown markers have exact, tested cleanup semantics.

**Tests:** Pure registry tests, background protocol tests, direct/extension facade tests, lifecycle/order tests.

**Status:** Not Started

### Task 3: Move uncertainty and owner resolution to the application authority

**Files:**

- Create: `apps/packages/ui/src/services/recipe-persistence-registry.ts`
- Replace: `apps/packages/ui/src/services/recipe-persistence-uncertainty.ts`
- Create: `apps/packages/ui/src/services/__tests__/recipe-persistence-registry.test.ts`
- Create: `apps/packages/ui/src/services/__tests__/recipe-persistence-authority.test.ts`
- Modify: `apps/packages/ui/src/entries/background.ts`
- Create: `apps/packages/ui/src/entries/__tests__/background.recipe-persistence-owner.test.ts`
- Modify: `apps/packages/ui/src/services/background-proxy.ts` only for the direct authority adapter if required.

**Interfaces:**

- Consumes: owner view and snapshot resolver from Tasks 1–2; an authority-local current-user lookup for the authenticated principal used by manual-bearer and cookie-session sources.
- Produces:

```ts
export type RecipeUncertaintyState = "clear" | "scoped" | "unknown_owner"

export class RecipePersistenceRegistry {
  read(id: string, ownerId: string | null): RecipeUncertaintyState
  markScoped(id: string, ownerId: string): void
  markUnknown(id: string): void
  clearScoped(id: string, ownerId: string): void
  forgetUnknown(id: string): void
}

export async function resolveRecipePersistenceOwnerView(): Promise<
  RecipePersistenceOwnerView | null
>

export async function readRecipePersistenceUncertainty(
  id: string,
  ownerId: string | null
): Promise<RecipeUncertaintyState>
```

- [ ] **Step 1: Write failing pure-registry tests**

Prove exact-ID and owner separation, same-owner clear, cross-owner non-read/non-clear, unknown quarantine blocking every owner, `forgetUnknown` clearing only unknown state, scoped marker surviving Forget, and new instances starting clean. Do not export a production global test-reset function.

- [ ] **Step 2: Write failing authority/protocol tests**

Direct WebUI: system and composer clients see the same registry instance.

Extension: mark from sidepanel, read from pop-out, close/unmount sidepanel, read remains locked; cross-owner clear does nothing; background restart creates a new registry while persisted `syncStatus:error` remains the durable fallback.

Owner resolution: background returns only `{ownerId, authorizationRevision}`. Manual bearer and cookie session call the authoritative current-user path and fail closed on 401/missing user; no decoded-but-unverified bearer claim qualifies. Direct WebUI may reuse `tldwAuth.getCurrentUser()`. Background must call its current-user endpoint through its local non-captured `tldwRequest` runtime rather than calling `bgRequest` or messaging itself. Message payloads and responses contain no key/token/header/snapshot.

Both direct and extension owner-resolution views use the same representative unsafe prompt route (`POST /api/v1/prompts/`) as v2 create; update IDs may change the endpoint suffix but cannot change the resolved transport base or auth source. A test must prove create and update resolution produce the same owner under one effective connection.

- [ ] **Step 3: Run registry/protocol tests and confirm RED**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/recipe-persistence-registry.test.ts \
  src/services/__tests__/recipe-persistence-authority.test.ts \
  src/entries/__tests__/background.recipe-persistence-owner.test.ts --reporter=dot
```

- [ ] **Step 4: Implement the pure registry and direct singleton facade**

Use nested maps/sets keyed by opaque `ownerId` and exact string ID; keep unknown quarantine separate. The facade chooses direct singleton only when no real extension runtime exists.

- [ ] **Step 5: Add extension background messages**

Add narrowly validated messages:

```ts
type RecipePersistenceMessage =
  | { type: "tldw:recipe-owner:resolve" }
  | { type: "tldw:recipe-uncertainty:read"; id: string; ownerId: string | null }
  | { type: "tldw:recipe-uncertainty:mark-scoped"; id: string; ownerId: string }
  | { type: "tldw:recipe-uncertainty:mark-unknown"; id: string }
  | { type: "tldw:recipe-uncertainty:clear-scoped"; id: string; ownerId: string }
  | { type: "tldw:recipe-uncertainty:forget-unknown"; id: string }
```

Reject empty/oversized IDs and malformed owner IDs. Never accept owner material or credentials from the page. Background resolves its own effective owner and cookie principal.

The background also passes its registry's `markScoped` as request-core's `RecipeDispatchAuthority.markDispatched`. Direct WebUI passes the app singleton. This pre-dispatch mark is mandatory for v2 mutations; if it cannot be written, request-core returns `not_dispatched` and does not fetch. Do not clear it merely because raw HTTP returned success—the sync layer clears it only after validating the response and completing local reconciliation.

- [ ] **Step 6: Implement extension client facade with no unsafe fallback**

Owner and registry messages fail closed when background messaging is unavailable. They never fall back to a page-local registry in a real extension because that would split sidepanel/pop-out state.

- [ ] **Step 7: Run tests shuffled to prove isolation**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/recipe-persistence-registry.test.ts \
  src/services/__tests__/recipe-persistence-authority.test.ts \
  src/entries/__tests__/background.recipe-persistence-owner.test.ts \
  --sequence.shuffle --sequence.seed=12984 --reporter=dot
```

- [ ] **Step 8: Update Backlog and commit Task 3**

```bash
backlog task edit TASK-12984.2.1 --append-notes "Task 3: app-wide/background owner authority and uncertainty registry complete; record shuffled protocol evidence." --plain
git add apps/packages/ui/src/services/recipe-persistence-registry.ts apps/packages/ui/src/services/recipe-persistence-uncertainty.ts apps/packages/ui/src/services/__tests__/recipe-persistence-registry.test.ts apps/packages/ui/src/services/__tests__/recipe-persistence-authority.test.ts apps/packages/ui/src/entries/background.ts apps/packages/ui/src/entries/__tests__/background.recipe-persistence-owner.test.ts apps/packages/ui/src/services/background-proxy.ts "backlog/tasks/task-12984.2.1 - Implement-recipe-persistence-owner-contract.md"
git commit -m "fix(prompts): share recipe uncertainty authority (TASK-12984.2.1)"
```

### Task 4: Use authoritative owner and authorization revision in both real surfaces

**Files:**

- Create: `apps/packages/ui/src/hooks/useRecipePersistenceOwner.ts`
- Create: `apps/packages/ui/src/hooks/__tests__/useRecipePersistenceOwner.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PromptSelect.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx`
- Modify: `apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx`
- Modify: `apps/packages/ui/src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx`
- Modify only if required for sanitized prop forwarding: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`, `ComposerToolbar.tsx`, `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`, `ControlRow.tsx`.

**Interfaces:**

- Consumes: `resolveRecipePersistenceOwnerView` from Task 3 and current capability endpoint/query.
- Produces:

```ts
export function useRecipePersistenceOwner(enabled: boolean): {
  owner: RecipePersistenceOwnerView | null
  loading: boolean
  refresh: () => Promise<RecipePersistenceOwnerView | null>
}
```

The builder receives `ownerId`; capability query identity receives both owner fields. No surface receives config snapshots or credentials.

- [ ] **Step 1: Write failing hook lifecycle tests**

Assert disabled mode performs no resolution; opening recipe mode masks persistence while resolving; closing ignores late results; owner/auth revision changes replace the view; extension resolution failures remain null; and refresh does not expose stale prior owner while pending.

- [ ] **Step 2: Write failing real-adapter behavior tests**

For both system and composer adapters, cover advanced normalized base, runtime-key override, manual/cookie source change, same-sub bearer rotation, and missing cookie principal. Reopen the real adapter after a mocked durable-marker failure and assert the returned authoritative owner remains locked without manually injecting transport scope.

Assert capability queries use `['promptCapabilities', ownerId, authorizationRevision]`, refetch on every recipe-mode open, pass undefined capabilities while fetching, and never enable from a stale authorized result.

- [ ] **Step 3: Run hook/adapter tests and confirm RED**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/hooks/__tests__/useRecipePersistenceOwner.test.tsx \
  src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx \
  src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx --reporter=dot
```

- [ ] **Step 4: Implement the owner hook and migrate both adapters**

Resolve only while recipe mode is open. On each open, await a fresh owner then a capability refetch for that exact owner/revision. Keep Apply independent. Remove `promptAssistBackendKey` from persistence ownership; retain it only where unrelated existing behavior still needs the global scope.

- [ ] **Step 5: Audit the four layout files**

If prop forwarding is required, add only typed props and pass-through values. Verify their rendered element order, class names, and control shape are byte-for-byte unchanged by inspecting the focused diff. Do not add wrapper elements.

- [ ] **Step 6: Run real-adapter, Track A, and shell tests to GREEN**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/hooks/__tests__/useRecipePersistenceOwner.test.tsx \
  src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx \
  src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx \
  src/components/Option/Playground/__tests__/ComposerToolbar.layout.guard.test.ts \
  src/components/Sidepanel/Chat/__tests__/SidepanelComposerControlArea.prompt-assist.test.tsx --reporter=dot
```

- [ ] **Step 7: Update Backlog and commit Task 4**

```bash
backlog task edit TASK-12984.2.1 --append-notes "Task 4: both real surfaces use authoritative owner plus credential revision; record reopen/capability/unchanged-layout evidence." --plain
git add apps/packages/ui/src/hooks/useRecipePersistenceOwner.ts apps/packages/ui/src/hooks/__tests__/useRecipePersistenceOwner.test.tsx apps/packages/ui/src/components/Common/PromptSelect.tsx apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx apps/packages/ui/src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx apps/packages/ui/src/components/Sidepanel/Chat/form.tsx apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx "backlog/tasks/task-12984.2.1 - Implement-recipe-persistence-owner-contract.md"
git commit -m "fix(chat): resolve recipe owner in real surfaces (TASK-12984.2.1)"
```

---

## Stage 3: Sync, Builder, and Explicit Recovery

**Goal:** Carry expected/actual owner through persistence and give users accurate, fail-closed recovery for every dispatch state.

**Success Criteria:** No pre-dispatch rejection creates uncertainty; dispatched ambiguity marks the actual owner; unknown dispatch quarantines the exact ID across owners and surfaces; only exact matching recovery clears state.

**Tests:** Real builder-to-transport integration, sync/manual/auto/pull/delete tests, editor recovery tests, locale/ICU checks.

**Status:** Not Started

### Task 5: Integrate owner-aware sync and uncertainty recovery

**Files:**

- Modify: `apps/packages/ui/src/services/prompt-sync.ts`
- Modify: `apps/packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/prompt-sync.auto-sync.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/prompt-sync.uncertainty.test.ts`
- Modify: `apps/packages/ui/src/db/dexie/helpers.ts`
- Modify: `apps/packages/ui/src/db/dexie/__tests__/prompt-rollback.test.ts`
- Modify: `apps/packages/ui/src/components/Common/PromptAssist/recipes/PromptRecipeBuilder.tsx`
- Modify: `apps/packages/ui/src/components/Common/PromptAssist/recipes/SingleFieldRecipeEditor.tsx`
- Modify: `apps/packages/ui/src/components/Common/PromptAssist/recipes/types.ts`
- Modify: `apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/common.json`
- Modify generated locale peer: `apps/packages/ui/src/public/_locales/en/common.json`

**Interfaces:**

- Consumes: owner view/hook, `RecipePersistenceRequestPolicy`, `RecipePersistenceDispatch`, and authority registry facade from Tasks 2–4.
- Produces:

```ts
export type RecipeSyncOwnership = Readonly<{
  dispatch: RecipePersistenceDispatch
  localId: string
}>

export type SyncResult = {
  success: boolean
  localId: string
  serverId?: number
  error?: string
  syncStatus: PromptSyncStatus
  failureKind?: "validation" | "invalid_server_payload" | "transient"
  recipeOwnership: RecipeSyncOwnership | null
}

export type RecipePersistenceInput = Readonly<{
  expectedOwnerId: string
}>
```

Every v2 create/update path receives `RecipePersistenceInput`. V1 callers omit it. Pull captures owner but never requires one.

- [ ] **Step 1: Write failing real-path create/update race tests**

Drive builder → auto/manual sync → Prompt Studio → apiSend → request-core with only storage/config/network boundaries mocked. For both create and update:

- open under owner A;
- switch backend before dispatch so actual B differs and assert no fetch because expected A mismatches;
- reopen under B and retry once;
- return malformed response after B dispatch;
- reject durable error marking;
- close/reopen both real adapters and assert B remains locked, A is not scoped-locked, Apply remains enabled, and no second mutation occurs.

Repeat for principal and auth-source changes. Include same-sub token rotation as an allowed owner-stable control.

- [ ] **Step 2: Write failing dispatch classification tests**

Assert exact behavior:

```ts
expect(classify({ state: "not_dispatched", actualOwnerId: null })).toBe("known_rejection")
expect(classify({ state: "dispatched", actualOwnerId: ownerB })).toBe("scoped_uncertain")
expect(classify({ state: "dispatched", actualOwnerId: null })).toBe("unknown_owner")
expect(classify({ state: "unknown", actualOwnerId: null })).toBe("unknown_owner")
```

Mismatched returned local ID is unknown-owner quarantine. Server body spoof attempts do not affect classification. Add response-certainty cases:

- valid 2xx response plus completed local reconciliation clears the exact scoped pre-dispatch marker;
- a typed validation/authorization/conflict rejection that contractually performs no mutation clears the exact scoped marker before exact rollback;
- a failure proven to occur before dispatch creates no marker and may retain existing local-pending recovery;
- malformed 2xx, connection loss, timeout after dispatch, and unclassified 5xx retain scoped uncertainty and cannot auto-retry;
- extension dispatch state `unknown` adds exact-ID unknown-owner quarantine even if the background may also retain a scoped pre-dispatch marker.

- [ ] **Step 3: Write failing cleanup and deletion tests**

Manual/auto create/update, keep-local, conflict-copy, keep-server, and pull clear only exact matching scoped markers after local reconciliation succeeds. Null/other owner does not clear. Ordinary deletion does not clear another owner's scoped marker or unknown quarantine. Exact matching-owner permanent deletion clears only its scoped marker.

- [ ] **Step 4: Write failing recovery UI tests**

Unknown-owner state shows accurate localized copy, disables Save and Update across source/owner changes, leaves Apply enabled, and exposes `Forget unresolved operation`. Clicking it shows an inline confirmation with duplicate-risk copy; Cancel changes nothing; Confirm clears only exact-ID unknown quarantine and does not call create/update/delete. Scoped uncertainty does not show Forget.

- [ ] **Step 5: Run focused sync/builder/editor tests and confirm RED**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/prompt-sync.structured-prompts.test.ts \
  src/services/__tests__/prompt-sync.auto-sync.test.ts \
  src/services/__tests__/prompt-sync.uncertainty.test.ts \
  src/db/dexie/__tests__/prompt-rollback.test.ts \
  src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx \
  src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx \
  src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx --reporter=dot
```

- [ ] **Step 6: Thread expected owner and discriminated result through sync**

Create/update require `expectedOwnerId` only when `prompt_schema_version === 2`. Return recipe ownership on success and every failure branch. Never re-resolve owner inside sync after response. Use only the transport result for uncertainty and cleanup.

- [ ] **Step 7: Implement builder classification and app-wide registry calls**

Known pre-dispatch rejection follows existing exact rollback. A typed server rejection proven not to mutate clears its pre-dispatch scoped marker before exact rollback. Valid success clears only after local reconciliation. Dispatched malformed/timeout/unclassified failure retains the pre-dispatch scoped marker and record. Missing/unknown dispatch ownership retains the record and adds unknown quarantine. Always invalidate prompt queries. Preserve local-pending behavior only for failures proven to occur before dispatch, and preserve the existing synchronous no-double-click guard.

- [ ] **Step 8: Implement explicit unknown-owner recovery**

Add a single inline confirmation state to the shared editor; do not stack a modal. Wire Confirm only to `forgetUnknown(id)`. Keep focus restoration, polite live status, 44px targets, and localized accessible names.

- [ ] **Step 9: Synchronize locales and run focused tests to GREEN**

```bash
cd apps
bun run --cwd extension locales:sync
cd packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/prompt-sync.structured-prompts.test.ts \
  src/services/__tests__/prompt-sync.auto-sync.test.ts \
  src/services/__tests__/prompt-sync.uncertainty.test.ts \
  src/db/dexie/__tests__/prompt-rollback.test.ts \
  src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx \
  src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx \
  src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx --reporter=dot
```

- [ ] **Step 10: Update Backlog and commit Task 5**

```bash
backlog task edit TASK-12984.2.1 --append-notes "Task 5: owner-aware sync, exact dispatch classification, matching cleanup, and explicit unknown-owner recovery complete; record RED/GREEN and locale evidence." --plain
git add apps/packages/ui/src/services/prompt-sync.ts apps/packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts apps/packages/ui/src/services/__tests__/prompt-sync.auto-sync.test.ts apps/packages/ui/src/services/__tests__/prompt-sync.uncertainty.test.ts apps/packages/ui/src/db/dexie/helpers.ts apps/packages/ui/src/db/dexie/__tests__/prompt-rollback.test.ts apps/packages/ui/src/components/Common/PromptAssist/recipes/PromptRecipeBuilder.tsx apps/packages/ui/src/components/Common/PromptAssist/recipes/SingleFieldRecipeEditor.tsx apps/packages/ui/src/components/Common/PromptAssist/recipes/types.ts apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx apps/packages/ui/src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx apps/packages/ui/src/assets/locale/en/common.json apps/packages/ui/src/public/_locales/en/common.json "backlog/tasks/task-12984.2.1 - Implement-recipe-persistence-owner-contract.md"
git commit -m "fix(prompts): recover ambiguous recipe writes safely (TASK-12984.2.1)"
```

### Task 6: Prove the revised contract and hand back to Task 8 review

**Files:**

- Create: `apps/packages/ui/src/services/__tests__/recipe-persistence-owner.contract.test.ts`
- Modify: `backlog/tasks/task-12984.2.1 - Implement-recipe-persistence-owner-contract.md` through Backlog MCP/CLI only.
- Do not modify the capability flag, E2E files from original Task 9, or composer layout.

**Interfaces:**

- Consumes: every interface produced by Tasks 1–5.
- Produces: one cross-layer contract suite and a clean review package suitable for independent Task 8 approval.

- [ ] **Step 1: Add one cross-layer contract matrix**

Use table-driven scenarios for WebUI system, WebUI composer, extension sidepanel, and extension pop-out across:

- manual API key;
- runtime API-key override;
- manual bearer and same-sub refresh;
- cookie session with and without authoritative principal;
- quickstart and advanced normalized base;
- backend/principal/org/auth-source change before dispatch and refresh;
- known pre-dispatch rejection, scoped ambiguous result, and unknown dispatch;
- close/reopen, cross-surface read, matching reconciliation, cross-owner non-clear, and confirmed Forget.

The matrix asserts zero duplicate remote mutations and unchanged local Apply in every failure case.

- [ ] **Step 2: Run the complete revised-owner gate shuffled**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/services/__tests__/recipe-persistence-owner-contract.test.ts \
  src/services/__tests__/recipe-request-snapshot.test.ts \
  src/services/__tests__/recipe-persistence-registry.test.ts \
  src/services/__tests__/recipe-persistence-authority.test.ts \
  src/services/__tests__/recipe-persistence-owner.contract.test.ts \
  src/services/__tests__/request-core.persistence-scope.test.ts \
  src/services/__tests__/api-send.test.ts \
  src/services/__tests__/prompt-sync.uncertainty.test.ts \
  src/db/dexie/__tests__/firefox-prompt-write-order.test.ts \
  src/db/dexie/__tests__/prompt-rollback.test.ts \
  src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.test.tsx \
  src/components/Common/PromptAssist/recipes/__tests__/PromptRecipeBuilder.dispatch.test.tsx \
  src/components/Common/PromptAssist/recipes/__tests__/SingleFieldRecipeEditor.test.tsx \
  src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx \
  src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx \
  src/entries/__tests__/background.recipe-persistence-owner.test.ts \
  --sequence.shuffle --sequence.seed=12984 --reporter=dot
```

- [ ] **Step 3: Run existing compatibility and Track A/Track B gates**

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run \
  src/components/Option/Prompt/__tests__/structured-prompt-utils.test.ts \
  src/components/Common/PromptAssist \
  src/services/__tests__/prompt-sync.structured-prompts.test.ts \
  src/services/__tests__/prompt-sync.auto-sync.test.ts \
  src/components/Option/Prompt/__tests__ --reporter=dot
cd ../..
bun run --cwd extension compile
```

Run the documented isolated exact test for any known 5-second aggregate timeout; do not raise timeouts or mask failures. Run backend capability/authorization tests to prove support remains false and v1/v2 server contracts remain unchanged.

- [ ] **Step 4: Run static, locale, and security-scope gates**

```bash
cd apps
bun run --cwd extension locales:sync
bun run --cwd extension locales:sync -- --check
bun run --cwd extension compile
cd packages/ui
./node_modules/.bin/eslint \
  src/services/recipe-persistence-owner.ts \
  src/services/recipe-persistence-registry.ts \
  src/services/recipe-persistence-uncertainty.ts \
  src/services/tldw/recipe-request-snapshot.ts \
  src/services/tldw/request-core.ts \
  src/services/api-send.ts \
  src/services/prompt-sync.ts \
  src/hooks/useRecipePersistenceOwner.ts \
  src/components/Common/PromptAssist/recipes/PromptRecipeBuilder.tsx \
  src/components/Common/PromptAssist/recipes/SingleFieldRecipeEditor.tsx
git diff --check
```

Run the package TypeScript check and save its output:

```bash
cd apps/packages/ui
./node_modules/.bin/tsc --noEmit --pretty false -p tsconfig.json > /tmp/task-12984-2-1-tsc.log 2>&1
! rg -n "recipe-persistence-owner|recipe-persistence-registry|recipe-persistence-uncertainty|recipe-request-snapshot|useRecipePersistenceOwner|PromptRecipeBuilder|PromptAssistComposerAction|PromptSelect" /tmp/task-12984-2-1-tsc.log
```

Expected: the first command either passes or retains only documented untouched baseline diagnostics; the `rg` command prints no touched-path diagnostic. Scan added lines for raw credentials, logging, unsafe HTML, `Record<string, any>`, TODO/FIXME, duplicated global-scope code, and style/layout changes. No Python file is expected; document the Bandit skip rather than scanning TypeScript.

- [ ] **Step 5: Independently review before declaring Task 8 approved**

Create a review package from this recovery plan's base through HEAD. The reviewer must trace one owner value from each real surface to request dispatch and back through uncertain reopen/reconciliation, inspect every registry clear site, and verify extension background authority. Any production fix returns to the owning task with strict RED/GREEN.

- [ ] **Step 6: Finalize the child task and commit contract coverage**

After independent approval:

```bash
backlog task edit TASK-12984.2.1 -s Done --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --append-final-summary "Implemented and independently approved one recipe-specific expected/actual owner contract across real WebUI/extension surfaces, request dispatch, sync, and uncertainty recovery. Record final tests and commit range." --plain
git add apps/packages/ui/src/services/__tests__/recipe-persistence-owner.contract.test.ts "backlog/tasks/task-12984.2.1 - Implement-recipe-persistence-owner-contract.md"
git commit -m "test(prompts): verify recipe owner contract (TASK-12984.2.1)"
```

Then return to the original plan at Task 9. Do not enable `single_text_recipe_v2` or change composer placement as part of this child task.
