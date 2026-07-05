# WebUI Setup Choice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-wizard WebUI setup choice that clearly explains WebUI setup versus API server `/setup`, opens API setup separately when safe, and prevents blocked backend state from entering the normal WebUI wizard.

**Architecture:** Add a small pure resolver/helper module and a focused `SetupEntryChoice` component under the existing onboarding area. Integrate that component into the shared `/setup` route so users see the decision before `UnifiedSetupWizard`. Keep all backend security behavior unchanged.

**Tech Stack:** React 18, TypeScript, Vitest, Testing Library, Playwright, Next.js WebUI package under `apps/tldw-frontend`, shared UI package under `apps/packages/ui`.

---

## Scope Check

This is one WebUI onboarding change. It touches no backend setup state machine and no provider-save behavior. It does not change root `/` first-run behavior; that can be handled in a separate spec if product decides root should also show the same choice.

## File Structure

- Create `apps/packages/ui/src/components/Option/Onboarding/setup-entry-choice-utils.ts`
  - Pure trigger and URL-resolution helpers.
  - No React imports and no browser globals except values passed in by callers.
- Create `apps/packages/ui/src/components/Option/Onboarding/SetupEntryChoice.tsx`
  - Plain-language setup choice UI.
  - Receives state, metadata, configured server URL, current origin, and callbacks.
- Create `apps/packages/ui/src/components/Option/Onboarding/__tests__/setup-entry-choice-utils.test.ts`
  - Fast resolver and trigger tests.
- Create `apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupEntryChoice.test.tsx`
  - Component behavior tests for copy, blocked state, link, fallback, and refresh.
- Modify `apps/packages/ui/src/routes/option-setup.tsx`
  - Show `SetupEntryChoice` before the wizard when backend setup is incomplete.
  - Hide the manual connection card while the choice is active.
- Modify `apps/packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx`
  - Route-level tests for `/setup` integration and manual-card hiding.
- Modify `apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts`
  - Add `/setup` smoke assertions for desktop/mobile layout and safe API setup link/fallback behavior.

## Task 1: Pure Trigger And API Setup URL Resolver

**Files:**
- Create: `apps/packages/ui/src/components/Option/Onboarding/setup-entry-choice-utils.ts`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/setup-entry-choice-utils.test.ts`

- [ ] **Step 1: Write failing tests for trigger rules**

Create the test file with fixtures for `FirstRunState` and `FirstRunMetadata`. Cover:

```ts
expect(shouldShowSetupEntryChoice(state("not_started"), metadata())).toBe(true)
expect(shouldShowSetupEntryChoice(state("in_progress"), metadata())).toBe(true)
expect(shouldShowSetupEntryChoice(state("first_chat_complete"), metadata())).toBe(true)
expect(shouldShowSetupEntryChoice(state("blocked"), metadata())).toBe(true)
expect(shouldShowSetupEntryChoice(state("completed"), metadata())).toBe(false)
expect(shouldShowSetupEntryChoice(state("skipped"), metadata())).toBe(false)
expect(shouldShowSetupEntryChoice(null, metadata())).toBe(false)
expect(shouldShowSetupEntryChoice(state("not_started"), null)).toBe(false)
expect(
  shouldShowSetupEntryChoice(state("not_started"), metadata({ setup_required: false }))
).toBe(false)
expect(
  shouldShowSetupEntryChoice(state("not_started"), metadata({ setup_completed: true }))
).toBe(false)
```

- [ ] **Step 2: Write failing tests for browser-openable URL resolution**

Cover the examples from the spec:

```ts
expect(resolveApiSetupUrl({
  metadata: metadata({ connection: { api_origin: "http://127.0.0.1:8000" } }),
  configuredServerUrl: null,
  currentOrigin: "http://127.0.0.1:8080",
})?.href).toBe("http://127.0.0.1:8000/setup")

expect(resolveApiSetupUrl({
  metadata: metadata({ connection: { api_origin: "http://app:8000" } }),
  configuredServerUrl: null,
  currentOrigin: "http://127.0.0.1:8080",
})).toBeNull()

expect(resolveApiSetupUrl({
  metadata: metadata({ connection: { api_origin: "http://server:8000" } }),
  configuredServerUrl: null,
  currentOrigin: "http://server:8080",
})?.href).toBe("http://server:8000/setup")
```

Also test same-origin rejection, `metadata.connection.frontend_origin` rejection, private LAN acceptance, configured server fallback, invalid URL rejection, and preserving an existing `/setup` path without appending `/setup/setup`.

- [ ] **Step 3: Run the failing resolver tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/setup-entry-choice-utils.test.ts
```

Expected: fail because the helper module does not exist.

- [ ] **Step 4: Implement the helper module**

Export these functions and types:

```ts
import type { FirstRunMetadata, FirstRunState } from "@/types/setup-onboarding"

export type ApiSetupUrlResolution = {
  href: string
  source: "metadata" | "configured_server"
}

export const mutableWebUiSetupStatuses = new Set([
  "not_started",
  "in_progress",
  "first_chat_complete"
])

export const setupEntryChoiceStatuses = new Set([
  ...mutableWebUiSetupStatuses,
  "blocked"
])

export const isBlockedSetupState = (state: FirstRunState | null): boolean =>
  state?.status === "blocked"

export const isMutableWebUiSetupState = (
  state: FirstRunState | null
): boolean => Boolean(state && mutableWebUiSetupStatuses.has(state.status))

export const shouldShowSetupEntryChoice = (
  state: FirstRunState | null,
  metadata: FirstRunMetadata | null
): boolean => {
  if (!state || !metadata) return false
  return (
    metadata.setup_required === true &&
    metadata.setup_completed === false &&
    setupEntryChoiceStatuses.has(state.status)
  )
}

export function resolveApiSetupUrl(input: {
  metadata: FirstRunMetadata | null
  configuredServerUrl?: string | null
  currentOrigin?: string | null
}): ApiSetupUrlResolution | null {
  // Try metadata.connection.api_origin, then configuredServerUrl.
  // Return null when neither candidate passes the browser-openable predicate.
}
```

Implementation notes:

- Parse candidates with `new URL(candidate)`.
- Accept only `http:` and `https:`.
- Reject candidate origins equal to `currentOrigin` or `metadata.connection.frontend_origin`.
- Accept loopback hosts: `localhost`, `127.0.0.1`, IPv4 `127.0.0.0/8`, and `[::1]`.
- Accept same hostname with different port when the current origin is parseable.
- Accept RFC1918 IPv4 ranges: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`.
- Accept DNS names that contain at least one dot.
- Reject single-label non-loopback hostnames unless the same-hostname/different-port rule applies.
- Build the returned href by replacing any path with `/setup`, except an existing `/setup` path should remain `/setup`.

- [ ] **Step 5: Run resolver tests green**

Run:

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/setup-entry-choice-utils.test.ts
```

Expected: pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add apps/packages/ui/src/components/Option/Onboarding/setup-entry-choice-utils.ts apps/packages/ui/src/components/Option/Onboarding/__tests__/setup-entry-choice-utils.test.ts
git commit -m "feat: add setup entry URL resolver"
```

## Task 2: SetupEntryChoice Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/Onboarding/SetupEntryChoice.tsx`
- Test: `apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupEntryChoice.test.tsx`
- Uses: `apps/packages/ui/src/components/Option/Onboarding/setup-entry-choice-utils.ts`

- [ ] **Step 1: Write failing component tests**

Mock state/metadata and render `SetupEntryChoice`. Cover:

- It renders one `h1` with text like `Choose where to set up tldw`.
- It explains that the user is in the WebUI and API server setup opens separately.
- `Set up in WebUI` calls `onStartWebUiSetup` when state is mutable.
- In `blocked` state, `Set up in WebUI` is disabled and the recovery copy is visible.
- `Open API server setup` is an `<a>` with `target="_blank"` and `rel="noopener noreferrer"`.
- Clicking the API setup link reveals `I finished API server setup`.
- Fallback guidance still shows `I finished API server setup` so users can re-check after manually opening API setup elsewhere.
- The refresh button calls `onRefreshSetupState`.
- When no browser-openable URL is available, fallback guidance is shown and no authoritative API setup link is rendered.
- Local-only and remote-setup warning copy changes based on `metadata.connection.browser_access` and `metadata.remote_setup_enabled`.

- [ ] **Step 2: Run the failing component tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/SetupEntryChoice.test.tsx
```

Expected: fail because the component does not exist.

- [ ] **Step 3: Implement `SetupEntryChoice`**

Use a focused prop type:

```ts
type SetupEntryChoiceProps = {
  state: FirstRunState | null
  metadata: FirstRunMetadata | null
  configuredServerUrl?: string | null
  currentOrigin?: string | null
  onStartWebUiSetup: () => void
  onRefreshSetupState: () => Promise<void> | void
}
```

Behavior:

- Resolve `currentOrigin` from `window.location.origin` only when the prop is absent and `window` exists.
- Use `resolveApiSetupUrl`.
- Use a real anchor for `Open API server setup`, not `window.open`.
- On anchor click, set local `apiSetupOpened` state to true.
- Always show the resolved URL as readable text under the link.
- Show fallback text when `resolveApiSetupUrl` returns null:
  `Open the API server setup page on the machine running tldw. For the default local install this is usually http://127.0.0.1:8000/setup.`
- Show `I finished API server setup` in both link and fallback states. In the link state, reveal it after the API setup link is clicked. In fallback state, show it immediately because the user may need to open the API setup URL manually.
- Disable the WebUI setup button when `isBlockedSetupState(state)` is true.
- Keep button text explicit: `Set up in WebUI`, `Open API server setup`, `I finished API server setup`.
- Keep the visual structure unframed beyond one route-level panel; do not nest cards.

- [ ] **Step 4: Run component tests green**

Run:

```bash
bunx vitest run src/components/Option/Onboarding/__tests__/SetupEntryChoice.test.tsx
```

Expected: pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add apps/packages/ui/src/components/Option/Onboarding/SetupEntryChoice.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupEntryChoice.test.tsx
git commit -m "feat: add setup entry choice component"
```

## Task 3: `/setup` Route Integration

**Files:**
- Modify: `apps/packages/ui/src/routes/option-setup.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx`

- [ ] **Step 1: Write failing `/setup` route tests**

Extend the existing mocked hook return shape to include `refresh`.

Add tests:

```ts
it("shows the setup entry choice before the wizard when backend setup is incomplete", () => {
  // state: not_started, metadata: setup_required true, setup_completed false
  // expect choice h1 visible
  // expect Connect your tldw server absent
  // expect UnifiedSetupWizard absent until Set up in WebUI is clicked
})
```

```ts
it("enters the existing wizard after choosing WebUI setup and can go back to choices", () => {
  // click Set up in WebUI
  // expect unified setup shell visible
  // click Back to setup choices
  // expect choice visible again
})
```

```ts
it("keeps blocked setup on the recovery choice instead of routing into the wizard", () => {
  // state: blocked
  // expect Set up in WebUI disabled
  // expect UnifiedSetupWizard absent
})
```

```ts
it("refreshes first-run state after API setup handoff", async () => {
  // click Open API server setup
  // click I finished API server setup
  // expect refresh called
})
```

Add a blocked transition test:

```ts
it("returns to the recovery choice if refreshed state becomes blocked after WebUI mode was selected", async () => {
  // render not_started, click Set up in WebUI
  // rerender with state: blocked
  // expect choice visible, Set up in WebUI disabled, UnifiedSetupWizard absent
})
```

Add recovery availability tests:

```ts
it("keeps the manual connection and recovery UI when setup metadata is missing", () => {
  // state: not_started, metadata: null, loading: false
  // expect Connect your tldw server visible
  // expect Setup operator recovery visible
  // expect setup entry choice absent
})

it("keeps the manual connection and recovery UI when setup state is missing", () => {
  // state: null, metadata: setup_required true/setup_completed false, loading: false
  // expect Connect your tldw server visible
  // expect Setup operator recovery visible
  // expect setup entry choice absent
})
```

- [ ] **Step 2: Run failing `/setup` route tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-setup-readiness.test.tsx
```

Expected: fail because the route still renders the wizard immediately.

- [ ] **Step 3: Integrate the choice into `option-setup.tsx`**

Implementation outline:

```ts
import { SetupEntryChoice } from "@/components/Option/Onboarding/SetupEntryChoice"
import { useConnectionActions, useConnectionState } from "@/hooks/useConnectionState"
import {
  isMutableWebUiSetupState,
  shouldShowSetupEntryChoice
} from "@/components/Option/Onboarding/setup-entry-choice-utils"
```

Route state:

```ts
const [setupEntryMode, setSetupEntryMode] = useState<"choice" | "webui">("choice")
const { serverUrl: configuredServerUrl } = useConnectionState()
const { state, metadata, loading, refresh, adoptState } = useSetupOnboarding()
const setupChoiceRequired = shouldShowSetupEntryChoice(state, metadata)
const showWizard = setupEntryMode === "webui" && isMutableWebUiSetupState(state)
const shouldShowChoice = setupChoiceRequired && !showWizard
```

Render rules:

- If `shouldShowChoice`, render `SetupEntryChoice` and hide the manual connection card and recovery panel.
- If `showWizard`, render a low-emphasis `Back to setup choices` button above `UnifiedSetupWizard`.
- If `setupEntryMode === "webui"` but `state` is no longer mutable, `showWizard` must be false and `shouldShowChoice` must become true when `setupChoiceRequired` is true.
- Keep the manual connection card and `SetupRequiredPanel` for metadata/state load failures and non-choice recovery states.
- Pass `configuredServerUrl` from `useConnectionState`, not the manual form's local `serverUrl` input.
- Pass `onRefreshSetupState={refresh}`.

- [ ] **Step 4: Run `/setup` route tests green**

Run:

```bash
bunx vitest run src/routes/__tests__/option-setup-readiness.test.tsx
```

Expected: pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add apps/packages/ui/src/routes/option-setup.tsx apps/packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx
git commit -m "feat: gate setup route with setup choice"
```

## Task 4: Browser Workflow And Final Verification

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts`
- Possibly modify: `apps/packages/ui/src/routes/__tests__/core-route-identity.test.tsx` if its route mocks need `refresh` or `serverUrl`

- [ ] **Step 1: Add `/setup` handoff smoke coverage**

Add a compact Playwright test:

```ts
test('explains API server setup before the WebUI wizard on /setup', async ({ page, diagnostics }) => {
  await prepareFirstRunPage(page);
  await installUnifiedFirstRunApi(page);

  await page.goto('/setup');

  await expect(page.getByRole('heading', { name: /choose where to set up tldw/i })).toBeVisible();
  await expect(page.getByRole('link', { name: /open api server setup/i })).toHaveAttribute(
    'href',
    'http://127.0.0.1:8000/setup'
  );
  await expect(page.getByText(/opens in a new tab/i)).toBeVisible();
  await page.getByRole('button', { name: /set up in webui/i }).click();
  await expect(page.getByRole('heading', { name: /first-time setup/i })).toBeVisible();

  await assertNoCriticalErrors(diagnostics);
});
```

- [ ] **Step 2: Add mobile `/setup` smoke coverage**

In the same Playwright file, add a compact mobile viewport check:

```ts
test('keeps the API setup choice usable on mobile /setup', async ({ page, diagnostics }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await prepareFirstRunPage(page);
  await installUnifiedFirstRunApi(page);

  await page.goto('/setup');

  await expect(page.getByRole('heading', { name: /choose where to set up tldw/i })).toBeVisible();
  await expect(page.getByRole('button', { name: /set up in webui/i })).toBeVisible();
  await expect(page.getByRole('link', { name: /open api server setup/i })).toBeVisible();

  await assertNoCriticalErrors(diagnostics);
});
```

- [ ] **Step 3: Add `/setup` mocked first-chat handoff coverage**

Add or update a Playwright test that starts at `/setup`, chooses WebUI setup,
and completes the same mocked first-chat path that the existing root first-run
test covers. Prefer extracting a local helper in
`unified-first-run-onboarding.spec.ts` if needed to avoid duplicating every
wizard click.

Required flow:

```ts
await prepareFirstRunPage(page);
const mock = await installUnifiedFirstRunApi(page);

await page.goto('/setup');
await expect(page.getByRole('heading', { name: /choose where to set up tldw/i })).toBeVisible();
await page.getByRole('button', { name: /set up in webui/i }).click();

// Proceed through the existing mocked first-chat path:
// - select Solo, Docker
// - acknowledge privacy/security
// - select OpenAI, provide key/model, validate, save, continue
// - accept ingest/audio/optional defaults
// - send test chat
// - assert first-source milestone appears

expect(mock.firstChatRequests).toHaveLength(1);
expect(mock.completeRequests).toHaveLength(1);
expect(mock.state.status).toBe('completed');
```

- [ ] **Step 4: Run focused Vitest suites**

Run from `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Option/Onboarding/__tests__/setup-entry-choice-utils.test.ts \
  src/components/Option/Onboarding/__tests__/SetupEntryChoice.test.tsx \
  src/routes/__tests__/option-setup-readiness.test.tsx \
  src/routes/__tests__/setup-status.test.ts
```

Expected: all pass.

- [ ] **Step 5: Run focused Playwright suite**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/unified-first-run-onboarding.spec.ts --reporter=line
```

Expected: all tests in the file pass.

- [ ] **Step 6: Run typecheck**

Run from `apps/tldw-frontend`:

```bash
bun run typecheck
```

Expected: no TypeScript errors.

- [ ] **Step 7: Run Bandit on touched Python scope**

This task should touch only TypeScript/React and Markdown. Record that Bandit is skipped because no Python code changed. Do not run Bandit unless Python files are touched during implementation.

- [ ] **Step 8: Commit verification updates**

```bash
git add apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts
```

If `core-route-identity.test.tsx` or other route tests needed mock shape updates, include them in the same commit:

```bash
git add apps/packages/ui/src/routes/__tests__/core-route-identity.test.tsx
```

Then commit:

```bash
git commit -m "test: cover setup choice first-run handoff"
```

## Final Checks

- [ ] Confirm `git status --short` only shows unrelated pre-existing workspace changes.
- [ ] Record verification commands and outcomes in the final response.
- [ ] Update `TASK-12123` with final summary, touched files, verification results, and Bandit skip reason.
- [ ] If implementation created commits, include the commit hashes in the final response.
