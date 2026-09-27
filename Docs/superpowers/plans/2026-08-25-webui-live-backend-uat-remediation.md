# WebUI Live-Backend UAT Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the eight initial live-backend WebUI defects and gate failures, then execute and remediate the complete Tier-1, Tier-2, and Tier-3 Playwright inventories against isolated real services.

**Architecture:** Keep production fixes at the first broken boundary: shared cancellation classification plus connection corroboration, explicit Research Workspace parent readiness, container-aware Prompt layout, and existing route-error infrastructure. Build deterministic UAT orchestration around disposable backend databases and the repository OpenAI-compatible model service, while reporting intercepted tests separately from genuine live-backend evidence.

**Tech Stack:** Next.js 16, React, TypeScript, Zustand, Vitest, Testing Library, Playwright, Bun, Node.js scripts, FastAPI/Uvicorn, the repository `mock_openai_server`, Backlog.md.

**Spec:** `Docs/superpowers/specs/2026-08-25-webui-live-backend-uat-remediation-design.md`

## Global Constraints

- Work only in `.worktrees/webui-live-uat-remediation` on `codex/webui-live-uat-remediation`; do not modify the user's dirty checkout.
- Baseline evidence is `origin/dev@b1d0aed671dcf45bbe4211a9690022c083c99feb`; synchronize with then-current `origin/dev` only after the initial red-green tasks and before final certification.
- Use `TLDW_E2E_ALLOW_OFFLINE=0`, one Playwright worker initially, explicit service URLs, and the isolated profile's synthetic single-user API key; never use a real credential or a second hard-coded key.
- Do not add commercial provider credentials, broadly suppress console errors, weaken assertions into catches, or skip after a test mutates server state.
- A mocked or intercepted Playwright case counts as executed UI/contract coverage, not live-backend evidence.
- Every newly confirmed product or gate-drift defect receives a searched, reviewable child task under `TASK-13124` before repository edits for that defect.
- Each production fix follows red-green-refactor and receives its own reviewable commit where dependencies permit.
- After three failed attempts on one issue, record the evidence and stop for architecture review.
- Run Bandit for touched Python files; record an explicit frontend-only skip when no Python implementation is touched.

## Stage Map

### Stage 1: Runtime and Connection Reliability
**Goal:** Establish a bounded development runtime and prevent false backend-unavailable UI.
**Success Criteria:** One bundler qualifies under the recorded guardrails or a bounded containment is documented; cancellations do not produce outage UI while genuine outages still do.
**Tests:** Runtime probe tests, package-script contract, request-event/background-proxy tests, WebLayout connection tests.
**Status:** Complete

### Stage 2: Workspace and Settings Defects
**Goal:** Correct Research initialization, Prompt responsive behavior, Settings form ownership, and Kanban gate drift.
**Success Criteria:** Focused unit/component tests and affected Playwright cases pass without speculative CSS or production-only test hooks.
**Tests:** Research hook/reconciliation tests, Prompt geometry test, Settings tests, all-pages Kanban recovery.
**Status:** Complete

### Stage 3: Deterministic Chat and Real-Server Gate
**Goal:** Prove character/persona continuity through a real backend and reduce the legacy 17-case suite to unique, honest coverage.
**Success Criteria:** Deterministic generation wiring works, no-provider behavior is truthful, and every legacy case is mapped before deletion or repair.
**Tests:** Character/Persona focused real-server cases, model readiness tests, maintained real-server gate.
**Status:** Complete

### Stage 4: Complete Tier-1–3 UAT Loop
**Goal:** Execute every listed Tier-1–3 test with live services available and remediate each confirmed defect.
**Success Criteria:** Exact denominators, mock inventory, evidence classifications, child tasks, fixes, and complete affected-tier reruns are recorded.
**Tests:** Complete `tier-1`, `tier-2`, and `tier-3` projects plus route sweep and dedicated Research/Chat suites.
**Status:** Complete

### Stage 5: Current-Dev Certification and Closeout
**Goal:** Reconcile current `origin/dev`, repeat final gates, and finalize Backlog records.
**Success Criteria:** Final evidence names the exact synchronized commit with no unexplained failure, hang, page error, or false outage popup.
**Tests:** Focused regressions, full tiers, typecheck, touched-scope lint, build when required, and applicable Bandit.
**Status:** Complete

## File Structure

- `apps/tldw-frontend/scripts/dev-runtime-uat.mjs`: orchestrates one bundler probe and writes bounded resource/health samples.
- `apps/tldw-frontend/scripts/dev-runtime-uat-lib.mjs`: pure process-table parsing and runtime threshold evaluation.
- `apps/tldw-frontend/scripts/__tests__/dev-runtime-uat.test.ts`: unit coverage for process discovery and pass/fail evaluation.
- `apps/tldw-frontend/__tests__/next-config-dev-watch-guard.test.ts`: locks the selected default and explicit alternate dev commands.
- `apps/packages/ui/src/services/request-events.ts`: canonical explicit-cancellation predicate and backend-unreachable event type.
- `apps/packages/ui/src/services/background-proxy.ts`: avoids diagnostics/events for cancellations while preserving real failures.
- `apps/tldw-frontend/lib/api.ts`: avoids request-history failures for explicitly aborted direct requests.
- `apps/tldw-frontend/components/layout/WebLayout.tsx`: corroborates candidate outage events before rendering recovery UI.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/use-source-saved-views.ts`: accepts narrow server-parent readiness.
- `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`: publishes workspace identity after successful upsert/reconciliation.
- `apps/packages/ui/src/components/Option/Prompt/index.tsx`: derives compact layout from actual workspace content width.
- `apps/packages/ui/src/components/Option/Settings/TldwConnectionSettings.tsx`: removes competing Ant Form initialization.
- `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`: aligns Kanban forced-error inventory with the canonical shared boundary.
- `apps/tldw-frontend/scripts/live-tier-uat/profile.mjs`: creates disposable paths, ports, environment, and service commands.
- `apps/tldw-frontend/scripts/live-tier-uat/inventory-api-mocks.mjs`: inventories API interception in Tier-1–3 source files.
- `apps/tldw-frontend/scripts/live-tier-uat/run.mjs`: starts/stops real backend, model service, and WebUI and executes list/run phases.
- `apps/tldw-frontend/scripts/live-tier-uat/report.mjs`: merges Playwright JSON and service evidence into deterministic Markdown.
- `Docs/superpowers/reviews/2026-08-25-real-server-workflow-coverage-map.md`: maps all 17 legacy cases to maintained coverage.
- `Docs/superpowers/reviews/2026-08-25-tier-1-3-live-uat-results.md`: records denominators, mock inventory, classifications, reruns, and final commit.

---

### Task 1: Measure and Select the Supported Development Runtime (`TASK-13124.1`)

**Files:**
- Create: `apps/tldw-frontend/scripts/dev-runtime-uat-lib.mjs`
- Create: `apps/tldw-frontend/scripts/dev-runtime-uat.mjs`
- Create: `apps/tldw-frontend/scripts/__tests__/dev-runtime-uat.test.ts`
- Modify: `apps/tldw-frontend/package.json`
- Modify: `apps/tldw-frontend/__tests__/next-config-dev-watch-guard.test.ts`
- Modify: `apps/tldw-frontend/README.md`

**Interfaces:**
- Produces: `parseProcessTable(text: string): ProcessRow[]`, `descendantUsage(rows: ProcessRow[], rootPid: number): { rssBytes: number; cpuPercent: number; pids: number[] }`, and `evaluateRuntime(samples: RuntimeSample[]): RuntimeEvaluation`.
- Produces: `bun run uat:dev-runtime -- --bundler=webpack|turbopack --port=<port> --warm-idle-ms=<milliseconds> --output=<json>`.
- Consumes: the existing `e2e:smoke:all-pages:gate` route traversal and `TLDW_WEB_AUTOSTART=false` Playwright contract.
- Consumes: an explicit healthy `TLDW_E2E_SERVER_URL` and matching `TLDW_E2E_API_KEY`; the probe fails closed instead of measuring an offline/fallback route sweep.

- [x] **Step 1: Write pure evaluator tests**

```ts
import { describe, expect, it } from "vitest"
import {
  descendantUsage,
  evaluateRuntime,
  parseProcessTable,
} from "../dev-runtime-uat-lib.mjs"

it("sums the Next server process tree instead of the launcher only", () => {
  const rows = parseProcessTable(`10 1 1024 0.1 bun\n11 10 2048 1.0 next-server\n12 11 4096 2.0 worker`)
  expect(descendantUsage(rows, 10)).toEqual({
    rssBytes: 7 * 1024,
    cpuPercent: 3.1,
    pids: [10, 11, 12],
  })
})

it("fails a responsive runtime whose idle growth exceeds two GiB", () => {
  expect(evaluateRuntime([
    { phase: "post-traversal", rssBytes: 10 * 2 ** 30, responsive: true },
    { phase: "post-idle", rssBytes: 13 * 2 ** 30, responsive: true },
    { phase: "second-pass", rssBytes: 13 * 2 ** 30, responsive: true },
  ])).toMatchObject({ qualified: false, reasons: ["idle_rss_growth"] })
})
```

- [x] **Step 2: Run the evaluator tests and observe red**

Run: `cd apps/tldw-frontend && bunx vitest run scripts/__tests__/dev-runtime-uat.test.ts`

Expected: FAIL because `dev-runtime-uat-lib.mjs` does not exist.

- [x] **Step 3: Implement process discovery and exact guardrails**

```js
export const MAX_RSS_BYTES = 16 * 2 ** 30
export const MAX_IDLE_GROWTH_BYTES = 2 * 2 ** 30

export function evaluateRuntime(samples) {
  const postTraversal = samples.find((sample) => sample.phase === "post-traversal")
  const postIdle = samples.find((sample) => sample.phase === "post-idle")
  const secondPass = samples.find((sample) => sample.phase === "second-pass")
  const reasons = []
  if (!samples.every((sample) => sample.responsive)) reasons.push("unresponsive")
  if (!secondPass) reasons.push("second_pass_missing")
  if (Math.max(...samples.map((sample) => sample.rssBytes)) >= MAX_RSS_BYTES) reasons.push("rss_limit")
  if (postTraversal && postIdle && postIdle.rssBytes - postTraversal.rssBytes > MAX_IDLE_GROWTH_BYTES) {
    reasons.push("idle_rss_growth")
  }
  return { qualified: reasons.length === 0, reasons }
}
```

The CLI must verify the explicit backend health endpoint, spawn `bun run dev:webpack` or `bun run dev:turbopack`, wait for HTTP readiness, run the complete all-pages route gate with WebUI autostart disabled and offline fallback forbidden, sample the whole descendant process tree, health-check through 20 warm-idle minutes, run a second critical-route pass, write JSON, and terminate only the process group it created.

- [x] **Step 4: Start one isolated backend profile, then run short probes**

Use the existing `scripts/onboarding-uat/profile.mjs` helpers to start the repository mock model service and one disposable backend shared by both candidate measurements. Health-check the backend and keep its profile, ports, and API key identical across candidates. Stop and classify an environment defect if the backend cannot become healthy; do not continue with offline fallback.

Run both commands with `--warm-idle-ms=60000`; require a JSON report and clean process teardown:

```bash
cd apps/tldw-frontend
TLDW_E2E_ALLOW_OFFLINE=0 TLDW_E2E_SERVER_URL=http://127.0.0.1:18180 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-UAT bun run uat:dev-runtime -- --bundler=webpack --port=18181 --warm-idle-ms=60000 --output=test-results/dev-runtime/webpack-short.json
TLDW_E2E_ALLOW_OFFLINE=0 TLDW_E2E_SERVER_URL=http://127.0.0.1:18180 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-UAT bun run uat:dev-runtime -- --bundler=turbopack --port=18182 --warm-idle-ms=60000 --output=test-results/dev-runtime/turbopack-short.json
```

- [x] **Step 5: Run the qualifying comparison and select the default from evidence**

Run each still-viable bundler with `--warm-idle-ms=1200000` against the same healthy backend profile. A candidate that has already crossed an irreversible guardrail in the clean short probe is terminated rather than held for the remaining idle interval. If exactly one qualifies, set `package.json` `dev` to that command and keep both named commands. If neither qualifies, keep `dev` unchanged, add an explicit `uat:dev` bounded command selected from the less severe report, and record both failed reasons on `TASK-13124.1` without calling either stable.

- [x] **Step 6: Lock the package-script contract and documentation**

After the comparison, add a literal assertion for the selected default to `next-config-dev-watch-guard.test.ts`; do not compute the expectation from the implementation under test. Also assert that both `dev:webpack` and `dev:turbopack` remain defined and distinct. Document the selected command, both report paths, the UAT-host thresholds, and that the red bottom-left number is Next.js development tooling rather than a tldw-owned counter. Because it reflects Next's captured errors, retain it as actionable UAT evidence.

- [x] **Step 7: Verify and commit**

Run: `cd apps/tldw-frontend && bunx vitest run scripts/__tests__/dev-runtime-uat.test.ts __tests__/next-config-dev-watch-guard.test.ts`

Run: `git diff --check`

Commit: `fix(webui): select evidence-backed development runtime`

---

### Task 2: Corroborate Backend-Unavailable Events (`TASK-13124.2`)

**Files:**
- Modify: `apps/packages/ui/src/services/request-events.ts`
- Modify: `apps/packages/ui/src/services/backend-unreachable.ts`
- Modify: `apps/packages/ui/src/services/background-proxy.ts`
- Create: `apps/packages/ui/src/services/__tests__/request-events.test.ts`
- Modify: `apps/tldw-frontend/lib/api.ts`
- Modify: `apps/packages/ui/src/services/__tests__/background-proxy.test.ts`
- Modify: `apps/tldw-frontend/lib/__tests__/api-client.fetch.test.ts`
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Modify: `apps/tldw-frontend/__tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx`

**Interfaces:**
- Produces: `isExplicitRequestCancellation(value: unknown): boolean` in `request-events.ts` for `AbortError`, `REQUEST_ABORTED`, and established abort messages.
- Consumes: `useConnectionActions().checkOnce({ force: true })`, `ConnectionPhase`, and the existing `tldw:backend-unreachable` event.

- [x] **Step 1: Add red cancellation and corroboration tests**

```ts
it("does not persist or emit an explicitly aborted request", async () => {
  const controller = new AbortController()
  controller.abort()
  await expect(bgRequest({ path: "/api/v1/health", abortSignal: controller.signal })).rejects.toMatchObject({
    name: "AbortError",
    code: "REQUEST_ABORTED",
  })
  expect(storage.set).not.toHaveBeenCalledWith("__tldwLastRequestError", expect.anything())
  expect(dispatchEvent).not.toHaveBeenCalled()
})

it("keeps a status-zero candidate hidden when the forced check reconnects", async () => {
  dispatchBackendUnreachable({ status: 0, message: "Failed to fetch" })
  expect(screen.queryByTestId("backend-unavailable-modal")).toBeNull()
  resolveForcedCheck(ConnectionPhase.CONNECTED)
  expect(screen.queryByTestId("backend-unavailable-modal")).toBeNull()
})
```

- [x] **Step 2: Run focused tests and observe red**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/background-proxy.test.ts lib/__tests__/api-client.fetch.test.ts __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx`

Expected: cancellation is recorded and/or the modal appears before connection corroboration.

- [x] **Step 3: Centralize explicit cancellation classification**

```ts
export const isExplicitRequestCancellation = (value: unknown): boolean => {
  if (!value || typeof value !== "object") return false
  const error = value as { name?: unknown; code?: unknown; message?: unknown }
  if (error.name === "AbortError" || error.code === "REQUEST_ABORTED") return true
  return typeof error.message === "string" && /(?:request\s+)?abort(?:ed|ing)?/i.test(error.message)
}
```

Use this predicate before `recordRequestError`, `notifyBackendUnavailable`, and direct `recordFailure`. Preserve a thrown cancellation object so callers still observe cancellation.

- [x] **Step 4: Make WebLayout display only corroborated candidates**

Store `backendUnavailableCandidate` separately from visible detail. On each event, increment a sequence ref, clear visible recovery, and force `checkOnce`. An effect promotes only the current candidate when checking has settled outside `CONNECTED`; it clears both candidate and detail when connected.

```tsx
if (!isChecking && phase === ConnectionPhase.CONNECTED && isConnected) {
  setBackendUnavailableCandidate(null)
  setBackendUnavailableDetail(null)
} else if (!isChecking && backendUnavailableCandidate) {
  setBackendUnavailableDetail(backendUnavailableCandidate)
}
```

- [x] **Step 5: Verify genuine outage and retry behavior**

Add a test where the forced check settles in `ConnectionPhase.ERROR`; assert sanitized method/path/message and Retry remain visible. Then make Retry settle connected and assert recovery UI clears.

- [x] **Step 6: Run adjacent tests and commit**

Run the Step 2 command plus `cd apps/tldw-frontend && bunx vitest run __tests__/components/layout/WebLayout.backend-unreachable.test.tsx ../packages/ui/src/store/__tests__/connection.test.ts`.

Commit: `fix(webui): corroborate backend outage events`

---

### Task 3: Gate Source Saved Views on Server Workspace Existence (`TASK-13124.3`)

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/use-source-saved-views.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-server-reconcile.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/use-source-saved-views.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage12.source-list-view-state.test.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`

**Interfaces:**
- Changes: `useSourceSavedViews(workspaceId, workspaceExists, currentState, onApplyState)`.
- Produces: `serverWorkspaceIdentity: string | null`; readiness is `serverWorkspaceIdentity === workspaceId`.
- Consumes: `reconcileResearchWorkspaceServerState(...): { workspaceReady: boolean; errors: string[] }`.

- [x] **Step 1: Add a red hook ordering test**

```tsx
const { rerender } = renderHook(
  ({ exists }) => useSourceSavedViews("workspace-new", exists, localState(), vi.fn()),
  { initialProps: { exists: false } },
)
expect(tldwClient.listWorkspaceSourceViews).not.toHaveBeenCalled()
rerender({ exists: true })
await waitFor(() => expect(tldwClient.listWorkspaceSourceViews).toHaveBeenCalledWith("workspace-new"))
```

- [x] **Step 2: Run the hook test and observe red**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/use-source-saved-views.test.tsx`

Expected: current signature lists immediately when `workspaceId` is non-null.

- [x] **Step 3: Add narrow identity readiness**

Reset `serverWorkspaceIdentity` synchronously when `activeWorkspaceId` changes. Publish it from the matching reconciliation request's `onWorkspaceReady` callback immediately after the parent upsert succeeds and only while the request sequence still matches. Record the full reconcile signature only after source reconciliation also completes without errors; saved views do not wait for that later work.

```tsx
const [serverWorkspaceIdentity, setServerWorkspaceIdentity] = React.useState<string | null>(null)
React.useLayoutEffect(() => setServerWorkspaceIdentity(null), [activeWorkspaceId])

onWorkspaceReady: () => {
  if (!cancelled && requestSeq === workspaceServerReconcileRequestSeqRef.current) {
    setServerWorkspaceIdentity(activeWorkspaceId)
  }
}

if (reconcileResult.errors.length === 0 && reconcileResult.workspaceReady) {
  workspaceServerReconcileSignatureRef.current = reconcileSignature
}
```

- [x] **Step 4: Guard every saved-view operation**

Include `workspaceExists` in the list effect, `available`, retry, and mutation preconditions. Identity/generation guards remain unchanged so late readiness for workspace A cannot unlock workspace B.

- [x] **Step 5: Add live request-order evidence**

In `research-workspace.real-backend.spec.ts`, collect request timestamps for workspace upsert and `/source-views`; create a fresh workspace and assert the successful upsert response completes before the first saved-view list request. Assert no `Source view not found` alert.

- [x] **Step 6: Verify and commit**

Run the Step 2 command plus the Stage-12 test and the focused live-server scenario.

Commit: `fix(research): wait for workspace before loading saved views`

---

### Task 4: Make Prompt Workspace Container-Responsive (`TASK-13124.4`)

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Prompt/index.tsx`
- Create: `apps/packages/ui/src/components/Option/Prompt/__tests__/PromptBody.responsive.test.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-2-features/prompts-workspace.spec.ts`

**Interfaces:**
- Produces: compact state derived from the `PromptBody` root container width rather than `window.innerWidth` alone.
- Preserves: current mobile controls, desktop sidebar, keyboard focus, and accessible names.

- [x] **Step 1: Reproduce on the clean bundle before changing CSS**

Add a Playwright geometry helper that opens `/prompts` at 1365x768 and 1365x900, dismisses the intentional tour, and asserts:

```ts
expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true)
await expectNoOverlap(page.getByTestId("prompt-sidebar"), page.getByTestId("prompts-search-control"))
await expect(page.getByTestId("prompts-add")).toBeEnabled()
await expect(page.getByTestId("prompts-search")).toBeEditable()
```

If both viewports pass twice with a clean `.next`, record gate drift on the task and correct only the stale audit setup. Continue to Step 2 only when the geometry test is red.

- [x] **Step 2: Run the focused geometry test and preserve the result**

Run: `cd apps/tldw-frontend && bunx playwright test e2e/workflows/tier-2-features/prompts-workspace.spec.ts --project=tier-2 --grep 'fits standard desktop content widths' --workers=1 --trace=on`

- [x] **Step 3: Stop conditional production remediation when the gate is green**

The clean geometry test passed twice at both reported 1365px viewports, so the
planned `ResizeObserver` production change was not justified and was not added.

- [x] **Step 4: Add complete acceptance-matrix regression coverage**

The geometry gate now covers 390x844, 1365x768, 1365x900, and 1536x960. It
asserts global overflow, expected sidebar visibility, sidebar/search geometry,
and enabled/editable primary controls. The full-suite run also exposed a lazy
editor page-object race; the harness now polls for either supported editor
instead of treating `Locator.isVisible()` as a wait.

- [x] **Step 5: Verify mobile, desktop, and wide viewports and commit**

Run the focused Tier-2 geometry test at all four acceptance widths and the
complete `prompts-workspace.spec.ts`. Component coverage was not added because
production responsive behavior was unchanged.

Commit: `test(prompts): harden responsive live-backend coverage`

---

### Task 5: Remove Settings Form Ownership Warning (`TASK-13124.6`)

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Settings/TldwConnectionSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/SetupRecoverySettings.test.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Modify: `apps/packages/ui/src/components/Layouts/__tests__/settings-layout-labels.test.tsx`
- Create: `apps/tldw-frontend/pages/settings/chat-macros.tsx`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/e2e/page-mapping.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/tier-1-critical/settings-core.spec.ts`

**Interfaces:**
- Consumes: parent `Form` `initialValues` and later `form.setFieldsValue` calls in `tldw.tsx`.
- Preserves: `rememberApiKey` component state for help copy and persistence result messaging.

- [x] **Step 1: Add a red warning regression**

Render `TldwSettings`, spy on `console.error`, wait for config hydration, and assert no Ant warning contains `initialValues` or `initialValue` for `rememberApiKey`.

- [x] **Step 2: Run focused Settings tests and observe red**

Run: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Option/Settings/__tests__/tldw-review-comments.test.tsx`

- [x] **Step 3: Remove the child initializer**

```tsx
<Form.Item name="rememberApiKey" valuePropName="checked">
  <Checkbox onChange={(event) => setRememberApiKey(event.target.checked)}>
    {label}
  </Checkbox>
</Form.Item>
```

Do not replace it with a controlled `checked` prop; Ant Form remains the field owner.

- [x] **Step 4: Lock heading hierarchy and navigation labels**

Assert the settings shell has one page-level `Settings` heading, `Setup & Recovery` is level 2, and Workflow Prompts and Prompt Studio links retain distinct accessible names.

Review also found `/settings/chat-macros` assigned to a nonexistent
`"experience"` group, which both failed TypeScript and removed the destination
from rendered navigation. Reassign it to `preferencesWorkflow`, add the missing
Next.js page that caused the restored link to reach a 404, register it in both
UAT page inventories, and lock the accessible link plus live route.

- [x] **Step 5: Verify save/persistence and commit**

Run focused unit tests, `settings-core.spec.ts --project=tier-1 --workers=1`,
and the existing device/session manual API-key persistence scenarios.

Commit: `fix(settings): keep connection form initialization authoritative`

---

### Task 6: Correct Kanban Forced-Error Gate Drift (`TASK-13124.8`)

**Files:**
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Modify: `apps/packages/ui/src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx`

**Interfaces:**
- Consumes: production `RouteErrorBoundary routeId="kanban" routeLabel="Kanban"` in `option-kanban-playground.tsx`.
- Produces: matching smoke inventory `{ path: "/kanban", routeId: "kanban", routeLabel: "Kanban" }`.

- [x] **Step 1: Lock the canonical route boundary contract**

Keep the existing unit case asserting `routeId="kanban"` and `routeLabel="Kanban"`; add a source/inventory assertion that the all-pages entry matches those values.

- [x] **Step 2: Run the focused forced-error scenario and observe red**

Run: `cd apps/tldw-frontend && bunx playwright test e2e/smoke/all-pages.spec.ts --grep 'Kanban Playground forced-error fixture' --workers=1`

Expected: the `kanban-playground` query value does not trigger the `kanban` boundary.

- [x] **Step 3: Correct only the stale smoke inventory**

```ts
{
  name: "Kanban Playground",
  path: "/kanban",
  routeId: "kanban",
  routeLabel: "Kanban",
}
```

Do not add a Kanban-specific crash hook or alter the production boundary.

- [x] **Step 4: Verify error and recovery paths and commit**

Run the focused Playwright scenario, click Retry after removing the query signal, and assert the normal board heading returns. Run the shared route-boundary unit test.

The focused live-backend scenario now removes the query through Next Router,
dismisses the expected development overlay, clicks Retry, and observes the
normal Kanban heading. The recovery interaction is scoped to Kanban so the
generic boundary loop does not unexpectedly load every target route. The
production-mode component test confirms the fixture remains inert outside
development/test builds.

Commit: `test(webui): align Kanban route recovery fixture`

---

### Task 7: Provide Deterministic Character and Persona Continuity (`TASK-13124.5`)

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase7-readiness.spec.ts`
- Modify: `apps/tldw-frontend/e2e/utils/api-assertions.ts`
- Modify: `apps/tldw-frontend/scripts/onboarding-uat/profile.mjs`
- Modify: `apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/ChatModelSelectorDropdown.tsx`
- Modify only if tracing proves a product defect: `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts`
- Modify only if tracing proves a product defect: `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- Modify only if tracing proves a product defect: `apps/packages/ui/src/hooks/chat/personaServerChat.ts`
- Modify only if tracing proves a product defect: `apps/packages/ui/src/utils/character-chat-session.ts`
- Modify only if tracing proves a product defect: `apps/packages/ui/src/hooks/handlers/messageHandlers.ts`
- Modify only if tracing proves a product defect: `apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts`
- Modify only if tracing proves a product defect: `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx`
- Modify only if tracing proves a product defect: `apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts`

**Interfaces:**
- Consumes: backend environment `CUSTOM_OPENAI_API_IP=http://127.0.0.1:<mock-port>/v1`, `CUSTOM_OPENAI_API_KEY=<isolated profile mock key>`, and `CUSTOM_OPENAI_API_MODEL=gpt-4`.
- Produces: reusable live-test readiness `{ providerId: "custom-openai-api"; modelId: string; reason?: string }` selected from advertised configured chat-capable models.
- Preserves: no-provider SEND gating with a truthful user-visible reason.

- [x] **Step 1: Reconcile overlap before edits**

Compare `origin/dev...codex/character-chat-phase8-continuity` and `origin/dev...codex/chat-character-overlay-tracked-identity` for the four conditional production files. Record overlapping commits/files on `TASK-13124.5`; port or avoid duplicated changes rather than editing around them.

- [x] **Step 2: Trace all five boundaries without code changes**

For disposable, tracked character, and tracked persona cases, capture usable-model metadata, create-chat payload identity, created chat ID, terminal streaming outcome, and post-reload chat identity. Attach the bounded trace to the Backlog task.

- [x] **Step 3: Add red live assertions for deterministic provider wiring**

```ts
expect(readiness).toMatchObject({ providerId: "custom-openai-api" })
expect(createPayload.character_id ?? createPayload.assistant_id).toBe(expectedIdentity)
await expect(assistantMessage).toContainText(/mock/i)
await page.reload()
await expect(trackedIdentityBadge).toContainText(expectedDisplayName)
```

- [x] **Step 4: Fix the first proven broken boundary only**

Tracing proved that identity transport, chat creation, message persistence, and the backend completion path were correct. The broken boundary was the fresh-page session latch: without a prior restore attempt, immediate persistence stayed disabled and a quick reload could lose `serverChatId` while retaining the selected character. The fix settles that latch when the scoped store has no valid session. A second bounded defect wired the already-computed model-usability state into the visible selector so blocked models no longer display a contradictory Healthy badge. The capture helper now waits for in-flight response parsing so a completed streaming call cannot disappear from evidence collection.

- [x] **Step 5: Preserve truthful no-provider behavior**

The Phase-7 gate selects a real provider-unconfigured or unavailable model advertised by the backend while the deterministic custom provider remains available for the callable case. It asserts the accessible setup status, warning selector, preserved draft/character, and absence of `/complete-v2`; this covers the blocked path without requiring a second backend process.

- [x] **Step 6: Verify deterministic live cases and commit**

Run the three Chat Cockpit real-server cases through the isolated backend/mock service, the focused hook tests for any touched production boundary, and Phase-7 readiness.

Verification: 3/3 Chat Cockpit live cases passed; blocked and callable Phase-7 cases passed; 51/51 focused unit/integration tests passed; touched frontend lint completed with zero errors (pre-existing explicit-any warnings remain in the large real-server spec). Bandit is not applicable because this task touches TypeScript/JavaScript only.

Commit: `fix(chat): prove tracked assistant continuity on live backend`

---

### Task 8: Replace the Legacy 17-Case Gate with Unique Live Coverage (`TASK-13124.7`)

**Files:**
- Create: `Docs/superpowers/reviews/2026-08-25-real-server-workflow-coverage-map.md`
- Modify: `apps/test-utils/real-server-workflows.ts`
- Modify: `apps/tldw-frontend/e2e/real-server-workflows.spec.ts`
- Modify: `apps/extension/tests/e2e/real-server-workflows.spec.ts`
- Modify: `apps/tldw-frontend/__tests__/e2e-harness-readiness.guard.test.ts`

**Interfaces:**
- Produces: one coverage-map row per existing test title with `behavior`, `current maintained coverage`, `live or intercepted`, `decision`, and `reason`.
- Produces: `resolveRunnableChatModel(metadata): { id: string; provider: string } | null` shared by retained provider-dependent cases.
- Consumes: deterministic custom-OpenAI provider configured in Task 7.

- [x] **Step 1: Extract and map all 17 tests before editing the suite**

Use `rg -n '^  test\(' apps/test-utils/real-server-workflows.ts` and inspect each title/body. The coverage-map document must contain exactly 17 source rows and one of three decisions: `delete-redundant`, `move-to-tier`, or `retain-live-gate`.

- [x] **Step 2: Add a red guard for map completeness and skip placement**

The guard test reads the map and source titles, asserts every legacy title appears exactly once, rejects `test.skip` after the first mutation marker in retained tests, and rejects empty `catch` blocks around assertions.

- [x] **Step 3: Delete redundant cases and move unique behavior**

Delete `delete-redundant` cases only after their mapped maintained test is named and passing. Move `move-to-tier` behavior into the named Tier-1–3 file. Keep only unique cross-surface or extension/WebUI parity behavior in `real-server-workflows.ts`.

- [x] **Step 4: Centralize current first-run and model readiness helpers**

```ts
const resolveRunnableChatModel = (items) =>
  items
    .map(normalizeModelMetadata)
    .find((item) => item.configured && item.capabilities.includes("chat")) ?? null
```

Retained provider-dependent cases call this before mutation and skip only when it returns null. CRUD/navigation cases do not consult model readiness.

- [x] **Step 5: Update current observable contracts**

Use the shared intentional-tour dismissal, exact action roles, current page identities, and current cleanup payload/identifier shapes. Replace breadcrumb-satisfiable text selectors with scoped roles or stable test IDs.

- [x] **Step 6: Run the reduced gate against real services and commit**

Run WebUI and extension retained gates, assert every remaining skip is counted with a pre-mutation reason, and run the map guard.

Commit: `test(e2e): reduce real-server workflows to honest coverage`

---

### Task 9: Build the Isolated Tier-1–3 UAT Runner (`TASK-13124.9`)

**Files:**
- Create: `apps/tldw-frontend/scripts/live-tier-uat/profile.mjs`
- Create: `apps/tldw-frontend/scripts/live-tier-uat/inventory-api-mocks.mjs`
- Create: `apps/tldw-frontend/scripts/live-tier-uat/report.mjs`
- Create: `apps/tldw-frontend/scripts/live-tier-uat/run.mjs`
- Create: `apps/tldw-frontend/scripts/__tests__/live-tier-uat.test.ts`
- Create: `Docs/superpowers/reviews/2026-08-25-tier-1-3-live-uat-results.md`
- Modify: `apps/tldw-frontend/package.json`

**Interfaces:**
- Produces: `bun run uat:live-tiers -- --projects=tier-1,tier-2,tier-3 --workers=1`.
- Produces: disposable profile with backend URL, WebUI URL, mock URL, fake API key, isolated users/media/notes/evaluation database paths, and log/report paths.
- Produces: JSON/Markdown inventory entries `{ project, file, line, matcher, kind: "intercepted" | "live" }`.

- [x] **Step 1: Test profile isolation, process teardown, and mock inventory**

```ts
it("places every mutable backend path under the run directory", () => {
  const profile = buildLiveTierProfile("/tmp/tldw-live-tier-run")
  expect(Object.values(profile.databasePaths).every((path) => path.startsWith(profile.runDir))).toBe(true)
})

it("marks page.route API fulfillment as intercepted coverage", () => {
  expect(inventorySource(`await page.route("**/api/v1/notes", route => route.fulfill({ json: [] }))`))
    .toEqual([expect.objectContaining({ kind: "intercepted", matcher: "**/api/v1/notes" })])
})
```

- [x] **Step 2: Run runner-unit tests and observe red**

Run: `cd apps/tldw-frontend && bunx vitest run scripts/__tests__/live-tier-uat.test.ts`

- [x] **Step 3: Implement disposable service orchestration**

Reuse `createRuntimeProfile()` and `buildBackendEnv()` from `scripts/onboarding-uat/profile.mjs` so the tier runner inherits the established isolated config, users database, per-user database roots, fixture allowlist, synthetic credentials, and secret-safe base environment. Start `.venv/bin/python -m mock_openai.server` from `mock_openai_server/` with a chosen free port, then `.venv/bin/python -m uvicorn tldw_Server_API.app.main:app` from the repository root with the profile environment, then the evidence-backed WebUI dev command. Health-check each service before Playwright and terminate only spawned process groups in reverse order.

Extend the environment returned by `buildBackendEnv()` with the deterministic custom-provider aliases required by Character/Persona live tests:

```js
{
  CUSTOM_OPENAI_API_IP: `${profile.mockUrl}/v1`,
  CUSTOM_OPENAI_API_KEY: backendEnv.OPENAI_API_KEY,
  CUSTOM_OPENAI_API_MODEL: "gpt-4",
}
```

The runner passes `buildBackendEnv()`'s single-user API key to Playwright; it must not introduce a second hard-coded credential that disagrees with the isolated backend profile.

- [x] **Step 4: Implement complete list/run phases**

For each project, run `playwright test --list --project=<project>` first and record the denominator. Then run the complete project with JSON and line reporters, `TLDW_E2E_ALLOW_OFFLINE=0`, explicit URLs, and `--workers=1`. Do not pass `--grep` in a certification run.

- [x] **Step 5: Generate mock/interception inventory and result report**

Use the TypeScript parser to find `page.route`, `context.route`, and `route.fulfill` in each listed Tier source file. Report intercepted cases separately and flag critical behavior with no live counterpart for manual classification.

- [x] **Step 6: Verify the runner on one bounded Tier-1 case, then list all tiers**

Run runner unit tests, a non-certifying Tier-1 smoke invocation, and complete `--list` for all three projects. Confirm cleanup leaves no listening service on the allocated ports.

- [x] **Step 7: Commit the runner**

Commit: `test(e2e): add isolated live tier UAT runner`

---

### Task 10: Execute, Remediate, Synchronize, and Certify (`TASK-13124` and `TASK-13124.9`)

**Files:**
- Modify per newly created child tasks only after Backlog duplicate search.
- Modify: `Docs/superpowers/reviews/2026-08-25-tier-1-3-live-uat-results.md`
- Modify: `backlog/tasks/task-13124*.md` only through official Backlog CLI/MCP.

**Interfaces:**
- Consumes: Tasks 1–9, the isolated runner, exact Playwright denominators, and current `origin/dev`.
- Produces: final evidence bound to one exact synchronized commit and finalized Backlog summaries.

- [x] **Step 1: Run complete Tier-1, Tier-2, and Tier-3 inventories**

Run: `cd apps/tldw-frontend && bun run uat:live-tiers -- --projects=tier-1,tier-2,tier-3 --workers=1`

Record for each project: listed, passed, failed, skipped, intercepted, live-evidence cases, elapsed time, service health before/after, and artifact paths.

- [x] **Step 2: Classify every non-pass exactly once**

Use `product defect`, `gate drift`, `optional capability unavailable`, or `environment defect`. Attach assertion, console/page error, failed request, backend correlation, and minimal reproduction. Ambiguous outcomes remain failures.

- [x] **Step 3: Create and execute one child task per confirmed new defect**

Search Backlog first. Create a child under `TASK-13124`, add the plan document, write a failing behavior regression, implement the smallest root-cause fix, run the focused test and adjacent suite, commit, and rerun the complete affected tier.

- [x] **Step 4: Repeat until affected complete tiers are clean**

Do not stop at focused green tests. Update exact rerun denominators and justified capability skips in the result document and child task notes.

- [x] **Step 5: Synchronize with then-current `origin/dev`**

Fetch, record the new `origin/dev` hash, inspect incoming changes, and rebase or merge without discarding either workstream. Resolve conflicts by preserving both intended behaviors. Run every focused regression after synchronization.

- [x] **Step 6: Run final certification gates**

Run complete Tier-1–3 again, the all-pages route sweep, Research real-backend suite, Chat Cockpit real-backend suite, reduced real-server gate, runtime probe, frontend typecheck, touched-scope ESLint, `git diff --check`, and an optimized build if Next/bundler configuration changed. Run Bandit on each touched Python scope or record the frontend-only skip.

- [x] **Step 7: Finalize tasks and commit evidence**

Mark acceptance criteria only from recorded evidence. Add modified files, red/green commands, final results, justified skips, exact commit, and final summaries to every child and the parent through Backlog CLI/MCP.

Commit: `docs(uat): record live tier certification results`

## Execution Mode

Execute inline in this session with `superpowers:executing-plans`. The requester asked to continue the full remediation loop but did not request subagent delegation, so no subagents are authorized.
