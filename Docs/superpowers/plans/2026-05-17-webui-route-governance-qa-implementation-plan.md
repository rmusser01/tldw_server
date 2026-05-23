# WebUI Route Governance QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent WebUI and extension UX regressions by turning route inventory, headings, command targets, capability states, responsive behavior, sidepanel availability, and browser evidence into enforceable governance checks.

**Architecture:** Build on the WP1 route contract, WP2 capability states, and WP4 landmark/responsive rules instead of adding another source of truth. Keep CI-suitable checks deterministic and fast, while preserving browser-observed screenshot and DOM evidence for route-family changes that cannot be reduced to static metadata.

**Tech Stack:** Next.js page inventory, shared `apps/packages/ui` route registries, extension route registries, Playwright smoke tests, Vitest, React Testing Library, Axe via `@axe-core/playwright`, existing smoke diagnostics, Backlog task records, Markdown evidence artifacts.

---

## Source Documents

- Backlog task: `TASK-418.10`
- Parent implementation plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- UX remediation spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Dependency plans:
  - `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md`

## Audit Findings Addressed

Task 12 is the final regression gate for all audit findings:

- `F1`: Navigation and IA source-of-truth drift.
- `F2`: Mobile overflow on core root pages.
- `F3`: Setup and onboarding shell regression.
- `F4`: Raw endpoint error becoming primary UI.
- `F5`: Model settings usability regression.
- `F6`: Chat-specific global chrome leaking into non-chat routes.
- `F7`: Internal translation keys leaking into UI.
- `F8`: Command palette label and target mismatch.
- `F9`: Capability and unsupported-state drift.
- `F10`: Media first-selection and mobile master-detail regression.
- `F11`: Settings IA regression.
- `F12`: Extension sidepanel availability drift.
- `F13`: Chat composer mobile collision.
- `F14`: Research, Knowledge, Chat, and Workspace ladder drift.
- `F15`: Missing or inconsistent root-page heading landmarks.
- `F16`: Settings routine versus high-risk action regression.
- `F17`: Route inventory and smoke coverage drift.
- `F18`: Hosted, beta, labs, debug, alias, and internal visibility drift.
- `F19`: Deprecated UI cleanup tracking where it blocks touched UX work.

## CI-Suitable Versus Manual Browser QA

### CI-Suitable Checks

These checks must run without subjective review:

- Route metadata coverage for audited root routes.
- Next page file discovery against route metadata and smoke inventory.
- Shared route registry coverage against route metadata.
- Extension route registry and sidepanel availability coverage.
- Command palette label and target agreement.
- Heading presence for user-facing root routes, with explicit exceptions.
- Smoke inventory coverage for user-facing routes, with explicit skip reasons.
- Route error boundary fixture coverage for representative non-core routes.
- Capability-state fixture coverage for routes that adopted WP2 states.
- Axe high-risk route checks.
- Core 390px page-level horizontal overflow checks.
- Deprecated component usage checks in touched UX files where cleanup is required.

### Manual Or Browser Evidence Checks

These checks remain evidence-backed but not fully deterministic:

- Before and after screenshots for changed visual routes.
- Human-readable notes for expected hosted, unavailable, or degraded states.
- Console and request error triage when a route has intentional self-hosted failures.
- Final route-family browser observations linked from Backlog tasks.
- Sidepanel screenshots when the route is extension-sidepanel reachable.

Manual evidence must be stored under the relevant task or review artifact path and referenced from the Backlog task final summary.

## Current QA Surface

| Area | Current owner | Notes for Task 12 |
| --- | --- | --- |
| Smoke inventory | `apps/tldw-frontend/e2e/smoke/page-inventory.ts` | Static list with categories, skips, and route names. Needs route metadata cross-check. |
| All-pages smoke | `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts` | Visits active pages, checks runtime errors, error boundaries, expected test ids, console and request failures. Needs governance assertions for headings and route metadata reasons. |
| Axe high-risk smoke | `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts` | High-risk route list and targeted Axe rules. Needs route metadata ownership and exception policy. |
| Stage 2 route contracts | `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts` | Checks selected placeholder and real routes. Needs to consume or validate route metadata. |
| Smoke diagnostics | `apps/tldw-frontend/e2e/smoke/smoke.setup.ts` | Collects console, page errors, and request failures. Needs owner and expiry discipline for allowlists. |
| Route registry | `apps/packages/ui/src/routes/route-registry.tsx` | Shared option route definitions. Needs metadata coverage tests from WP1. |
| Extension route registry | `apps/tldw-frontend/extension/routes/route-registry.tsx` | Extension options route definitions. Needs sidepanel and visibility agreement checks. |
| Sidepanel registry | `apps/packages/ui/src/routes/sidepanel-route-registry.tsx` | Sidepanel route availability. Needs explicit route metadata validation. |
| Command palette tests | `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx` | Existing shortcut and modal tests. Needs route label and target agreement tests. |
| Route visibility tests | `apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts` | Hosted visible route coverage. Needs expansion through route metadata. |

## Non-Goals

- Do not build a full visual regression system for every page.
- Do not require screenshots for routes untouched by an implementation slice.
- Do not make all warnings fail CI without route-owned allowlist rules.
- Do not replace Playwright smoke infrastructure.
- Do not duplicate the WP1 route metadata source.
- Do not add backend APIs for governance.
- Do not hide routes to make tests pass.
- Do not collapse manual browser QA into unreviewed generated artifacts.

## File Structure

### New Files

- `apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts`
  - Verifies audited route metadata covers Next pages, shared route registry routes, extension routes, aliases, hosted-only routes, labs routes, debug/internal routes, and smoke inventory entries.
- `apps/packages/ui/src/routes/__tests__/route-governance.sidepanel-availability.test.ts`
  - Verifies sidepanel route availability matches route metadata and extension registry targets.
- `apps/packages/ui/src/components/Common/__tests__/CommandPalette.route-targets.test.tsx`
  - Verifies command labels, route targets, visibility, and alias handling match route metadata.
- `apps/tldw-frontend/e2e/smoke/route-heading-governance.spec.ts`
  - Verifies user-facing root routes expose one `h1` or an approved metadata exception.
- `apps/tldw-frontend/e2e/smoke/route-responsive-governance.spec.ts`
  - Verifies representative core routes have no page-level horizontal overflow at 390px and extension sidepanel widths where relevant.
- `apps/tldw-frontend/e2e/smoke/route-capability-state-governance.spec.ts`
  - Verifies representative WP2 routes render capability states instead of raw endpoint text.
- `apps/tldw-frontend/e2e/smoke/route-evidence-protocol.md`
  - Documents where before and after screenshots, DOM snapshots, console triage, and known skips are recorded.

### Modified Files

- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.helpers.ts`
- `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`
- `apps/tldw-frontend/e2e/smoke/smoke.setup.ts`
- `apps/tldw-frontend/package.json`
- `apps/packages/ui/src/routes/route-registry.tsx`
- `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- `apps/packages/ui/src/routes/option-route-visibility.ts`
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
- `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
- `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- `apps/packages/ui/src/components/Common/CommandPaletteHost.tsx`

### Existing Tests To Extend

- `apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts`
- `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts`
- `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-clipper.test.ts`
- `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts`
- `apps/packages/ui/src/routes/__tests__/sidepanel-home-resolver.test.tsx`
- `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx`
- `apps/packages/ui/src/components/Common/__tests__/CommandPaletteHost.test.tsx`
- `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`

## Governance Contracts

### Route Metadata Required Fields

Task 12 depends on the WP1 route contract. If WP1 has not landed the final shape, use its current metadata and extend it only in the WP1 task. The governance tests require these fields to exist somewhere in route metadata:

```ts
export type RouteSurfaceClass =
  | "default_self_hosted"
  | "advanced_self_hosted"
  | "hosted_only"
  | "admin_operator"
  | "extension_sidepanel"
  | "labs_beta"
  | "internal_debug"
  | "legacy_alias"
  | "redirect"
  | "deprecated"

export type RouteGovernanceRecord = {
  path: string
  canonicalPath: string
  label: string
  group: string
  surfaceClass: RouteSurfaceClass
  aliases: string[]
  appearsInSmokeInventory: boolean
  appearsInCommandPalette: boolean
  sidepanelAvailability: "available" | "not_available" | "extension_only"
  requiresH1: boolean
  h1ExceptionReason?: string
  responsiveGate: "core_390" | "sidepanel" | "not_required"
  capabilityFixture?: string
  manualEvidenceRequired: boolean
}
```

### Smoke Inventory Rules

Every user-facing route must have exactly one of these states:

- Included in `PAGES` and active.
- Included in `PAGES` with a skip reason backed by route metadata.
- Excluded because metadata classifies it as `internal_debug`, `deprecated`, or a non-user-facing child route.

Skip reasons must be stable and owner-readable. A skip cannot be "flaky" without an owner, date, and route-specific rationale in metadata or an adjacent test comment.

### Heading Rules

Every user-facing root route needs one semantic `h1` unless route metadata declares one of these exceptions:

- `legacy_alias`: route redirects to a canonical route.
- `redirect`: route is intentionally redirect-only.
- `internal_debug`: route is not a user-facing product surface.
- `hosted_only`: route is hidden or rerouted in self-hosted mode.
- `embedded_tool`: route intentionally renders inside a non-page shell with a parent landmark.

### Command Target Rules

Every command palette route command must have:

- User-facing label matching route metadata.
- Target path matching canonical path or an allowed alias.
- Visibility aligned with hosted, labs, debug, and sidepanel policy.
- No duplicate command with the same user-facing label and different target.

### Capability-State Rules

Every representative WP2 route fixture must assert:

- Raw endpoint text is not the primary route state.
- User-language diagnosis is visible.
- At least one recovery action or next step is visible.
- Diagnostics details remain available behind disclosure when the route exposes raw technical detail.

### Responsive Rules

Core routes must be checked at 390px:

- `/chat`
- `/media`
- `/settings`
- `/settings/model`
- `/prompts`
- `/workspace-playground`
- `/setup`
- `/sources`
- `/mcp-hub`
- `/stt`
- `/tts`
- `/chat-workspace`

Sidepanel-reachable routes must also be checked at sidepanel width when their implementation changes.

## Implementation Tasks

### Task 1: Lock Route Inventory And Metadata Coverage

**Files:**
- Create: `apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Reference: `apps/packages/ui/src/routes/route-registry.tsx`
- Reference: `apps/tldw-frontend/extension/routes/route-registry.tsx`

- [ ] **Step 1: Write failing metadata coverage tests**

Create `route-governance.metadata-coverage.test.ts` to assert:

```ts
import { describe, expect, it } from "vitest"
import { ROUTE_DEFINITIONS } from "../route-registry"
import { ROUTE_GOVERNANCE } from "../route-governance"

describe("route governance metadata coverage", () => {
  it("covers every shared option route", () => {
    const metadataPaths = new Set(ROUTE_GOVERNANCE.map((route) => route.path))
    const missing = ROUTE_DEFINITIONS
      .filter((route) => route.kind === "options")
      .map((route) => route.path)
      .filter((path) => !metadataPaths.has(path))

    expect(missing).toEqual([])
  })

  it("requires explicit smoke inventory decisions", () => {
    for (const route of ROUTE_GOVERNANCE) {
      expect(typeof route.appearsInSmokeInventory).toBe("boolean")
      if (!route.appearsInSmokeInventory) {
        expect(["internal_debug", "deprecated", "legacy_alias", "redirect", "hosted_only"]).toContain(
          route.surfaceClass
        )
      }
    }
  })
})
```

Use the actual WP1 route metadata export name instead of `ROUTE_GOVERNANCE` if WP1 lands a different name.

- [ ] **Step 2: Run the metadata coverage test to verify it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts
```

Expected: FAIL until WP1 metadata is complete and exported.

- [ ] **Step 3: Add page-inventory validation**

Extend `page-inventory.ts` or an adjacent test to validate:

- Every active `PAGES` entry has route metadata.
- Every metadata route marked `appearsInSmokeInventory` exists in `PAGES`.
- Every skipped page has a non-empty skip reason and metadata-compatible surface class.
- Page categories can be derived from metadata or are checked against metadata groups.

- [ ] **Step 4: Run inventory tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit inventory governance**

```bash
git add apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts apps/tldw-frontend/e2e/smoke/page-inventory.ts apps/tldw-frontend/e2e/smoke/all-pages.spec.ts
git commit -m "test: govern route inventory coverage"
```

### Task 2: Govern Headings, Labels, And Command Targets

**Files:**
- Create: `apps/tldw-frontend/e2e/smoke/route-heading-governance.spec.ts`
- Create: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.route-targets.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- Modify: `apps/packages/ui/src/components/Common/CommandPaletteHost.tsx`

- [x] **Step 1: Write heading governance test**

Create `route-heading-governance.spec.ts`:

```ts
import { test, expect, seedAuth } from "./smoke.setup"
import { PAGES } from "./page-inventory"
import { waitForAppShell } from "../utils/helpers"

const HEADING_LOAD_TIMEOUT = 30_000

test.describe("Route heading governance", () => {
  for (const entry of PAGES.filter((page) => !page.skip)) {
    test(`${entry.path} has a governed page heading or exception`, async ({ page }) => {
      await seedAuth(page)
      await page.goto(entry.path, { waitUntil: "domcontentloaded", timeout: HEADING_LOAD_TIMEOUT })
      await waitForAppShell(page, HEADING_LOAD_TIMEOUT)

      const h1Count = await page.locator("h1").count()
      expect(h1Count, `${entry.path} must have one h1 or a metadata exception`).toBe(1)
    })
  }
})
```

Before implementation, this simple test will fail on valid alias and exception routes. Replace the raw `h1Count` assertion with route metadata exception logic before committing.

- [x] **Step 2: Run heading test to verify failures**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/smoke/route-heading-governance.spec.ts --reporter=line
```

Expected: FAIL until metadata exceptions are wired.

- [x] **Step 3: Add metadata-backed heading exceptions**

Update the test so:

- `requiresH1 === true` requires exactly one `h1`.
- `requiresH1 === false` requires `h1ExceptionReason`.
- Alias and redirect routes assert the final canonical path or redirect panel.
- Internal/debug routes are excluded from user-facing heading enforcement.

- [x] **Step 4: Add command target tests**

Create `CommandPalette.route-targets.test.tsx` to assert:

- Route commands use metadata labels.
- "Go to Chat" targets `/chat`.
- Hosted-only, labs, internal, debug, redirect, and deprecated routes follow visibility policy.
- No two route commands share the same label with different targets.
- Aliases either do not appear as separate commands or are labeled as aliases.

- [x] **Step 5: Run heading and command tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/__tests__/CommandPalette.route-targets.test.tsx apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx apps/packages/ui/src/components/Common/__tests__/CommandPaletteHost.test.tsx
```

Expected: PASS.

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/smoke/route-heading-governance.spec.ts --reporter=line
```

Expected: PASS.

- [x] **Step 6: Commit heading and command governance**

```bash
git add apps/tldw-frontend/e2e/smoke/route-heading-governance.spec.ts apps/tldw-frontend/__tests__/smoke/route-heading-governance.metadata.test.ts apps/packages/ui/src/components/Common/__tests__/CommandPalette.route-targets.test.tsx apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx apps/packages/ui/src/components/Common/CommandPalette.tsx apps/packages/ui/src/components/Common/CommandPaletteHost.tsx
git commit -m "test: govern headings and command targets"
```

Implementation outcome:
- Added metadata-backed command labels and h1 policy helpers in route metadata.
- Added command palette route-target governance and frontend-owned route heading governance tests.
- Browser QA initially found `/media` and `/notes` heading failures; `/media` needed route test mocks, and `/notes` is now a metadata-backed h1-policy exception because user-authored note content can contain document h1s.
- Kept CommandPaletteHost and shortcut behavior unchanged because the new target contract passed without host changes.
- Follow-up review moved the metadata-only heading governance test out of `@tldw/ui` so the package no longer imports frontend smoke inventory.
- Follow-up review also added a non-empty browser route guard and made explicit `requiresH1: false` opt-outs require `h1ExceptionReason` directly.

### Task 3: Govern Sidepanel And Hosted Visibility

**Files:**
- Create: `apps/packages/ui/src/routes/__tests__/route-governance.sidepanel-availability.test.ts`
- Modify: `apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts`
- Modify: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts`
- Modify: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-clipper.test.ts`
- Modify: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts`
- Modify: `apps/packages/ui/src/routes/option-route-visibility.ts`
- Modify: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`

- [ ] **Step 1: Write sidepanel governance tests**

Assert:

- Every route marked `sidepanelAvailability: "available"` exists in the shared sidepanel registry or extension registry as appropriate.
- Every route in a sidepanel registry is marked available in route metadata.
- Flashcards, chat, clipper, persona, and companion sidepanel routes keep their current handoff tests.
- Internal/debug sidepanel routes are excluded from default user navigation.

- [ ] **Step 2: Write hosted visibility tests**

Expand `option-route-visibility.test.ts` so hosted-visible paths come from metadata or are checked against metadata. Assert:

- Hosted-visible paths are not internal/debug.
- Hosted-hidden paths have an explicit reason.
- Audio explainer routes `/stt` and `/tts` remain visible if hosted copy depends on them.

- [ ] **Step 3: Run sidepanel and hosted tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/route-governance.sidepanel-availability.test.ts apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-clipper.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
```

Expected: PASS.

- [ ] **Step 4: Commit sidepanel and hosted governance**

```bash
git add apps/packages/ui/src/routes/__tests__/route-governance.sidepanel-availability.test.ts apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-clipper.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts apps/packages/ui/src/routes/option-route-visibility.ts apps/packages/ui/src/routes/sidepanel-route-registry.tsx apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx
git commit -m "test: govern sidepanel route availability"
```

### Task 4: Govern Capability And Raw Error States

**Files:**
- Create: `apps/tldw-frontend/e2e/smoke/route-capability-state-governance.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/smoke.setup.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Modify: representative route tests from WP2, WP10, WP11A, and WP11B as needed

- [ ] **Step 1: Write capability governance fixtures**

Create Playwright fixtures that simulate or assert representative states for:

- `/sources`
- `/scheduled-tasks`
- `/integrations`
- `/settings/model`
- `/evaluations`
- `/mcp-hub`
- `/skills`
- `/tts`
- `/speech`
- `/data-tables`

Each fixture must assert:

- No raw `Not Found (GET` style text is primary route UI.
- A user-language diagnosis is visible.
- A retry, setup, connect, configure, or diagnostic action is visible.
- Raw endpoint detail is absent or behind disclosure.

- [ ] **Step 2: Add smoke allowlist discipline**

In `smoke.setup.ts`, require allowlist entries to include:

- `id`
- `scope`
- `pattern`
- route scope or global rationale
- `owner`
- `expiresOn`

Fail tests when an allowlist rule is expired or ownerless.

- [ ] **Step 3: Run capability governance tests**

Run:

```bash
bunx playwright test apps/tldw-frontend/e2e/smoke/route-capability-state-governance.spec.ts --reporter=line
```

Expected: PASS after route fixtures are aligned with WP2 capability states.

- [ ] **Step 4: Commit capability governance**

```bash
git add apps/tldw-frontend/e2e/smoke/route-capability-state-governance.spec.ts apps/tldw-frontend/e2e/smoke/smoke.setup.ts apps/tldw-frontend/e2e/smoke/all-pages.spec.ts
git commit -m "test: govern route capability states"
```

### Task 5: Govern Responsive And Accessibility Gates

**Files:**
- Create: `apps/tldw-frontend/e2e/smoke/route-responsive-governance.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.helpers.ts`
- Modify: `apps/tldw-frontend/package.json`

- [x] **Step 1: Write 390px overflow governance**

Create `route-responsive-governance.spec.ts` with route metadata-driven checks for:

- `/chat`
- `/media`
- `/settings`
- `/settings/model`
- `/prompts`
- `/workspace-playground`
- `/setup`
- `/sources`
- `/mcp-hub`
- `/stt`
- `/tts`
- `/chat-workspace`

Assert:

```ts
const overflow = await page.evaluate(() => {
  const root = document.scrollingElement ?? document.documentElement
  return root.scrollWidth - root.clientWidth
})
expect(overflow, `${route.path} has page-level horizontal overflow`).toBeLessThanOrEqual(1)
```

- [x] **Step 2: Add sidepanel-width checks**

For routes marked sidepanel available and changed by the current PR, run a sidepanel viewport check. Start with:

- `/chat`
- `/flashcards`
- `/companion`
- `/persona`

Use route metadata to expand this set as sidepanel support grows.

- [x] **Step 3: Align high-risk Axe routes with metadata**

Update `stage4-axe-high-risk-routes.spec.ts` so high-risk routes are selected from metadata or checked against metadata. Every manual high-risk route entry needs a rationale.

- [x] **Step 4: Add package scripts**

Add scripts to `apps/tldw-frontend/package.json`:

```json
{
  "e2e:smoke:route-governance": "playwright test e2e/smoke/route-responsive-governance.spec.ts --reporter=line --workers=1",
  "e2e:smoke:governance-gate": "bun run e2e:smoke:all-pages:gate && bun run e2e:smoke:stage4 && bun run e2e:smoke:route-governance"
}
```

The route-governance script intentionally scopes to the stable WP12 responsive/sidepanel governance spec. An attempted broader bundle with existing route-heading and route-capability governance suites exposed unrelated baseline failures and is not used for this Task 5 gate.

If repository script conventions avoid shell chaining, create a small Node runner instead of adding an `&&` script.

- [x] **Step 5: Run responsive and Axe tests**

Run:

```bash
bun run e2e:smoke:route-governance
```

Expected: PASS.

Run:

```bash
bun run e2e:smoke:stage4
```

Expected: PASS.

- [x] **Step 6: Commit responsive and accessibility governance**

```bash
git add apps/tldw-frontend/e2e/smoke/route-responsive-governance.spec.ts apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.spec.ts apps/tldw-frontend/e2e/smoke/stage4-axe-high-risk-routes.helpers.ts apps/tldw-frontend/package.json
git commit -m "test: govern responsive route behavior"
```

### Task 6: Define Browser Evidence Protocol

**Files:**
- Create: `apps/tldw-frontend/e2e/smoke/route-evidence-protocol.md`
- Modify: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md` only if the user asks to append final evidence links
- Modify: Backlog tasks for completed slices during implementation

- [ ] **Step 1: Write evidence protocol**

Create `route-evidence-protocol.md` documenting:

- Required before and after observations for changed visual routes.
- Required viewport list: desktop, 390px mobile, and sidepanel width when relevant.
- Screenshot naming convention.
- DOM snapshot naming convention.
- Console and request triage format.
- Known-skip format.
- Backlog task final-summary fields.

- [ ] **Step 2: Add evidence template**

Include this template:

```md
## Route Evidence

- Route:
- Slice:
- Date:
- Viewport:
- State:
- Screenshot:
- DOM or browser observation:
- Console/request notes:
- Expected capability failures:
- Regressions found:
- Follow-up task:
```

- [ ] **Step 3: Link protocol from governance plan or task docs**

Reference the protocol from Task 12 implementation PRs and route-family Backlog tasks. Do not append screenshot links to the original audit unless the user explicitly asks for a consolidated final report update.

- [ ] **Step 4: Commit evidence protocol**

```bash
git add apps/tldw-frontend/e2e/smoke/route-evidence-protocol.md
git commit -m "docs: define route evidence protocol"
```

### Task 7: Run Final Governance Gate And Close Coverage Matrix

**Files:**
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/package.json`
- Modify: active Backlog task for the implementation slice

- [ ] **Step 1: Run all-pages hard gate**

From `apps/tldw-frontend`, run:

```bash
bun run e2e:smoke:all-pages:gate
```

Expected: PASS, or every failure is tied to a route metadata reason and a follow-up task.

- [ ] **Step 2: Run Stage 4 Axe gate**

From `apps/tldw-frontend`, run:

```bash
bun run e2e:smoke:stage4
```

Expected: PASS.

- [ ] **Step 3: Run route governance gate**

From `apps/tldw-frontend`, run:

```bash
bun run e2e:smoke:route-governance
```

Expected: PASS.

- [ ] **Step 4: Run route metadata unit tests**

From repository root, run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts apps/packages/ui/src/routes/__tests__/route-governance.sidepanel-availability.test.ts apps/packages/ui/src/components/Common/__tests__/CommandPalette.route-targets.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Update Backlog closure fields**

Update the active implementation task with:

- Findings closed.
- Route rows changed.
- Tests run.
- Browser evidence paths.
- Known skips.
- Backend dependencies deferred to separate tasks.
- Any route rows intentionally left open.

- [ ] **Step 6: Commit final governance closeout**

```bash
git add apps/tldw-frontend/e2e/smoke/all-pages.spec.ts apps/tldw-frontend/e2e/smoke/page-inventory.ts apps/tldw-frontend/package.json
git commit -m "test: close route governance gate"
```

## Acceptance Criteria

- Route inventory cannot silently omit user-facing page-file routes.
- Shared route registry, extension route registry, page inventory, sidepanel matrix, and command palette are checked against one route metadata contract.
- Every user-facing root route has one `h1` or an approved metadata exception.
- Command palette route commands have labels, targets, and visibility that agree with metadata.
- Smoke inventory skips require explicit route-owned reasons.
- Representative capability-state fixtures prove raw endpoint text is not primary UI.
- Representative core routes fail on page-level horizontal overflow at 390px.
- Stage 4 high-risk Axe route list is metadata-backed or metadata-checked.
- Manual browser QA protocol exists for before and after screenshots, DOM observations, console triage, and known skips.
- Final Backlog task records findings closed, routes changed, tests run, evidence paths, known skips, and deferred backend dependencies.

## Verification Commands

Run these before considering Task 12 complete:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts apps/packages/ui/src/routes/__tests__/route-governance.sidepanel-availability.test.ts apps/packages/ui/src/components/Common/__tests__/CommandPalette.route-targets.test.tsx
```

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/option-route-visibility.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-clipper.test.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx apps/packages/ui/src/components/Common/__tests__/CommandPaletteHost.test.tsx
```

From `apps/tldw-frontend`, run:

```bash
bun run e2e:smoke:all-pages:gate
```

```bash
bun run e2e:smoke:stage4
```

```bash
bun run e2e:smoke:route-governance
```

From repository root, run:

```bash
git diff --check
```

```bash
bunx tsc --noEmit
```

## Review Notes For Implementers

- Start with metadata coverage tests. Do not add more smoke inventory rows before proving metadata ownership.
- Keep CI gates deterministic. Put subjective visual judgment in browser evidence, not fragile pixel assertions.
- Every skip needs a route-owned reason. A skip without owner, scope, and route metadata is not governance.
- Keep manual screenshots scoped to changed routes. Full visual regression for every page remains out of scope.
- Do not fix route-family UX regressions inside Task 12 unless they are trivial test harness wiring. Create a route-family follow-up task when the fix belongs to a product surface.
- Treat warning allowlists as debt with owner and expiry, not permanent silence.
- Keep Backlog closure factual: routes changed, findings closed, tests run, evidence captured, known skips, and deferred backend dependencies.
