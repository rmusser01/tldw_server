# WebUI Route Contract Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a canonical route metadata contract that makes WebUI and extension route labels, aliases, visibility, sidepanel availability, command palette targets, and smoke inventory coverage explicit and testable.

**Architecture:** Add a metadata layer beside the existing route registries first, then validate existing registries against it before using it to drive behavior. Keep route elements and aliases compatible while moving user-facing route decisions into typed data. This slice closes route-governance UX issues without redesigning page content or changing backend APIs.

**Tech Stack:** React, TypeScript, Next.js page wrappers, shared `@tldw/ui` route registries, `react-router-dom`, Vitest, Playwright, existing smoke inventory, Backlog.md task tracking.

---

## Source Documents

- Parent plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Source spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Source audit: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`
- Planning Backlog task: `TASK-419`
- Parent planning task: `TASK-418`

## Scope

### Findings Closed Or Partially Closed

- F1: Navigation and IA are fragmented across multiple source-of-truth systems.
- F8: Command palette contains route-label mismatch.
- F12: Extension sidepanel route availability is not aligned with shared intent.
- F17: Route inventory and smoke coverage are not authoritative.
- F18: Specialized, hosted, beta, and debug routes need a visibility policy.

### Primary Route Rows

All 74 audited root/top-level route rows get metadata. First-code focus:

`/`, `/chat`, `/search`, `/reading`, `/review`, `/audio`, `/prompt-studio`,
`/workspace-playground`, `/billing`, `/account`, `/signup`, `/documentation`,
`/composer-variants-preview`, `/onboarding-test`, `/vn-assets`, `/vn-play`.

### Out Of Scope

- No visual redesign.
- No page-local UX remediation except the command target correction.
- No backend API changes.
- No removal of existing aliases or direct URLs.
- No broad smoke-suite rewrite beyond route-contract validation.

## Metadata Contract

Create a typed route contract that can describe each audited route without
requiring every route to be shown in navigation.

Recommended file:

- Create: `apps/packages/ui/src/routes/route-metadata.ts`

Recommended exports:

```ts
export type RouteSurface =
  | "default_self_hosted"
  | "advanced_self_hosted"
  | "hosted_only"
  | "admin_operator"
  | "extension_sidepanel"
  | "labs_beta"
  | "internal_qa_debug"
  | "legacy_alias"
  | "redirect"
  | "deprecated"

export type RouteGroup =
  | "start"
  | "chat"
  | "knowledge"
  | "media_library"
  | "settings"
  | "operations"
  | "workspace"
  | "audio"
  | "study"
  | "safety"
  | "specialized"
  | "documentation"
  | "account"

export type RouteAvailability = "web" | "extension_options" | "extension_sidepanel"

export type RouteMetadata = {
  path: string
  canonicalPath: string
  label: string
  group: RouteGroup
  surface: RouteSurface
  availability: RouteAvailability[]
  aliases?: string[]
  redirectsTo?: string
  smoke: "include" | "exclude" | "manual"
  commandPalette: "show" | "hide" | "alias_only"
  nav: "primary" | "secondary" | "hidden"
  requiresAuth?: boolean
  requiresBackend?: boolean
  rationale: string
}
```

Keep labels in English in this contract for testable user-facing route identity.
Do not wire locale/i18n migration in this slice.

## File Map

### Create

- `apps/packages/ui/src/routes/route-metadata.ts`
  - Owns typed metadata for audited routes.
  - Contains helpers such as `getRouteMetadata(path)`,
    `getCanonicalRoutePath(path)`, `isRouteVisibleForSurface(path, surface)`,
    and `getRoutesForSmokeInventory()`.

- `apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts`
  - Verifies all 74 audited routes have metadata.
  - Verifies aliases resolve to canonical routes.
  - Verifies every route has a visibility classification and smoke policy.

- `apps/packages/ui/src/routes/__tests__/route-registry.visibility.test.ts`
  - Verifies option registry routes have corresponding metadata.
  - Verifies metadata does not claim a route exists when no page/registry path
    exists unless it is a documented redirect, placeholder, or external page.

- `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts`
  - Verifies sidepanel route availability is declared in metadata.
  - Verifies extension sidepanel routes are not inferred from options routes.

### Modify

- `apps/packages/ui/src/routes/route-registry.tsx`
  - Keep existing route elements.
  - Add metadata imports only if needed for validation or visibility filtering.

- `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
  - Keep existing route elements.
  - Validate sidepanel routes against metadata.

- `apps/tldw-frontend/extension/routes/route-registry.tsx`
  - Preserve existing `nav` behavior during migration.
  - Later replace duplicated route labels with metadata only after tests pass.

- `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
  - Validate sidepanel availability against shared metadata.

- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
  - First add validation against metadata.
  - Generate from metadata only after validation is stable.

- `apps/packages/ui/src/components/Common/CommandPalette.tsx`
  - Fix "Go to Chat" so it targets `/chat`, or relabel it if it intentionally
    targets `/`.
  - Prefer metadata-backed navigation commands after direct bug fix is covered.

- `apps/packages/ui/src/components/Common/CommandPaletteHost.tsx`
  - Only modify if command availability needs host-level filtering.

- `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
  - Validate shortcuts against canonical metadata.

- `apps/packages/ui/src/components/Layouts/ModeSelector.tsx`
  - Validate route targets against canonical metadata if it owns route jumps.

- `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`
  - Extend from placeholder route contract checks to metadata contract checks.

### Existing Tests To Reuse

- `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx`
- `apps/packages/ui/src/routes/__tests__/sources-route-registry.test.tsx`
- `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`
- `apps/packages/ui/src/routes/__tests__/integrations-route.test.tsx`
- `apps/packages/ui/src/routes/__tests__/mcp-hub-route.test.tsx`
- `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-chat.test.ts`
- `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`

## Route Metadata Bootstrap Set

The first implementation should include all audited routes from the spec Route
Coverage Matrix. Use this list as the acceptance source:

```text
/
/setup
/login
/signup
/account
/profile
/privileges
/config
/billing
/404
/chat
/quick-chat-popout
/persona
/characters
/companion
/agents
/agent-tasks
/chat-workflows
/chat-workspace
/knowledge
/search
/research
/workspace-playground
/document-workspace
/repo2txt
/model-playground
/writing-playground
/presentation-studio
/audiobook-studio
/media
/media-multi
/review
/media-trash
/items
/collections
/reading
/notes
/shared
/chatbooks
/chatbooks-playground
/sources
/connectors
/integrations
/scheduled-tasks
/watchlists
/workflow-editor
/settings
/admin
/mcp-hub
/acp-playground
/prompts
/prompt-studio
/dictionaries
/world-books
/speech
/stt
/tts
/audio
/evaluations
/flashcards
/quiz
/moderation-playground
/content-review
/claims-review
/data-tables
/chunking-playground
/kanban
/skills
/vn-assets
/vn-play
/documentation
/notifications
/composer-variants-preview
/onboarding-test
```

## Implementation Tasks

### Task 0: Baseline And Branch Hygiene

**Files:**
- Reference: `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`
- Reference: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Backlog: create or update an implementation task before product code edits.

- [ ] **Step 1: Verify branch and dirty worktree**

Run:

```bash
git branch --show-current
git status --short
```

Expected:
- You know the current branch.
- Existing unrelated dirty files are left untouched.

- [ ] **Step 2: Create implementation Backlog task**

Create a task named `Implement WebUI route contract and visibility policy`.

Expected:
- The task links this plan, the parent plan, and source audit.
- The task lists findings F1, F8, F12, F17, F18.

- [ ] **Step 3: Capture baseline route evidence**

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line
```

Expected:
- Current baseline is recorded before route contract edits.
- Failures are copied into the Backlog task rather than silently ignored.

### Task 1: Add Route Metadata Types And Coverage Tests

**Files:**
- Create: `apps/packages/ui/src/routes/route-metadata.ts`
- Create: `apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts`

- [ ] **Step 1: Write failing coverage test**

Add `AUDITED_ROOT_ROUTES` inside the test, using the 74-route list from this
plan.

Test expectations:

```ts
import { describe, expect, it } from "vitest"
import {
  getRouteMetadata,
  ROUTE_METADATA
} from "../route-metadata"

describe("route metadata coverage", () => {
  it("defines metadata for every audited root route", () => {
    for (const route of AUDITED_ROOT_ROUTES) {
      expect(getRouteMetadata(route), route).toBeDefined()
    }
  })

  it("defines required user-facing metadata fields", () => {
    for (const metadata of ROUTE_METADATA) {
      expect(metadata.path).toMatch(/^\\//)
      expect(metadata.canonicalPath).toMatch(/^\\//)
      expect(metadata.label.trim()).not.toHaveLength(0)
      expect(metadata.group).toBeTruthy()
      expect(metadata.surface).toBeTruthy()
      expect(metadata.availability.length).toBeGreaterThan(0)
      expect(metadata.rationale.trim()).not.toHaveLength(0)
    }
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/route-metadata.coverage.test.ts
```

Expected:
- Fails because `route-metadata.ts` does not exist or has empty coverage.

- [ ] **Step 3: Implement metadata skeleton**

Create `route-metadata.ts` with the typed contract and placeholder-complete
metadata for all 74 audited routes. Use conservative classifications:
- hidden nav for aliases, hosted-only routes, debug/QA routes, labs routes.
- include smoke for default self-hosted routes.
- manual smoke for routes requiring setup or unstable external state.
- exclude smoke only when the route is explicitly internal, hosted-only, or
  covered by another canonical path.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
bunx vitest run src/routes/__tests__/route-metadata.coverage.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/route-metadata.ts apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts
git commit -m "feat: add webui route metadata contract"
```

### Task 2: Validate Option Route Registry Against Metadata

**Files:**
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Create: `apps/packages/ui/src/routes/__tests__/route-registry.visibility.test.ts`
- Reference: `apps/tldw-frontend/pages`

- [ ] **Step 1: Write failing route-registry validation test**

Test expectations:
- every non-dynamic `ROUTE_DEFINITIONS` option path has metadata
- every metadata path marked `web` has either a route-registry entry, a Next page
  file, or an explicit redirect/placeholder rationale
- no route marked `internal_qa_debug` is primary nav
- no route marked `hosted_only` is default self-hosted nav

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
bunx vitest run src/routes/__tests__/route-registry.visibility.test.ts
```

Expected:
- Fails on missing metadata integration or classification mismatches.

- [ ] **Step 3: Add helper exports only if needed**

If the test cannot inspect route definitions cleanly, export pure helpers from
`route-metadata.ts` rather than changing route elements.

Do not refactor lazy route imports in this task.

- [ ] **Step 4: Fix classifications and registry references**

Update metadata, not route behavior, unless an existing route target is clearly
wrong and covered by test.

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
bunx vitest run src/routes/__tests__/route-registry.visibility.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/routes/route-metadata.ts apps/packages/ui/src/routes/__tests__/route-registry.visibility.test.ts
git commit -m "test: validate webui route visibility metadata"
```

### Task 3: Validate Extension Options And Sidepanel Availability

**Files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
- Create: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts`

- [ ] **Step 1: Write failing sidepanel availability test**

Test expectations:
- every shared sidepanel route has `extension_sidepanel` availability metadata
- extension options routes that appear in nav have metadata labels/groups
- `/error-boundary-test`, `/__debug__/sidepanel-chat`, and other debug routes
  are marked `internal_qa_debug`
- extension sidepanel route availability is not inferred from web route presence

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
bunx vitest run src/routes/__tests__/route-registry.sidepanel-availability.test.ts
```

Expected:
- Fails on missing sidepanel metadata or inconsistent debug classification.

- [ ] **Step 3: Add metadata or validation adapters**

Prefer pure test helpers that read source registries and metadata. Do not move
extension route definitions yet unless required.

- [ ] **Step 4: Run sidepanel route tests**

Run:

```bash
bunx vitest run src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/routes/__tests__/route-registry.sidepanel-chat.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/route-metadata.ts apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts
git commit -m "test: validate extension sidepanel route availability"
```

### Task 4: Fix Command Palette Route Target Trust

**Files:**
- Modify: `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Common/CommandPaletteHost.tsx`

- [ ] **Step 1: Write failing command target test**

Add a test asserting:
- command id `nav-chat` has label "Go to Chat"
- command id `nav-chat` has `targetPath === "/chat"`
- selecting the command navigates to `/chat`

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
bunx vitest run src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx
```

Expected:
- Fails because current `nav-chat` navigates to `/` and has `targetPath: "/"`.

- [ ] **Step 3: Fix command target**

Change the command action and `targetPath` to `/chat`.

Do not broaden command palette route generation in the same commit unless the
child implementation task explicitly expands scope.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
bunx vitest run src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Common/CommandPalette.tsx apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx
git commit -m "fix: route chat command to chat page"
```

### Task 5: Add Smoke Inventory Validation

**Files:**
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts`
- Reference: `apps/packages/ui/src/routes/route-metadata.ts`

- [ ] **Step 1: Write failing smoke inventory validation**

Add a test or script that checks:
- routes marked `smoke: "include"` appear in `PAGES`
- routes marked `smoke: "manual"` are excluded only with a reason
- routes marked `internal_qa_debug` are not treated as normal product pages
- aliases in `PAGES` point at canonical metadata rows

- [ ] **Step 2: Run smoke validation**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line
```

Expected:
- Fails until inventory and metadata agree.

- [ ] **Step 3: Align inventory with metadata**

Prefer validation over generation in this first slice. If generation is easier
later, make it a separate small commit after validation passes.

- [ ] **Step 4: Run smoke validation again**

Run:

```bash
bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/tldw-frontend/e2e/smoke/page-inventory.ts apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts
git commit -m "test: validate smoke inventory against route metadata"
```

### Task 6: Final Route Contract Gate

**Files:**
- Reference all files changed in Tasks 1-5.
- Backlog: update the active implementation task.

- [ ] **Step 1: Run route metadata and command tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/route-metadata.coverage.test.ts src/routes/__tests__/route-registry.visibility.test.ts src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run route contract smoke test**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line
```

Expected: PASS or documented environment-specific skip with evidence.

- [ ] **Step 3: Run diff checks**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Update Backlog task**

Record:
- findings closed: F1, F8, F12, F17, F18
- route metadata coverage count
- route inventory validation result
- sidepanel matrix validation result
- command target validation result
- known skips or deferred routes

- [ ] **Step 5: Commit final task notes if needed**

```bash
git add backlog/tasks/<task-file>.md
git commit -m "docs: record route contract verification"
```

## Review Checklist

Before opening an implementation PR:

- [ ] Every audited route has metadata.
- [ ] Every route has one visibility classification.
- [ ] Aliases and redirects keep existing direct URLs compatible.
- [ ] `/chat` command palette target is truthful.
- [ ] Sidepanel availability is explicit.
- [ ] Smoke inventory is checked against metadata.
- [ ] Debug, QA, labs, hosted-only, and deprecated routes are not promoted as
  normal self-hosted product routes.
- [ ] No page-level visual redesign was included.
- [ ] No backend API was changed.
- [ ] Browser or Playwright evidence is attached for changed route behavior.

## Planning Verification

After editing this plan, run:

```bash
rg -n 'T[O]DO|T[B]D|FIX[M]E|\\.\\.\\.|\\bmaybe\\b|\\bprobably\\b|\\bshould consider\\b' Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md 'backlog/tasks/task-419 - Plan-WebUI-route-contract-and-visibility-policy-implementation.md'
rg -n '[[:blank:]]$|[^\\x00-\\x7F]' Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md 'backlog/tasks/task-419 - Plan-WebUI-route-contract-and-visibility-policy-implementation.md'
git diff --check -- Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md 'backlog/tasks/task-419 - Plan-WebUI-route-contract-and-visibility-policy-implementation.md'
```

Expected:
- Placeholder scan exits 1 with no output.
- ASCII/trailing whitespace scan exits 1 with no output.
- `git diff --check` exits 0.
