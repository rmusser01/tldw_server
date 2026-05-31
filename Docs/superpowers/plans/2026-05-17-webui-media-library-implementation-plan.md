# WebUI Media Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve large-library and bulk-review power while improving first selection, narrow-width browsing, recovery, trash safeguards, object terminology, sharing, Chatbooks, and notifications.

**Architecture:** Add a small route-job and alias policy layer, then make focused route and component changes that clarify the job of each media/library page. Reuse existing `ViewMediaPage`, `MediaReviewPage`, `MediaTrashPage`, `CollectionsPlaygroundPage`, `ItemsWorkspace`, `NotesManagerPage`, `SharedWithMe`, `ChatbooksPlaygroundPage`, and notifications code instead of replacing the media or library runtime.

**Tech Stack:** React, TypeScript, React Router, Next.js pages, Ant Design, shared WebUI state components, TanStack Query, Vitest, Testing Library, Playwright, Backlog.md task tracking.

---

## Source Documents

- Source spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Parent implementation plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Planning Backlog task: `TASK-418.5`
- Parent planning Backlog task: `TASK-418`
- Dependency plan: `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`
- Dependency plan: `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
- Dependency plan: `Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md`

## Findings Closed Or Supported

- F10: `/media` and related review pages need usable list-detail behavior at narrow widths.
- F2 support: setup, unavailable, and empty media states must point to the next action in user language.
- F1 support: route names, aliases, and headings must explain media, review, library, sharing, and Chatbook jobs.
- F18 support: beta, unsupported, disconnected, and degraded states must not look like user failure.
- F15 support: dense filters, bulk actions, jobs, and recovery controls must remain discoverable for power users.

## Route Scope

Primary implementation routes:

- `/media`
- `/media-multi`
- `/review`
- `/media-trash`
- `/items`
- `/collections`
- `/reading`
- `/notes`
- `/shared`
- `/chatbooks`
- `/chatbooks-playground`
- `/notifications`

Related route targets:

- `/settings/tldw`
- `/settings/health`
- `/workspace-playground`
- `/watchlists`

## Out Of Scope

- No backend ingestion, media processing, notification, sharing, or Chatbooks API changes.
- No broad media redesign or design-system replacement.
- No removal of existing filters, bulk selection, keyboard shortcuts, import/export, or job tracking.
- No route renaming. Existing URLs and aliases stay valid.
- No changes to unrelated knowledge, chat, research, settings, or admin routes.

## Current Code Evidence

- `apps/packages/ui/src/routes/option-media-view-route-registry.tsx` registers `/media` and `/media-trash`.
- `apps/packages/ui/src/routes/option-media-review-route-registry.tsx` redirects `/review` to `/media-multi` and registers `/media-multi`.
- `apps/tldw-frontend/pages/review.tsx` redirects `/review` to `/media-multi`.
- `apps/tldw-frontend/pages/reading.tsx` redirects `/reading` to `/collections`.
- `apps/packages/ui/src/routes/route-registry.tsx` registers `/shared`, `/chatbooks`, `/collections`, and `/notes`.
- `apps/tldw-frontend/pages/chatbooks.tsx` and `apps/tldw-frontend/pages/chatbooks-playground.tsx` both load `option-chatbooks-playground`.
- `apps/tldw-frontend/pages/items.tsx` loads `option-items`.
- `apps/tldw-frontend/pages/shared.tsx` loads `option-shared-with-me`.
- `apps/packages/ui/src/routes/option-media.tsx` renders `ViewMediaPage` inside `RouteErrorBoundary`.
- `apps/packages/ui/src/routes/option-media-multi.tsx` renders connection, demo, unsupported, and `MediaReviewPage` states inside `RouteErrorBoundary`.
- `apps/packages/ui/src/routes/option-media-trash.tsx` renders `MediaTrashPage` inside `RouteErrorBoundary`.
- `apps/packages/ui/src/routes/option-collections.tsx`, `apps/packages/ui/src/routes/option-notes.tsx`, and `apps/packages/ui/src/routes/option-items.tsx` already use `RouteErrorBoundary`.
- `apps/packages/ui/src/routes/option-shared-with-me.tsx` renders `SharedWithMe` without the route-boundary wrapper used by nearby routes.
- `apps/packages/ui/src/routes/option-chatbooks-playground.tsx` renders `ChatbooksPlaygroundPage` without the route-boundary wrapper used by nearby routes.
- `apps/packages/ui/src/components/Review/ViewMediaPage.tsx` owns media search, filters, bulk mode, quick ingest, library tools, media trash link, content viewer, selection recovery, and navigation persistence.
- `apps/packages/ui/src/components/Media/ContentViewer.tsx` owns the selected-item empty state and quick ingest affordance.
- `apps/packages/ui/src/components/Review/MediaReviewPage.tsx` owns the three-panel review workflow, filters, results, reading pane, selection status, batch trash handoff, and keyboard shortcuts.
- `apps/packages/ui/src/components/Review/ResizablePanels.tsx` controls the media review narrow-width tab layout.
- `apps/packages/ui/src/components/Review/MediaTrashPage.tsx` owns restore, permanent delete, empty trash, retention policy, search, pagination, bulk restore, and bulk delete behavior.
- `apps/packages/ui/src/components/Option/Collections/index.tsx` owns the Collections tabs: Reading List, Highlights, Templates, Digest Schedules, and Import/Export.
- `apps/packages/ui/src/components/Option/Items/ItemsWorkspace.tsx` owns saved item search, filters, bulk actions, item status, favorites, tag edits, and generated outputs.
- `apps/packages/ui/src/components/Option/SharedWithMe/index.tsx` owns shared workspace listing, open, and clone actions.
- `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx` owns export, import, job tracker, polling, cleanup, download, cancel, and remove actions.
- `apps/tldw-frontend/pages/notifications.tsx` owns notification inbox state, preferences, refresh, snooze, dismiss, mark read, link routing, and stream/poll fallback.

## Route Job Taxonomy

Use this route-job policy as the implementation source of truth. The wording can be adapted to existing localization keys, but the concept, primary job, and primary action must be visible through headings, labels, empty states, tests, or route metadata.

| Route | Concept | Primary job | Primary action | Power-user job |
| --- | --- | --- | --- | --- |
| `/media` | Media inspector | Search and inspect one media or note item at a time with content, metadata, and analysis actions. | Select first item or quick ingest | Filter, resume last item, bulk-select, create collections, discuss media, generate study assets. |
| `/media-multi` | Media review | Triage and compare many media or note items in a dense review layout. | Review results | Filter, batch select, compare, export, trash handoff, keyboard navigation. |
| `/review` | Legacy review alias | Preserve old review entrypoint by sending users to Media Review. | Continue to Media Review | Keep old links functional without duplicate UI. |
| `/media-trash` | Media trash | Restore or permanently delete trashed media with visible retention policy. | Restore selected | Search, bulk restore, permanent delete, empty trash, inspect failures. |
| `/items` | Saved items | Manage saved reading or content items across status, tags, favorites, and bulk actions. | Review saved items | Filter, tag, favorite, set status, generate output, bulk edit. |
| `/collections` | Collections | Manage reading list, highlights, templates, digests, and import/export. | Add or review reading item | Switch tabs, bulk manage reading items, manage templates and digest schedules. |
| `/reading` | Reading alias | Preserve reading-list entrypoint by sending users to Collections. | Continue to Collections | Keep old links functional and land on the reading tab when supported. |
| `/notes` | Notes | Create, find, edit, and organize notes. | Create or search note | Keyword, dock, edit, recover, and cross-link notes. |
| `/shared` | Shared workspaces | Open or clone workspaces shared by another user. | Open shared workspace | Clone, inspect owner/access, recover from empty or unsupported states. |
| `/chatbooks` | Chatbooks | Export or import portable bundles and track jobs. | Export or import chatbook | Filter content, select types, download jobs, cancel active work, clean old jobs. |
| `/chatbooks-playground` | Chatbooks alias | Preserve old playground entrypoint by loading Chatbooks. | Continue to Chatbooks | Keep old links and launcher ids functional. |
| `/notifications` | Notifications | Review reminders and job notices, then act or tune preferences. | View notification | Mark read, snooze, dismiss, route to linked work, change preferences. |

## Alias And Canonical Route Policy

| Alias route | Canonical target | Required behavior |
| --- | --- | --- |
| `/review` | `/media-multi` | Redirect or navigate with replace; keep E2E coverage for old links. |
| `/reading` | `/collections` | Redirect or navigate with replace; land on Reading List tab when route state or query support exists. |
| `/chatbooks-playground` | `/chatbooks` | Keep Next page compatibility, but global launchers should use `/chatbooks`. |
| `/items` | `/items` | Keep as a direct saved-items page, not a redirect to Collections. |
| `/shared` | `/shared` | Keep direct shared-workspace inbox semantics. |

If WP1 route metadata already owns canonical aliases, use that contract instead of creating a parallel map. If not, create a pure local helper for this route family.

## File Ownership Map

### Route Job And Alias Contract

- Create: `apps/packages/ui/src/routes/media-library-route-jobs.ts`
  - Own media/library route jobs, primary labels, and canonical targets.
  - Keep pure TypeScript. No React hooks, network calls, or component imports.

- Create: `apps/packages/ui/src/routes/__tests__/media-library-route-jobs.test.ts`
  - Assert every WP8 route has a concept, primary job, primary action, and canonical route target when relevant.

- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
  - Add `/reading` and `/chatbooks-playground` aliases only if WP1 did not already centralize alias handling.
  - Preserve direct `/items`, `/shared`, `/chatbooks`, `/collections`, and `/notes`.

- Modify: `apps/packages/ui/src/routes/option-media-view-route-registry.tsx`
  - Preserve `/media` and `/media-trash`.
  - Attach metadata only if route registry tests require it.

- Modify: `apps/packages/ui/src/routes/option-media-review-route-registry.tsx`
  - Preserve `/review` redirect to `/media-multi`.
  - Assert redirect is documented as an alias, not a second review page.

- Modify: `apps/tldw-frontend/pages/review.tsx`
  - Preserve redirect to `/media-multi`.

- Modify: `apps/tldw-frontend/pages/reading.tsx`
  - Preserve redirect to `/collections`.

- Modify: `apps/tldw-frontend/pages/chatbooks-playground.tsx`
  - Preserve old URL compatibility while route launchers use `/chatbooks`.

- Test: `apps/packages/ui/src/routes/__tests__/option-media-route-guards.test.tsx`
  - Assert route boundaries for `/media`, `/media-multi`, `/media-trash`, `/shared`, and `/chatbooks` after wrappers are aligned.

- Test: `apps/packages/ui/src/routes/__tests__/media-library-aliases.test.tsx`
  - Assert `/review`, `/reading`, and `/chatbooks-playground` canonical behavior.

### Media Inspector

- Modify: `apps/packages/ui/src/routes/option-media.tsx`
  - Preserve `RouteErrorBoundary`.
  - Add route job metadata only if needed.

- Modify: `apps/packages/ui/src/components/Review/ViewMediaPage.tsx`
  - Preserve search, filters, bulk selection, library tools, trash link, quick ingest, favorites, collections, study pack, flashcards, chat handoff, and keyboard shortcuts.
  - Improve first-selection and narrow-width behavior without removing existing dense controls.

- Modify: `apps/packages/ui/src/components/Media/ContentViewer.tsx`
  - Clarify empty detail state for no selection, empty library, deleted selection, and stale selection recovery.
  - Keep quick ingest and keyboard shortcut affordances.

- Modify if needed: `apps/packages/ui/src/components/Review/hooks/useMediaNavigationState.ts`
  - Keep last item resume and stale-selection behavior.
  - Add tests before changing persistence or recovery logic.

- Modify if needed: `apps/packages/ui/src/utils/media-navigation-resume.ts`
  - Keep resume data pure and testable.

- Test: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.connection.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.search-experience.integration.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.stage13.error-handling.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.stage14.bulk-actions.test.tsx`
- Create or modify: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.first-selection-mobile.test.tsx`

### Media Review

- Modify: `apps/packages/ui/src/routes/option-media-multi.tsx`
  - Preserve connection, demo, unavailable, and route error states.
  - Align unsupported states with WP2 capability vocabulary where shared helpers exist.

- Modify: `apps/packages/ui/src/components/Review/MediaReviewPage.tsx`
  - Preserve three-panel layout, filters, results, reading pane, batch bar, selection status, trash handoff, and keyboard shortcut behavior.
  - Clarify empty results versus no selection versus bulk selection states.

- Modify: `apps/packages/ui/src/components/Review/ResizablePanels.tsx`
  - Preserve mobile tab behavior and 390px usability.
  - Add stable landmarks and dimensions only where tests prove layout shifts or overflow.

- Modify: `apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx`
  - Preserve card density, virtualization, selected states, and bulk mode.

- Modify: `apps/packages/ui/src/components/Review/MediaReviewReadingPane.tsx`
  - Preserve reading pane and detail states.
  - Clarify no detail or failed detail state in user language.

- Modify: `apps/packages/ui/src/components/Review/MediaReviewBatchBar.tsx`
  - Preserve high-throughput batch actions and selection warnings.

- Test: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.mobile-parity.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.focus-recovery.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.touch-status-a11y.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage7.three-panel.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage5.batch-toolbar.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage5.export-trash-handoff.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-media-multi.connection-state.test.tsx`

### Media Trash

- Modify: `apps/packages/ui/src/routes/option-media-trash.tsx`
  - Preserve `RouteErrorBoundary`.

- Modify: `apps/packages/ui/src/components/Review/MediaTrashPage.tsx`
  - Keep restore, permanent delete, empty trash, search, pagination, and bulk actions.
  - Make retention policy and destructive-action consequence visible before delete actions.
  - Preserve confirm-danger use for permanent deletion.

- Test: `apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.connection.test.tsx`
- Test: `apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.stage2-3.test.tsx`
- Create or modify: `apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.safeguards.test.tsx`

### Library Routes

- Modify: `apps/packages/ui/src/routes/option-items.tsx`
  - Preserve `RouteErrorBoundary`.

- Modify: `apps/packages/ui/src/components/Option/Items/ItemsWorkspace.tsx`
  - Keep saved item filters, tags, favorites, status changes, bulk actions, and output generation.
  - Clarify relationship to Collections and Reading List without redirecting this page.

- Modify: `apps/packages/ui/src/routes/option-collections.tsx`
  - Preserve `RouteErrorBoundary`.

- Modify: `apps/packages/ui/src/components/Option/Collections/index.tsx`
  - Keep tabs and beta alert.
  - Clarify Reading List, Highlights, Templates, Digest Schedules, and Import/Export jobs.

- Modify if needed: `apps/packages/ui/src/components/Option/Collections/ReadingList/ReadingItemsList.tsx`
  - Preserve reading list density and bulk behavior.
  - Add empty-state and recovery text only where tests prove confusion.

- Modify: `apps/packages/ui/src/routes/option-notes.tsx`
  - Preserve `RouteErrorBoundary`.

- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
  - Clarify notes as user-created research notes, not media transcripts.
  - Preserve note creation, search, editing, and keyword workflows.

- Test: `apps/packages/ui/src/components/Option/Items/__tests__/ItemsWorkspace.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Collections/__tests__/CollectionsPlaygroundPage.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.test.tsx`
- Create or modify: `apps/packages/ui/src/routes/__tests__/media-library-aliases.test.tsx`

### Sharing, Chatbooks, And Notifications

- Modify: `apps/packages/ui/src/routes/option-shared-with-me.tsx`
  - Add `RouteErrorBoundary` with route id `shared`.
  - Preserve `SharedWithMe` behavior.

- Modify: `apps/packages/ui/src/components/Option/SharedWithMe/index.tsx`
  - Clarify empty state, error state, open action, clone action, owner, and access level.
  - Keep clone modal and workspace target.

- Modify: `apps/packages/ui/src/routes/option-chatbooks-playground.tsx`
  - Add `RouteErrorBoundary` with route id `chatbooks`.
  - Preserve `ChatbooksPlaygroundPage`.

- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
  - Keep export, import, job tracker, polling, cancel, remove, cleanup, and download.
  - Clarify export versus import direction and job state.
  - Use WP2 capability language for unavailable Chatbooks support where shared helpers exist.

- Modify: `apps/tldw-frontend/pages/notifications.tsx`
  - Clarify inbox empty, loading, error, preferences unavailable, snooze, and linked route behavior.
  - Keep stream/poll fallback and current notification actions.

- Test: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ContentTypePicker.error-state.test.tsx`
- Create or modify: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx`
- Create or modify: `apps/packages/ui/src/components/Option/SharedWithMe/__tests__/SharedWithMe.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-4-admin/notifications.spec.ts`

## Implementation Tasks

### Task 0: Implementation Setup And Evidence Refresh

**Files:**
- Reference: `Docs/superpowers/plans/2026-05-17-webui-media-library-implementation-plan.md`
- Reference: `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`
- Reference: `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
- Reference: `Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md`
- Backlog: create or update the implementation Backlog task before product code edits.

- [ ] **Step 1: Verify branch and dirty worktree**

Run:

```bash
git branch --show-current
git status --short
```

Expected:
- The implementation branch is known.
- Existing unrelated dirty files are left untouched.

- [ ] **Step 2: Create or update implementation Backlog task**

Expected:
- The task links this plan, the parent plan, the source spec, and the audit report.
- The task lists F10, F2 support, F1 support, F18 support, and F15 support.
- The task states that product code edits are limited to the WP8 files in this plan.

- [ ] **Step 3: Capture current browser baseline**

Use Playwright or the in-app browser for:
- `/media` at desktop and 390px width
- `/media-multi` at desktop and 390px width
- `/media-trash`
- `/collections`
- `/notes`
- `/shared`
- `/chatbooks`
- `/notifications`
- `/review`
- `/reading`
- `/chatbooks-playground`

Expected:
- Each primary route has a screenshot or DOM observation covering heading, primary action, empty or setup state, and first visible recovery path.
- Alias routes record target behavior.
- Observations are linked from the Backlog task.

### Task 1: Lock Media Library Route Jobs And Aliases

**Files:**
- Create: `apps/packages/ui/src/routes/media-library-route-jobs.ts`
- Create: `apps/packages/ui/src/routes/__tests__/media-library-route-jobs.test.ts`
- Create or modify: `apps/packages/ui/src/routes/__tests__/media-library-aliases.test.tsx`
- Modify if needed: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify if needed: `apps/packages/ui/src/routes/option-media-review-route-registry.tsx`
- Modify if needed: `apps/tldw-frontend/pages/review.tsx`
- Modify if needed: `apps/tldw-frontend/pages/reading.tsx`
- Modify if needed: `apps/tldw-frontend/pages/chatbooks-playground.tsx`

- [ ] **Step 1: Write failing route-job tests**

Create `media-library-route-jobs.test.ts`:

```ts
import { describe, expect, it } from "vitest"

import {
  MEDIA_LIBRARY_ROUTE_JOBS,
  getMediaLibraryRouteJob,
} from "../media-library-route-jobs"

const expectedRoutes = [
  "/media",
  "/media-multi",
  "/review",
  "/media-trash",
  "/items",
  "/collections",
  "/reading",
  "/notes",
  "/shared",
  "/chatbooks",
  "/chatbooks-playground",
  "/notifications",
]

describe("media library route jobs", () => {
  it("defines one user job for every WP8 route", () => {
    expect(MEDIA_LIBRARY_ROUTE_JOBS.map((job) => job.route).sort()).toEqual(
      expectedRoutes.sort(),
    )

    for (const route of expectedRoutes) {
      const job = getMediaLibraryRouteJob(route)
      expect(job).toBeDefined()
      expect(job?.primaryJob).toMatch(/\w/)
      expect(job?.primaryActionLabel).toMatch(/\w/)
      expect(job?.concept).toMatch(
        /media|review|trash|library|notes|sharing|chatbook|notification|alias/,
      )
    }
  })
})
```

- [ ] **Step 2: Run test to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/media-library-route-jobs.test.ts
```

Expected:
- FAIL because `media-library-route-jobs.ts` does not exist.

- [ ] **Step 3: Implement route-job policy**

Create `media-library-route-jobs.ts`:

```ts
export type MediaLibraryConcept =
  | "media"
  | "review"
  | "trash"
  | "library"
  | "notes"
  | "sharing"
  | "chatbook"
  | "notification"
  | "alias"

export type MediaLibraryRouteJob = {
  route: string
  concept: MediaLibraryConcept
  label: string
  primaryJob: string
  primaryActionLabel: string
  canonicalRoute?: string
}

export const MEDIA_LIBRARY_ROUTE_JOBS: MediaLibraryRouteJob[] = [
  {
    route: "/media",
    concept: "media",
    label: "Media Inspector",
    primaryJob: "Search and inspect one media or note item at a time.",
    primaryActionLabel: "Select Item",
  },
  {
    route: "/media-multi",
    concept: "review",
    label: "Media Review",
    primaryJob: "Triage and compare many media or note items.",
    primaryActionLabel: "Review Results",
  },
  {
    route: "/review",
    concept: "alias",
    label: "Review",
    primaryJob: "Legacy entrypoint for Media Review.",
    primaryActionLabel: "Continue to Media Review",
    canonicalRoute: "/media-multi",
  },
  {
    route: "/media-trash",
    concept: "trash",
    label: "Media Trash",
    primaryJob: "Restore or permanently delete trashed media.",
    primaryActionLabel: "Restore Selected",
  },
  {
    route: "/items",
    concept: "library",
    label: "Saved Items",
    primaryJob: "Manage saved reading and content items.",
    primaryActionLabel: "Review Saved Items",
  },
  {
    route: "/collections",
    concept: "library",
    label: "Collections",
    primaryJob: "Manage reading list, highlights, templates, digests, and import/export.",
    primaryActionLabel: "Review Reading List",
  },
  {
    route: "/reading",
    concept: "alias",
    label: "Reading",
    primaryJob: "Legacy entrypoint for the reading list inside Collections.",
    primaryActionLabel: "Continue to Collections",
    canonicalRoute: "/collections",
  },
  {
    route: "/notes",
    concept: "notes",
    label: "Notes",
    primaryJob: "Create, find, edit, and organize notes.",
    primaryActionLabel: "Create Note",
  },
  {
    route: "/shared",
    concept: "sharing",
    label: "Shared With Me",
    primaryJob: "Open or clone workspaces shared by another user.",
    primaryActionLabel: "Open Shared Workspace",
  },
  {
    route: "/chatbooks",
    concept: "chatbook",
    label: "Chatbooks",
    primaryJob: "Export or import portable bundles and track jobs.",
    primaryActionLabel: "Export or Import",
  },
  {
    route: "/chatbooks-playground",
    concept: "alias",
    label: "Chatbooks Playground",
    primaryJob: "Legacy entrypoint for Chatbooks.",
    primaryActionLabel: "Continue to Chatbooks",
    canonicalRoute: "/chatbooks",
  },
  {
    route: "/notifications",
    concept: "notification",
    label: "Notifications",
    primaryJob: "Review reminders and job notices, then act or tune preferences.",
    primaryActionLabel: "View Notification",
  },
]

export const getMediaLibraryRouteJob = (
  route: string,
): MediaLibraryRouteJob | undefined =>
  MEDIA_LIBRARY_ROUTE_JOBS.find((job) => job.route === route)
```

- [ ] **Step 4: Add alias tests**

Test expectations:
- `/review` targets `/media-multi`.
- `/reading` targets `/collections`.
- `/chatbooks-playground` targets `/chatbooks` or loads the same component while launchers use `/chatbooks`.
- `/items` and `/shared` remain direct routes.

- [ ] **Step 5: Run route tests**

Run:

```bash
bunx vitest run src/routes/__tests__/media-library-route-jobs.test.ts src/routes/__tests__/media-library-aliases.test.tsx src/routes/__tests__/option-media-route-guards.test.tsx
```

Expected:
- PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/routes/media-library-route-jobs.ts apps/packages/ui/src/routes/__tests__/media-library-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/media-library-aliases.test.tsx apps/packages/ui/src/routes/route-registry.tsx apps/packages/ui/src/routes/option-media-review-route-registry.tsx apps/tldw-frontend/pages/review.tsx apps/tldw-frontend/pages/reading.tsx apps/tldw-frontend/pages/chatbooks-playground.tsx
git commit -m "test: lock media library route jobs"
```

### Task 2: Make Media Inspector First Selection And Narrow Layout Usable

**Files:**
- Modify: `apps/packages/ui/src/routes/option-media.tsx`
- Modify: `apps/packages/ui/src/components/Review/ViewMediaPage.tsx`
- Modify: `apps/packages/ui/src/components/Media/ContentViewer.tsx`
- Modify if needed: `apps/packages/ui/src/components/Review/hooks/useMediaNavigationState.ts`
- Modify if needed: `apps/packages/ui/src/utils/media-navigation-resume.ts`
- Create or modify: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.first-selection-mobile.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.connection.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.search-experience.integration.test.tsx`

- [ ] **Step 1: Write failing first-selection tests**

Test expectations:
- Empty library state offers Quick Ingest as the primary next action.
- Non-empty library with no selected item explains selection and keeps keyboard shortcut help discoverable.
- Stale selection notice appears when a saved selection cannot be restored.
- Detail fetch failure offers retry and keeps the selected item visible.
- At 390px, the media list and content detail do not overlap and the selected item can be reached.

- [ ] **Step 2: Run tests to verify failure or current coverage**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Review/__tests__/ViewMediaPage.first-selection-mobile.test.tsx src/components/Review/__tests__/ViewMediaPage.connection.test.tsx src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx src/components/Review/__tests__/ViewMediaPage.search-experience.integration.test.tsx
```

Expected:
- FAIL on missing first-selection or narrow-layout assertions, or PASS only if current UI already meets the contract.

- [ ] **Step 3: Implement minimal first-selection and layout fixes**

Implementation rules:
- Keep the left search/filter/sidebar workflow.
- Keep `Bulk` mode, library tools, trash link, favorites, collections, and Quick Ingest.
- Do not auto-select a result if it would break existing resume, permalink, or keyboard behavior.
- If auto-selection is introduced, make it deterministic, tested, and reversible through selection controls.
- Prefer a narrow-width tab, drawer, or stacked layout using existing responsive patterns over a new layout framework.

- [ ] **Step 4: Run focused tests**

Run:

```bash
bunx vitest run src/components/Review/__tests__/ViewMediaPage.first-selection-mobile.test.tsx src/components/Review/__tests__/ViewMediaPage.connection.test.tsx src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx src/components/Review/__tests__/ViewMediaPage.search-experience.integration.test.tsx
```

Expected:
- PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/option-media.tsx apps/packages/ui/src/components/Review/ViewMediaPage.tsx apps/packages/ui/src/components/Media/ContentViewer.tsx apps/packages/ui/src/components/Review/hooks/useMediaNavigationState.ts apps/packages/ui/src/utils/media-navigation-resume.ts apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.first-selection-mobile.test.tsx apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.connection.test.tsx apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.search-experience.integration.test.tsx
git commit -m "fix: improve media inspector first selection"
```

### Task 3: Preserve Dense Media Review While Clarifying Selection And Recovery

**Files:**
- Modify: `apps/packages/ui/src/routes/option-media-multi.tsx`
- Modify: `apps/packages/ui/src/components/Review/MediaReviewPage.tsx`
- Modify: `apps/packages/ui/src/components/Review/ResizablePanels.tsx`
- Modify: `apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx`
- Modify: `apps/packages/ui/src/components/Review/MediaReviewReadingPane.tsx`
- Modify: `apps/packages/ui/src/components/Review/MediaReviewBatchBar.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/option-media-multi.connection-state.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.mobile-parity.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.focus-recovery.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.touch-status-a11y.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage7.three-panel.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage5.batch-toolbar.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage5.export-trash-handoff.test.tsx`

- [ ] **Step 1: Write failing review workflow tests**

Test expectations:
- 390px viewport uses the existing tabbed or collapsed review layout without hidden primary actions.
- No-selection state tells the user how to choose an item.
- Empty results state differs from disconnected or unsupported media state.
- Selection status remains visible and announced.
- Batch trash handoff offers a clear path to `/media-trash`.
- Keyboard shortcut help remains discoverable without occupying the primary workflow.

- [ ] **Step 2: Run tests to verify failure or current coverage**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-media-multi.connection-state.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.mobile-parity.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.focus-recovery.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.touch-status-a11y.test.tsx src/components/Review/__tests__/MediaReviewPage.stage7.three-panel.test.tsx src/components/Review/__tests__/MediaReviewPage.stage5.batch-toolbar.test.tsx src/components/Review/__tests__/MediaReviewPage.stage5.export-trash-handoff.test.tsx
```

Expected:
- FAIL on missing state or narrow-layout expectations, or PASS only if current UI already satisfies the contract.

- [ ] **Step 3: Implement minimal review improvements**

Implementation rules:
- Preserve current filter sidebar, results panel, reading pane, and batch bar.
- Preserve virtualized or dense result rendering.
- Preserve selection warning thresholds and open-all limits.
- Keep advanced controls visible or one interaction away.
- Use WP2 capability vocabulary where unsupported media states are route-level states.

- [ ] **Step 4: Run focused tests**

Run:

```bash
bunx vitest run src/routes/__tests__/option-media-multi.connection-state.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.mobile-parity.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.focus-recovery.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.touch-status-a11y.test.tsx src/components/Review/__tests__/MediaReviewPage.stage7.three-panel.test.tsx src/components/Review/__tests__/MediaReviewPage.stage5.batch-toolbar.test.tsx src/components/Review/__tests__/MediaReviewPage.stage5.export-trash-handoff.test.tsx
```

Expected:
- PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/option-media-multi.tsx apps/packages/ui/src/components/Review/MediaReviewPage.tsx apps/packages/ui/src/components/Review/ResizablePanels.tsx apps/packages/ui/src/components/Review/MediaReviewResultsList.tsx apps/packages/ui/src/components/Review/MediaReviewReadingPane.tsx apps/packages/ui/src/components/Review/MediaReviewBatchBar.tsx apps/packages/ui/src/routes/__tests__/option-media-multi.connection-state.test.tsx apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.mobile-parity.test.tsx apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.focus-recovery.test.tsx apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage6.touch-status-a11y.test.tsx apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage7.three-panel.test.tsx apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage5.batch-toolbar.test.tsx apps/packages/ui/src/components/Review/__tests__/MediaReviewPage.stage5.export-trash-handoff.test.tsx
git commit -m "fix: clarify media review selection states"
```

### Task 4: Strengthen Media Trash Recovery And Destructive Safeguards

**Files:**
- Modify: `apps/packages/ui/src/routes/option-media-trash.tsx`
- Modify: `apps/packages/ui/src/components/Review/MediaTrashPage.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.connection.test.tsx`
- Modify: `apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.stage2-3.test.tsx`
- Create or modify: `apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.safeguards.test.tsx`

- [ ] **Step 1: Write failing trash policy tests**

Test expectations:
- Retention policy is visible when reported by the server.
- Unknown retention policy is explicit.
- Empty trash, permanent delete, and delete selected all require confirmation.
- Restore and bulk restore are visually distinct from permanent deletion.
- Search-empty state offers clear filter recovery.
- Partial bulk failure surfaces counts and does not imply full success.

- [ ] **Step 2: Run tests to verify failure or current coverage**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Review/__tests__/MediaTrashPage.connection.test.tsx src/components/Review/__tests__/MediaTrashPage.stage2-3.test.tsx src/components/Review/__tests__/MediaTrashPage.safeguards.test.tsx
```

Expected:
- FAIL on missing safeguard assertions, or PASS only if current UI already satisfies the contract.

- [ ] **Step 3: Implement minimal trash improvements**

Implementation rules:
- Keep current endpoint paths and batching behavior.
- Keep `useConfirmDanger`.
- Keep restore actions primary in recovery contexts.
- Do not add destructive shortcuts.
- Keep pagination and search behavior.

- [ ] **Step 4: Run focused tests**

Run:

```bash
bunx vitest run src/components/Review/__tests__/MediaTrashPage.connection.test.tsx src/components/Review/__tests__/MediaTrashPage.stage2-3.test.tsx src/components/Review/__tests__/MediaTrashPage.safeguards.test.tsx
```

Expected:
- PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/option-media-trash.tsx apps/packages/ui/src/components/Review/MediaTrashPage.tsx apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.connection.test.tsx apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.stage2-3.test.tsx apps/packages/ui/src/components/Review/__tests__/MediaTrashPage.safeguards.test.tsx
git commit -m "fix: clarify media trash recovery"
```

### Task 5: Clarify Library Route Jobs Without Flattening Them Together

**Files:**
- Modify: `apps/packages/ui/src/routes/option-items.tsx`
- Modify: `apps/packages/ui/src/components/Option/Items/ItemsWorkspace.tsx`
- Modify: `apps/packages/ui/src/routes/option-collections.tsx`
- Modify: `apps/packages/ui/src/components/Option/Collections/index.tsx`
- Modify if needed: `apps/packages/ui/src/components/Option/Collections/ReadingList/ReadingItemsList.tsx`
- Modify: `apps/packages/ui/src/routes/option-notes.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Items/__tests__/ItemsWorkspace.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Collections/__tests__/CollectionsPlaygroundPage.test.tsx`
- Modify or create: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.test.tsx`
- Modify: `apps/tldw-frontend/pages/reading.tsx`

- [ ] **Step 1: Write failing library route tests**

Test expectations:
- `/items` is described as saved items with status, tags, favorites, and bulk actions.
- `/collections` is described as Reading List, Highlights, Templates, Digests, and Import/Export.
- `/reading` aliases to Collections and preserves reading-list intent.
- `/notes` is described as notes, not transcripts or media items.
- Empty states identify the object type and the next action.

- [ ] **Step 2: Run tests to verify failure or current coverage**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Items/__tests__/ItemsWorkspace.test.tsx src/components/Option/Collections/__tests__/CollectionsPlaygroundPage.test.tsx src/components/Notes/__tests__/NotesManagerPage.test.tsx src/routes/__tests__/media-library-aliases.test.tsx
```

Expected:
- FAIL on missing route-job or empty-state assertions, or PASS only if current UI already satisfies the contract.

- [ ] **Step 3: Implement minimal library copy and route fixes**

Implementation rules:
- Keep `/items`, `/collections`, and `/notes` separate.
- Keep Collections tabs.
- Keep saved item bulk actions and generated-output flow.
- Keep Notes create, edit, search, and keyword behavior.
- Do not create a generic "Library" hub.

- [ ] **Step 4: Run focused tests**

Run:

```bash
bunx vitest run src/components/Option/Items/__tests__/ItemsWorkspace.test.tsx src/components/Option/Collections/__tests__/CollectionsPlaygroundPage.test.tsx src/components/Notes/__tests__/NotesManagerPage.test.tsx src/routes/__tests__/media-library-aliases.test.tsx
```

Expected:
- PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/option-items.tsx apps/packages/ui/src/components/Option/Items/ItemsWorkspace.tsx apps/packages/ui/src/routes/option-collections.tsx apps/packages/ui/src/components/Option/Collections/index.tsx apps/packages/ui/src/components/Option/Collections/ReadingList/ReadingItemsList.tsx apps/packages/ui/src/routes/option-notes.tsx apps/packages/ui/src/components/Notes/NotesManagerPage.tsx apps/packages/ui/src/components/Option/Items/__tests__/ItemsWorkspace.test.tsx apps/packages/ui/src/components/Option/Collections/__tests__/CollectionsPlaygroundPage.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.test.tsx apps/tldw-frontend/pages/reading.tsx
git commit -m "fix: clarify media library route jobs"
```

### Task 6: Clarify Shared Workspaces And Chatbooks Direction

**Files:**
- Modify: `apps/packages/ui/src/routes/option-shared-with-me.tsx`
- Modify: `apps/packages/ui/src/components/Option/SharedWithMe/index.tsx`
- Modify: `apps/packages/ui/src/routes/option-chatbooks-playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ContentTypePicker.error-state.test.tsx`
- Create or modify: `apps/packages/ui/src/components/Option/SharedWithMe/__tests__/SharedWithMe.test.tsx`
- Create or modify: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx`
- Modify if needed: `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- Modify if needed: `apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx`

- [ ] **Step 1: Write failing shared and Chatbooks tests**

Test expectations:
- `/shared` route uses `RouteErrorBoundary`.
- Shared empty state names shared workspaces and distinguishes open from clone.
- Shared error state offers retry or recovery when existing hooks can support it.
- `/chatbooks` route uses `RouteErrorBoundary`.
- Chatbooks export and import tabs state direction explicitly.
- Job tracker states are visible for pending, active, completed, failed, cancelled, and removable jobs.
- Header shortcut launches `/chatbooks`, not the legacy playground path.

- [ ] **Step 2: Run tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-media-route-guards.test.tsx src/components/Option/SharedWithMe/__tests__/SharedWithMe.test.tsx src/components/Option/Chatbooks/__tests__/ContentTypePicker.error-state.test.tsx src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx src/components/Layouts/__tests__/HeaderShortcuts.test.tsx
```

Expected:
- FAIL on missing wrappers, state labels, or launcher expectations.

- [ ] **Step 3: Add wrappers and clarify states**

Implementation rules:
- Keep shared workspace open and clone actions.
- Keep Chatbooks export/import/job tracker/polling behavior.
- Do not hide job controls in a separate page.
- Keep `/chatbooks-playground` compatible while launcher labels use Chatbooks.

- [ ] **Step 4: Run focused tests**

Run:

```bash
bunx vitest run src/routes/__tests__/option-media-route-guards.test.tsx src/components/Option/SharedWithMe/__tests__/SharedWithMe.test.tsx src/components/Option/Chatbooks/__tests__/ContentTypePicker.error-state.test.tsx src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx src/components/Layouts/__tests__/HeaderShortcuts.test.tsx
```

Expected:
- PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/option-shared-with-me.tsx apps/packages/ui/src/components/Option/SharedWithMe/index.tsx apps/packages/ui/src/routes/option-chatbooks-playground.tsx apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ContentTypePicker.error-state.test.tsx apps/packages/ui/src/components/Option/SharedWithMe/__tests__/SharedWithMe.test.tsx apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx apps/packages/ui/src/components/Layouts/header-shortcut-items.ts apps/packages/ui/src/components/Layouts/__tests__/HeaderShortcuts.test.tsx
git commit -m "fix: clarify sharing and chatbooks routes"
```

### Task 7: Make Notifications Status And Recovery Clear

**Files:**
- Modify: `apps/tldw-frontend/pages/notifications.tsx`
- Modify or create: `apps/tldw-frontend/e2e/workflows/tier-4-admin/notifications.spec.ts`

- [ ] **Step 1: Write failing notification workflow tests**

Test expectations:
- Loading state is distinct from empty inbox.
- Empty inbox says there are no notifications yet and preserves preferences access.
- Error state offers refresh or diagnostics.
- Preferences unavailable state offers retry.
- Notification link routing preserves safe same-origin navigation.
- Snoozed list states and cancel-snooze behavior are discoverable.

- [ ] **Step 2: Run notification E2E or focused page test**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-4-admin/notifications.spec.ts --reporter=line
```

Expected:
- FAIL on missing status or recovery coverage, or PASS only if current UI already satisfies the contract.

- [ ] **Step 3: Implement minimal notification state fixes**

Implementation rules:
- Keep stream subscription and polling fallback.
- Keep mark read, snooze, dismiss, cancel snooze, and preferences.
- Do not introduce a new notification API.
- Keep link routing constrained to same-origin URLs and known route types.

- [ ] **Step 4: Run notification verification**

Run:

```bash
bunx playwright test e2e/workflows/tier-4-admin/notifications.spec.ts --reporter=line
```

Expected:
- PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/tldw-frontend/pages/notifications.tsx apps/tldw-frontend/e2e/workflows/tier-4-admin/notifications.spec.ts
git commit -m "fix: clarify notification recovery states"
```

### Task 8: Route Family Browser QA

**Files:**
- Modify if needed: `apps/tldw-frontend/e2e/workflows/media-review.spec.ts`
- Modify if needed: `apps/tldw-frontend/e2e/workflows/media-navigation-ux-verification.spec.ts`
- Modify if needed: `apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts`
- Modify if needed: `apps/tldw-frontend/e2e/workflows/tier-4-admin/notifications.spec.ts`
- Create if needed: `apps/tldw-frontend/e2e/workflows/media-library-route-family.spec.ts`
- Backlog: update implementation task with before and after observations.

- [ ] **Step 1: Extend browser checks where unit coverage cannot prove behavior**

Cover:
- `/media` first selection, empty library, quick ingest action, and 390px layout.
- `/media-multi` three-panel or tabbed review, selection status, batch handoff, and 390px layout.
- `/media-trash` retention policy, restore, delete confirmation, and empty/search-empty states.
- `/collections` Reading List tab and route purpose.
- `/notes` note purpose and first action.
- `/shared` empty/error/open/clone states.
- `/chatbooks` export/import/jobs direction.
- `/notifications` loading, empty, error, preferences, snooze, and link actions.
- `/review`, `/reading`, and `/chatbooks-playground` alias behavior.

- [ ] **Step 2: Run parent-required E2E checks**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/media-review.spec.ts e2e/workflows/media-navigation-ux-verification.spec.ts e2e/workflows/tier-2-features/chatbooks.spec.ts --reporter=line
```

Expected:
- PASS for media review, media navigation, and Chatbooks flows.

- [ ] **Step 3: Run notifications E2E if touched**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-4-admin/notifications.spec.ts --reporter=line
```

Expected:
- PASS if notifications changed.

- [ ] **Step 4: Run route-family E2E if added**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/media-library-route-family.spec.ts --reporter=line
```

Expected:
- PASS, or the test is not created because focused route/component tests and browser snapshots already cover the changed pages.

- [ ] **Step 5: Capture browser observations**

Expected:
- Before and after observations record heading, primary action, empty/setup state, overflow check, and alias result for every route in scope.
- Any unverified route names the blocker, server state, and exact command attempted.

- [ ] **Step 6: Commit verification updates**

If `media-library-route-family.spec.ts` was not created, omit that path from
the `git add` command.

```bash
git add apps/tldw-frontend/e2e/workflows/media-review.spec.ts apps/tldw-frontend/e2e/workflows/media-navigation-ux-verification.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/chatbooks.spec.ts apps/tldw-frontend/e2e/workflows/tier-4-admin/notifications.spec.ts apps/tldw-frontend/e2e/workflows/media-library-route-family.spec.ts
git commit -m "test: verify media library route family"
```

## Full Verification

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-media-route-guards.test.tsx src/routes/__tests__/option-media-multi.connection-state.test.tsx
```

Run additional focused tests when touched:

```bash
bunx vitest run src/routes/__tests__/media-library-route-jobs.test.ts src/routes/__tests__/media-library-aliases.test.tsx src/components/Review/__tests__/ViewMediaPage.first-selection-mobile.test.tsx src/components/Review/__tests__/ViewMediaPage.connection.test.tsx src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx src/components/Review/__tests__/ViewMediaPage.search-experience.integration.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.mobile-parity.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.focus-recovery.test.tsx src/components/Review/__tests__/MediaReviewPage.stage6.touch-status-a11y.test.tsx src/components/Review/__tests__/MediaReviewPage.stage7.three-panel.test.tsx src/components/Review/__tests__/MediaReviewPage.stage5.batch-toolbar.test.tsx src/components/Review/__tests__/MediaReviewPage.stage5.export-trash-handoff.test.tsx src/components/Review/__tests__/MediaTrashPage.connection.test.tsx src/components/Review/__tests__/MediaTrashPage.stage2-3.test.tsx src/components/Review/__tests__/MediaTrashPage.safeguards.test.tsx src/components/Option/Items/__tests__/ItemsWorkspace.test.tsx src/components/Option/Collections/__tests__/CollectionsPlaygroundPage.test.tsx src/components/Notes/__tests__/NotesManagerPage.test.tsx src/components/Option/SharedWithMe/__tests__/SharedWithMe.test.tsx src/components/Option/Chatbooks/__tests__/ContentTypePicker.error-state.test.tsx src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.jobs.test.tsx src/components/Layouts/__tests__/HeaderShortcuts.test.tsx
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/media-review.spec.ts e2e/workflows/media-navigation-ux-verification.spec.ts e2e/workflows/tier-2-features/chatbooks.spec.ts --reporter=line
```

Run if notifications changed:

```bash
bunx playwright test e2e/workflows/tier-4-admin/notifications.spec.ts --reporter=line
```

If route-family E2E is added, run:

```bash
bunx playwright test e2e/workflows/media-library-route-family.spec.ts --reporter=line
```

Run from the repo root:

```bash
git diff --check
```

Expected final state:
- `/media` is usable at 390px and has clear first-selection or empty-library behavior.
- `/media-multi` preserves dense review, filters, batch selection, and recovery paths.
- `/review`, `/reading`, and `/chatbooks-playground` have documented canonical alias behavior.
- `/media-trash` shows retention policy and destructive safeguards.
- `/items`, `/collections`, and `/notes` stay distinct and understandable.
- `/shared` and `/chatbooks` explain object type, direction, and next action.
- `/notifications` explains inbox state and recovery without exposing implementation details first.

## Acceptance Criteria

- Every route in `/media`, `/media-multi`, `/review`, `/media-trash`, `/items`, `/collections`, `/reading`, `/notes`, `/shared`, `/chatbooks`, `/chatbooks-playground`, and `/notifications` has a distinct route job documented in code and verified by tests.
- `/media` has usable list-detail behavior at 390px.
- `/media-multi` preserves existing filter, bulk, keyboard, compare, export, and trash-handoff workflows.
- `/review`, `/reading`, and `/chatbooks-playground` preserve existing direct URLs with canonical target behavior.
- `/media-trash` exposes retention policy and permanent-delete safeguards.
- `/items`, `/collections`, and `/notes` preserve separate product meanings.
- `/shared` and `/chatbooks` expose object type, direction, and primary next action.
- Browser QA covers every changed route or records the exact blocker.

## Rollback Plan

- Revert route-job helper and tests if WP1 central route metadata makes the helper redundant.
- Revert alias changes route by route if direct URLs or Next page behavior regress.
- Revert media inspector changes separately from media review changes.
- Revert trash safeguards separately from library copy changes.
- Preserve tests that document existing behavior unless the behavior itself is intentionally reverted.

## Handoff Notes

- Start with route-job and alias tests. This prevents confusing `/media`, `/media-multi`, `/review`, `/items`, and `/collections`.
- Treat high-throughput review as a core workflow, not an edge case. Bulk filters, selection, keyboard navigation, and trash handoff must remain fast.
- Do not turn `/collections` into a generic library hub or redirect `/items` into Collections.
- Do not use Chatbooks playground terminology as the primary label when the route is `/chatbooks`.
- Browser verification is required for narrow-width media layouts because unit tests cannot prove visual overlap.
