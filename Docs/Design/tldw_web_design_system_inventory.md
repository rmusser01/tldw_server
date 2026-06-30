# tldw Web Design System Inventory

Date: 2026-05-05

## Purpose

This inventory is the Stage 2 follow-up to the tldw Web Design System
Contract. It maps the current shared UI and near-duplicate component surface
after the proof-surface PR merged. The goal is to make the next migrations
small and reviewable, not to start a broad rewrite.

## Scope

Inventory source of truth:

- Contract: `Docs/Design/tldw_web_design_system_contract.md`
- Shared UI package: `apps/packages/ui/src`
- WebUI wrapper components: `apps/tldw-frontend/components`
- Browser-extension routes and shared routes under `apps/packages/ui/src/routes`
- Proof surface implemented by PR #1272

This inventory covers the contract-required categories:

- `Button`
- `StatusBadge`
- `Badge`
- `Alert`
- `FeatureEmptyState`
- `EmptyState`
- Loading states
- Recovery banners
- Page shells
- Modal footers
- Admin and health panels

## Current Shared Exports

`apps/packages/ui/src/components/ui` is now the canonical design-system
primitive surface for new shared primitives. Its package-level exports are:

- Root barrel: `apps/packages/ui/src/components/ui/index.ts`
- State subpath export: `@tldw/ui/components/ui/state`
- Asset export: `@tldw/ui/assets/tailwind.css`

The current `components/ui` tiers are:

- `primitives`: `Alert`, `Badge`
- `feedback`: `EmptyState`, `LoadingState`
- `layout`: `ModalFooter`
- `state`: `ActionGroup`, `DiagnosticRow`, `StatePanel`,
  `RecoveryCallout`, `PermissionNotice`, `SetupRequiredPanel`

The state registry is owned by `apps/packages/ui/src/design-system/states.ts`.
State token aliases are owned by
`apps/packages/ui/src/assets/tailwind-shared.css`.

## Inventory

### `Button`

Canonical owner today:

- `apps/packages/ui/src/components/Common/Button.tsx`

Near-duplicates and consumers:

- `apps/tldw-frontend/components/ui/Button.tsx`
- Direct AntD `Button` usage in health, admin, Playground, Prompt Studio,
  Writing Playground, media and notes.
- Shared state primitives already consume `Common/Button`.

Decision:

- Keep `Common/Button` as the canonical tldw-owned button for now.
- Do not migrate it during documentation or proof-surface follow-ups.

Migration target:

- Add a `components/ui` action export or wrapper in a dedicated migration.
- Start with Chat/Playground buttons that express product action hierarchy.
- Leave AntD table and form mechanics alone.

### `StatusBadge`

Canonical owner today:

- `apps/packages/ui/src/components/Common/StatusBadge.tsx` for generic
  demo/warning/error status.

Near-duplicates and consumers:

- `apps/packages/ui/src/components/Option/Collections/common/StatusBadge.tsx`
- `apps/packages/ui/src/components/Option/Evaluations/components/StatusBadge.tsx`
- `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioStatusBadge.tsx`
- `apps/packages/ui/src/components/Option/Prompt/SyncStatusBadge.tsx`
- AntD `Tag` usage.

Decision:

- Treat generic status chips as design-system work.
- Domain badges may remain if they map domain status to canonical badge
  variants and include readable labels.

Migration target:

- Prefer `components/ui/primitives/Badge` for generic status display.
- Convert domain wrappers to thin semantic adapters around `Badge` when their
  surface enters the migration queue.

### `Badge`

Canonical owner today:

- `apps/packages/ui/src/components/ui/primitives/Badge.tsx`

Near-duplicates and consumers:

- `apps/tldw-frontend/components/ui/Badge.tsx`
- AntD `Tag` and `Badge` usage in Chat, Playground, health, admin, Prompt
  Studio, Writing Playground, VN assets and evaluations.

Decision:

- `components/ui/Badge` is the canonical primitive.
- The WebUI-local `Badge` is a migration candidate because it duplicates
  variants and styling.

Migration target:

- Export or use the shared `Badge` through the WebUI alias path before deleting
  the WebUI-local duplicate.
- Replace AntD `Tag` only when the tag is product status, not when it is table
  or form mechanics.

### `Alert`

Canonical owner today:

- `apps/packages/ui/src/components/ui/primitives/Alert.tsx`

Near-duplicates and consumers:

- AntD `Alert` remains in health, admin, Prompt Studio, integrations, Quiz,
  Writing Playground and document-generator flows.
- `DismissibleBetaAlert` already consumes shared `Alert`.

Decision:

- Use shared `Alert` for product messages and inline feedback.
- Use state primitives for recovery/setup/permission states.
- AntD `Alert` can remain for un-migrated feature-local mechanics.

Migration target:

- First migrate Chat/Playground error and warning banners that users read as
  product state.
- Do not attempt a repo-wide AntD `Alert` replacement.

### `FeatureEmptyState`

Canonical owner today:

- `apps/packages/ui/src/components/Common/FeatureEmptyState.tsx`

Near-duplicates and consumers:

- Heavy use across Notes, Prompt, Dictionaries, Media, Review, WorldBooks,
  Skills, Knowledge, Workspace Playground and extension routes.
- It still uses AntD `Button` internally.

Decision:

- Keep as compatibility wrapper for existing surfaces.
- It overlaps with `components/ui/feedback/EmptyState`.

Migration target:

- Make `FeatureEmptyState` a thin wrapper around `EmptyState` before
  high-volume migrations.
- Migrate new surfaces directly to `EmptyState`.

### `EmptyState`

Canonical owner today:

- `apps/packages/ui/src/components/ui/feedback/EmptyState.tsx`

Near-duplicates and consumers:

- `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/empty.tsx`
- WebUI local empty patterns.
- AntD `Empty`.
- Many `FeatureEmptyState` consumers.

Decision:

- `EmptyState` is the canonical primitive for empty-but-actionable state.
- The proof surface does not yet exercise it deeply.

Migration target:

- Start with Playground and Sidepanel Chat empty states because Chat/Playground
  is the first migration queue item and already has focused tests.

### Loading States

Canonical owner today:

- `apps/packages/ui/src/components/ui/feedback/LoadingState.tsx`

Near-duplicates and consumers:

- `apps/packages/ui/src/components/Common/UnifiedLoadingState.tsx`
- `apps/tldw-frontend/components/ui/Skeleton.tsx`
- AntD `Spin` and `Skeleton`.
- Chat composer local skeletons.
- Route/readiness loading state.

Decision:

- `LoadingState` is the canonical primitive for shared product loading.
- Domain skeletons may remain when they preserve layout-specific affordances.

Migration target:

- Normalize user-facing Chat/Playground loading/retrying states first.
- Leave dense table skeletons and layout-preserving composer skeletons until
  their owning surface migrates.

### Recovery Banners

Canonical owner today:

- `components/ui/state` for product state language.
- `apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx` for
  backend recovery orchestration.

Near-duplicates and consumers:

- `ConnectionProblemBanner`
- `ConnectFeatureBanner`
- `WorkspaceConnectionGate`
- `apps/packages/ui/src/components/Sidepanel/Chat/ConnectionBanner.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundChatErrorBanner.tsx`
- Route and global error boundaries.

Decision:

- `components/ui/state` owns product state language.
- `BackendUnavailableRecovery` should stay in `Common` because it orchestrates
  behavior rather than only styling.

Migration target:

- Convert Chat/Playground connection and error banners to `RecoveryCallout` or
  `StatePanel` when their state is unavailable, degraded, retrying, blocked or
  auth-required.

### Page Shells

Canonical owner today:

- `apps/packages/ui/src/components/Common/PageShell.tsx`

Near-duplicates and consumers:

- `OptionLayout`
- `options-route-shell.tsx`
- `sidepanel-route-shell.tsx`
- WebUI page wrappers.
- Feature-local layout containers.

Decision:

- `Common/PageShell` remains the current page-width wrapper.
- A fuller `components/ui/layout/PageShell` should wait until route shell
  requirements are clearer.

Migration target:

- Do not move page shells in the Chat/Playground slice except to remove local
  one-off page padding when a route already uses `PageShell`.

### Modal Footers

Canonical owner today:

- `apps/packages/ui/src/components/ui/layout/ModalFooter.tsx`

Near-duplicates and consumers:

- Many AntD `Modal footer={...}` arrays in Playground, Common Playground,
  Prompt Studio, Writing Playground, Notes and media.
- WebUI local `ConfirmDialog` footer uses WebUI-local `Button`.

Decision:

- `ModalFooter` is the canonical footer layout for tldw-owned modal actions.
- AntD modal mechanics remain allowed.

Migration target:

- Start with Playground modal footers:
  `PlaygroundStartupTemplateModal`, `PlaygroundContextWindowModal`,
  `PlaygroundImageGenModal`, `PlaygroundRawRequestModal`,
  `PlaygroundMcpSettingsModal`, and `Common/Playground/DocumentGeneratorDrawer`.

### Admin And Health Panels

Canonical owner today:

- `components/ui/state` primitives for state panels.
- AntD `Card`, `Table`, `Descriptions`, `Space` remain mechanics.

Near-duplicates and consumers:

- `apps/packages/ui/src/components/Option/Settings/health-status.tsx`
- `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx`
- `apps/packages/ui/src/routes/option-settings-health.tsx`
- `apps/packages/ui/src/routes/option-admin-server.tsx`
- `apps/tldw-frontend/pages/settings/health.tsx`
- `apps/tldw-frontend/pages/admin/server.tsx`
- Other admin routes still use AntD and local patterns.

Decision:

- Proof surface is complete enough for state language.
- Admin tables/forms are not design-system primitives.

Migration target:

- Later Settings/account-security migration can continue admin/health state
  cleanup.
- Do not expand admin migration before Chat/Playground, ingestion/library,
  jobs, MCP/ACP and evaluations queues.

## Proof-Surface Consumers

The v1 proof surface now has a concrete shared-state footprint:

- `apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx`
- `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
- `apps/tldw-frontend/components/ErrorBoundary.tsx`
- `apps/tldw-frontend/components/networking/ConfigurationErrorScreen.tsx`
- `apps/tldw-frontend/components/networking/ConfigurationGuard.tsx`
- `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- `apps/packages/ui/src/routes/option-setup.tsx`
- `apps/packages/ui/src/routes/option-settings-health.tsx`
- `apps/packages/ui/src/components/Option/Settings/health-status.tsx`
- `apps/packages/ui/src/routes/option-admin-server.tsx`
- `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx`
- `apps/tldw-frontend/pages/setup.tsx`
- `apps/tldw-frontend/pages/settings/health.tsx`
- `apps/tldw-frontend/pages/admin/server.tsx`

The proof surface intentionally leaves other admin routes and feature pages for
later queues.

## First Migration Queue: Chat And Playground

The contract prioritizes Chat and Playground first. The inventory suggests this
bounded slice:

1. Normalize empty and disconnected states:
   `Option/Playground/PlaygroundEmpty.tsx`,
   `Sidepanel/Chat/empty.tsx`, and `Sidepanel/Chat/ConnectionBanner.tsx`.
2. Normalize Playground error and recovery banners:
   `Option/Playground/PlaygroundChatErrorBanner.tsx`,
   `Option/Playground/PlaygroundComposerNotices.tsx`, and
   `Common/Playground/DocumentGeneratorDrawer.tsx` error states.
3. Normalize product status badges and chips, not every visual pill:
   `Sidepanel/Chat/StatusDot.tsx`,
   `Sidepanel/Chat/SaveStatusIcon.tsx`,
   `Option/Playground/ResearchRunStatusStack.tsx`,
   `Option/Playground/VoiceChatIndicator.tsx`, and
   `Common/Playground/PlaygroundUserMessage.tsx`.
4. Normalize modal footers in Playground before touching Prompt Studio or
   Writing Playground.
5. Add a small static guard that prevents new local Chat/Playground recovery,
   alert, empty-state and status-badge patterns after the migration.

Recommended first PR boundary:

- Use EmptyState, LoadingState, RecoveryCallout and Badge where the user-facing state is
  empty, loading, unavailable, degraded, retrying, blocked, ready or error.
- Keep composer-specific visual controls such as `IconButton`, `SendButton`,
  `TokenMeter`, `Pill`, `SourceChip` and `FacetRow` local until a dedicated
  composer-component migration exists.
- Keep AntD mechanics for `Popover`, `Tooltip`, `Image`, `Modal`, `Collapse`,
  forms and tables.
- Do not introduce the full `Button` migration unless the Chat/Playground slice
  needs an action-wrapper export to avoid duplicating button semantics.

Suggested focused verification for that PR:

```bash
cd apps/tldw-frontend
bunx vitest run \
  ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx \
  ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx \
  ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx \
  ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConnectionBanner.test.tsx \
  ../packages/ui/src/components/Sidepanel/Chat/__tests__/empty.test.tsx
```

Add or update tests if the migration changes accessible labels, actions,
diagnostics, or state text.

### Shared UI Product-State Guard

Run `bun run verify:design-system-state` from `apps/packages/ui` before adding
or changing shared UI product-state surfaces. New recovery, loading, empty,
status, readiness, setup, auth, and permission UI in `apps/packages/ui/src`
should use the design-system primitives under `src/components/ui` and
`src/design-system/states.ts`.

Existing shared UI product-state debt is tracked in
`apps/packages/ui/scripts/design-system-product-state-baseline.json`. Do not add
new baseline entries unless a migration exception has an owner, reason,
replacement, and queue; remove stale baseline entries when a migration removes
the matching finding. When the guard reports a duplicate-suffixed finding ID for
another occurrence of the same rule/path/subject, preserve that exact ID in the
baseline so one legacy exception cannot cover a new occurrence.

## Non-Goals

- Do not remove Ant Design.
- Do not migrate every `Button`, `Tag`, `Alert`, `Empty`, `Spin` or `Skeleton`
  reference in one PR.
- Do not move `Common/Button` or `Common/PageShell` wholesale without a focused
  compatibility plan.
- Do not delete the WebUI-local `components/ui` folder until its consumers have
  clear shared-package replacements.
- Do not expand admin migration beyond `/admin/server` as part of the
  Chat/Playground queue.

## Evidence Notes

Inventory evidence came from targeted file scans and direct reads of these
anchor files:

- `apps/packages/ui/src/components/ui/index.ts`
- `apps/packages/ui/src/components/ui/primitives/Alert.tsx`
- `apps/packages/ui/src/components/ui/primitives/Badge.tsx`
- `apps/packages/ui/src/components/ui/feedback/EmptyState.tsx`
- `apps/packages/ui/src/components/ui/feedback/LoadingState.tsx`
- `apps/packages/ui/src/components/ui/layout/ModalFooter.tsx`
- `apps/packages/ui/src/components/ui/state/*`
- `apps/packages/ui/src/components/Common/Button.tsx`
- `apps/packages/ui/src/components/Common/StatusBadge.tsx`
- `apps/packages/ui/src/components/Common/FeatureEmptyState.tsx`
- `apps/packages/ui/src/components/Common/PageShell.tsx`
- `apps/tldw-frontend/components/ui/Button.tsx`
- `apps/tldw-frontend/components/ui/Badge.tsx`
- `apps/tldw-frontend/components/ui/Skeleton.tsx`
- `apps/tldw-frontend/components/ui/ConfirmDialog.tsx`
- `apps/packages/ui/src/components/Option/Settings/health-status.tsx`
- `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx`
- `apps/packages/ui/src/components/Option/Playground` 
- `apps/packages/ui/src/components/Common/Playground` 
- `apps/packages/ui/src/components/Sidepanel/Chat`
