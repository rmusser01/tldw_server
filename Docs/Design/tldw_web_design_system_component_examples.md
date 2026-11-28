# Shared design-system component examples

Use these patterns when migrating WebUI or extension code in `apps/packages/ui`.
Feature components own loading, error, and data decisions. The shared primitives
render those decisions, and the state registry supplies canonical labels and
severity. [Design-system contract](tldw_web_design_system_contract.md) ·
[Migration tracker](../superpowers/plans/2026-05-14-design-system-remaining-work-tracker-issue-map.md)

## Choose a primitive

| What the user needs | Use |
| --- | --- |
| A collection or region is loading | `LoadingState` with a named status or live announcement. |
| A loaded collection has no items or matches | `EmptyState` with an explanation and an action when one is available. |
| A whole region needs to explain a product state and the next action | `StatePanel`. Use `SetupRequiredPanel` or `PermissionNotice` for those specific states. |
| A blocked workflow needs an actionable recovery path | `RecoveryCallout` with the cause and the shortest recovery action. |
| A local operation fails while the surrounding content remains usable | `Alert` next to the affected control or content. |
| A row or summary needs a brief, readable state label | `Badge` with text. Use the registry for a canonical state or to map severity to a variant. |

## Loading and empty collections

Show the empty state after a successful load confirms there are no items. For
filtered results, explain how to change or clear the filters. Keep usable data
visible during a background refresh and announce progress close to that data.
Show a visible loading label on first render. A status element that stays
mounted can also announce later updates. [WCAG status technique](https://www.w3.org/WAI/WCAG22/Techniques/aria/ARIA22) ·
[Recent Study Sessions](../../apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx)

```tsx
import type { ReactNode } from "react"
import { EmptyState, LoadingState } from "@/components/ui"

export function SourcesLoading({ loading, children }: {
  loading: boolean
  children: ReactNode
}) {
  return (
    <>
      <p role="status">
        {loading ? "Loading workspace sources" : ""}
      </p>
      <div aria-busy={loading}>
        {loading ? <LoadingState mode="skeleton" rows={3} /> : children}
      </div>
    </>
  )
}

export function NoSources({ onAddSource }: { onAddSource: () => void }) {
  return (
    <EmptyState
      variant="inline"
      title="No sources yet"
      description="Add a source to make it available in this workspace."
      primaryAction={{ label: "Add source", onClick: onAddSource }}
    />
  )
}
```

## Product state and recovery

Use `StatePanel` when the state applies to a whole section. For recovery,
`RecoveryCallout` uses the same registry and action props. Name the affected
operation, give the best next action, and keep technical detail in the
diagnostics area when available. Choose assertive announcements for blocking
errors and polite announcements for progress. Keep action buttons outside the
alert message. [WCAG error announcement technique](https://www.w3.org/WAI/WCAG22/Techniques/aria/ARIA19) ·
[Presentation Studio product states](../../apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioIndex.tsx) ·
[Server recovery](../../apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx)

```tsx
import { RecoveryCallout, StatePanel } from "@/components/ui"

export function WorkspaceRequired({ onChooseWorkspace }: {
  onChooseWorkspace: () => void
}) {
  return (
    <StatePanel
      state="blocked"
      title="Choose a workspace"
      message="Sources belong to a workspace. Select one to continue."
      primaryAction={{ label: "Choose workspace", onClick: onChooseWorkspace }}
    />
  )
}

export function SourcesUnavailable({ onRetry }: { onRetry: () => void }) {
  return (
    <RecoveryCallout
      state="unavailable"
      title="Workspace sources could not load"
      message={<span role="alert">Check your server connection to load workspace sources.</span>}
      primaryAction={{ label: "Try again", onClick: onRetry }}
    />
  )
}
```

## Compact status labels

Use `getDesignSystemState` when a lightweight component needs a canonical label
or severity. Map API results to a known state within the feature. For a
domain-specific status such as a reading state, preserve its own label and reuse
the registry's severity mapping when it applies. Translate feature copy through
the feature's existing translation namespace. [State registry](../../apps/packages/ui/src/design-system/states.ts) ·
[Source status badges](../../apps/packages/ui/src/components/Option/Sources/SourceStatusPanels.tsx) ·
[Collection status badges](../../apps/packages/ui/src/components/Option/Collections/common/StatusBadge.tsx)

```tsx
import { Badge, getBadgeVariantForDesignSystemSeverity } from "@/components/ui"
import { getDesignSystemState } from "@/design-system"

export function SourceStatus({ stateKey }: { stateKey: "ready" | "degraded" }) {
  const state = getDesignSystemState(stateKey)
  return (
    <Badge variant={getBadgeVariantForDesignSystemSeverity(state.severity)}>
      {state.label}
    </Badge>
  )
}
```

## Verify a migration

Cover loading, empty, ready, and applicable recovery transitions in the
feature's tests. Assert the visible label, discoverable action, and result of
invoking it. For a status or alert announcement, assert the relevant role.
Run the product-state verifier and record before and after baseline counts for
the migrated area. Review affected shared visuals in the WebUI and extension as
directed by the checklist. [CI gate path](tldw_web_design_system_ci_gate_path.md) ·
[Visual QA checklist](tldw_web_design_system_visual_qa_checklist.md)

From the repository root, run the verifier and inspect the existing tests for
examples of semantic assertions.

```bash
cd apps/packages/ui
bun run verify:design-system-state
bun run test src/design-system/__tests__/states.test.ts src/components/ui/state/__tests__/state-primitives.test.tsx
```
