# Design System Product-State Guard Design

Date: 2026-05-06
Topic: Guarded foundation for shared UI product-state migration
Status: Draft for spec review and user review

## Goal

Add a guarded foundation for the tldw Web Design System by preventing new
product-state divergence in `apps/packages/ui/src`.

The app is not live yet, so the priority is not user-visible polish on one
surface. The priority is making the shared UI package converge on the design
framework before feature migrations accelerate. The guard should make that
convergence explicit: new shared UI product-state code must use the design
system, or it must declare itself as a temporary migration exception with a
clear replacement path.

## Current Context

The repo already has the core design-system foundation on `origin/dev`:

- Contract: `Docs/Design/tldw_web_design_system_contract.md`
- Inventory: `Docs/Design/tldw_web_design_system_inventory.md`
- Shared UI primitives: `apps/packages/ui/src/components/ui`
- Product-state primitives: `apps/packages/ui/src/components/ui/state`
- State registry: `apps/packages/ui/src/design-system/states.ts`
- Shared token aliases: `apps/packages/ui/src/assets/tailwind-shared.css`

The inventory identifies the shared UI package as the first place to govern
product-state patterns. The contract's later migration queue still matters, but
this design focuses on the foundation layer before broad feature migration.

## Scope

Included:

- Static guard over `apps/packages/ui/src`.
- Product-state pattern detection for shared UI code.
- A checked-in baseline of existing exceptions.
- Rule tests and report tests.
- A package script such as `bun run verify:design-system-state`.
- Documentation updates that make this the enforcement path for shared UI
  product-state work.

Excluded:

- Repo-wide enforcement over `apps/tldw-frontend`.
- Browser-extension-specific app code outside shared UI.
- Visual redesign work.
- Full Button, PageShell, token, color, radius, or spacing migration.
- A custom ESLint plugin.
- Removing Ant Design.

Ant Design remains allowed as a mechanics substrate. The guard should only flag
AntD or local components when they express product-state language that belongs
in the design system.

## Product-State Definition

The guard applies to UI that communicates application or system state to the
user.

Flagged product-state categories:

- Availability: backend offline, server unavailable, disconnected.
- Readiness: ready, degraded, setup required, blocked.
- Permission and auth: sign in required, API key required, admin only.
- Workflow state: empty, loading, retrying, failed, recoverable error.
- Status chips: queued, running, complete, failed, inactive, enabled,
  disabled, connected, disconnected.
- Recovery affordances: retry, open setup, open diagnostics, switch server,
  reload.

Allowed mechanical or metadata categories:

- AntD table, form, modal, tooltip, popover, drawer, card, descriptions,
  select, switch, and input mechanics.
- Static labels that are not state.
- Domain metadata chips such as tag names, source types, provider names, model
  families, file types, or user-authored categories.
- Layout skeletons that preserve geometry and are not user-facing loading
  messages.

Ambiguous cases should start as baseline entries rather than hard failures, so
the team can classify them without blocking unrelated work.

## Migration Model

The guard supports three states for any finding:

1. `allowed_legacy_exception`
   A known current pattern. It remains temporarily allowed because it predates
   the guard or belongs to a scheduled migration slice. It must include a
   replacement and migration queue.

2. `active_migration_target`
   A known pattern selected for cleanup in the next migration slice. It remains
   visible in reports and should be removed from the baseline when migrated.

3. `blocked`
   A new or unapproved product-state pattern. The verification command fails
   until the code uses a design-system primitive or an explicit exception is
   added with rationale.

The long-term target is zero shared-package product-state exceptions. The
baseline exists to let the guard ship early without forcing a broad rewrite.
Every baseline entry is migration debt with an owner, reason, replacement, and
queue.

Command behavior:

- Findings with no matching baseline entry are `blocked`; the command exits
  nonzero.
- Findings matched by `allowed_legacy_exception` entries are reported as
  legacy debt; the command exits zero.
- Findings matched by `active_migration_target` entries are reported in a
  separate higher-priority group; the command exits zero, but the report should
  show they are selected for the next cleanup slice.

Expansion order after the shared package reaches zero exceptions:

1. `apps/packages/ui/src`
2. Shared browser-extension routes through shared imports
3. `apps/tldw-frontend` local wrappers and app-specific pages
4. Stricter token, color, radius, and layout guards

## Guard Rules

The initial guard should be a small repo-owned policy engine, not a dependency
heavy lint stack.

Rule groups:

- `antd-product-state-import`
  Flags direct use of AntD product-state components such as `Alert`, `Tag`,
  `Badge`, `Empty`, `Spin`, or `Result` in governed product-state contexts.
  AntD mechanics remain allowed.

- `local-recovery-banner`
  Flags local components whose names or filenames indicate duplicated recovery
  semantics, for example `*ErrorBanner`, `*ConnectionBanner`,
  `*UnavailableBanner`, or `*RecoveryBanner`.

- `local-empty-state`
  Flags local empty-state components such as `*Empty`, `*EmptyState`, or
  feature-specific empty wrappers that should use `EmptyState`.

- `local-loading-state`
  Flags user-facing local loading-state components or hardcoded loading
  messages that should use `LoadingState` or a canonical state primitive.

- `local-status-badge`
  Flags local generic status chips such as `*StatusBadge`, `*StatusTag`,
  `*StatusChip`, or direct product-state tag variants that should map through
  `Badge` or the state registry.

- `canonical-state-label`
  Flags hardcoded canonical state labels outside `design-system/states.ts`,
  known adapters, tests, localization files, or approved baseline entries.
  Examples: `Unavailable`, `Setup required`, `Sign in required`,
  `Permission denied`, `Degraded`, `Retrying`, `Blocked`, `Ready`.

The first implementation should prefer explicit rules and understandable
output over exhaustive detection. False positives should become allowed
mechanics or baseline entries with a migration note. False negatives can be
covered as the migration proceeds.

Product-state context detection:

The guard should not flag every AntD `Alert`, `Tag`, `Badge`, `Empty`, `Spin`,
or `Result` import by import name alone. It should flag a finding when the
import is used in JSX and at least one product-state signal is present:

- The file, component, or function name matches a product-state pattern such as
  `*Status*`, `*Error*`, `*Empty*`, `*Loading*`, `*Recovery*`,
  `*Connection*`, `*Unavailable*`, `*Readiness*`, or `*Permission*`.
- Nearby literal text, props, or translation defaults include canonical state
  labels or recovery actions such as unavailable, degraded, retrying, blocked,
  setup, sign in, permission denied, retry, diagnostics, or reconnect.
- The JSX usage sets status/severity props that communicate application state,
  such as `type="error"`, `status="warning"`, `color="success"`, or
  equivalent variant mappings.
- The component renders primary recovery actions such as retry, open setup,
  open settings, copy diagnostics, switch server, or reload.

The same import should not be flagged when usage is clearly mechanical or
metadata-only, such as table rendering, form validation plumbing, popover/modal
mechanics, static domain tags, provider/model chips, or file-type/source-type
labels.

## Allowed Roots And Adapters

The guard scans `apps/packages/ui/src`, but canonical design-system
implementation files are not legacy violations just because they implement the
approved primitives.

Default canonical roots:

- `src/components/ui/primitives/Alert.tsx`
- `src/components/ui/primitives/Badge.tsx`
- `src/components/ui/feedback/EmptyState.tsx`
- `src/components/ui/feedback/LoadingState.tsx`
- `src/components/ui/layout/ModalFooter.tsx`
- `src/components/ui/state/ActionGroup.tsx`
- `src/components/ui/state/DiagnosticRow.tsx`
- `src/components/ui/state/PermissionNotice.tsx`
- `src/components/ui/state/RecoveryCallout.tsx`
- `src/components/ui/state/SetupRequiredPanel.tsx`
- `src/components/ui/state/StatePanel.tsx`
- `src/components/ui/**/index.ts`
- `src/design-system/states.ts`
- `src/design-system/index.ts`
- `src/assets/tailwind.css`
- `src/assets/tailwind-shared.css`
- `src/**/*.test.ts`
- `src/**/*.test.tsx`
- `src/**/__tests__/**`
- `src/assets/locale/**`
- `src/public/_locales/**`

These files still need ordinary unit tests, but they should not be counted as
legacy product-state drift by this guard. For example, `LoadingState` can
implement loading mechanics, and `StatePanel` can render canonical labels from
the state registry.

New files under `src/components/ui/**` or `src/design-system/**` are not
automatically exempt. They should either be added to the explicit canonical-root
list as part of a design-system primitive change, or be scanned by the guard.
This prevents a duplicate local product-state component from bypassing the
guard just because it was placed under the canonical namespace.

Canonical roots should also have a lightweight conformance rule: if a canonical
file imports or renders an AntD product-state primitive, it should do so only to
implement the approved tldw-owned primitive semantics, not to introduce a second
parallel state language.

Domain adapters may also be allowed when they are thin mappings into the design
system. An adapter exception must be explicit, path-scoped, and include the
canonical primitive or registry it maps to. A domain adapter should be allowed;
a duplicate local design language should be baselined for migration.

## Baseline File

The baseline is checked in beside the guard script.

Example shape:

```json
[
  {
    "id": "local-recovery-banner:src/components/Sidepanel/Chat/ConnectionBanner.tsx:ConnectionBanner",
    "path": "src/components/Sidepanel/Chat/ConnectionBanner.tsx",
    "rule": "local-recovery-banner",
    "subject": "ConnectionBanner",
    "state": "allowed_legacy_exception",
    "owner": "design-system",
    "reason": "Existing Chat disconnected-state surface before the shared UI guard.",
    "replacement": "RecoveryCallout or StatePanel",
    "migrationQueue": "chat-playground-product-state"
  }
]
```

Baseline rules:

- Matching must use a stable per-finding identity, not only `path` plus
  `rule`.
- The initial identity should include `rule`, `path`, and `subject`, where
  `subject` is the imported symbol, component/function name, JSX identifier, or
  canonical label that triggered the finding.
- A text fingerprint may be added for label findings when `subject` is not
  enough to distinguish multiple matches.
- Entries must include `id`, `path`, `rule`, `subject`, `state`, `owner`,
  `reason`, `replacement`, and `migrationQueue`.
- Baseline entry `state` values must be `allowed_legacy_exception` or
  `active_migration_target`. `blocked` is emitted for unbaselined findings and
  must not be stored in the baseline.
- Removed baseline entries must not silently reappear.
- New baseline entries are allowed only when intentionally documenting a
  migration exception.
- The report should print remaining baseline totals grouped by rule and queue.
- Baseline entries that do not match any live finding should be reported as
  stale. Stale entries exit zero in Stage A, but they should be removed in the
  same cleanup PR that removes the underlying violation.

This prevents a new violation in an already-baselined file from being hidden by
the older baseline entry.

## Architecture

Proposed files:

- `apps/packages/ui/scripts/verify-design-system-product-state.mjs`
  CLI entrypoint. Scans governed files, loads rules and baseline, prints a
  report, and exits nonzero on new unbaselined findings.

- `apps/packages/ui/scripts/design-system-product-state-rules.mjs`
  Rule definitions and helpers. Owns forbidden imports, local-component naming
  checks, canonical-label checks, allowed mechanics, and classification.

- `apps/packages/ui/scripts/design-system-product-state-baseline.json`
  Current known exceptions with migration metadata.

- `apps/packages/ui/scripts/__tests__/design-system-product-state-rules.test.mjs`
  Fixture-style tests for rule matching, allowed mechanics, baseline behavior,
  and report formatting.

- `apps/packages/ui/package.json`
  Adds `verify:design-system-state`.

Data flow:

1. Collect source files under `apps/packages/ui/src`.
2. Apply canonical-root handling and explicit adapter exceptions for this guard.
3. Parse each remaining file for imports, component/function names, JSX
   identifiers, and canonical state labels.
4. Classify AntD product-state imports using the context signals above, not by
   import name alone.
5. Emit raw findings with rule id, path, subject, stable finding id, reason,
   and suggested replacement.
6. Subtract matching baseline entries by finding id.
7. Mark baseline entries with no matching live finding as stale.
8. Fail on new unbaselined findings.
9. Print new findings, active migration targets, allowed legacy exceptions,
   stale baseline entries, and remaining baseline totals.

The implementation should use structured parsing where practical for imports
and JSX identifiers. Limited text checks are acceptable for hardcoded canonical
labels. A full ESLint plugin is a later option only if this guard becomes too
large for a standalone script.

## Report Format

The report must be actionable enough to fix without reading the guard code.

Example:

```text
Design-system product-state guard failed

apps/packages/ui/src/components/Sidepanel/Chat/ConnectionBanner.tsx
  rule: local-recovery-banner
  reason: Use RecoveryCallout or StatePanel for unavailable/degraded/retrying states.
  replacement: RecoveryCallout or StatePanel

apps/packages/ui/src/components/Option/Foo/FooStatusBadge.tsx
  rule: local-status-badge
  reason: Product status chips should map through Badge or the state registry.
  replacement: Badge plus a domain-to-state adapter
```

Successful runs should still summarize baseline debt:

```text
Design-system product-state guard passed
Baseline exceptions: 18
  local-status-badge: 7
  local-recovery-banner: 5
  local-empty-state: 4
  local-loading-state: 2
Stale baseline entries: 0
```

## Testing Strategy

Focused tests:

- Direct AntD `Alert`, `Tag`, `Badge`, `Empty`, `Spin`, and `Result` product
  state snippets are flagged.
- Local `FooStatusBadge`, `FooEmpty`, `FooLoadingState`, and
  `FooConnectionBanner` snippets are flagged.
- Hardcoded canonical state labels are flagged outside allowed registry,
  adapter, test, and localization contexts.
- AntD `Table`, `Modal`, `Tooltip`, `Popover`, `Drawer`, `Card`,
  `Descriptions`, `Select`, `Switch`, and inputs are not flagged as mechanics.
- AntD `Alert`, `Tag`, `Badge`, `Empty`, `Spin`, and `Result` imports are not
  flagged unless JSX usage also has product-state context signals.
- Metadata tags for provider, model, source type, file type, and user-authored
  tags are not flagged as product-state chips.
- A known violation passes only when a baseline entry matches its stable
  finding id.
- A new violation in a file with an existing baseline entry still fails when
  its stable finding id does not match the baseline.
- `active_migration_target` entries exit zero but are grouped separately from
  passive legacy exceptions.
- A baseline entry with no matching live finding is reported as stale.
- Explicit canonical design-system files are not reported as legacy drift, but
  new unlisted files under `src/components/ui/**` or `src/design-system/**` are
  still scanned or must be added intentionally to the canonical-root list.
- Report output includes path, rule, reason, and replacement.

Real scan smoke:

```bash
cd apps/packages/ui
bun run verify:design-system-state
```

The command should be suitable for CI once the initial baseline is checked in.

## Rollout

Stage A: report mode with baseline

- Scan all of `apps/packages/ui/src`.
- Check in the initial baseline.
- Fail only on new unbaselined findings.
- Print baseline totals so cleanup remains visible.

Stage B: slice-by-slice burn-down

- Each migration PR removes a few baseline entries.
- The guard fails if a removed entry returns.
- Baseline entries must retain migration notes while they exist.

Stage C: zero-baseline shared UI

- Remove baseline mode for `apps/packages/ui/src` once no product-state
  exceptions remain.
- Expand the guard to app-specific WebUI and extension code.
- Consider stricter token, color, radius, and layout guards after product-state
  semantics are stable.

## Success Criteria

This design succeeds when:

- One command verifies shared UI product-state compliance.
- The command fails on a new local product-state duplicate in
  `apps/packages/ui/src`.
- The command does not fail on allowed AntD mechanics or domain metadata.
- Every existing exception has a documented replacement and migration queue.
- The design-system contract or inventory points to the guard as the shared UI
  enforcement mechanism.
- The baseline count can only stay flat or go down unless a new exception is
  deliberately added with rationale.

## Risks And Mitigations

- Risk: noisy false positives slow down unrelated feature work.
  Mitigation: start with a baseline and explicit allowed mechanics.

- Risk: regex-only checks miss real patterns or flag harmless text.
  Mitigation: parse imports and identifiers structurally where practical; use
  text checks only for canonical labels.

- Risk: the baseline becomes a junk drawer.
  Mitigation: require replacement, reason, and migration queue for every entry,
  and print grouped counts on every run.

- Risk: guard scope grows into a broad visual-style policy too early.
  Mitigation: keep this design limited to product-state semantics and defer
  token/color/layout enforcement.

## Open Questions For Implementation Planning

- Which parser should the script use: existing TypeScript compiler APIs,
  Babel parser, or a lightweight hybrid?
- Should the initial baseline include only current violations discovered by the
  first scan, or should it also include ambiguous warnings for near-duplicates?
- Should the CI hook land immediately, or should the first PR add the script and
  tests only, then wire CI after one clean local run?
