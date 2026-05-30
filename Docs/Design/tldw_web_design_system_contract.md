# tldw Web Design System Contract

Date: 2026-05-04

## Purpose

This contract defines the first governed design system for the tldw_server WebUI
and browser extension. The goal is not to redesign every screen at once. The
goal is to stop product surfaces from inventing local visual language when the
shared UI layer already has enough foundation to become authoritative.

Version 1 is governance-first. It defines ownership, token semantics, component
tiers, state language, Ant Design policy, testing expectations, and rollout
order. Implementation should start with setup, recovery, and admin health
because those screens exercise the most important product states: unavailable,
setup required, auth required, permission denied, degraded, retrying, blocked,
empty, loading, error, and ready.

## Approved Direction

- The contract governs both WebUI and browser extension from day one.
- `apps/packages/ui/src` is the shared implementation source of truth.
- Ant Design remains allowed as an implementation substrate.
- tldw-owned wrappers and documented patterns own product semantics.
- Setup, recovery, and admin health are the first proof surface.
- Chat, ingestion, library, jobs, MCP, evaluations, and settings migrate later.

## Current Repo Anchors

The current frontend already has design-system building blocks:

- Shared UI package: `apps/packages/ui/src`
- Shared asset entrypoint: `apps/packages/ui/src/assets/tailwind.css`
- Shared theme variables: `apps/packages/ui/src/assets/tailwind-shared.css`,
  imported by `tailwind.css`
- Tailwind token mapping: `apps/tldw-frontend/tailwind.config.js`
- Shared primitive exports: `apps/packages/ui/src/components/ui`
- Broader reusable components: `apps/packages/ui/src/components/Common`
- Feature surfaces: `apps/packages/ui/src/components/Option`,
  `apps/packages/ui/src/components/Sidepanel`, `apps/packages/ui/src/components/Chat`,
  and related feature directories
- Frontend architecture guide: `apps/DEVELOPMENT.md`

Those anchors mean this should be a formalization effort first, not a greenfield
component-library project.

## Authority And Scope

The design system is the product UI contract for the WebUI and extension. It
defines what shared UI decisions mean across both surfaces:

- Semantic tokens
- Component ownership
- Layout patterns
- Product state language
- Accessibility requirements
- Test obligations
- Migration rules

Ant Design is not the product design system. Engineers may use AntD for hard UI
mechanics, but product state, action hierarchy, diagnostics, setup guidance, and
readiness language must come from tldw-owned wrappers or documented patterns.

## System Layers

### 1. Tokens

Tokens define shared semantic values for both WebUI and extension. Existing
tokens such as `--color-bg`, `--color-surface`, `--color-primary`,
`--color-success`, `--color-warn`, `--color-danger`, `--color-text`,
`--radius-*`, `--shadow-*`, and font tokens remain the base.

Version 1 should extend the system toward product states. Recommended semantic
state tokens include:

- `--state-ready`
- `--state-unavailable`
- `--state-setup-required`
- `--state-auth-required`
- `--state-permission-denied`
- `--state-degraded`
- `--state-retrying`
- `--state-blocked`
- `--state-empty`
- `--state-loading`
- `--state-error`

Tailwind mappings should expose state tokens through readable names. Tests
should assert semantic usage and accessible text, not raw color values.

Version 1 state tokens should alias existing semantic color variables first.
Do not introduce a new palette during the proof-surface slice unless the current
tokens cannot meet accessibility requirements.

| State token | Initial alias |
| --- | --- |
| `--state-ready` | `--color-success` |
| `--state-unavailable` | `--color-danger` |
| `--state-setup-required` | `--color-warn` |
| `--state-auth-required` | `--color-warn` |
| `--state-permission-denied` | `--color-danger` |
| `--state-degraded` | `--color-warn` |
| `--state-retrying` | `--color-primary` |
| `--state-blocked` | `--color-danger` |
| `--state-empty` | `--color-muted` |
| `--state-loading` | `--color-muted` |
| `--state-error` | `--color-danger` |

### 2. Primitives

Primitives are small reusable components with stable semantics. The canonical
home is `apps/packages/ui/src/components/ui`.

Existing canonical `components/ui` primitives include:

- `Badge`
- `Alert`
- `EmptyState`
- `LoadingState`
- `ModalFooter`

Existing near-primitives and migration candidates include:

- `Button`, currently under `components/Common`
- `StatusBadge`, currently under `components/Common`
- `FeatureEmptyState`, currently under `components/Common`
- Setup and backend recovery components under `components/Common`

Version 1 proof-surface primitives should define or normalize:

- `RecoveryCallout`
- `DiagnosticRow`
- `ActionGroup`
- `StatePanel`
- `PermissionNotice`
- `SetupRequiredPanel`

Later inventory and migration candidates include:

- `StatusBadge`
- `PageHeader`
- `FieldRow`
- Shared table shell
- Shared form section
- `Button` wrapper or `components/ui` export

Primitives should avoid feature-specific behavior. They should express stable
structure, accessible labels, variants, sizing, density, and interaction states.

### 3. Patterns

Patterns are composed UI rules for workflows that recur across product surfaces.
For version 1, proof-surface patterns are:

- Setup gate
- Backend recovery
- Health diagnostics
- Admin readiness panel
- Permission denial
- Retry flow

Inventory and later migration patterns include:

- Loading-to-ready transition
- Empty-but-actionable state
- Destructive confirmation

Patterns should document required content regions, allowed actions, state
transitions, diagnostics behavior, and testing hooks.

### 4. Product Surfaces

Product surfaces consume primitives and patterns rather than inventing local
styling. Feature directories own domain behavior and page composition, but they
must not define new generic button, alert, badge, empty-state, modal-footer,
recovery, or status styling.

The first proof surface is setup, recovery, and admin health. Later migration
surfaces are:

1. Chat and Playground
2. Ingestion and Library
3. Jobs and scheduled tasks
4. MCP and ACP surfaces
5. Evaluations
6. Settings and account/security screens

## Component Ownership

### `components/ui`

`apps/packages/ui/src/components/ui` is the canonical design-system package. It
owns product-agnostic primitives and structural elements.

The proof-surface slice should not migrate `Button` wholesale. It may add a
`components/ui` wrapper or re-export only if setup, recovery, and admin health
need it. Full `Button` ownership migration belongs to the inventory and
migration stage.

Examples:

- Alerts
- Badges
- Buttons
- Empty/loading states
- Modal footers
- Page headers
- State panels
- Recovery callouts
- Diagnostic rows
- Field rows
- Action groups

### `components/Common`

`apps/packages/ui/src/components/Common` may keep broader shared product
components. New common components should consume `components/ui` primitives
internally. Generic components currently in `Common` should be moved into
`components/ui` or wrapped by `components/ui` as migration work allows.

Examples that may remain in `Common` if they contain product behavior:

- Backend recovery orchestration
- Command palette
- Chat sidebar
- Quick ingest workflow
- Markdown rendering
- Provider/model selection
- Tutorial runtime

### Feature Directories

Feature directories under `components/Option`, `components/Sidepanel`,
`components/Chat`, `components/Media`, and similar paths own domain behavior,
data loading, page composition, and feature-specific copy.

They should not own reusable product-state styling. If a feature needs a new
status, recovery, alert, empty-state, or action pattern, update this design
system contract or add a documented shared pattern before adding one-off local
classes.

## Ant Design Policy

Ant Design remains available as an implementation substrate.

Direct AntD use is allowed for:

- Table mechanics
- Selects and date pickers
- Drawers and modals
- Tooltips
- Tabs
- Form validation plumbing
- Feature-local prototypes that do not define product-level state language

tldw-owned wrappers or documented patterns are required for:

- Buttons and action hierarchy
- Alerts, banners, recovery states, and diagnostics
- Status badges and readiness labels
- Job state and permission labels
- Empty states and setup guidance
- Page shells and headers
- Local toolbars
- Admin and health panels
- Modal footers
- Destructive confirmation flows

The practical rule is: engineers may use AntD to implement behavior, but users
should experience tldw_server's design language, not AntD defaults.

## Canonical State Language

Product states are first-class design-system concepts. The implementation plan
should introduce a shared state registry or equivalent typed map so WebUI and
extension code use the same state keys, labels, severity, action rules, and
diagnostic behavior.

Each canonical state must define:

- Stable state key
- Readable label
- Severity
- Icon role
- Color token
- Copy pattern
- Allowed primary action
- Allowed secondary actions
- Diagnostic behavior
- Test expectation

Version 1 states:

| State key | Label | Severity | Token | Diagnostics | Primary action guidance | Test expectation |
| --- | --- | --- | --- | --- | --- | --- |
| `ready` | Ready | success | `--state-ready` | Hidden by default | Continue or open the feature | Readable ready label is present |
| `unavailable` | Unavailable | error | `--state-unavailable` | Visible when available | Try again | Retry action and failing target are present |
| `setup_required` | Setup required | warning | `--state-setup-required` | Collapsed unless setup details exist | Open setup | Setup action is present |
| `auth_required` | Sign in required | warning | `--state-auth-required` | Hidden unless request details exist | Sign in or add API key | Auth action is present |
| `permission_denied` | Permission denied | error | `--state-permission-denied` | Visible for admin/developer surfaces | Request access or switch account | Permission label and recovery action are present |
| `degraded` | Degraded | warning | `--state-degraded` | Visible when dependency details exist | Review diagnostics | Limitation and diagnostics affordance are present |
| `retrying` | Retrying | info | `--state-retrying` | Hidden unless retry fails | Wait or cancel when supported | Busy state is announced |
| `blocked` | Blocked | error | `--state-blocked` | Visible when root cause exists | Follow the next required fix | Blocking reason and next action are present |
| `empty` | Empty | neutral | `--state-empty` | Hidden by default | Create, import, or connect source | Empty state includes an action or explanation |
| `loading` | Loading | neutral | `--state-loading` | Hidden by default | Wait | Loading state is accessible and non-color-only |
| `error` | Error | error | `--state-error` | Visible when available | Retry or open diagnostics | Error label, retry path, and diagnostics affordance are present |

Secondary actions may include reload, open diagnostics, open settings, switch
server, switch account, copy diagnostics, or dismiss. The state registry should
mark which secondary actions are allowed for each state rather than leaving
feature screens to invent local action sets.

## Recovery Pattern

Backend recovery and setup recovery must not be full-screen one-offs. They are
instances of the `RecoveryState` pattern.

Required structure:

1. Summary: what is wrong in plain language.
2. Likely cause: server stopped, URL mismatch, missing auth, setup incomplete,
   permission denied, degraded dependency, or unknown error.
3. Primary next action: the shortest path to progress.
4. Secondary actions: retry, reload, open diagnostics, open settings, switch
   server, or sign in where applicable.
5. Diagnostics: request method, path, configured server URL, status, raw
   message, and structured diagnostics when available.
6. Recovery status: idle, retrying, failed, or recovered.

Diagnostics should be visible by default when they are available on admin,
health, setup, and developer-facing screens. For ordinary user workflows,
diagnostics may be collapsed behind a clearly labeled control.

## Content Rules

Design-system copy should be concise and actionable.

- Name the state before explaining details.
- Put the next action near the problem.
- Avoid vague phrases such as "something went wrong" when the system has a
  concrete method, path, status, missing permission, or setup requirement.
- Do not use color as the only signal.
- Use the same labels across WebUI and extension.
- Developer diagnostics may be technical, but the primary message should remain
  understandable without reading logs.

## Accessibility Rules

Version 1 must require:

- Keyboard-reachable primary and secondary actions.
- Visible focus states via shared focus tokens.
- ARIA labels for icon-only actions.
- `role="alert"` or appropriate live regions for blocking alerts and recovery
  changes.
- Text labels for every status, not color-only badges.
- Reduced-motion compliance for loading/retrying states.
- High-contrast compatibility for state tokens.

## Testing And Enforcement

The design system should use contract tests before visual-regression tooling.

Version 1 tests should verify:

- Shared primitives expose stable variants and accessible labels.
- Setup, recovery, and admin health screens use canonical state labels.
- Recovery UI exposes a primary next action.
- Diagnostics render when diagnostic details exist.
- WebUI and extension import shared assets/components where applicable.
- New shared UI exports are available through `components/ui`.
- Feature screens in the proof surface consume shared primitives instead of
  recreating local alert, badge, empty-state, and recovery styling.

Tests should assert semantics, behavior, and accessibility, not raw color
values.

Recommended enforcement sequence:

1. Focused Vitest coverage for proof-surface states.
2. Static guards for canonical state labels and shared exports.
3. Browser-observed visual QA for shared visual changes, using
   `Docs/Design/tldw_web_design_system_visual_qa_checklist.md`.
4. Optional Storybook or visual-regression coverage after the contract has
   stabilized.

## Rollout Plan

### Stage 1: Contract

Publish this contract and treat it as the source of truth for new WebUI and
extension surfaces.

### Stage 2: Inventory

Create a lightweight inventory of primitives and near-duplicates:

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

The inventory should identify canonical owners, duplicates, migration targets,
and proof-surface consumers.

Stage 2 inventory lives at
`Docs/Design/tldw_web_design_system_inventory.md`. Treat that document as the
current migration map for component ownership decisions until a later migration
updates it.

### Stage 3: Proof Surface

Apply the contract to setup, recovery, and admin health. This means proving the
contract can govern important states, not redesigning every admin screen.

The version 1 proof-surface boundary is:

- `apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx`
- `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
- `apps/tldw-frontend/components/ErrorBoundary.tsx`
- `apps/tldw-frontend/components/networking/ConfigurationGuard.tsx`
- `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- `apps/packages/ui/src/routes/option-settings-health.tsx`
- `apps/tldw-frontend/pages/settings/health.tsx`
- `apps/packages/ui/src/routes/option-setup.tsx`
- `apps/tldw-frontend/pages/setup.tsx`
- `apps/packages/ui/src/routes/option-admin-server.tsx`

Other admin routes are later migration candidates unless a proof-surface change
must touch them to preserve navigation or shared state contracts.

Required proof points:

- Backend unavailable
- Setup required
- Auth required
- Permission denied
- Degraded server/dependency
- Retrying
- Ready
- Diagnostics available
- Empty-but-actionable admin state

### Stage 4: Enforcement

Add focused tests and static guards for the proof surface and shared primitives.
Block new local one-off recovery, alert, status, and empty-state patterns in the
proof surface.

### Stage 5: Migration Queue

Prioritize later migrations:

1. Chat and Playground
2. Ingestion and Library
3. Jobs and scheduled tasks
4. MCP and ACP
5. Evaluations
6. Settings and account/security

## Non-Goals For Version 1

- Rebuilding all frontend components.
- Removing Ant Design.
- Introducing a full Storybook requirement.
- Redesigning every WebUI and extension screen.
- Creating a brand-new visual identity independent of existing tokens.
- Blocking feature work outside the proof surface before migration paths exist.

## Success Criteria

Version 1 succeeds when:

- New WebUI and extension work has a single documented UI contract.
- Setup, recovery, and admin health can be implemented without one-off state
  styling.
- Engineers know whether a component belongs in `components/ui`,
  `components/Common`, or a feature directory.
- AntD usage is constrained by product semantics.
- Product states have stable labels, actions, and diagnostics behavior.
- Tests can detect drift in the proof surface.
