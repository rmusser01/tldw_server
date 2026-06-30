# Route Evidence Protocol

This protocol defines the browser-observed evidence expected for WebUI and extension route-family changes. It complements the deterministic WP12 route governance checks; it does not replace Playwright, Vitest, Axe, route metadata, or smoke inventory gates.

Use it when a slice changes a route's visible layout, page state, navigation, empty state, recovery path, sidepanel availability, or responsive behavior. Do not collect screenshots for untouched routes just to pad an evidence record.

## Evidence Location

Store evidence under the implementation task or review artifact for the slice that changed the route:

- Preferred review artifact path: `Docs/Reviews/assets/<yyyy-mm-dd>-<task-or-slice-slug>/`
- Preferred task artifact path: `Docs/Plans/artifacts/<yyyy-mm-dd>-<task-or-slice-slug>/`
- Temporary local screenshots are acceptable during investigation, but final Backlog notes must identify whether the artifact is committed, attached to a PR, or intentionally not retained.

Do not append screenshot links to the original WebUI audit report unless the user explicitly asks for a consolidated final report update.

## Required Coverage

For every changed visual route, capture before and after observations unless there is no meaningful before state because the route or state is newly introduced. In that case, record `Before: new route/state`.

Required viewports:

- Desktop: `1440x900`, unless a route-family plan names a different desktop size.
- Mobile: `390x844`.
- Extension sidepanel: `360x720` when the route is sidepanel-reachable or the change affects extension layout, route availability, or shared sidepanel components.

Required states:

- First-time or empty state when the route has one.
- Returning or populated state when fixtures or local data make it available.
- Degraded, unavailable, or capability-limited state when the slice changes recovery, diagnostics, provider readiness, or backend-dependent UI.
- Error state when the slice changes error presentation or recovery.

If a state cannot be reached, record why and whether it is blocked by fixture coverage, local backend setup, authentication, or product behavior.

## Observation Requirements

Each route evidence record must answer:

- What route and state was observed.
- What workflow or primary action was checked.
- Whether primary actions, labels, navigation, and recovery paths were visible without relying on implementation-language details.
- Whether the page avoided obvious horizontal overflow, overlapping controls, hidden primary actions, or unreadable compressed content at the required viewport.
- Whether console errors, page errors, and failed requests were absent or categorized as expected capability failures.
- Whether any regression was found, fixed in the slice, or deferred with a follow-up task.

Screenshots alone are not sufficient. Pair each screenshot with a short DOM or browser observation that names the user-visible result.

## Naming Conventions

Use lowercase route-family slugs, task ids, and stable state names.

Screenshot files:

```text
<task-id>-<route-slug>-<viewport>-<state>-<phase>.png
```

DOM or browser observation files:

```text
<task-id>-<route-slug>-<viewport>-<state>-<phase>.md
<task-id>-<route-slug>-<viewport>-<state>-<phase>.json
```

Allowed `phase` values:

- `before`
- `after`
- `regression`
- `fix`
- `follow-up`

Examples:

```text
task-418.8.5-audio-desktop-empty-before.png
task-418.8.5-audio-mobile-ready-after.png
task-418.10.6-settings-sidepanel-unavailable-after.md
```

Route slug rules:

- `/settings/model` becomes `settings-model`.
- `/chat-workspace` becomes `chat-workspace`.
- Extension hash routes drop the hash prefix, so `#/knowledge/thread/:threadId` becomes `knowledge-thread`.
- Dynamic path values use the route pattern or fixture name, not private local ids.

## Console And Request Triage

Record console and request notes in this format:

```md
### Console And Request Triage

- Console errors:
- Page errors:
- Failed requests:
- Expected capability failures:
- Unexpected failures:
- Action taken:
```

Use `None observed` only when the browser run actually checked the category. Do not use it when the browser tool did not collect console, page error, or network data.

Expected capability failures must include:

- Route or workflow affected.
- User-facing recovery or explanation visible on the page.
- Technical detail location when diagnostics are intentionally disclosed.
- Owner or follow-up task if the failure is not expected long term.

Unexpected failures must either be fixed in the slice or linked to a follow-up task before the slice is closed.

## Known Skip Format

Known skips must be route-specific and time-bounded. Avoid generic labels such as `flaky`, `not working`, or `backend issue`.

```md
### Known Skip

- Route:
- State:
- Viewport:
- Reason:
- Owner:
- Added:
- Expires or review trigger:
- Follow-up task:
- User impact:
```

A skip is valid only when it explains why the evidence cannot be collected now and what must happen before it is removed.

## Route Evidence Template

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

## Backlog Final Summary Fields

Every route-family Backlog final summary should include:

- Routes and viewports observed.
- Evidence artifact paths or PR attachment location.
- Browser tool used and whether it captured screenshots, DOM, console, page errors, and requests.
- Deterministic checks run, with pass/fail status.
- Expected capability failures and known skips.
- Regressions found and whether they were fixed or deferred.
- Unverified areas and why they remain unverified.
- Follow-up task ids for deferred work.

For documentation-only slices that define or update the protocol, record that no live browser evidence was required and list the document verification performed instead.

## Closure Rules

A route-family slice is not ready to close until one of these is true for each changed route:

- Evidence is recorded for required desktop and mobile viewports.
- Sidepanel evidence is recorded for sidepanel-reachable routes.
- A deterministic smoke or E2E check covers the same route/state and the Backlog summary explains why additional manual screenshots are unnecessary.
- A known skip records the blocker, owner, review trigger, and follow-up task.

Do not claim route-family browser QA is complete from static code inspection alone. If browser verification is unavailable, explicitly record the limitation and use code evidence only as a fallback.
