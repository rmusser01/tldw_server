# tldw WebUI And Extension Visual QA Checklist

Date: 2026-05-30

## Purpose

This checklist defines the browser-observed visual QA expected for shared
design-system changes that can affect both the WebUI and browser extension. It
turns visual review from an ad hoc screenshot request into a repeatable
evidence path for migration PRs.

This document complements the route evidence protocol in
`apps/tldw-frontend/e2e/smoke/route-evidence-protocol.md`. Use that protocol
for route-family screenshots, naming, console/request triage, and known skips.
Use this checklist to decide whether a design-system PR needs WebUI,
extension, or sidepanel visual evidence at all.

## When This Checklist Applies

Apply this checklist when a PR changes any shared visual contract that can
alter rendered UI:

- shared primitives in `apps/packages/ui/src/components/ui`
- shared product-state wrappers in `apps/packages/ui/src/components/Common`
- shared sidepanel components under `apps/packages/ui/src/components/Sidepanel`
- shared route/layout shells consumed by WebUI and extension surfaces
- design-system tokens, semantic state mapping, spacing, radius, typography, or
  shared Tailwind assets
- migration PRs that replace local AntD product-state UI with tldw-owned
  primitives or wrappers

Do not require visual QA for purely internal helpers, type-only changes,
translation-only edits that do not change layout, backend-only changes, or
tests that do not alter rendered UI.

## Required Evidence By Change Type

| Change type | Minimum visual evidence |
| --- | --- |
| Token, spacing, radius, typography, or global shared CSS | One representative WebUI route and one extension/sidepanel surface that consume the affected token or class |
| `components/ui` primitive | A focused component/unit test plus one real consuming WebUI route; add extension/sidepanel evidence when the primitive is used there |
| `components/Common` product-state wrapper | The highest-risk consuming WebUI route and any extension route that imports the wrapper |
| Sidepanel shared component | Extension sidepanel viewport evidence; add WebUI evidence only when the same component appears in WebUI |
| Route/layout shell | Desktop and mobile WebUI evidence; add extension sidepanel evidence when the shell is shared with or launches from extension UI |
| Product-state migration | Before/after count evidence from the product-state verifier plus browser evidence for one representative migrated state |

If a deterministic Playwright/E2E test exercises the same route, viewport, and
state, a PR may cite that test instead of adding committed screenshots. The
Backlog task or PR body must state why the automated check is equivalent.

## Viewports

Use the route evidence protocol defaults unless a route-family plan names a
different viewport:

- Desktop WebUI: `1440x900`
- Mobile WebUI: `390x844`
- Extension sidepanel: `360x720`

For compact shared components, also check the smallest realistic container
where the component appears. Examples include sidepanel cards, modal footers,
table cells, drawers, and cockpit rails.

## What To Inspect

Every visual QA note should cover the user-visible result, not implementation
details alone:

- primary action is visible and not visually demoted by the migration
- secondary/destructive actions still have clear hierarchy
- status, recovery, loading, empty, and blocked states remain distinguishable
  without color alone
- labels and descriptions do not duplicate, truncate, overlap, or wrap into
  unusable controls
- keyboard focus remains visible on migrated controls
- hover, disabled, loading, selected, and error states remain legible
- no obvious horizontal overflow appears at the required viewport
- extension sidepanel layout still fits without hiding the primary workflow
- console/page/request failures are either absent or categorized with a visible
  recovery path

Screenshots are supporting evidence. A screenshot without a short observation
that names what was checked is not enough.

## Extension Evidence

Extension evidence is required when any changed shared component or token is
used by:

- sidepanel chat, clipper, flashcards, knowledge, or settings surfaces
- `apps/packages/ui/src/routes/sidepanel-*`
- extension entry shells, route launchers, or handoff controls
- shared components imported by both WebUI and extension routes

Prefer packaged extension smoke when the harness is available and stable. If
packaged launch is unavailable on the host, record the exact launch failure and
fall back to the closest available debug route or shared component evidence.
Do not present debug-route evidence as packaged extension proof.

## Before And After Evidence

For migration PRs, collect before/after visual evidence when the old and new
states are meaningfully different. A before screenshot may be omitted only when:

- the route or state is newly introduced
- the PR is documentation-only
- the visual change is fully covered by a deterministic before/after assertion
- the current baseline cannot be run, and the Backlog note records why

When a before screenshot is omitted, state the reason explicitly.

## Artifact Location

Use the route evidence protocol artifact paths:

- `Docs/Reviews/assets/<yyyy-mm-dd>-<task-or-slice-slug>/`
- `Docs/Plans/artifacts/<yyyy-mm-dd>-<task-or-slice-slug>/`

Temporary screenshots may stay local during investigation. The final Backlog
summary must say whether evidence is committed, attached to the PR, captured by
automation, or intentionally not retained.

## PR Body Checklist

For design-system PRs with visual impact, include a short `Visual QA` section:

```md
## Visual QA

- Surfaces checked:
- Viewports:
- States:
- Evidence:
- Extension evidence:
- Console/request triage:
- Known skips:
```

Use `Not applicable` only with a concrete reason, such as `type-only change` or
`docs-only checklist update`.

## Backlog Notes

Backlog implementation notes for visual-impact design-system tasks should
record:

- changed shared component, token, wrapper, or route shell
- WebUI route(s), extension route(s), and viewports observed
- automated visual/E2E checks run, with pass/fail status
- screenshot or artifact paths when retained
- expected capability failures and known skips
- product-state verifier before/after counts when the PR changes baseline debt
- follow-up task ids for deferred evidence or regressions

## Known Skip Standard

Visual QA skips must be precise and removable. Use the known-skip format from
the route evidence protocol and include the affected shared component or token.

Valid examples:

- packaged extension smoke cannot launch on the current host; debug route was
  checked instead; follow-up requires known-good packaged extension environment
- backend-dependent degraded state could not be reached because the local
  fixture lacks the required provider failure; follow-up task owns fixture
  coverage

Invalid examples:

- flaky
- not tested
- browser unavailable
- probably fine

## Closure Rule

A visual-impact design-system PR is ready to close when one of these is true:

- required WebUI and extension evidence is recorded
- deterministic Playwright/E2E coverage verifies the same route, viewport, and
  state
- the PR is docs-only or non-visual, and the PR/Backlog note explains why
- a known skip records the blocker, owner, review trigger, and user impact

Do not claim Browser/WebUI/extension visual QA is complete from static code
inspection alone when the change affects rendered UI.
