# ADR-079: Advisory UI Design Drift Reporting

**Status:** Proposed
**Date:** 2026-10-20
**Backfilled from:** not backfilled
**Decision owner:** Repository owner; direction approved in this design session
**Related task:** TASK-45.44.16
**Related spec/plan:** [Design spec](../superpowers/specs/2026-10-20-design-system-drift-reporting-design.md)

## Decision

Report token, color, radius, and layout drift across repository-owned product UI
through a standalone advisory verifier, keeping existing and new findings visible
without blocking builds or merges.

## Context

The product-state guard already governs state primitives and labels. Styling
drift extends beyond that guard, while legitimate theme, chart, editor, and layout
values make blanket raw-value bans inaccurate. Admin UI also has its own semantic
token vocabulary. The owner selected repo-wide scanning with no current blocking.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Immediate required enforcement | Contradicts the owner's advisory-only choice and requires more rule-accuracy evidence. |
| New-code-only scan or suppression baseline | Hides existing migration work and fails the repo-wide reporting goal. |
| ESLint plus a CSS checker | Adds multiple configuration and reporting paths for one inventory. |
| Unscoped text matching | Cannot reliably distinguish product styling from data and prose. |

## Consequences

All four UI roots are scanned on each report: shared UI, WebUI, extension, and
admin UI. Existing token authorities remain intact. Layout values are review
candidates; findings do not imply visual defects. Intentional exceptions require
narrow scope and a reason, and stale entries stay visible.

Findings exit successfully. Scan failures remain distinguishable and visible in
a nonrequired CI job. Required product-state and token-sync checks retain their
behavior. Blocking enforcement requires a separate owner decision and policy
change; it cannot be enabled by debt counts or elapsed time.

## Follow-up

Approve the written spec, then plan and implement the reporter and nonrequired
CI integration. Public tracking: https://github.com/rmusser01/tldw_server/issues/1673.
