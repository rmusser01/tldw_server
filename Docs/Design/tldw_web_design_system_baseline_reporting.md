# tldw Web Design-System Baseline Reporting

Date: 2026-05-22

## Purpose

The product-state guard is the live source of truth for remaining WebUI and
extension design-system migration debt. GitHub issues own public tracker state,
and Backlog.md records execution evidence, but both must be refreshed from the
verifier output rather than hand-counted.

This document defines the reporting and cleanup workflow for migration PRs.

## Report Sections

Run the verifier from the UI package:

```bash
bun run verify:design-system-state
```

The report includes detailed finding lists plus grouped summaries:

- `Baseline exceptions`: active migration targets plus allowed legacy entries
  that still match live findings.
- `By product area`: counts grouped by the tracker path ownership map from the
  remaining-work design.
- `By rule`: counts by guard rule, such as `antd-product-state-import` or
  `canonical-state-label`.
- `By migration queue`: counts by the baseline entry migration queue.
- `Stale baseline cleanup summary`: baseline rows that no longer match a live
  finding and should be removed instead of carried forward.

Stale baseline entries are warnings, not verifier failures, so migration PRs
must actively remove stale rows before recording final counts.

## Migration PR Workflow

1. Before changing code, run `bun run verify:design-system-state` and record the
   relevant product-area total, rule split, and migration queue count.
2. Migrate a narrow set of product-state UI to shared design-system primitives
   or the design-system state registry.
3. Remove the migrated baseline entries from
   `apps/packages/ui/scripts/design-system-product-state-baseline.json`.
4. Re-run `bun run verify:design-system-state`.
5. If `Stale baseline cleanup summary` lists entries owned by the PR scope,
   remove those baseline rows and re-run the verifier.
6. Update the GitHub product-area or governance issue first with the before
   count, after count, rule split, verifier command, and PR link.
7. Update the Backlog task with the same evidence and note that GitHub owns the
   canonical current count.

## Closure Rules

- A product-area issue can close only after the current verifier output reports
  zero baseline exceptions for that area.
- A governance issue can close when it delivers the promised guard, policy,
  CI path, ownership decision, documentation, or visual QA artifact and records
  verification.
- If a new baseline exception is intentionally introduced, the baseline reason
  must include the owning tracker issue, and the `migrationQueue` must match the
  tracker queue slug.

