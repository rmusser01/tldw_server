# Harden design-system baseline reporting and stale-entry cleanup

## Purpose

Reduce tracker drift by making the product-state verifier report grouped totals that are useful for issue updates and by ensuring stale baseline entries are removed during migration PRs.

## Scope

Included guard, doc, CI, or ownership decision:

- Improve baseline reporting around product-area totals, rule splits, and stale entries.
- Document how migration PRs refresh counts and remove resolved baseline entries.
- Preserve the approved source-of-truth model: GitHub owns mutable tracker state, and Backlog.md owns execution notes and PR evidence.

## Non-Goals

- This track does not migrate product-area baseline entries.
- This track does not create public GitHub issues without human approval.
- This track does not make Backlog.md the canonical current-count tracker.

## Done Criteria

- Durable artifact exists and is linked from the epic.
- Verification or review path is documented.
- Follow-up migration tasks know how to use the artifact.

## Tracking

- Parent epic: https://github.com/rmusser01/tldw_server/issues/1655
- Backlog task: TASK-45.44.14 (`backlog/tasks/task-45.44.14 - Harden-design-system-baseline-reporting-and-stale-entry-cleanup.md`)
- PRs:

| PR | Notes |
| --- | --- |
| TBD | |
