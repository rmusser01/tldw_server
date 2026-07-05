# ADR-003: Jobs Vs Scheduler Default

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** `AGENTS.md`
**Decision owner:** User + prior project guidance
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md`

## Decision

Use Jobs by default for new user-visible work that needs admin or ops visibility, and use Scheduler for internal orchestration where dependency handling is central.

## Context

The project has both Jobs and Scheduler systems. Future contributors need a durable default to avoid ad hoc queue/orchestration choices.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Jobs for all async work | Internal dependency orchestration fits Scheduler better and does not always need user/admin controls. |
| Scheduler for all async work | User-facing work often needs pause, resume, drain, retries, quotas, RLS, status endpoints, and worker processes. |
| Decide per feature with no default | Repeated debates and inconsistent ownership would slow implementation and increase maintenance cost. |

## Consequences

New user-visible features or work needing admin controls should use Jobs. Internal orchestration with task dependencies, idempotency keys, and registered handlers should use Scheduler. Recurring schedules use APScheduler to enqueue into whichever backend the feature chooses.

## Follow-up

Later ADR inventory work should identify any module-specific exceptions.
