# ADR-001: ADR Workflow And Governance

**Status:** Accepted
**Date:** 2026-06-02
**Backfilled from:** not backfilled
**Decision owner:** User + Codex collaboration session
**Related task:** TASK-506, TASK-507, TASK-508
**Related spec/plan:** `Docs/superpowers/specs/2026-06-02-adr-workflow-adoption-design.md`, `Docs/superpowers/plans/2026-06-02-adr-workflow-adoption-stage-1-implementation-plan.md`

## Decision

Use `Docs/ADR/` as the canonical home for Architecture Decision Records and require ADR assessment for substantial specs, implementation plans, and PRs.

## Context

Architecture decisions existed in scattered design docs, plans, review packets, and embedded ADR-like sections. The project needs a lightweight durable record that explains why architectural rules exist without replacing Backlog.md, Superpowers specs, implementation plans, or module documentation.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Big-bang migration | Too much churn and too high a risk of converting stale decisions into accepted policy. |
| Decision index before ADRs | Safer for audit, but delays the actual ADR workflow. |
| Module-by-module only | Too passive; it would not establish a repo-wide standard. |

## Consequences

Significant durable architecture decisions need ADRs. Substantial specs, plans, and PRs need an explicit ADR assessment. Accepted ADRs are immutable except for supersession metadata. Backfilled decisions use source metadata rather than pretending they were written at decision time.

## Follow-up

Create follow-up Backlog tasks for the decision inventory, module-by-module backfill, and possible global Superpowers updates.
