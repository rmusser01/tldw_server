# ADR-012: Evaluations Resource ID Prefixes

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Evals/Evals-Plan-1.md`, `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`
**Decision owner:** Human requester approval of TASK-518 continuation
**Related task:** TASK-518
**Related spec/plan:** `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`

## Decision

Use OpenAI-style type-prefixed IDs for primary Evaluations API resources: `eval_` for evaluations, `run_` for evaluation runs, and `dataset_` for datasets.

## Context

The Evaluations API exposes OpenAI-compatible resource surfaces for evaluations, runs, and datasets. These IDs are returned through public API responses, stored in the evaluations database, used by tests, and passed across service and runner boundaries.

The current implementation generates these prefixes in `EvaluationsDatabase.create_evaluation`, `EvaluationsDatabase.create_run`, and `EvaluationsDatabase.create_dataset`. `UnifiedEvaluationService.create_run` also pre-generates a `run_` ID before durable run persistence so audit records, idempotent schedulers, and task registration can refer to the same run consistently.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Bare UUIDs for all resources | Removes quick resource-type recognition from logs, API clients, and human debugging. |
| Integer IDs | Makes cross-database migration, per-user storage, and API exposure more collision-prone and less OpenAI-compatible. |
| Prefix only at the API layer | Creates a split identity model where API IDs and persisted IDs differ, increasing translation and idempotency risk. |

## Consequences

Evaluations resource IDs are part of the public API contract. New primary resource families should choose explicit, non-overlapping prefixes instead of reusing these prefixes.

The prefix identifies resource type only. It must not be used as an authorization signal, ownership proof, or database-routing subject.

Tests and documentation should continue to assert the prefix contract for evaluations, runs, and datasets when those IDs are exposed to callers.

## Follow-up

- Use this ADR as the covering record for INV-010.
- Create a new ADR before replacing the prefixed ID contract or adding a primary Evaluations resource family with ambiguous ID semantics.
