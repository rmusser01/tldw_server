# ADR-013: Evaluations Deletion Lifecycle

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Evals/Evals-Plan-1.md`, `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`
**Decision owner:** Human requester approval of TASK-518 continuation
**Related task:** TASK-518
**Related spec/plan:** `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`

## Decision

Use soft deletes for evaluation definitions and hard deletes for datasets in the Evaluations API lifecycle.

## Context

Evaluation definitions can have associated runs, audit records, result history, and user-visible references. Deleting an evaluation should hide it from normal get/list/update paths without immediately erasing its historical relationship to runs and audit trails.

Datasets are user-managed payload containers. The current implementation treats dataset deletion as storage cleanup: `EvaluationsDatabase.delete_dataset` removes the dataset row, while evaluation deletion sets `deleted_at` and normal evaluation queries filter `deleted_at IS NULL`.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Hard-delete evaluations | Risks losing history and breaking run/audit references tied to a deleted evaluation definition. |
| Soft-delete datasets | Keeps potentially large sample payloads around after users explicitly request cleanup. |
| Use one deletion mode for every resource | Ignores the different retention needs of evaluation definitions versus dataset payload storage. |

## Consequences

Evaluation read, list, and update paths must filter out soft-deleted rows unless a future recovery/admin workflow explicitly opts into viewing them.

Dataset deletion is destructive. Callers and endpoints must preserve appropriate permission checks before invoking it, and future dataset recovery semantics require a new decision.

The deletion lifecycle is independent of database backend choice. SQLite and PostgreSQL implementations should preserve the same logical behavior even if column types differ.

## Follow-up

- Use this ADR as the covering record for INV-011.
- Create a new ADR before adding dataset recovery, evaluation hard-delete, or retention-window behavior that changes these lifecycle semantics.
