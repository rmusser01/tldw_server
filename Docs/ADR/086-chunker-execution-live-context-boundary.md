# ADR-086: Chunker Execution Live-Context Boundary

**Status:** Proposed
**Date:** 2026-11-02
**Backfilled from:** not backfilled
**Decision owner:** Requester + Codex collaboration; detailed boundary awaiting requester approval
**Related task:** TASK-15413
**Related spec/plan:** [Shared execution design](../superpowers/specs/2026-11-02-chunker-shared-execution-refactor-design.md); implementation plan not started

## Decision

Route Chunker's three public execution methods through an internal execution
service whose dependencies point toward neutral helpers and a live context,
not back toward the Chunker facade or sibling pipelines.

This is a proposed decision. The requester selected the internal-subsystem
approach B and authorized adding this record; the revised design and detailed
boundary are not yet approved for implementation.

## Context

`Chunker.chunk_text`, `chunk_text_with_metadata` and `chunk_text_generator`
repeat preparation and strategy setup but have different cache, metadata,
exception and iterator behavior. Keeping all execution in the facade makes
those responsibilities harder to review independently. Earlier process_text
and hierarchy extractions demonstrate the local service/context pattern, but
no existing accepted ADR governs the new execution-module boundary.

The proposed specification establishes an explicit, tested one-way import
boundary. That is a durable module-boundary decision under ADR-001 and the ADR
workflow, even without a public API change or repository-wide policy. This
record is made at design time, not a backfill or a rewrite of an accepted ADR.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Extract only shared setup helpers | Smaller change, but leaves the three execution bodies and their coordination in the facade; the requester selected approach B instead. |
| Let execution helpers import the facade or sibling pipelines | Creates reverse dependencies and initialization-order hazards, and couples component tests to the orchestration layer. |
| Move registry, result-cache and mutable strategy ownership into a new executor | Expands the selected scope and risks changing cache identity, overrides, locking and generator timing. |

## Consequences

- `chunker.py` delegates to `execution.service`; execution components depend
  only on their internal models/helpers and existing neutral Chunking modules.
- New execution code does not import `Chunker`, `chunker.py`, `process_text`
  or `hierarchical`, including through local imports. Existing parent-package
  bootstrap eagerly exporting the facade is not redesigned by this decision.
- `Chunker` retains config, strategy/result-cache ownership and existing
  helper/override seams. The service receives a narrow live context rather
  than copying facade state or constructing a second registry.
- Observability is injected from the facade through forwarding hooks, not by
  importing the facade to resolve module-level logging or metric functions.
- Three path-specific entries retain existing behavior; this is not authority
  to unify cache usage, exception regions, metadata mutation or generator locks.
- Static boundary contracts and fresh-process import smoke tests enforce the
  dependency direction and exercise real package initialization.
- The tradeoff is one additional delegation layer and explicit context/hook
  wiring, in exchange for independent component testing and clearer ownership.
- The scope is the new execution subsystem. It introduces no public API,
  configuration, dependency, persistence or security-policy change and does
  not retroactively impose new ownership on existing sibling subsystems.

## Follow-up

Obtain requester approval of the revised design and this Proposed ADR before
planning implementation. Keep its status Proposed until that approval is
explicit. The later implementation task/plan and PR should link this decision
and the detailed compatibility/verification contracts in the specification.
If the accepted durable boundary later changes, use a new ADR and supersession
metadata rather than rewriting accepted rationale.
