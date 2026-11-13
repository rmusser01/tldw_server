# ADR-097: Parent-Supervised Standard-Library Regex for Moderation

**Status:** Accepted
**Date:** 2026-11-02
**Backfilled from:** not backfilled
**Decision owner:** Requester; written design approved on 2026-11-02
**Related task:** TASK-15414
**Related spec/plan:** Docs/superpowers/specs/2026-11-02-moderation-killable-stdlib-re-design.md

**Identity correction (2026-11-07):** Previously ADR-086 on this unmerged branch;
renumbered after rebasing because dev independently assigned that number to
provider credential scope. Interim ADR-096 was also claimed by PR3436 while
verification ran, so publication uses ADR-097. Accepted decision and rationale
are unchanged.

## Decision

Execute Moderation's runtime matching/redaction with standard-library `re` only
in a parent-supervised killable child, blocking affected content and surfacing a
sanitized admin incident when execution cannot complete safely.

## Context

Moderation currently runs unbounded search/iteration/substitution in application
execution paths. Scan chunking and syntactic safety heuristics do not establish
a runtime termination boundary. The requester chose preserving stdlib semantics
and isolating failure over switching regex engines.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Third-party regex timeouts | Requester rejected changing engines; not the requested process boundary. |
| Thread/future timeout | Stops waiting without establishing termination of the regex operation. |
| Shared generic process pool | Does not give a request simple ownership of a killable child; pool recovery and queued work complicate failure containment. |

## Consequences

The parent owns admission, an operation-wide deadline, bounded IPC, termination,
reaping, and fail-closed caller behavior. No overload/error inline fallback is
allowed. Successful matching retains the existing evaluator algorithms and
stdlib semantics. Process startup/serialization add overhead. Default values and
the detailed integration contract are documented in the approved linked design.

Operational reporting is independent of optional content-review capture and
must not contain failed content or invent a policy match. Reporter failure never
permits content through. Existing DB/admin surfaces are reused where suitable.

This is not a complete OS sandbox or a guarantee against host-wide resource
exhaustion. Document platform-specific memory/CPU guarantees and deployment limits.
Initial policy compilation/linting and coverage of previously unmoderated LLM
surfaces are outside this decision's first implementation scope.

## Follow-up

The builder check found no blocking questions, and the requester approved the
complete design on 2026-11-02. Write an incremental implementation plan with
compatibility, process-lifecycle, privacy, and caller integration tests.
