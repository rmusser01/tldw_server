# ADR-007: Research Workspace Canonical First-Slice Shell

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md`, `Docs/superpowers/specs/2026-05-06-tldw-product-roadmap-design.md`, `Docs/superpowers/plans/2026-05-06-tldw-product-roadmap-first-slice-implementation-plan.md`
**Decision owner:** Human requester approval of TASK-509 inventory defaults
**Related task:** TASK-514
**Related spec/plan:** `Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md`

## Decision

Use `ResearchWorkspace` as the canonical shell for the first roadmap slice while keeping `ChatWorkspace` and `DocumentWorkspace` as specialized routes or modes instead of deleting or fully merging them.

## Context

The workspace roadmap needs one product model for sources, selected sources, chat, quick notes, generated artifacts, saved workspaces, source transfer, local persistence, and server sync boundaries. `ResearchWorkspace` already contains the broadest version of that model and is the best first-slice shell. `ChatWorkspace` and `DocumentWorkspace` still validate important workflows, but they should not define parallel product models during this slice.

The first slice should consolidate the model before consolidating routes. Route consolidation remains a later decision.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Fully merge `ChatWorkspace` and `DocumentWorkspace` into `ResearchWorkspace` immediately | Too much route and workflow churn for the first slice; it risks breaking existing chat-first and document-focused flows before the shared model is stable. |
| Keep all three workspaces as independent product models | Creates duplicated state, divergent persistence semantics, and unclear roadmap ownership. |
| Create a new roadmap workspace from scratch | Duplicates existing `ResearchWorkspace` capabilities and delays first-value work. |

## Consequences

`ResearchWorkspace` owns the first-slice canonical workspace direction. `ChatWorkspace` and `DocumentWorkspace` remain available as specialized entry points or modes. Implementation plans should update the canonical model or write a new ADR before changing route ownership semantics.

Server sync should use the existing `/api/v1/workspaces` family first, while browser-local workspace state remains a responsive cache and offline-friendly UI surface.

## Follow-up

- Use this ADR as the covering record for `INV-017` and the route-consolidation context in `INV-019`.
- If later work fully merges or removes `ChatWorkspace` or `DocumentWorkspace`, create a superseding ADR.
