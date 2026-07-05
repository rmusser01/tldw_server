# ADR-009: Quick Chat Docs Assistant Modes

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Design/Quick_Chat_Docs_Assistant.md`
**Decision owner:** Human requester approval of TASK-509 inventory defaults
**Related task:** TASK-514
**Related spec/plan:** `Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md`

## Decision

Quick Chat uses three explicit modes: `Chat`, `Docs Q&A`, and `Browse Guides`.

## Context

Quick Chat needs to provide lightweight help without replacing the main chat UI or creating a new backend surface. The helper supports normal conversation, retrieval-grounded documentation answers, and deterministic local guidance from tutorials and curated workflow cards.

The modal owns mode switching and route handoff. `Docs Q&A` uses the existing RAG search API with a docs-focused retrieval profile. `Browse Guides` uses local tutorial and workflow-guide registries so guidance remains fast and deterministic even when retrieval quality varies.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Single chat mode for all helper interactions | Blurs retrieval-grounded documentation answers, deterministic guide browsing, and normal model conversation into one unclear path. |
| Add a new backend endpoint for Quick Chat docs help | Existing RAG search already provides the needed retrieval surface; a new endpoint would add ownership and testing burden without a clear need. |
| Generate workflow cards dynamically | Curated/static workflow cards and validated overrides are more deterministic and easier to test for route-aware help. |

## Consequences

Quick Chat mode selection must preserve the intended send path: standard chat for `Chat`, docs-scoped RAG for `Docs Q&A`, and local tutorial/workflow lookup for `Browse Guides`.

Docs retrieval profile settings, citation formatting, and suggested-page generation are part of the Quick Chat contract. Browse-mode cards remain curated or validated rather than free-form generated.

## Follow-up

- Use this ADR as the covering record for `INV-020`.
- If Quick Chat later adds a dedicated backend API or generated workflow-card system, create a superseding ADR or a new ADR for that durable change.
