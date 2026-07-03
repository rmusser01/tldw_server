# ADR-014: Evaluations OpenAI-Compatible Schemas

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Evals/Evals-Plan-1.md`, `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`
**Decision owner:** Human requester approval of TASK-518 continuation
**Related task:** TASK-518
**Related spec/plan:** `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`

## Decision

Use separate request and response schemas for Evaluations API resources while preserving OpenAI-compatible response conventions such as `object`, Unix `created` timestamps, and list wrappers.

## Context

The Evaluations API needs to expose OpenAI-compatible evaluation, run, and dataset workflows while still supporting tldw-specific evaluation types and metadata.

The schema modules keep request payloads, update payloads, and response payloads separate. `openai_eval_schemas.py` explicitly documents the OpenAI-compatible conventions, and `evaluation_schemas_unified.py` extends the current API surface while preserving resource `object` fields, `created` timestamps, and dedicated response models.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Reuse database row dictionaries as API responses | Leaks persistence details, backend-specific timestamp formats, and internal fields into the public API. |
| Use one shared model for create, update, and response | Blurs required input fields, partial update semantics, generated fields, and output-only metadata. |
| Drop OpenAI-compatible response conventions | Breaks client expectations for resource objects, Unix timestamps, and list response shape. |

## Consequences

Schema changes must preserve the public API shape unless a superseding ADR explicitly changes compatibility goals.

Internal database fields can evolve independently from API responses, but conversion code must keep generated IDs, resource `object` values, timestamps, and list wrappers consistent.

tldw-specific extensions should be added through explicit schema fields or metadata rather than by leaking internal persistence rows.

## Follow-up

- Use this ADR as the covering record for INV-013.
- Create a new ADR before changing Evaluations API compatibility away from the OpenAI-style resource/list response shape.
