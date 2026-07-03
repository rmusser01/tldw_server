# ADR-015: Evaluations Existing Evaluator Integration

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Evals/Evals-Plan-1.md`, `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`
**Decision owner:** Human requester approval of TASK-518 continuation
**Related task:** TASK-518
**Related spec/plan:** `Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md`

## Decision

Integrate Evaluations API orchestration by wrapping and delegating to existing evaluator modules instead of rewriting evaluator logic inside the API runner or endpoints.

## Context

The Evaluations module already has dedicated evaluator implementations for GEval, RAG, response quality, OCR, propositions, and related workflows. The API runner and service layer need to orchestrate persistence, run state, progress, webhooks, and result shaping without duplicating the scoring implementations.

The current runner imports and delegates to existing components such as `ms_g_eval.run_geval`, `RAGEvaluator`, `ResponseQualityEvaluator`, proposition evaluation helpers, and the unified RAG pipeline. `UnifiedEvaluationService` maps high-level evaluation requests to these dedicated evaluator services.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Rewrite evaluator logic inside `eval_runner.py` | Duplicates tested scoring behavior and increases drift between API and non-API evaluation paths. |
| Put evaluator-specific logic directly in FastAPI endpoints | Couples routing, auth, persistence, and scoring logic, making tests and future evaluator additions harder. |
| Replace local evaluator modules with a new external-only evaluation package | Would discard existing project-specific RAG, OCR, response-quality, and proposition evaluation behavior. |

## Consequences

The API runner and service layer own orchestration, persistence, state transitions, and response shaping. Dedicated evaluator modules own scoring behavior.

Adding a new evaluator should register or map to a dedicated evaluator implementation rather than embedding scoring logic in endpoint handlers.

Some adapters may still need sync-to-async bridging, provider resolution, and result normalization at the runner/service boundary.

## Follow-up

- Use this ADR as the covering record for INV-015.
- Create a new ADR before replacing the wrapper/delegation strategy with a centralized evaluator rewrite.
