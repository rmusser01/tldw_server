# Evaluations Review Fixes Design

## Goal

Evaluation ownership, webhook dispatch, and backend selection must preserve stable identity boundaries across single-user and multi-user deployments. The fixes tracked in this PR close review findings where auth metadata, webhook identity, and PostgreSQL-backed evaluation services could drift from the configured runtime.

## Decisions

- Heavy-admin checks use the same truthy configuration parsing as the rest of AuthNZ.
- Evaluation persistence receives stable user identifiers, not raw API keys or bearer tokens.
- Batch evaluation flows pass normalized webhook identity context through the same path as single evaluation flows.
- Webhook storage follows the active evaluations backend instead of forcing SQLite when PostgreSQL is configured.
- Tests cover the public endpoint behavior and backend adapter selection so the identity and storage choices remain explicit.

## Implementation Plan

The staged implementation is tracked in [IMPLEMENTATION_PLAN_evals_review_fixes.md](../Plans/IMPLEMENTATION_PLAN_evals_review_fixes.md).
