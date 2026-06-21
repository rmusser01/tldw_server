# Implementation Plan: ACP Support Guardrails (#2399)

## Stage 1: Registry And Setup-Guide Guardrails

**Goal**: Reconcile stale Aider registry expectations and add focused setup-guide coverage for Aider, Continue, and the seeded custom profile.
**Success Criteria**: Tests assert Aider remains `documented_unverified` / `documented_only` as an unverified `aider-acp` external adapter candidate; Continue remains documented-only with no ACP command; custom remains template-only and non-runnable.
**Tests**: Focused pytest for `test_acp_agent_registry.py`, `test_registry_entrypoint_strategy.py`, `test_acp_health.py`, and `test_acp_certification_smoke.py`.
**Status**: Complete

## Stage 2: Cross-Surface Conservative Audit

**Goal**: Audit compatibility docs, `agents.yaml`, bundled runner config, helper manifests, and Agent Registry UI for overclaims.
**Success Criteria**: No surface claims live ACP support for Aider, Continue, or generic custom profiles without evidence; any drift found is corrected with minimal changes.
**Tests**: Targeted grep/audit commands, focused UI Vitest only if UI changes are needed.
**Status**: Complete

## Stage 3: Verification And Tracker Closeout

**Goal**: Record evidence in Backlog and GitHub, then open the PR for #2399.
**Success Criteria**: Focused tests pass, `git diff --check` passes, Bandit is run for touched Python scope, Backlog task is finalized, and #2399/#2398 are updated with PR evidence.
**Tests**: Focused pytest, Bandit on touched Python tests/helpers when applicable, `git diff --check`.
**Status**: Complete
