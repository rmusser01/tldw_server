# Research Workspace Provider Availability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Research Workspace Studio from treating catalog-visible but unusable LLM providers as generation-ready.

**Architecture:** Add backend readiness metadata to `/api/v1/llm/providers` and `/api/v1/llm/models/metadata`, then let the existing frontend model normalization and chat-model filtering consume those fields. Studio should inherit the existing model prerequisite path rather than adding a separate provider checker.

**Tech Stack:** FastAPI, pytest, React, Vitest, Ant Design Select.

---

## Stage 1: Backend Provider Readiness

**Goal:** Mark provider catalog entries unavailable when their endpoint is blocked by egress policy, lacks usable credentials, or cannot map to a chat provider id.

**Success Criteria:** `/api/v1/llm/providers` includes `availability`, `provider_enabled`, `readiness_reason_code`, `readiness_message`, and `chat_provider` for affected providers.

**Tests:** Add focused pytest coverage for egress-blocked Ollama, unreachable local endpoint, missing custom OpenAI credentials, and unsupported catalog aliases without real LLM calls.

**Status:** Complete

- [x] Write failing backend tests in `tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py`.
- [x] Run those tests and confirm they fail for missing readiness metadata.
- [x] Implement minimal readiness helpers in `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`.
- [x] Run the focused backend tests until green.

## Stage 2: Model Metadata Propagation

**Goal:** Flatten provider readiness onto each chat model so UI model services can filter unavailable choices.

**Success Criteria:** `/api/v1/llm/models/metadata` entries inherit provider `availability`, `provider_enabled`, `readiness_reason_code`, `readiness_message`, and `chat_provider`.

**Tests:** Extend backend metadata tests and frontend normalization tests to prove readiness fields survive normalization.

**Status:** Complete

- [x] Add failing metadata assertions to the backend readiness tests.
- [x] Add failing Vitest coverage in `apps/packages/ui/src/services/__tests__/tldw-api-client.models-normalization.test.ts`.
- [x] Implement metadata propagation in `llm_providers.py` and `model-normalization.ts`.
- [x] Run the focused backend and frontend tests until green.

## Stage 3: Studio Gating And Copy

**Goal:** Ensure unavailable provider classes are filtered out of Studio model choices or shown as precise prerequisites instead of generic artifact failures.

**Success Criteria:** Studio excludes unavailable models from selectable chat models, preserves stale-selection warnings for already selected unavailable models, and surfaces actionable setup copy.

**Tests:** Extend `TldwModels.test.ts` and `StudioPane.stage1.test.tsx` for egress-blocked, unreachable, missing-credential, and unsupported-alias model states.

**Status:** Complete

- [x] Add failing frontend tests for unavailable readiness statuses and Studio prerequisite copy.
- [x] Implement minimal frontend filtering and prerequisite copy.
- [x] Run the focused Research Workspace Studio tests until green.

## Stage 4: Verification And Documentation

**Goal:** Record the fixed behavior and remaining environmental limits honestly.

**Success Criteria:** Focused tests pass, static diff checks pass, Bandit is run for touched Python code, and UAT documentation/backlog notes are updated.

**Tests:** Run focused pytest, focused Vitest, `git diff --check`, and Bandit over touched backend code.

**Status:** Complete

- [x] Run focused backend tests.
- [x] Run focused frontend tests.
- [x] Run `git diff --check`.
- [x] Run Bandit for `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`.
- [x] Update `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md` and Backlog task notes.
