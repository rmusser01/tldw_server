# PR 2565 CI Fixes After Dev Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #2565 onto latest `dev` and fix actionable GitHub Actions failures without changing the standalone MCP docs feature scope.

**Architecture:** Treat the failures as independent regressions in existing backend/frontend contracts. Reproduce each focused failure first, then make the smallest code or test-contract change that matches the intended behavior on the rebased baseline.

**Tech Stack:** Python 3.12 pytest, FastAPI/Pydantic schemas, SQLite-backed integration tests, Bun/Playwright frontend gates where touched.

---

## Stage 1: Backend Regression Reproduction
**Goal**: Confirm the CI backend failures are present on the rebased branch.
**Success Criteria**: Focused pytest commands reproduce the expected audio, Fish S2, Guardian outcomes, and document rebased/local-only pass outcomes for Chatbooks, persona alias, and sandbox.
**Tests**:
- `python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_audio_files_preflight.py -q`
- `python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_fish_s2_reference_endpoints.py::test_import_fish_s2_references_returns_partial_provider_errors -q`
- `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py::test_create_chatbook_export_writes_requested_v1_1_manifest_version tldw_Server_API/tests/integration/test_chatbook_integration.py::TestErrorScenarios::test_export_with_database_error -q`
- `python -m pytest tldw_Server_API/tests/Chat_NEW/integration/test_chat_persona_exemplars_integration.py::test_chat_completion_accepts_persona_id_alias_before_removal_date_when_resolvable tldw_Server_API/tests/Chat_NEW/integration/test_chat_persona_exemplars_integration.py::test_chat_completion_rejects_unresolvable_persona_id_alias tldw_Server_API/tests/Chat_NEW/integration/test_chat_persona_exemplars_integration.py::test_chat_completion_rejects_persona_id_alias_after_removal_date tldw_Server_API/tests/Chat_NEW/integration/test_chat_persona_exemplars_integration.py::test_chat_completion_persona_id_alias_before_removal_date_with_embeddings_strategy_uses_character_context -q`
- `python -m pytest tldw_Server_API/tests/Guardian/test_comprehensive_edge_cases.py::TestNotificationDeliveryEdgeCases::test_generic_notify_adds_timestamp -q`
- `python -m pytest tldw_Server_API/tests/sandbox/test_execution_concurrency_cap.py::test_background_admission_renews_queued_claim_while_waiting -q`
**Status**: Complete

## Stage 2: Backend Fixes
**Goal**: Fix confirmed backend regressions at their source.
**Success Criteria**: Focused failing tests pass after minimal changes.
**Files**:
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_voices.py`
- `tldw_Server_API/app/core/Monitoring/notification_service.py`
- `tldw_Server_API/app/core/Sandbox/service.py`
**Tests**: Re-run every command from Stage 1 after each relevant fix.
**Status**: Complete

**Notes**:
- Chatbooks and persona alias tests passed on the rebased baseline, so no code changes were needed there.
- Sandbox focused tests passed locally, but the renewal loop now renews queued claims immediately after denied admission to reduce the CI lease race.

## Stage 3: Frontend Gate Investigation
**Goal**: Determine whether UX gate failures still apply after the dev rebase and whether they are repo issues or stale CI/environment issues.
**Success Criteria**: Reproduce and fix focused frontend gate failures, or document that the rebased branch/backend fixes do not touch the failing frontend contracts.
**Tests**:
- Inspect `apps/tldw-frontend/e2e/smoke/stage6-interaction-stage1.spec.ts`
- Inspect `apps/tldw-frontend/e2e/smoke/stage6-interaction-stage2.spec.ts`
- Inspect `apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts`
- `bunx vitest run ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx`
- `bunx playwright test e2e/workflows/onboarding-ingestion-first.spec.ts --project=chromium --reporter=line`
**Status**: Complete

**Notes**:
- Fixed the standalone smoke workflow start command to use the built Next standalone server.
- Fixed post-onboarding media readiness to accept the runtime single-user API key override used by web runtime auth.
- Updated the onboarding evidence spec to seed shared auth, stub runtime config, and stub model metadata/provider warmups.

## Stage 4: Verification, Tracking, and Push
**Goal**: Verify changed scope, update Backlog task `TASK-12088`, commit, and push the rebased branch.
**Success Criteria**: Focused tests pass, Bandit runs on touched backend paths, diff is reviewed, task notes/final summary are updated, commit and push succeed.
**Tests**:
- Focused pytest commands from Stage 1.
- `python -m bandit -r <touched_backend_paths> -f json -o /tmp/bandit_pr2565_ci_fixes.json`
- `git diff --check`
**Status**: Complete

**Verification so far**:
- `python -m pytest ... -q` focused backend set: 24 passed.
- `bunx vitest run ...`: 6 files, 99 tests passed.
- `bunx playwright test e2e/workflows/onboarding-ingestion-first.spec.ts --project=chromium --reporter=line`: 2 passed.
- `python -m bandit -q -r <touched backend files> -f json -o /tmp/bandit_pr2565_ci_fixes.json`: 0 findings.
- `git diff --check`: passed.
- `bun run build:prod`: inconclusive locally. The build stayed silent with stagnant CPU for several minutes; after interrupting, the follow-on token-sync verifier failed against the partial `.next` output.
