---
id: TASK-13376
title: Qualify complete-app provider setup and first-document workflow
status: In Progress
assignee: []
created_date: '2026-09-26 16:38'
updated_date: '2026-09-26 19:11'
labels:
  - distribution
  - qualification
  - webui
dependencies:
  - TASK-13343
references:
  - Docs/Design/2026-09-20-complete-app-distribution-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-09-26-complete-app-provider-document-qualification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Explicit unresolved product acceptance checkpoint from the user-approved review corrections. TASK-13265 is the completed design task and cannot stand in for pending implementation or qualification. Initial wizard progression in WP1 does not prove that a newcomer can configure a provider and use documents. This task qualifies that ordinary browser workflow before an installer is advertised as a complete usable application; it does not authorize publication or change native-platform/core-format requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 From a fresh signed extracted paired candidate outside a checkout, complete ordinary provider setup with a deterministic mock using the real WebUI and no manual backend master key or frontend-server URL wiring.
- [ ] #2 Use ordinary UI controls to ingest a Markdown document, find its content through search, and complete a chat with the configured mock provider; fail on setup or application errors.
- [ ] #3 Stop/start preserves provider configuration and document data; record exact candidate source, platform, browser and novice instructions with full-setup evidence distinct from initial-wizard checks.
- [ ] #4 Keep full-provider/document qualification false until all required workflows pass; retain the complete native platform and core-format matrix as separate required product gates.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the approved provider, Markdown ingestion, search and chat workflow on the existing signed extracted candidate outside the checkout.
2. Record concrete failures and trace them through existing UI/backend paths; present any product behavior change for review before implementing it.
3. Add bounded qualification coverage and rerun the workflow, including stop/start persistence, with exact source/platform/browser evidence. Keep full setup false until every acceptance criterion passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created as an explicit acceptance checkpoint while correcting TASK-13343. No test success is claimed and no requirement is waived.

Continuation on 2026-09-26: working only in complete-app-wp1. Existing qualification source 84d54884346f548379f44fea53c09320b4a5fb6c; local documentation HEAD d3d1083adc24fbb64e5d35c5d0e46311a736a453. No new product decisions, publication or requirement waivers authorized. Inspecting ordinary existing controls first.

Fresh signed source-84 candidate extracted outside checkout at /private/tmp/task13376-provider-probe-20260926/bundle. Disposable local registry and instance on 127.0.0.1:19080; repository mock provider on private instance network only. Actual WebUI selected Solo Docker, acknowledged privacy, configured Custom OpenAI-compatible with mock URL, validated models, saved provider, kept existing ingest defaults, deferred audio/RAG/storage and skipped optional MCP; Send test chat completed and navigated to Home. Home blocks document workflow with Restore media access / Single-user API key. UI Retry reproduces the same blocker. No master key entered. Source inspection identifies usePostOnboardingMediaReadiness.hasRequiredAuth checking only apiKey/runtime override, omitting cookie-session recognized by existing TldwAuth.isAuthenticated. Proposed bounded correction presented for approval; no production/test code changed. Full provider/document evidence remains false.

Reproduction manifest SHA256 39e09e851d90d25e0e3c213bb359bcf25494622ddac85d5930d37160841774a5, linux/arm64 on macOS Docker Desktop, Codex in-app Chromium (exact version not collected). Failure evidence retained at /private/tmp/task13376-provider-probe-20260926/workflow-failure.json. Existing hook suite: 2 tests pass, Node 26 emits localStorage experimental warning; neither test covers cookie-session. Markdown ingestion, search, ordinary application chat and their persistence checks not run because Home requires a manual master key. Proposed fix remains pending approval; no production or test edits. Bandit N/A for task notes and private observation JSON only.

Cleanup verified: signed stop helper stopped the owned application while retaining its instance state and data/config volumes. Exact recorded mock and private registry container IDs removed. Unrelated tldw_postgres_test and tldw_workspace_12020_50_pg remain running. Pending approval; task remains In Progress.

User approved the bounded readiness correction on 2026-09-26: recognize the existing single-user cookie-session source, retain a real listMedia access probe before ready, add regression tests, then resume signed-candidate qualification. Approval does not waive failures or authorize unrelated redesign/publication.

Approved readiness fix adds only the existing single-user cookie-session auth source to the local precheck; the real listMedia probe and its error handling remain mandatory. New behavior tests observed red 5 failed/7 passed, then focused hook+Home setup flow green 18 passed. Scoped ESLint and matching shared-code formatting pass. Broader core-route-identity result and baseline comparison tracked separately. Bandit not applicable: changed production code is TypeScript only, no Python changes.

Stage 1 verified: hook12+Home setup6 =18 passed; read-only provider_readiness_review independently reran18 and found no Critical/Important/Minor issue. Shared style retained; ESLint and formatting/diff checks pass. Baseline extracted d3d1083adc reproduces identical core-route-identity first-heading failure (1 failed/6 passed), with modified hook mocked; private baseline output /private/tmp/task13376-baseline-d3d1083adc/core-route.log retained. No unrelated test fix or success claim. Stage2 fresh signed build and full UI qualification pending.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
