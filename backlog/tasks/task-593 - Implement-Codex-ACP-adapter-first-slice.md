---
id: TASK-593
title: Implement Codex ACP adapter first slice
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-06-02 00:46
labels:
- ACP
- Codex
- agents
- implementation
dependencies: []
documentation:
- Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
- Docs/superpowers/plans/2026-06-01-codex-acp-adapter-implementation-plan.md
priority: high
ordinal: 593
modified_files:
- Docs/superpowers/plans/2026-06-01-codex-acp-adapter-implementation-plan.md
- apps/packages/ui/src/services/acp/types.ts
- apps/packages/ui/src/services/acp/readiness.ts
- apps/packages/ui/src/services/acp/__tests__/readiness.test.ts
- apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx
- apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved first-slice Codex ACP adapter implementation plan. Scope includes canonical external_acp_adapter backend/API strategy support, dynamic registry metadata persistence, seeded Codex codex-acp profile/docs, passive Go runner readiness and explicit launch rules, frontend structured readiness gating, certification helper compatibility, and verification. Live Codex ACP certification and Codex app-server support remain follow-up work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canonical external_acp_adapter strategy is implemented and legacy adapter_acp is accepted only as an import compatibility alias.
- [ ] #2 Backend/API/static fallback/DB registry metadata expose adapter readiness without stale OPENAI_API_KEY-only Codex semantics.
- [ ] #3 Go runner uses passive no-spawn readiness for inventory/initialize and explicit acp_command launch rules for ACP sessions.
- [ ] #4 Frontend ACP session creation gates on structured entrypoint readiness and does not let stale is_configured override blocked states.
- [ ] #5 Docs and seeded Codex profile describe pinned codex-acp 0.15.0 without overclaiming live certification.
- [ ] #6 Focused backend, Go, frontend, Bandit, and diff hygiene verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete and approved. Commit d7651601c3 implements canonical external_acp_adapter strategy normalization, legacy adapter_acp import aliasing in registry/schema/endpoint helpers, Codex static fallback external-adapter/delegated-auth semantics, endpoint test update for canonical forwarded strategy, and live_certification_required setup-step coverage. Verification: focused pytest command passed locally with 36 passed and 6 warnings using /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv; git diff --check passed; Bandit on touched backend files passed with no findings. Review: spec compliance passed after adding explicit ACPAgentEntrypointStatus legacy alias coverage; code quality issues were fixed before approval.

Task 2 complete and approved. Commit e86d935951 exposes ACP adapter readiness metadata, separates display-agent binary readiness from external adapter availability, blocks mutable npx @latest adapter invocations, adds credential/runtime/adapter metadata to registry and public entrypoint status payloads, maps new blocker codes into setup guide copy, and keeps delegated adapter credentials passive without OPENAI_API_KEY blocking. Verification: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py` passed with 47 passed and 6 warnings; `git diff --check` passed; Bandit on touched backend files passed with no findings. Review: spec compliance and code quality approved locally after delegated reviewers stalled due agent latency.

Task 3 complete and approved. Commit b3cb9dcc94 persists external ACP adapter metadata in the dynamic agent registry, adds schema version 15 migration/defaults for adapter package/version/version policy/install source/credential policy/runtime backend, preserves the fields through register/update/reload API paths, and keeps legacy adapter_acp normalized on DB write. Verification: focused pytest `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py` passed with 81 passed and 6 warnings; `git diff --check` passed. Bandit on touched production files exited 1 only for existing ACP_Sessions_DB.py baseline findings outside the Task 3 diff: B105 at line 205 and B608 at lines 1058, 1063, 1099, 1117, 1947, 2162, and 2247; endpoint/schema/registry files had zero findings. Review: delegated spec review approved; local code quality review approved with the baseline Bandit note recorded.

Task 4 complete and approved. Commit c97f4d0cc5 seeds the Codex registry row as an experimental external_acp_adapter profile using `codex-acp` 0.15.0 metadata, delegated adapter credentials, and live_certification_required, then updates the ACP compatibility matrix and active/published getting-started docs without claiming live certification. Verification: seeded Codex test was added red first and failed before the YAML change; focused registry test command `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py` passed with 29 passed and 6 warnings after the amended published-doc consistency fix; `git diff --check` passed. Bandit not applicable because Task 4 changed only docs, YAML, and tests. Review: local spec and code quality review approved after confirming no mutable runtime config, stale Codex OPENAI_API_KEY-only copy, or live support overclaim remained in changed Codex sections.

Task 5 complete and approved. Commit ca98282828 adds Go runner config parsing for ACP entrypoint/adapter metadata, explicit launch resolution that starts external_acp_adapter profiles through acp_command without falling back to the display command, passive initialize/agent-list readiness that never spawns downstream agents or runs mutable npx @latest inventory checks, and runner agent/list readiness metadata including adapter_docs_url. Verification: red Go test run with `GOCACHE=/private/tmp/tldw-go-cache go test ./internal/config ./internal/acp` failed before implementation on missing config fields and missing resolver; green command `cd tools/tldw-agent && GOCACHE=/private/tmp/tldw-go-cache go test -count=1 ./internal/config ./internal/acp` passed after amendments; `git diff --check 0a8f2b7e41 ca98282828` passed. Review: delegated spec review initially rejected missing config-test coverage for adapter_docs_url/adapter_package/runtime_backend and missing agent/list adapter_docs_url; both were fixed before local spec/code quality approval. Bandit not applicable for Go-only task.

Task 6 complete and reviewed. Frontend ACP types now mirror structured entrypoint readiness, `readiness.ts` exposes `isACPAgentReadyToStart` and `buildACPAgentSetupSummary`, and the ACP create modal uses structured readiness for agent cards, default selection, disabled submit state, setup tooltip/copy, and an inline external-adapter badge. Review found two P2 issues: secondary blockers could override `primary_blocker`, and cached agent data could be reset back to a blocked backend default; both were fixed by making `primary_blocker` authoritative and by resetting before applying the first structured-ready agent while preserving an already-ready current selection. Verification: red UI readiness test failed on missing helpers before implementation; review follow-up red test failed on blocker precedence before the fix; focused UI command `./node_modules/.bin/vitest run src/services/acp/__tests__/readiness.test.ts src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts` passed with 2 files and 11 tests; `git diff --check` passed. UI package typecheck required `NODE_OPTIONS=--max-old-space-size=8192` and then failed on existing non-ACP baseline errors in QuickIngest, Layout, Playground, Sidepanel, onboarding, option-index, and quick-ingest-open files, with no ACP file errors reported. Bandit not applicable for TypeScript-only production changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
