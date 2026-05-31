---
id: TASK-576
title: Address unified solo onboarding UAT blockers
status: In Progress
labels:
- onboarding
- uat
- webui
- quick-ingest
documentation:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
- Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
- Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md
- Makefile
- Dockerfiles/docker-compose.webui.yml
- apps/tldw-frontend/pages/_app.tsx
- apps/tldw-frontend/__tests__/app/app-layout.test.tsx
- apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts
- apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and implement repairs required for a clean first-time solo-user walkthrough: root setup entry, WebUI auth handoff, first chat completion, first-source ingest, and fresh-install UAT verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design spec created and reviewed: Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md. Implementation plan created: Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md. Cleanup preflight completed on 2026-05-31: pruned stale Git worktree metadata, removed 123 clean merged non-current worktrees without force, skipped 13 dirty/untracked worktrees, and verified free space at 134GiB. Stage 1 complete: generic first-run routes now target `/`, `/` is the unified setup host bypass, and explicit character-chat onboarding remains character-specific. Verification: `bunx vitest run __tests__/app/app-layout.test.tsx --reporter=default` passed. Stage 2 complete: quickstart now passes the generated single-user API key into the WebUI compose command without echoing the resolved key, the WebUI compose environment preserves `NEXT_PUBLIC_X_API_KEY`, runtime bootstrap has first-run key coverage, and `TldwApiClient` seeds first-run quickstart config from the public key and current WebUI origin. Verification: `bunx vitest run __tests__/frontend-quickstart-networking.test.ts __tests__/extension/runtime-bootstrap.test.ts --reporter=default` passed; `bunx vitest run src/services/__tests__/tldw-api-client.quickstart-auth.test.ts --reporter=default` passed. Next step is Stage 3 authenticated post-onboarding readiness gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
