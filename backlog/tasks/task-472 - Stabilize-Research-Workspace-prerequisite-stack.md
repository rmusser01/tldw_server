---
id: TASK-472
title: Stabilize Research Workspace prerequisite stack
status: Done
labels:
- research-workspace
- packaging
- webui
- backend
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
- https://github.com/rmusser01/tldw_server/pull/2018
- Docs/superpowers/plans/2026-05-23-research-workspace-prerequisite-stack-packaging-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Package the existing Research Workspace WIP from the dirty checkout into a clean reviewable branch based on origin/main. Scope includes the hard /research-workspace route replacement, legacy storage inventory gate, server bootstrap/trust panel wiring, migration protocol API, and related docs/tests without unrelated chat/sidebar/writing changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Clean branch is based on `origin/main` and contains only the Research Workspace prerequisite stack, with chat-workspace, prototype-workspace, and writing/playground changes excluded.
- [x] Active WebUI and extension routes use `/research-workspace` and do not register, alias, or redirect `/workspace-playground`.
- [x] Legacy local storage is inventoried and gated so unknown or unmapped data is never deletion-eligible.
- [x] Backend migration protocol and source/status/capability APIs are included with focused pytest coverage and Bandit verification for touched Python.
- [x] Frontend telemetry, prefill, trust panel, route, tutorial, and legacy inventory behavior have focused Vitest/e2e coverage.
- [x] Real backend plus CDP browser smoke validates `/research-workspace`, old-route 404 behavior, and backend status/capability calls where feasible.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Packaged the Research Workspace hard route replacement into `codex/research-workspace-prereq-stack`, keeping Research and Research Workspace separate and removing active `/workspace-playground` route files without adding aliases or redirects.
- Included first-class source status and capability projection APIs, migration protocol APIs, server-backed workspace bootstrap, legacy storage inventory, telemetry/prefill renames, and browser-extension/WebUI route test updates.
- Preserved the Shared Workspaces model in the trust/capabilities surface by exposing MCP Hub, ACP, Sandbox, sharing, migration, and provider readiness as workspace services.
- CDP smoke found and fixed three follow-up issues: AntD `destroyOnClose` usage in Research Workspace header, invalid rich-description paragraph nesting in `EmptyState`, and package-level Vitest missing the `wxt/browser` alias used by workspace API tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Research Workspace prerequisite stack is packaged on the clean branch with backend migration/status/capability APIs, WebUI route replacement, extension handoff updates, legacy migration safeguards, trust panel wiring, and focused coverage. Live validation used a real FastAPI backend and Playwright/CDP against the Next.js WebUI: `/research-workspace` rendered the trust panel, source status and capability calls returned 200, and `/workspace-playground` returned 404 without redirecting. Draft PR opened at https://github.com/rmusser01/tldw_server/pull/2040; draft status is intentional because the AI-generated PR merge gate requires the human requester to own the final Change summary before merge.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
