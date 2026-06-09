---
id: TASK-481.14
title: PR 5 - Notes tags terminology and organization semantics
status: Done
labels:
- notes
- ux
- webui
- frontend
parent_task_id: TASK-481
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 5 from the staged notes UX remediation plan: present one user-facing concept, Tags, while preserving the existing keywords API/client implementation contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User-facing Notes organization copy uses Tags in both app and public extension English locale sources.
- [x] #2 Printable single-note HTML uses Tags as the visible metadata label while Markdown frontmatter and JSON exports keep the keywords field.
- [x] #3 Directly connected Web Clipper capture copy remains Tags and the save-flow tests continue to pass.
- [x] #4 No database, backend API, TypeScript field, or storage rename from keywords to tags is introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation completed for PR5. Changed only user-facing Notes locale strings and the printable HTML metadata label; retained keywords in API payloads, TypeScript data fields, Markdown frontmatter, JSON exports, test IDs, and backend/storage naming. Added notes-tags-terminology.locale.test.ts for app/public locale copy and Web Clipper tag copy. Updated export-utils.test.ts to assert the printable HTML label says Tags while existing tests continue to assert markdown/json keywords output. Verification: RED vitest run failed on old copy and print label; GREEN focused run passed 12/12; broader focused run passed 37/37. Browser check reached http://localhost:8080/notes but was blocked by ServerReadinessGate because no backend was available on 8000 and this worktree lacks .venv/.env. The in-app browser could not set the offline localStorage bypass due read-only evaluation context, and the javascript URL attempt was blocked by browser policy. Bandit skipped because this is frontend-only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 5 completed for the staged notes UX remediation plan. Focused tests: RED run failed as expected on old keyword copy and print label; GREEN run passed with 12/12 tests. Broader related run passed with 37/37 tests across Notes terminology/export, Notes keyword-count/management flows, and Web Clipper save flow. Browser check: Next dev server reached /notes, but this local environment has no backend on 8000 and no worktree .venv/.env to start one; the route stayed behind ServerReadinessGate. The in-app browser cannot set the app's offline localStorage bypass due its read-only evaluation context, and the javascript URL attempt was blocked by browser policy, so rendered Notes controls need verification in an environment with backend readiness or preconfigured E2E storage.
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
