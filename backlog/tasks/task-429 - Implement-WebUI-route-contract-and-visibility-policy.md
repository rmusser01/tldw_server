---
id: TASK-429
title: Implement WebUI route contract and visibility policy
status: Done
labels:
- ux
- webui
- extension
- routes
- implementation
priority: high
modified_files:
- apps/packages/ui/src/routes/route-metadata.ts
- apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.visibility.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-availability.test.ts
- apps/packages/ui/src/components/Common/CommandPalette.tsx
- apps/packages/ui/src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx
- apps/tldw-frontend/e2e/smoke/page-inventory.ts
- apps/tldw-frontend/e2e/smoke/route-contract-stage2.spec.ts
- backlog/tasks/task-429 - Implement-WebUI-route-contract-and-visibility-policy.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first code slice from the WebUI/extension UX remediation audit: add a canonical route metadata contract for audited root routes, validate route registry and extension sidepanel availability against it, align smoke inventory expectations, and fix the command palette Chat target. Scope maps audit findings F1, F8, F12, F17, and F18. No page-level visual redesign or backend API changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typed route metadata exists for all 74 audited root/top-level routes.
- [x] #2 Route metadata tests cover required fields, aliases, visibility classifications, and sidepanel availability.
- [x] #3 Route registry and smoke inventory validation pass against metadata.
- [x] #4 Command palette `Go to Chat` targets `/chat` truthfully.
- [x] #5 Focused Vitest and Playwright route-contract checks are run or skips are documented.
- [x] #6 Audited root-route smoke inventory entries are generated from route metadata, with manual entries retained only for child/special routes.
- [x] #7 Command palette audited navigation targets resolve through route metadata.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a pure route metadata contract for the audited root-route set plus
extension sidepanel/debug sidepanel rows. Validation tests cover metadata
coverage, required fields, alias canonicalization, option-route ownership
through shared registry or Next pages, sidepanel availability, and hidden policy
for hosted/debug/legacy routes.

Smoke inventory now includes `/integrations` and `/scheduled-tasks` because both
are audited root routes with smoke-included metadata. Fixed the command palette
Chat command to navigate to `/chat` instead of `/`.

Verification:

- `bunx vitest run src/routes/__tests__/route-metadata.coverage.test.ts src/routes/__tests__/route-registry.visibility.test.ts src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx` passed.
- `TLDW_WEB_AUTOSTART=false bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --grep "route metadata smoke policy" --reporter=line` passed.
- `git diff --check` passed.
- Full `bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --reporter=line` was attempted with elevated local-port access. The dev server started, but the route browser assertion timed out at `/connectors` because the app stayed in `Retrying server readiness` without a backend. The metadata-policy check in that same file passed.
- Full `tsc --noEmit --project tsconfig.json` was attempted using the frontend TypeScript binary. It failed on existing repo-wide type errors. One new readonly availability type issue exposed by that run was fixed before final focused tests passed.

Bandit skipped because no Python/backend code changed.

Follow-up ownership cleanup:

- Audited root-route smoke entries are now generated from `ROUTE_METADATA`.
- `page-inventory.ts` keeps explicit entries only for child routes, auth/billing children, admin children, and other routes outside the audited metadata contract.
- Metadata-owned routes marked `smoke: "exclude"` are excluded from the smoke inventory, and `manual` routes become skipped page entries with their metadata rationale.
- The route-contract metadata policy now rejects duplicate smoke paths and excluded metadata routes appearing in the inventory.
- Command palette route commands now resolve through `getCommandPaletteTarget`; `Go to MCP Hub` now targets `/mcp-hub` instead of `/settings/mcp-hub`.
- The new duplicate-path check exposed a pre-existing duplicate `/chat/settings` entry in `page-inventory.ts`; the duplicate was removed.

Follow-up verification:

- `bunx vitest run src/routes/__tests__/route-metadata.coverage.test.ts src/routes/__tests__/route-registry.visibility.test.ts src/routes/__tests__/route-registry.sidepanel-availability.test.ts src/components/Common/__tests__/CommandPalette.shortcuts.test.tsx` passed with 18 tests.
- `TLDW_WEB_AUTOSTART=false bunx playwright test e2e/smoke/route-contract-stage2.spec.ts --grep "route metadata smoke policy" --reporter=line` passed.
- `git diff --check` passed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first WebUI/extension UX remediation code slice for route governance. Added `route-metadata.ts` with typed route identity, visibility, smoke, command, and sidepanel availability metadata for the 74 audited root routes and sidepanel-only/debug routes. Added focused tests for metadata coverage, registry ownership, sidepanel availability, command palette routing, smoke inventory ownership, duplicate path prevention, and excluded-route prevention. The smoke inventory now derives audited root-route entries from metadata and keeps explicit entries for child/special routes only. Fixed `Go to Chat` to target `/chat` and `Go to MCP Hub` to target `/mcp-hub` through metadata-owned command targets. Verification passed for the focused Vitest route/command suite, the metadata-only Playwright smoke-policy check, and `git diff --check`. Full browser route-contract sweep and full package typecheck were attempted but are blocked by existing environment/repo issues documented in implementation notes.
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
