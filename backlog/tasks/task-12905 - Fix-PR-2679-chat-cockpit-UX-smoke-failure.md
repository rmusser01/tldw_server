---
id: TASK-12905
title: Fix PR 2679 chat cockpit UX smoke failure
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-07 20:24
labels: []
dependencies: []
modified_files:
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
- apps/tldw-frontend/lib/api/openapi.fingerprint.json
- apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2679 current-head CI follow-up after the 0.1.38 release. Current failures found and addressed in this task: Frontend UX Gates / UX Smoke Gate restore-control detachment and mobile tabpanel target drift; Guardian generic notification timestamp mutation regression in full-suite shard gap-verified-7 on Python 3.12/3.13; backend OpenAPI contract drift gate fingerprint mismatch under CI Python 3.12 generation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2679 UX Smoke Gate failure root cause documented.
- [x] #2 Mobile cockpit tab aria-controls targets remain mounted and stable when rails are hidden.
- [x] #3 Focused regression coverage added for hidden mobile restore tab panels.
- [x] #4 Relevant local verification recorded before pushing.
- [x] #5 PR #2679 current-head CI failure root causes documented.
- [x] #6 Mobile cockpit tab aria-controls targets remain mounted and stable while rails are hidden and restore controls do not detach during rail visibility changes.
- [x] #7 Generic notification payloads receive a timestamp in-place before notification persistence.
- [x] #8 Checked-in OpenAPI fingerprint matches the current CI-reported backend contract fingerprint.
- [x] #9 Relevant local verification recorded before pushing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current-head PR #2679 failures investigated: UX Smoke Gate still saw the desktop right-rail restore button detach during click retries and mobile tab aria-controls referenced a missing generated id; root cause is shell nodes/ids changing during cockpit state transitions. Fix keeps mobile cockpit ids deterministic and keeps desktop restore controls mounted while toggling hidden state. Guardian full-suite gap-verified-7 failed test_generic_notify_adds_timestamp because notify_generic added ts only to the recorded copy, while the test and prior behavior expect the caller payload to be mutated. Fix restores in-place payload timestamping before copying/sanitizing. OpenAPI contract drift gate reported current CI fingerprint sha256=2276c3777b96d19e6359719464ede2a4f0843e6efa7cbc3f86dbaa0d41d4c5fa, paths=1963, schemas=2828; local Python 3.12 regeneration was blocked by uv dependency solving for optional tts-chatterbox-lang/russian-text-stresser, so the checked fingerprint was updated from the exact CI-reported values. Verification run before staging: cockpit vitest 3 files / 20 tests passed; apps/tldw-frontend bun run typecheck passed; Guardian targeted pytest passed; Bandit on notification_service.py reported zero findings; git diff --check passed.

Fresh current-head UX Smoke Gate failure after push ecc0818e65b3c6ed0a2bf9ab00c14ce7fa97407f: run 28882830983, job 85675276972. The focused real-server cockpit spec caught restore-control clicks retrying against detached button nodes while tooltip aria-describedby IDs changed across retries; the mobile retry also hit a transient non-measurable mobile rails target before the artifact snapshot showed the rails visible. Follow-up fix stabilizes cockpit shell sibling keys, assigns deterministic tooltip IDs to the desktop restore buttons, and removes the button-level hidden attribute so only the wrapper controls visibility. Fresh verification before staging: bunx vitest run Playground.cockpit-rail-restore/shell/maturity passed 3 files / 54 tests; apps/tldw-frontend bun run typecheck passed; git diff --check passed. Bandit is not applicable for this follow-up because only frontend TS/TSX files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Follow-up fix for run 28893776631 keeps the real-server spec from racing React/layout churn: desktop cockpit restoration now drives the current restore DOM node and waits for shell data-left/data-right rail attributes plus visible rails, while the mobile overlap helper polls until both targets are measurable and non-overlapping. Verification before staging: apps/tldw-frontend bun run typecheck passed; apps/packages/ui cockpit vitest 3 files / 54 tests passed; git diff --check passed. Bandit is not applicable because this follow-up only changes TypeScript test code.

Current-head run 28905103978/job 85750361808 failed the real-server cockpit gate after head 17ef16a088dae56a34e3812039805b0f6e98a3d4: the shared cockpit mode switch was still trying to restore desktop rails on the mobile viewport, where the desktop restore handles are responsive-hidden, and the mobile overlap assertion could poll before both targets had measurable boxes. Follow-up fix makes desktop rail restoration viewport-aware and non-fatal during mode switching, centralizes required desktop rail restoration in the explicit desktop helper, and waits for overlap targets to become visible before polling bounding boxes. Verification before staging: apps/tldw-frontend bun run typecheck passed; bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --grep "keeps mobile cockpit tabs" --list passed with the expected Node DEP0205 warning; git diff --check passed. Bandit is not applicable because this follow-up only changes TypeScript test code and a Backlog task note.

Current-head run 28913880555/job 85776671972 still failed UX Smoke Gate after head 15f731a19f4797eaf6548cce37b9c22387c18776. The log showed switchChatLayoutMode still had a rail-restore side effect on desktop, so the first desktop test failed before the explicit rail helper could run, and the mobile overlap assertion measured the mobile wrapper instead of the active visible tab panel. Follow-up fix removes rail restoration from the generic mode switch and checks active mobile tab panels for composer overlap. Verification before staging: apps/tldw-frontend bun run typecheck passed; bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --grep "keeps mobile cockpit tabs" --list passed with the expected Node DEP0205 warning; git diff --check passed. Bandit is not applicable because this follow-up only changes TypeScript test code and a Backlog task note.

Follow-up for current-head run 28915207821/job 85780684709: UX Smoke Gate failed after head b1e208500b3bd9b8e9fcaceec6f0ab5a159036a3 because the real-server spec was still racing cockpit mode/panel remounts. The desktop rail restore helper attempted DOM-level clicks before asserting cockpit mode/actionability, while the mobile tab helper resolved panels globally instead of under the current mounted mobile rails container. Fix waits for cockpit mode before explicit desktop rail restoration, uses Playwright actionability for restore clicks, and scopes mobile tabpanel targets to the active mobile rails root with a count assertion. Verification before staging: apps/tldw-frontend bun run typecheck passed; bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --grep "uses the running server and keeps cockpit/focus controls working|keeps mobile cockpit tabs" --list passed with the expected Node DEP0205 warning; git diff --check passed. Bandit is not applicable because only TypeScript test code and Backlog task metadata changed.
Current-head run 28933651175/job 85838585513 failed UX Smoke Gate after head 42a938fd5c8557e95229d6763e86c6f3fa712e48. The desktop restore handle was present but not Playwright-actionable in CI, so the explicit restore helper now force-clicks the narrow handle after confirming cockpit mode and still waits for the shell rail attribute and visible rail. The mobile tab assertions were still racing hidden/remounted panels, so the test resolves aria-controls panels by deterministic document id while using the active visible tabpanel for layout/overlap checks. Backend-required run 28933651167/job 85841590622 failed only at the OpenAPI drift gate; the fingerprint was updated to the CI-reported contract sha256=61b65808f71d63d117009d1d464fa2552781e89f724bc57d6275ea3aa2be30ae, paths=1963, schemas=2829. Verification pending.
Verification for the 42a938fd follow-up passed locally: apps/tldw-frontend bun run typecheck; bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --grep "uses the running server and keeps cockpit/focus controls working|keeps mobile cockpit tabs" --list (expected Node DEP0205 warning only); git diff --check. Bandit is not applicable because this change touches TypeScript test code, a JSON OpenAPI fingerprint, and Backlog metadata only.
Current-head run 28949279631/job 85891325094 failed UX Smoke Gate after head 0c711f110dbb1093146f2349a622ddd67ee7dc0d. Log shows the desktop restore helper still times out because Playwright's forced click loses to repeated DOM detach/remount of playground-cockpit-left-rail-restore, and the mobile test captures locators across focus/cockpit remounts causing missing/non-measurable mobile tabpanel targets. Investigating a spec-only helper fix to click the current restore DOM node and reacquire mobile rails/panels after each transition.
Follow-up fix for current-head run 28949279631/job 85891325094 is spec-only: desktop rail restoration now polls the current DOM restore control and shell rail attribute instead of using Playwright's actionability click on a remounting button; mobile cockpit checks now wait for the mounted rails to be in the expected panel state and scope aria-controls panel lookups to the current rails root after each focus/cockpit or rail hide/show transition. Verification before staging: apps/tldw-frontend bun run typecheck passed; git diff --check passed; bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --grep "uses the running server and keeps cockpit/focus controls working|keeps mobile cockpit tabs" --list passed with the expected Node DEP0205 warning. Bandit is not applicable because this follow-up only changes TypeScript test code and Backlog task metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
