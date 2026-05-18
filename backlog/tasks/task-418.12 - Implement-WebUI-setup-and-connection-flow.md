---
id: TASK-418.12
title: Implement WebUI setup and connection flow
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-18 04:36
labels:
- ux
- webui
- extension
- implementation
- setup
dependencies: []
documentation:
- Docs/superpowers/plans/2026-05-17-webui-setup-connection-flow-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md
parent_task_id: TASK-418
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Task 3 WebUI/extension UX remediation slice for first-run setup, home resolution, auth/account placeholders, hosted-only routes, redirects, and 404 recovery. Preserve existing self-host and hosted product intent, keep OnboardingConnectForm and route aliases, avoid backend API changes, and ground changes in the approved setup connection-flow plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Connection UX state matrix is locked with focused tests or documented as already covered.
- [x] #2 / route behavior is explicit for hosted, first-run incomplete, character-chat intent, first-run complete, and degraded states.
- [x] #3 /setup renders as a setup-only shell with one semantic setup heading and no chat/sidebar chrome.
- [x] #4 Self-host /login and hosted /login behavior are explicit and covered by tests.
- [x] #5 Hosted-only OSS placeholders for signup/account/billing/auth recovery routes expose route context and appropriate primary actions instead of defaulting to chat.
- [x] #6 /profile, /config, /privileges, and /404 recovery labels/actions are explicit and covered by tests.
- [x] #7 Browser or Playwright QA evidence is recorded for setup/recovery routes, with environment gaps documented.
- [x] #8 No backend auth API changes, broad shell redesign, or route renaming are included.
- [x] #9 Focused unit/E2E checks and git diff --check are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Task 1 connection state matrix: baseline focused tests passed after installing UI dependencies in the clean worktree. Added pure deriveConnectionUxState matrix coverage in src/types/__tests__/connection.test.ts while preserving existing store-side onboarding action coverage in src/store/__tests__/connection.test.ts. Verification: bunx vitest run src/types/__tests__/connection.test.ts src/store/__tests__/connection.test.ts passed 2 files / 27 tests.

Task 2 / resolver behavior: added src/routes/__tests__/option-index.setup-flow.test.tsx to lock hosted home, first-run onboarding, character-chat return intent, completed-first-run companion home, automatic beginOnboarding for unconfigured first-run users, and checkOnce refresh for completed users. Existing option-index implementation satisfied the matrix, so no product route code changed. Verification: bunx vitest run src/routes/__tests__/option-index.setup-flow.test.tsx src/routes/__tests__/core-route-identity.test.tsx passed 2 files / 13 tests.

Task 3 /setup shell: added setup-only shell assertions to OnboardingConnectForm.design-system.test.tsx and changed option-setup.tsx so the route owns exactly one semantic h1 (`Setup Wizard`) while the setup-required StatePanel handles the connection action as `Connect your server`. Verification: bunx vitest run src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx src/routes/__tests__/core-route-identity.test.tsx passed 2 files / 11 tests.

Task 4 login and hosted-only placeholders: added unit coverage for self-host /login redirect/recovery, hosted /login auth form, hosted-only account/billing/signup/auth placeholder pages in self-host and hosted deployment modes, and RoutePlaceholder settings-child CTA suppression. Patched self-host /login to render RouteRedirect to /settings/tldw with preserved query/hash instead of a blank interim page. Added HostedOnlyRoutePlaceholder so OSS hosted-only pages use /settings/tldw + Open Local Auth Settings in self-host mode and /login + Open Login in hosted mode, with plannedPath route context on every owned placeholder. Expanded hosted-placeholder-routes e2e coverage to include magic-link and verify-email and to assert deployment-specific primary CTAs. Verification: bunx vitest run __tests__/navigation/login-page.test.tsx __tests__/navigation/hosted-placeholder-pages.test.tsx __tests__/navigation/route-placeholder-component.test.tsx passed 3 files / 23 tests.

Task 4 additional focused verification: bunx vitest run __tests__/navigation/login-page.test.tsx __tests__/navigation/hosted-placeholder-pages.test.tsx __tests__/navigation/route-placeholder-component.test.tsx __tests__/navigation/route-redirect-component.test.tsx passed 4 files / 28 tests; git diff --check passed for the slice.

Task 5 profile/config/privileges/404 recovery: added page-level coverage for /profile and /config settings recovery, privileges redirect source/destination context, and 404 keyboard/action recovery. Patched /privileges to use route-specific redirect title/description, changed 404 and shared redirect/root fallback labels from chat-specific language to Open Home, and mirrored the 404 home label in the extension route shell. Verification: bunx vitest run __tests__/navigation/login-page.test.tsx __tests__/navigation/hosted-placeholder-pages.test.tsx __tests__/navigation/route-placeholder-component.test.tsx __tests__/navigation/route-redirect-component.test.tsx __tests__/navigation/not-found-page.test.tsx passed 5 files / 34 tests; rg found no remaining WebUI Go to Chat/not-found-go-chat/route-redirect-go-chat recovery strings; git diff --check passed.

Task 6 browser QA: added e2e/workflows/setup-connection-flow.spec.ts covering / and /setup first-run states plus /login, /account, /signup, /billing, and 404 recovery at 1440px desktop and 390px mobile. The first Playwright run exposed a real Next shell gap: /setup still rendered the WebUI header/sidebar because the outer Next app shell did not treat /setup as setup-only. Patched pages/_app.tsx to bypass app gates and hide header/sidebar for /setup, and added app-layout unit coverage for that behavior. Verification: bunx vitest run __tests__/app/app-layout.test.tsx __tests__/navigation/login-page.test.tsx __tests__/navigation/hosted-placeholder-pages.test.tsx __tests__/navigation/route-placeholder-component.test.tsx __tests__/navigation/route-redirect-component.test.tsx __tests__/navigation/not-found-page.test.tsx passed 6 files / 45 tests. Browser QA: bunx playwright test e2e/workflows/setup-connection-flow.spec.ts --reporter=line passed 4 tests at desktop/mobile. Initial sandbox run failed to bind port 8080 with EPERM; reran with approved escalation for local Playwright dev server.

Final verification gate: package UI focused unit gate passed (bunx vitest run src/types/__tests__/connection.test.ts src/store/__tests__/connection.test.ts src/routes/__tests__/option-index.setup-flow.test.tsx src/components/Option/Onboarding/__tests__/OnboardingConnectForm.design-system.test.tsx src/routes/__tests__/core-route-identity.test.tsx, 5 files / 44 tests; existing react-i18next warning only). Frontend focused unit gate passed (bunx vitest run __tests__/app/app-layout.test.tsx __tests__/navigation/login-page.test.tsx __tests__/navigation/hosted-placeholder-pages.test.tsx __tests__/navigation/route-placeholder-component.test.tsx __tests__/navigation/route-redirect-component.test.tsx __tests__/navigation/not-found-page.test.tsx, 6 files / 45 tests). Playwright browser gate passed after selector fixes (bunx playwright test e2e/login.spec.ts e2e/workflows/setup-connection-flow.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts --reporter=line, 14 tests). git diff --check passed. Scope check: no backend APIs, backend auth code, route renames, or broad visual redesign included. Bandit is not applicable to this frontend-only touched scope; no Python backend files were modified.

Final known skips/blockers: none for the touched frontend scope. Bandit was documented as not applicable because no Python backend files were changed. Playwright needed escalated local dev-server binding after sandbox EPERM on port 8080; the rerun passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the setup/connection-flow remediation slice and completed PR #1837 review fixes. Review follow-up moved the /setup route labels through the existing option i18n namespace, added English locale entries for runtime and extension locale bundles, and made the setup/recovery Playwright route QA deployment-aware for hosted and self-host/default modes. Verification after review fixes: package UI focused Vitest gate passed 5 files / 44 tests; frontend focused Vitest gate passed 6 files / 45 tests; i18n namespace smoke passed 1 file / 1 test; default/self-host Playwright route gate passed 12 tests; hosted-mode Playwright route gate passed 12 tests; git diff --check passed. Bandit remains not applicable because the touched scope is frontend TypeScript/JSON/backlog only.
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
