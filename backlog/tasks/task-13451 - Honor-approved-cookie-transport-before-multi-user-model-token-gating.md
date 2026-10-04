---
id: TASK-13451
title: Honor approved cookie transport before multi-user model token gating
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TldwModelsService checks multi-user accessToken before the shared cookie-transport approval helper. Honor an approved cookie transport first, without broadening the helper contract or bypassing live protected-profile authorization. This is a separate bounded follow-up to completed TASK-13408.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A helper-approved multi-user cookie transport discovers models without a JavaScript access token after protected-profile authorization.
- [x] #2 Unapproved credential-free configurations and failed protected-profile authorization remain fail-closed, including cached catalogs.
- [x] #3 Cookie catalogs remain isolated from token and unauthenticated cache namespaces and are not persisted.
- [x] #4 Native red/green regression tests, scoped checks, and public PR evidence are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: reproduce the admission-order bug at the existing transport approval boundary; move that check before the token branch; run focused native model and transport regressions; review and publish a separate PR. ADR assessment: no new ADR; routine correction reuses existing cookie approval and cache/authentication boundaries. Related historical task TASK-13408 is unchanged. Base origin/dev c95e41fc62a07e55fd74052023826c71b2a2c789.
Verification: original native model suite 54/54 passed. Eight new helper-boundary regressions failed against the original admission order; all 62 model tests passed after the minimal cookie-first move. Broader native Vitest: 125/125 tests across 7 suites (model service, chat-model fetch/filter, model normalization, browser networking, quickstart request core and auth). The actual canonical helper remains active for all pre-existing cases; only the new transport-contract tests override its approval result. Canonical unapproved multi-user/no-token, advanced, cross-origin, and missing-auth-source rejection remains intact. Deferred protected-profile authorization proves metadata is not requested before admission. Profile failures 401/403/503 withhold an already populated memory catalog; cookie results do not persist and do not reuse token/key/none namespaces.
Scoped ESLint: 0 errors, 1 unchanged baseline warning. Scoped TypeScript comparison: 9 baseline and 9 current diagnostics, no new diagnostics. git diff --check passed. Bandit attempted in project venv but unavailable (No module named bandit); touched code is TypeScript, outside Bandit coverage. Self-review found no further source changes necessary; independent PR review remains pending. Production diff is 2 inserted lines and 1 deleted line in TldwModels.ts; protected-profile authorization, cookie approval helper, and cache implementation are unchanged. Source SHA256 before: 36cb2954e38538bfc441452b08c46539af19b188dd9aea3d8df89448793ddf94; after: 99cd70edc8ba5a8d6d1bbe3d7acce61a79f6a6502c9dadc64a09ca66bf663a47.
Published separate public PR https://github.com/rmusser01/tldw_server/pull/3185 on codex/model-cookie-first-gate against dev. Implementation and regression criteria are verified; task remains In Progress until review, required CI, and normal integration. No merge or bypass attempted. The human-written Change summary gate is still pending for this separate PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Minimal production change honors an already approved cookie transport before requiring a JavaScript JWT. Protected-profile authorization and non-persistent isolated cookie caches remain unchanged. Native regression evidence: 54 baseline passed; 8 new red failures; 62 focused green and 125 broader passed. Public PR3185 is pending review and required checks.
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
