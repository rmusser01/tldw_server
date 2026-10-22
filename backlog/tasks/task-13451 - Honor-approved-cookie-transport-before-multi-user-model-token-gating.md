---
id: TASK-13451
title: Honor approved cookie transport before multi-user model token gating
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the generic model-client admission order: an already approved cookie transport must reach protected-profile authorization before multi-user JavaScript-token gating. The canonical helper remains restricted to exact-origin single-user quickstart cookies; this task does not add native tokenless multi-user or backend cookie authentication. Approval-override tests cover only the consumer contract. Separate bounded follow-up to completed TASK-13408.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A helper-approved multi-user cookie transport discovers models without a JavaScript access token after protected-profile authorization.
- [x] #2 Unapproved credential-free configurations and failed protected-profile authorization remain fail-closed, including cached catalogs.
- [x] #3 Cookie catalogs remain isolated from token and unauthenticated cache namespaces and are not persisted.
- [x] #4 Native red/green regression tests, scoped checks, and public PR evidence are recorded.
- [x] #5 The actual native helper, without approval substitution, admits exact-origin single-user quickstart cookies and rejects tokenless multi-user cookies.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: reproduce the admission-order bug at the existing transport approval boundary; move that check before the token branch; run focused native model and transport regressions; review and publish a separate PR. ADR assessment: no new ADR; routine correction reuses existing cookie approval and cache/authentication boundaries. Related historical task TASK-13408 is unchanged. Base origin/dev c95e41fc62a07e55fd74052023826c71b2a2c789.
Verification: original native model suite 54/54 passed. Eight new helper-boundary regressions failed against the original admission order; all 62 model tests passed after the minimal cookie-first move. Broader native Vitest: 125/125 tests across 7 suites (model service, chat-model fetch/filter, model normalization, browser networking, quickstart request core and auth). The actual canonical helper remains active for all pre-existing cases; only the new transport-contract tests override its approval result. Canonical unapproved multi-user/no-token, advanced, cross-origin, and missing-auth-source rejection remains intact. Deferred protected-profile authorization proves metadata is not requested before admission. Profile failures 401/403/503 withhold an already populated memory catalog; cookie results do not persist and do not reuse token/key/none namespaces.
Scoped ESLint: 0 errors, 1 unchanged baseline warning. Scoped TypeScript comparison: 9 baseline and 9 current diagnostics, no new diagnostics. git diff --check passed. Bandit attempted in project venv but unavailable (No module named bandit); touched code is TypeScript, outside Bandit coverage. Self-review found no further source changes necessary; independent PR review remains pending. Production diff is 2 inserted lines and 1 deleted line in TldwModels.ts; protected-profile authorization, cookie approval helper, and cache implementation are unchanged. Source SHA256 before: 36cb2954e38538bfc441452b08c46539af19b188dd9aea3d8df89448793ddf94; after: 99cd70edc8ba5a8d6d1bbe3d7acce61a79f6a6502c9dadc64a09ca66bf663a47.
Published separate public PR https://github.com/rmusser01/tldw_server/pull/3185 on codex/model-cookie-first-gate against dev. Implementation and regression criteria are verified; task remains In Progress until review, required CI, and normal integration. No merge or bypass attempted. The human-written Change summary gate is still pending for this separate PR.
Qodo finding 6dbb452e-9278-4ba3-afb1-c4a3ff3598a2 / inline comment 4179450707 correctly identified that the real helper cannot approve multi-user cookies. Scope is explicitly corrected: generic consumer precedence only, not new native multi-user authentication. Added actual-helper tests without an approval override for native credential-free single-user cookies (ordinary and forced refresh) and native multi-user/no-token rejection. Existing approval-override cases are explicitly named and documented as consumer-contract tests. Production source and prepared +2/-1 patch remain unchanged; no native auth helper/backend changes are proposed.
Qodo follow-up verification: 65/65 focused native model-service tests passed, including three new actual-helper cases with no approval substitution. Broader native matrix passed 128/128 tests across seven suites. Scoped ESLint remains 0 errors/1 unchanged warning; TypeScript comparison remains 9 baseline/9 current diagnostics, none introduced. git diff --check passed. Production SHA256 remains 99cd70edc8ba5a8d6d1bbe3d7acce61a79f6a6502c9dadc64a09ca66bf663a47 and source-only +2/-1 patch is unchanged. PR scope and test names now explicitly distinguish generic approved-transport consumer behavior from native single-user-only approval. Resolving the exact Qodo thread with this scope correction and verified actual-helper evidence, not claiming new native multi-user support.
2026-10-19 requester authorized PR3185 completion including review and CI. Original head65/65 focused tests pass. Preparing an isolated current-dev integration candidate while PR3197 qualifies; no remote refresh until this PR is next in merge order. Independent requesting-code-review of original patch found no actionable findings, but current-dev cache schema5, authorization scopes, cookie principal identity and cooldown metadata require regression fixture updates. Scope remains admission ordering and tests; no new ADR or native multi-user cookie support.
Current-dev integration verification: original head65/65 focused; unadapted integration80passed2failed because forced refresh now writes cooldown metadata. Updated only new fixtures to schema5/authorization scope, real profile identity, and exact allowed cooldown writes (no catalog persistence). Adapted service82/82 and seven connected suites135/135 pass. Removing only production correction reproduces8failed74passed; restored correction82/82 passes. Scoped ESLint0errors/1pre-existing unused inputMods warning. Scoped TypeScript baseline/candidate40 identical diagnostics after normalizing shifted line numbers; no new diagnostics and no clean full-workspace typecheck claim. Bandit invoked from project venv on touched scope; TypeScript semantic security is outside its coverage. No ADR required; existing authorization and cache policy preserved. Cleanup-batch human Change summary waiver explicitly granted by requester in originating chat on2026-10-19.
2026-10-20 publication qualification: rebased the independently approved5998522fbc candidate onto dev6a2cb72577 after PR3197 merged. Range-diff preserves the implementation patch exactly; no conflict resolutions. Fresh focused native model tests82/82 and canonical task contracts4/4 pass on this base. Prior seven-suite connected run135/135, eight red regression failures without the production correction, scoped lint and identical40 baseline/candidate TypeScript diagnostics remain applicable because the frontend patch and tooling are unchanged. Implementation and local verification are complete; normal PR3185 integration still requires all seven exact-head hosted gates at publication. No bypass or clean full-workspace typecheck claim. Added TASK13866 final cleanup record in the same completion batch.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the minimal generic model-client cookie approval precedence correction, preserving native single-user-only approval, protected-profile authorization, and isolated non-persistent cookie catalogs. Native actual-helper and consumer-contract regression coverage is explicit. Current-base focused82/82 and task-format4/4 pass; seven connected suites135/135 pass and removing the correction reproduces eight failures. Scoped lint has no errors and TypeScript introduces no diagnostics over40 baseline findings. Independent requesting-code-review approved the implementation; hosted checks and integration are tracked by PR3185. The requester waived the cleanup-batch human Change summary gate. TypeScript semantic security remains outside Bandit coverage.
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
