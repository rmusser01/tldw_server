---
id: TASK-13385
title: Recover pending VN generation commands after reload
status: In Progress
assignee: []
created_date: 2026-09-27 15:25
updated_date: 2026-09-27 17:31
labels:
- vn-assets
- frontend
- recovery
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/2021
- https://github.com/rmusser01/tldw_server/pull/3015
- https://github.com/rmusser01/tldw_server/pull/3028
documentation:
- Docs/Design/VN_PENDING_COMMAND_RECOVERY.md
- IMPLEMENTATION_PLAN_vn_command_recovery.md
- IMPLEMENTATION_PLAN_vn_command_recovery_pr_review.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Requester-approved continuation of #2021 after PR #3015 / TASK-13378. Persist unresolved Start/Retry payloads and idempotency keys in tab-scoped sessionStorage, bind recovery to server and verified principal, expose explicit same-request recovery after reload without automatic generation. No backend Jobs/output semantics changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reload restores unresolved Start/Retry commands; explicit recovery uses the exact original payload/key without automatic POST.
- [x] #2 Server/account boundaries, logout and stale callbacks cannot replay another principal command; storage has no credentials or prompts.
- [x] #3 Acknowledged commands clear; changed Retry provenance cannot replace unresolved requests; malformed/unavailable storage is visible.
- [x] #4 Focused storage/workbench regressions, full VN frontend suite, typecheck, scoped lint and browser verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Inventory across registered worktrees found maximum ID 13384. CLI auto-allocation selected already-used 13379; the newly created duplicate was archived through CLI without altering the unrelated Chat Macros task. Explicit unused ID 13385 is the authoritative VN slice record.

Approved design and three-stage plan added. Branch codex/vn-command-recovery reuses the VN worktree on dev f2830058d5; prior completion commit retained on original branch and cherry-picked as 6bbe71ab9e. Frontend-only closed session journal, verified account lifecycle and explicit same-payload/key recovery implemented. No automatic generation. Storage failures fail closed; unreadable data requires warned discard confirmation. Account changes reject pre-send replay; epochs and per-command tokens fence stale responses/locks; logout invalidates entries. Canonical profile 404/410 permits existing authenticated /auth/me compatibility path, never cached identity.

Red evidence: original Retry remount could not recover before implementation. Account/lifecycle tests failed with an extra cross-account POST and an incorrectly released newer lock before fixes. Legacy single-user verification could not enable generation before fallback. Initial Start remount invocation also hit a slow initial-load timeout, not claimed as recovery bug evidence. Existing mounted retry tests adapted to explicit recovery. Latest all 67 VN frontend tests passed; scoped ESLint zero warnings; typecheck passed before final compatibility follow-up and is rerunning. Browser verification in progress. No backend, dependency or unrelated changes.

Independent read-only review found two concrete recovery defects. Documented pre-admission VN 409 errors retained an unrecoverable command, and focus/pageshow could invalidate initial detail loading without starting a replacement. Two rejection regressions failed journal length 1 versus 0; real fetch-backed Retry normalization failed to preserve the documented code. Explicit deferred-read tests failed generation GET count 1 versus 2 before the fix (initial wait-based versions timed out; count assertion gives direct evidence). Fixed only known POST Start/Retry HTTP409 code envelopes using a closed canonical-message allowlist; unknown/in-progress conflicts and 5xx stay ambiguous, raw nested details remain private. Successful boundary verification restarts details while pre-send verification does not. Twelve targeted regressions/security controls passed, then all 71 VN tests plus 25 fetch-client tests passed (96 total, 12.21s), typecheck and scoped ESLint zero warnings passed. Follow-up independent review pending. Browser full Next workbench with isolated HTTP fixtures passed Start and Retry lost-response/reload/exact-body-key replay/acknowledgement cleanup, no automatic POST, Retry original source41 despite status42, desktop1440x1000 and mobile390x844 non-overlapping recovery controls. No real generation, live GPU or backend acceptance claimed. Bandit on unchanged VN Python baseline returned zero findings/errors; Bandit does not support the touched TypeScript. No dependency policy changes. Temporary shared UI dependency link will be removed after final checks. The colliding local CLI-created archived13379 record is not staged or published.

Follow-up review found the real VN contract uses detail.code, not provider detail.error_code. The earlier fixture and terminal browser claim used the wrong field; those did not establish actual VN rejection handling and are corrected here rather than treated as valid contract evidence. Backend-shaped Retry fixture genuinely failed with missing ApiError.errorCode before correcting the parser. API detail type now admits optional code; narrow method/path/status/code allowlist retained. Added provider-shaped-field negative control. Final 71 VN plus26 fetch-client tests passed (97 total,13.23s), typecheck and scoped ESLint0warnings passed. Corrected terminal browser verification pending. Focus/pageshow lifecycle fix accepted by reviewer with no additional lifecycle findings. Temporary UI link removal exposed isolated frontend resolution failure (react/jsx-dev-runtime); checking Next supported NODE_PATH process-only resolution so preview can run without repository dependency links. No shared environment/dependency-policy edits.

Final follow-up reviewer found no remaining actionable defects after detail.code correction; 14 independent targeted transport checks passed. Process-only NODE_PATH preview resolution instead triggered existing i18next-icu/intl-messageformat ESM resolution failure. Restored only the known-working link to the existing shared UI installation for the live local preview; this dependency symlink remains uncommitted, no package installs/versions or shared environments changed. Prior existing apps and frontend node_modules links remain untouched. Normal preview points to API8000 with no fake credential; isolated browser intercepts all fixture HTTP calls. Corrected browser validation still pending; failed module-setup attempts are not feature regression evidence.

Final corrected full-browser check used backend detail.code/message/details/retryable and the real API client: exactly one Retry POST, canonical public message, empty journal, Start enabled, raw message/details absent and no pageerror after clean reload. Normal preview now serves /vn-assets HTTP200 at http://127.0.0.1:8128, APIbase8000, no fake environment credential; existing dependency link retained locally only while preview runs. Isolated browser sessions closed. Final independent review clear. All three plan stages complete; own implementation plan is removed per repository completion policy, preserved historically in local implementation commit3f32c5be34. No full-project frontend/backend or live-GPU acceptance run; scoped suite covered71 VN and26 real fetch-client cases,97passed13.23s, tsc0, scopedESLint0warnings and diffcheck0. Bandit only unchanged VN Python baseline0findings0errors6764lines; unsupported TypeScript assessed by scoped lint, privacy tests and independent review. No backend/Jobs/output changes. New feature not pushed or merged; integration needs requester choice. Colliding CLI-created archive13379 and preview-only UI link excluded from commits.

Requester selected option2 (push and create PR), not merge. Fetched dev718c191082f1d6372fb6fe000ac763dcc07ffcbd; its four new commits touch only unrelated audio resampling/test and task13304/13382 records. Unpublished branch rebased conflict-free from old8b12335c171aaa3a40bc38a2076c4f89962e674f; range-diff shows all3patches unchanged and unrelated base files match dev byte-for-byte. Fresh97tests passed11.44s, typecheck0, scopedESLint0warnings and diffcheck0 on rebased head802bef1194309bb7543e7a0ccbf2395527f9da03. No remote branch or existing PR found; publish normally without force. Prior independent review and browser evidence apply to unchanged source patches, not a claim of completed new-head hosted review. Human-written Change summary and live CI remain merge gates; no merge or automation requested.

Published requester-approved branch codex/vn-command-recovery by normal push, verified local/remote head01a964685dbe0b667ac85c898ab97496c63d6a06, and created PR3028 against dev: https://github.com/rmusser01/tldw_server/pull/3028. Attached PR to this chat. Human-owned Change summary explicitly pending; AI-authored summary is not a substitute. No merge or auto-merge enabled. Task implementation remains Done; hosted CI/review and human summary are PR integration gates, not claims of completion. Final tracking-only commit will be pushed normally after verifying remote ownership.

Requester explicitly authorized protected rebase of PR3028 onto latest dev, scoped remediation of posted Qodo findings and a normal merge only after complete current-head review and all live required gates. Human-written Change summary was provided and published verbatim before this authorization; design approval was never merge authorization. Verified clean owned local/remote head5de2ed11671f593968a86aef24d3be0422488f75; latest dev35d6dd90d4c3b703a753efdbd926e30af4f9eac5 contains only unrelated MCP tests/task records. Rebase was conflict-free and all five prior patches are unchanged in range-diff. Local preview-only UI dependency link and CLI-created colliding archive13379 remain untracked and excluded. PR review plan: IMPLEMENTATION_PLAN_vn_command_recovery_pr_review.md. Fresh verification and new-head reviews remain pending.

Fresh verification after rebase onto dev35d6dd90d4c3b703a753efdbd926e30af4f9eac5: all97 VN/frontend real fetch-client tests passed (15.28s), frontend typecheck passed, scoped ESLint zero warnings and diff checks passed. Unrelated MCP tests/task13358/task13380/base workflows match dev byte-for-byte. Bandit unchanged VN Python baseline returned zero findings/errors; it does not scan touched TypeScript. Only scoped design wording distinguishes approved design from human-summary/current-head review/CI merge gates; no production behavior changed. Qodo discussion4116043591 was posted before the human summary, while merge/auto-merge were intentionally disabled. Human summary now published verbatim, explicit gated merge authorization received; reply and exact-new-head hosted reviews will follow protected publication. Task remains In Progress until verified merge.
Protected rebase published as581979a4a87543aedd73ac0f11667c2dc253e73b with explicit lease on full remote5de2ed11671f593968a86aef24d3be0422488f75; head/base/clean tracked checkout and verbatim human summary reverified. Qodo exact-head full-diff reassessment comment5858115068 found no production defects and accepted the human-gate clarification. It flagged duplicate task final-summary end markers and stale review-plan statuses; synchronize these through official Backlog mutation and own plan edit. CodeRabbit full review trigger5858113221 is running and exact-head CI/license audit pending. No merge attempted. Task remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented requester-approved tab-scoped pending Start/Retry command recovery. Exact original payload/key survives reload and changed Retry provenance; recovery is explicit with no automatic POST. Verified account/server boundaries and per-command fencing protect stale callbacks; malformed/unavailable storage fails closed. Known VN pre-admission conflicts unlock the pack while ambiguous responses retain recovery. Qualified locally with97passing tests, typecheck, scoped lint, unchanged-backend Bandit baseline and desktop/mobile full-browser smoke with isolated HTTP fixtures. No remaining actionable production review findings in Qodo's complete exact-head reassessment. Current-head CodeRabbit review, hosted CI and verified merge remain pending; integration and live backend/GPU acceptance are not claimed.
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
