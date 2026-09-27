---
id: TASK-13385
title: Recover pending VN generation commands after reload
status: In Progress
assignee: []
created_date: '2026-09-27 15:25'
updated_date: '2026-09-27 15:57'
labels:
  - vn-assets
  - frontend
  - recovery
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
  - 'https://github.com/rmusser01/tldw_server/pull/3015'
documentation:
  - Docs/Design/VN_PENDING_COMMAND_RECOVERY.md
  - IMPLEMENTATION_PLAN_vn_command_recovery.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Requester-approved continuation of #2021 after PR #3015 / TASK-13378. Persist unresolved Start/Retry payloads and idempotency keys in tab-scoped sessionStorage, bind recovery to server and verified principal, expose explicit same-request recovery after reload without automatic generation. No backend Jobs/output semantics changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reload restores unresolved Start/Retry commands; explicit recovery uses the exact original payload/key without automatic POST.
- [ ] #2 Server/account boundaries, logout and stale callbacks cannot replay another principal command; storage has no credentials or prompts.
- [ ] #3 Acknowledged commands clear; changed Retry provenance cannot replace unresolved requests; malformed/unavailable storage is visible.
- [ ] #4 Focused storage/workbench regressions, full VN frontend suite, typecheck, scoped lint and browser verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Inventory across registered worktrees found maximum ID 13384. CLI auto-allocation selected already-used 13379; the newly created duplicate was archived through CLI without altering the unrelated Chat Macros task. Explicit unused ID 13385 is the authoritative VN slice record.

Approved design and three-stage plan added. Branch codex/vn-command-recovery reuses the VN worktree on dev f2830058d5; prior completion commit retained on original branch and cherry-picked as 6bbe71ab9e. Frontend-only closed session journal, verified account lifecycle and explicit same-payload/key recovery implemented. No automatic generation. Storage failures fail closed; unreadable data requires warned discard confirmation. Account changes reject pre-send replay; epochs and per-command tokens fence stale responses/locks; logout invalidates entries. Canonical profile 404/410 permits existing authenticated /auth/me compatibility path, never cached identity.

Red evidence: original Retry remount could not recover before implementation. Account/lifecycle tests failed with an extra cross-account POST and an incorrectly released newer lock before fixes. Legacy single-user verification could not enable generation before fallback. Initial Start remount invocation also hit a slow initial-load timeout, not claimed as recovery bug evidence. Existing mounted retry tests adapted to explicit recovery. Latest all 67 VN frontend tests passed; scoped ESLint zero warnings; typecheck passed before final compatibility follow-up and is rerunning. Browser verification in progress. No backend, dependency or unrelated changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Independent read-only review found two concrete recovery defects. Documented pre-admission VN 409 errors retained an unrecoverable command, and focus/pageshow could invalidate initial detail loading without starting a replacement. Two rejection regressions failed journal length 1 versus 0; real fetch-backed Retry normalization failed to preserve the documented code. Explicit deferred-read tests failed generation GET count 1 versus 2 before the fix (initial wait-based versions timed out; count assertion gives direct evidence). Fixed only known POST Start/Retry HTTP409 code envelopes using a closed canonical-message allowlist; unknown/in-progress conflicts and 5xx stay ambiguous, raw nested details remain private. Successful boundary verification restarts details while pre-send verification does not. Twelve targeted regressions/security controls passed, then all 71 VN tests plus 25 fetch-client tests passed (96 total, 12.21s), typecheck and scoped ESLint zero warnings passed. Follow-up independent review pending. Browser full Next workbench with isolated HTTP fixtures passed Start and Retry lost-response/reload/exact-body-key replay/acknowledgement cleanup, no automatic POST, Retry original source41 despite status42, desktop1440x1000 and mobile390x844 non-overlapping recovery controls. No real generation, live GPU or backend acceptance claimed. Bandit on unchanged VN Python baseline returned zero findings/errors; Bandit does not support the touched TypeScript. No dependency policy changes. Temporary shared UI dependency link will be removed after final checks. The colliding local CLI-created archived13379 record is not staged or published.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
