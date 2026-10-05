# Submission outcomes and guarded composer recovery

**Task:** TASK-13454.1 under TASK-13454/TASK-13453. Approved design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md; accepted D7 design governs persistence follow-up.
**Baseline:** origin/dev c2260945ee, integrated10d6abff30. Draft-transition delivery and its completed plan archived214d97d402.
**Architecture:** reuse ChatSubmitResult, normal/native helpers, queue issue handling, existing composer attempt token and explicit accepted source/bind. A submitted result means the helper's defined submission completed; it is not proof that every latest turn reached the server. Retain conservative legacy special-mode void behavior until those senders expose outcomes. Do not add an in-memory save authority, implicit retry, outbox or new API. ADR-049/050 apply; no new ADR required for this plumbing. D7 idempotent creation, settlement/import/freshness/Sync contracts and all-surface Temporary transcript retention remain later deliveries.

## Stage 1: Propagate actual outcomes
**Goal:** Stop converting compact normal/native failure or cancellation into implicit success.
**Success Criteria:** Existing helper result is returned, verified native completion returns submitted, handled native errors return failed, preflight blocks return skipped/failed. Compact queue throws for unsuccessful structured results using the existing helper, retaining the item.
**Tests:** Real useMessage/controller/model normal success and provider failure; native ACK, missing settlement, pre-admission interruption, late disconnect after ACK, rejected preflight; actual queue callback failure/success.
**Status:** Complete

## Stage 2: Restore untouched failed text
**Goal:** Keep unsuccessful sends usable in both composers without overwriting newer work.
**Success Criteria:** Only the current attempt may restore its captured text into an untouched empty composer owned by the same account/address or its accepted first-ID bind. New input/attempt/view/account supersedes restoration. Unknown native outcome remains visibly recoverable and never automatically replays.
**Tests:** Structured failed/skipped and thrown send; newer text/newer attempt; New/navigation/account/workspace; accepted bind; failed-send callback on both real surface adapters; queue accepted does not invoke restoration.
**Status:** Complete

## Stage 3: Review and verify
**Goal:** Deliver the bounded unit with honest evidence and no data-loss/owner regression.
**Success Criteria:** Affected suites, WebUI typecheck, baseline-relative touched lint, diff/task checks pass; two already authorized independent reviewers have no remaining Critical/Important findings. Commit only related files/tasks.
**Tests:** Existing composer recovery/attempt/reset, actual send/history, queue dispatch and voice/source guard suites. Bandit N/A TS/docs-only. Live backend/native shell and full saving status remain program gates.
**Status:** Complete

Delivery evidence: seven meaningful sender reds established discarded normal results, native success/failure void outcomes and skipped preparation; two actual compact callback/queue reds established consumed failed/skipped requests. Shared rejection hook had one red; both deployed surface afterSend callbacks had two restoration reds. Initial core feature tests failed on the missing recovery method, then passed for exact source/bind and edit/image/attempt/owner boundaries. Review found saved-address pending input deletion. The first lockless cold test passed because comparison conservatively skipped a raced removal; a serial Web Locks fixture matching Chromium produced two meaningful local/native cold-reopen failures. Protect every current pending attempt while keeping fresh marker retirement separately qualified; both failures now pass.

Final checks: 17 affected suites / 292 tests pass; WebUI typecheck passes; baseline-relative lint adds no diagnostics and new files have none; diff/task normalization pass. Both authorized independent source reviewers found no remaining Critical/Important finding after the saved-source correction. They did not run tests or live/native-shell journeys. Bandit N/A TS/docs-only scope. Latest dev re-fetched unchanged c2260945ee. Source restoration is observational: no enqueue, retry, controller selection or recovery-record dismissal; structured submitted is not universally server saving proof. Legacy special-mode void results, unknown first creation, truthful durable Saved status, complete attachment recovery and accepted D7 backend/Temporary/Sync requirements remain tracked by the broader program.
