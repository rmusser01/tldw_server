# UAT359 chat startup timeout implementation plan

> Execute inline with test-driven development. Root owns independent review, commits, and native acceptance; do not change shared runtimes or browser settings.

**Task:** TASK-13260.277.9.
**Goal:** Fresh Chat uses the documented 120-second first-output budget, preserves explicit custom budgets, and reports the actual timeout phase and budget.
**Approved design:** Root approved the bounded shared startup-default and typed-error repair. Preserve existing stream idle policy, request scope, explicit Retry identity, and cancellation behavior.
**Architecture:** A small chat-timeout module supplies the startup default to the service and Settings and identifies locally generated startup/idle errors. The service preserves those errors; the existing assistant error formatter produces actionable, serializable diagnostics.
**Tech stack:** TypeScript, existing browser streaming transport, Vitest fake timers, i18next.

## Stage 1: Causal regression coverage
**Goal:** Reproduce both the default mismatch and hidden diagnostic at the real transport/formatter boundary.
**Success Criteria:** A configured fresh profile with no timeout override can receive its first token after 10 seconds; a custom 10-second limit still aborts once with an accurate diagnostic. No automatic POST replay occurs.
**Files:** services/tldw/__tests__/TldwChat.timeout.integration.test.ts.
**Tests:** Real TldwChatService, chatRagMethods, bgStream and assistant formatter; replace only remote fetch/config storage. SSE heartbeats keep transport alive while fake time crosses the startup deadline. Cover missing/invalid defaults, the actual 120-second deadline, custom deadline, visible-output idle timeout, and caller Stop.
**Status:** Complete
- [x] Real transport regressions RED: 8 failed, 1 Stop control passed; `/private/tmp/uat359-timeout-red.log`. Missing/invalid budgets abort at10s, explicit120s deadline fails early, startup/idle diagnostic identity is replaced by the generic wrapper.

## Stage 2: Minimal policy and diagnostic repair
**Goal:** Align only the startup fallback with the documented Settings contract and retain local timeout identity.
**Success Criteria:** Runtime and Balanced Settings share `CHAT_STARTUP_TIMEOUT_DEFAULT_MS = 120_000`; positive configured startup limits win; `ChatStreamTimeoutError` carries phase and effective timeout, survives service wrapping, and gets a phase-specific assistant summary/hint/detail.
**Files:** services/tldw/chat-timeouts.ts; services/tldw/TldwChat.ts; components/Option/Settings/TldwTimeoutSettings.tsx; utils/chat-error-message.ts; two timeout expectations in services/__tests__/tldw-chat.message-sanitization.test.ts.
**Tests:** Stage 1 suite plus existing abort lifecycle, unavailable-model integration, error formatting, and actual Settings timeout suites.
**Status:** Complete
- [x] Add the shared startup default and typed local timeout error.
- [x] Preserve the typed timeout through the service and formatter without broadening arbitrary provider-error exposure.
- [x] Run RED cases to GREEN and preserve Stop, idle policy, model recovery, and custom settings controls.
- The existing message-sanitization suite confirmed exactly two deliberate timeout-wrapper contract failures, with15 controls still passing, before its expectations were updated to require the precise typed timeout. `/private/tmp/uat359-existing-contract.log` retains that comparison.

## Stage 3: Verification and native handoff
**Goal:** Provide independently reviewable source and exact evidence with limits.
**Success Criteria:** Targeted suites pass, scoped lint adds no errors, TypeScript diagnostics on touched files are checked, and root receives the fresh-profile native acceptance recipe.
**Tests:** Local Vitest only, scoped ESLint, TypeScript diagnostics. Bandit is not applicable to TypeScript-only changes.
**Status:** In Progress
- [x] Review changed paths and run surrounding tests once.
- [x] Record results through the official Backlog CLI and report remaining native acceptance.
- [ ] Root independent source review and fresh-profile native acceptance.

### Verification

- `/private/tmp/uat359-timeout-green.log`:6 suites/59 tests passed, including real Settings save/reload/preset/custom controls and numeric-label accessibility.
- `/private/tmp/uat359-timeout-final.log`:6 suites/66 tests passed, including final service/formatter/transport behavior, message sanitization, scope/Stop/Retry controls, and request-core timeout compatibility. Counts overlap and are not additive.
- `/private/tmp/uat359-eslint.json`:0 errors; only two unchanged `any` warnings in TldwChat outside changed lines. New timeout module, formatter, Settings change and transport test have0 warnings.
- `/private/tmp/uat359-typescript.log`: shared UI compiler exits2 with repository-wide fixture diagnostics; no diagnostic belongs to a changed/new UAT359 file. Nearby untouched TldwChat.abort and Settings accessibility fixtures have existing spread/mock-type diagnostics. No full-repository typecheck success is claimed.
- Bandit does not analyze TypeScript/TSX; no Python source changed.

### Native acceptance handoff

Use a fresh profile with configured server/provider and no saved startup timeout. Exercise a real first response taking more than10 seconds and less than120 seconds. Confirm normal completion, one canonical user/assistant pair, and reload. Deliberately save a shorter custom startup limit through Settings and confirm one failed POST with a startup-specific explanation and its actual budget; explicit Retry must retain its existing client correlation. The local provider's performance and backend/database behavior remain root's native checks. No shared browser, runtime, provider, database, or git state was changed by this implementation.

## Retained diagnosis

PG captures `.tmp/uat363-repair1-20260920/pg-multi-034` through `039` show HTTP200 but no completed assistant. The frozen PG console `console-2026-09-20T23-58-26-990Z.log` lines185/187 report `Chat response timed out before any visible output arrived.` Runtime fallback was10 seconds; Settings advertises120 seconds. The outer service Error retained the cause but exposed only `Stream completion failed` to the formatter. This establishes the client timeout/diagnostic defect; it does not establish why the provider's first visible output was delayed.
