# PR2761 chat hook state repairs — TASK-12116

## Scope and approach

This batch changes only `useVoiceChatMessages`, `useComposerTokens`, their existing test suites, and this evidence. No rule settings or dependencies change. The previous four-rule inventory attributed four refs diagnostics to the voice hook and five to the token hook.

| Stage | Goal and acceptance | Result |
| --- | --- | --- |
| Reproduce | Deferred saves must not discard a newer voice turn; rendered IDs must clear without artificial rerenders; empty conversations must retain zero when sending starts. | Six new behavioral regressions failed; six existing/characterization cases passed. |
| Repair | Make rendered voice state reactive, guard asynchronous turn cleanup by identity, and preserve the last non-streaming token estimate including zero. | Both focused suites pass all 12 tests. |
| Verify | Exercise adjacent stream behavior, check all compiler rules in touched files, and run a focused no-emit typecheck. | Results below. |

## Observed defects and changes

**Voice turn persistence race.** Both `finalizeAssistant` and the partial-response `failTurn` awaited persistence and then unconditionally cleared `currentTurnRef`. Both UI callers start finalization without awaiting it. A second turn can therefore begin while the first save remains pending; completion of the first save erased the second turn, silently dropping its subsequent deltas. The new deferred-save tests reproduced this in both completion paths: the second assistant remained at `Second ▋` instead of receiving `Second response▋`.

Cleanup now checks that the current turn is still the same object that was saved. Existing synchronous reset/abandon semantics, message text, history, persistence payloads, and interruption metadata are retained. Rendered `activeAssistantId` is React state, set when a turn begins and cleared alongside its imperative ref; it updates when reset or asynchronous completion changes no store messages. Tests observe those updates without manually rerendering. The existing store boundary mock now notifies React when its message/history setters run, matching the production store's subscription behavior and ensuring that intermediate pre-save renders are exercised.

**Stale conversation token cache.** A nonempty conversation cached 11 tokens in the regression. Clearing all messages displayed zero but returned before updating that cache. Starting another send read the old cache and displayed 11 again. The hook now computes a pure non-streaming estimate and retains that value in React state. A guarded state adjustment records changed estimates, including zero. While sending, estimation remains suspended and the last non-streaming total stays fixed; completion refreshes it. Existing image-generation message filtering is unchanged. No render-time ref access or effect-driven derived state is introduced.

## Verification

- RED: **6 failures / 6 passes** across the two existing suites, including both persistence races, reset and both awaited-completion ID updates, and the empty-conversation cache.
- GREEN: **12 passing** across both focused suites.
- Adjacent stream validation: **34 passing in four suites**, including the existing voice-stream defaults and interrupt tests. Existing i18next initialization warnings remain.
- Focused ESLint uses the existing shared configuration and temporarily enforces refs, immutability, set-state-in-effect, and preserve-manual-memoization for the two hooks and two suites. **Zero errors**; four pre-existing warnings remain (two `any` annotations in token parameters and two unused voice-store bindings). Purity/static-components/use-memo remain enabled by the shared configuration. No repository configuration was changed.
- Focused TypeScript validation: **exit 0, no diagnostics**, with a temporary config extending the existing frontend config, explicitly listing the four touched TypeScript files and existing ambient declarations, and resolving the existing frontend Node type definitions. `noEmit` is true and incremental output is disabled. This checks their imported type graph under current frontend settings; it is not a repository-wide or strict-mode certification. The first temporary-config attempts exposed only path discovery / missing automatic Node type-root setup, corrected without repository changes or installation.
- `git diff --check` passes for the touched source/test files.
- Bandit was invoked from the project virtual environment on the two hooks. It cannot parse TypeScript and reports two syntax-parser errors; this is **not** a successful security scan. No Python code changes are present.

Local small evidence files: `/tmp/pr2761-chat-hooks-red.log`, `/tmp/pr2761-chat-hooks-green.log`, `/tmp/pr2761-chat-hooks-validation.log`, `/tmp/pr2761-chat-hooks-lint.log`, `/tmp/pr2761-chat-hooks-typecheck.log`, and `/tmp/pr2761-chat-hooks-tsconfig.json`. No builds, installations, emitted JavaScript, build-info files, or repository-wide audit JSON were generated.
