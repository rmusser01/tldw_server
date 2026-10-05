# Character conversation creation setup plan

**Goal:** The first native character conversation captures the provider, model, and supported explicit sampling values that were validated for its first turn.

**Task/spec:** TASK-13454, second delivery within Stage 1 of `Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md`. Stage 1's broader preflight, recovery, drafts and greeting startup criteria stay open.

**Baseline:** dev `75ab224081bf140ef52017c1a9b0a04f6878d488`; safe-clear delivery `7496dc9a6c`. The completed safe-clear plan is preserved in that commit and removed per repository policy.

**Architecture:** Add existing `ChatSessionCreate` fields to the current fenced `createChat` caller. Reuse the immutable settings copy and existing explicit provider/model resolver. Do not create a separate settings store, change public APIs, remap frequency/presence penalties to repetition penalty, or send fields the create schema does not support.

**ADR assessment:** No new ADR. Existing ADR-049/050 and accepted behavior snapshot decision 002 govern ownership and creation. This passes supported inputs to that contract; it does not expand snapshot or mutation authority.

## Stage 1: Reproduce creation/first-turn setup disagreement

**Goal:** Show that native creation omits the setup selected for the turn.
**Success Criteria:** Tests fail on absent model/provider/sampling fields, including explicit zero values and provider-qualified selection.
**Tests:** Existing mounted `useChatActions.history-selection.test.tsx` real send path with external create and completion doubles.
**Status:** Complete.

- [x] Run the existing full native/history actions suite before edits.
- [x] Extend the existing first-create ACK/fence cases with explicit temperature/top-p zero and a literal expected create payload. Add a provider-qualified selection case with a conflicting default provider; the qualified provider must win for both create and dispatch.
- [x] Run those cases red before changing production code. Keep the existing no-publication/no-dispatch checks for a stale creation response.

## Stage 2: Capture the supported creation inputs

**Goal:** Persist the validated turn's supported creation setup.
**Success Criteria:** Create and completion use the same normalized model/provider; temperature and top-p explicit overrides are passed without turning undefined values into defaults.
**Tests:** Stage 1 regressions.
**Status:** Complete.

- [x] In `apps/packages/ui/src/hooks/chat/native-history-character-send.ts`, extend the existing create body with `provider`, `model`, and defined `temperature`/`top_p` fields from the captured settings. Keep all surrounding lease checks, request scope, and workspace routing.
- [x] Do not send unsupported create fields, redefine the backend default sampling values, or convert completion-only penalties.
- [x] Run the targeted regressions green.

## Stage 3: Verify preserved dispatch and ownership

**Goal:** Check that saved owners, unsupported inputs and late responses retain their guards.
**Success Criteria:** Existing full history-actions suite passes; frontend typecheck and touched lint add no errors/warnings.
**Tests:** Complete existing actions suite, WebUI typecheck, baseline-relative ESLint, `git diff --check`.
**Status:** Complete.

- [x] Run the full actions suite after the change. Existing native owner, account, workspace, settings drift, repeated activity and unsupported input tests remain relevant.
- [x] Run WebUI typecheck and touched-file ESLint, distinguishing baseline formatting/warnings. Bandit does not apply to this TS/document-only scope.
- [x] Request independent review from the two already authorized reviewers. Fix important findings and reverify.

## Stage 4: Record and commit the delivery

**Goal:** Preserve concrete progress without closing incomplete Stage 1 acceptance criteria.
**Success Criteria:** Reviewed code/tests and canonical task record committed; completed plan removed after its delivery commit.
**Tests:** Canonical Backlog normalization and clean staged diff.
**Status:** In Progress.

- [x] Update TASK-13454 via backlog-py with reproduction, checks, review results and remaining work. Link this plan and archive the safe-clear plan reference to commit `7496dc9a6c`.
- [ ] Commit only this unit and its tracking records with `fix(chat): capture selected setup at character creation (TASK-13454)`.
- [ ] Remove only this completed plan; continue the approved shared setup preflight, source draft retention, recovery and greeting workflow.

## Verification record

Native/history actions baseline: 61 passed. New create input regressions: three meaningful failures on omitted provider/model/sampling. Final complete suite, including explicit zero and omitted defaults: 62 passed. WebUI typecheck exit 0. Touched ESLint zero errors/no new warnings; the unchanged action-test baseline has 103 warnings. Prettier flags both files on baseline and delivery. `git diff --check` clean. Both independent reviewers traced the create schema/endpoint/factory and found no Critical/Important issues. No backend changes; Bandit not applicable to this TS/document-only scope. This proves client creation inputs and preserved guards through external server doubles; end-to-end reopen remains a later whole-journey gate.
