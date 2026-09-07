# Writing Continuation Service Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make non-chat Writing Playground Predict and Fill instructions user-editable through shared Service Prompts without changing continuation behavior.

**Architecture:** Add two literal registry definitions and client defaults, then load one scope-bound snapshot at the existing generation call site. Reuse TldwChatService's existing signal and requestScope options; keep lifecycle ownership local to Writing Playground rather than introducing a new generation engine.

**Tech Stack:** Python registry/FastAPI, TypeScript/React shared UI, pytest, Vitest/Testing Library.

**Spec:** `Docs/Design/writing-continuation-service-prompts.md`

**Tracking:** TASK-13216

## Global Constraints

- No new API, database schema, settings system or dependency.
- These prompts apply only to non-chat Generate. Chat mode continues to use its existing messages and explicit context; it must not fetch either definition.
- Braces in authored instructions are literal, not variables.
- Authentication, scope, validation and non-404 failures do not become defaults.
- Manual Stop preserves already accepted partial output and its existing undo behavior.
- Do not erase unrelated user edits or introduce account-wide manuscript cleanup.
- Preserve main-checkout edits; work in `.worktrees/writing-continuation-service-prompts` on `codex/writing-continuation-service-prompts`.
- Activate `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate` before Python commands.

## Task 1: Stage 1 — Registry, defaults and Settings

**Goal:** Both definitions can be edited/reset and loaded through existing APIs.
**Success Criteria:** Exact default bytes and literal metadata agree across Python and TypeScript; each definition has an independent Settings entry and all older-server paths work.
**Status:** Not Started

**Files:**
- Modify `tldw_Server_API/app/core/Prompt_Management/service_prompts.py`.
- Modify `apps/packages/ui/src/services/tldw/domains/service-prompts.ts` (KnownServicePromptId).
- Modify `apps/packages/ui/src/services/service-prompts.ts` (render definitions and packaged fallback allowlist).
- Modify `apps/packages/ui/src/services/tldw-server.ts` (LEGACY_SERVICE_PROMPT_DEFAULTS).
- Modify `apps/packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx`.
- Modify mirrored `apps/packages/ui/src/{assets/locale/en,public/_locales/en}/settings.json`.
- Test `tldw_Server_API/tests/Prompt_Management/test_service_prompts.py` and `test_service_prompts_api.py`.
- Test `apps/packages/ui/src/services/__tests__/service-prompts.test.ts` and `src/services/tldw/domains/__tests__/service-prompts.test.ts`.
- Test `apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx`.

**Interfaces:** Existing definitions gain `writing.continuation.predict` and `writing.continuation.fill`, each with `{system: string}`. No new public function.

- [ ] Extend `EXPECTED_REGISTRY` and the existing registry-driven API cases. Add explicit default/literal tests using the current imports:

```python
@pytest.mark.parametrize("mode,default", [
    ("predict", "Continue the text from the prompt. Respond with only the continuation."),
    ("fill", "Fill in the missing text between the prefix and suffix. Respond with only the missing text."),
])
def test_continuation_defaults_are_literal(mode: str, default: str) -> None:
    """Keep existing continuation instructions and literal authored braces."""
    definition = get_service_prompt_definition(f"writing.continuation.{mode}")
    assert dict(definition.default_parts) == {"system": default}
    assert definition.parts[0].mode == "literal"
    assert definition.parts[0].required_variables == ()
```

- [ ] Extend the existing client fallback matrices for catalog 404, omitted ID and detail 404. Assert errors other than 404 propagate, braces are unchanged, and each ID resolves independently.
- [ ] Extend Settings fixture definitions and its parameterized save/reset cases with both IDs; use existing System instructions label and new definition labels `Writing continuation: Predict` / `Writing continuation: Fill`.
- [ ] Run these tests red before registration. Use the test commands in Stage 3, narrowed to the changed files.
- [ ] Add two `ServicePromptDefinition` entries following existing Writing Agent literal entries. Shared description: `Controls non-chat continuation instructions. Context, fill templates, stopping rules and provider settings remain fixed.` Workflow: `writing.continuation`, label `Writing Playground continuation`. Add matching client IDs, literal render schemas, frozen packaged defaults and fallback allowlist entries. Add matching Settings metadata and mirrored English strings.
- [ ] Run focused suites green, compare default strings to `PREDICT_SYSTEM_PROMPT`/`FILL_SYSTEM_PROMPT`, and lint touched files before committing this stage with its Backlog record.

## Task 2: Stage 2 — Scoped generation integration

**Goal:** Both continuation paths consume the selected prompt without stale writes.
**Success Criteria:** One snapshot per non-chat generation; no prompt lookup in chat mode; late callbacks cannot change a different operation or binding; Stop retains valid partial output.
**Status:** Not Started

**Files:**
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx` only around continuation generation/lifecycle.
- Modify `apps/packages/ui/src/components/Option/WritingPlayground/hooks/utils.ts` only if replacing now-unused exported constants with shared default references; retain exports needed by other callers/tests.
- Test `apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx` (reuse its existing component harness).
- Test `apps/packages/ui/src/services/tldw/__tests__/TldwChat.abort.test.ts` only if additional transport assertions are necessary; transport already supports scope and abort.

**Interfaces:** Consume `loadServicePromptSnapshot(ids, {signal}) -> Promise<ServicePromptSnapshot>`. Snapshot exposes `definitions`, `requestScope`, `scopeSignal`, `scopeInvalidatedSignal`, `release()`. Pass existing `TldwChatOptions.signal` and `.requestScope`; no transport API change.

- [ ] Extend the component harness with a mocked snapshot loader, controllable scope controller, deferred lookup/results/chunks and a release spy. Preserve the existing `seedWritingSession`, `getEditor`, `streamCalls`, `sendCalls`, `sendResponses` helpers and default fixtures so unrelated revision tests remain unchanged. Add a non-streaming Predict case in this shape:

```tsx
mockState.storageValues.set("selectedModel", "mock-model")
seedWritingSession({ prompt: "Opening", settings: { token_streaming: false } })
mockState.sendResponses.push(" ending")
render(<WritingPlayground />)
fireEvent.click(screen.getByTestId("writing-topbar-generate"))
await waitFor(() => expect(mockState.sendCalls).toHaveLength(1))
expect(mockState.sendCalls[0]?.options).toMatchObject({
  systemPrompt: "Continue in {my style}.",
  requestScope: expect.objectContaining({ userId: 42 }),
  signal: expect.any(AbortSignal)
})
await waitFor(() => expect(getEditor()).toHaveValue("Opening ending"))
```

- [ ] Parameterize Predict/Fill and send/stream coverage. Assert exact messages, provider knobs, stop strings and insertion results, not source-code text. Add a chat-mode case asserting zero continuation snapshot calls and unchanged explicit message precedence.
- [ ] Add deferred cases for Stop during lookup, scope change during lookup/generation, session/scene changes, unmount, late callbacks after a new request, rejected/malformed snapshot, failed generation, normal completion and manual-stop partial output. Assert no stale history/logprobs/error/cleanup writes and that leases release once. Include same-user token-refresh compatibility at the shared snapshot layer if not already covered.
- [ ] Run new cases red. Establish request identity and controller synchronously before lookup. Capture the editor binding and pre-request text; check live refs rather than the callback's captured session value. Do not use React `isGenerating` alone as the in-flight lock.
- [ ] Replace only the non-chat system selection with the snapshot result:

```ts
const id = plan.mode === "fill"
  ? "writing.continuation.fill"
  : "writing.continuation.predict"
const snapshot = await loadServicePromptSnapshot([id], { signal: controller.signal })
const systemPrompt = snapshot.definitions[id]?.parts.system
if (typeof systemPrompt !== "string" || !systemPrompt.trim()) {
  throw new Error("Writing continuation instructions are unavailable.")
}
// After ownership and signal checks, spread into the existing request options:
const scopedOptions = {
  ...generationRequestOptions,
  systemPrompt,
  signal: snapshot.scopeSignal,
  requestScope: snapshot.requestScope
}
```

- [ ] Keep lookup inside cleanup ownership: late snapshots release immediately; validated active snapshots release in final cleanup. Check ownership after every await and before streamed text/logprob callbacks and finalization. Use a separate manual-stop flag from invalidation so only user Stop commits partial output. Invalidation must not restore an old binding or overwrite independent edits. Gate cleanup so an old operation cannot clear a new operation's refs/loading state.
- [ ] Wire Stop, binding changes and unmount into the controller and existing transport cancellation. Add the scope-invalidated listener only while the request owns its snapshot; remove it before release. Preserve existing revision action behavior using the shared service.
- [ ] Run new and existing component/transport tests green, lint touched files and self-review against the spec before committing this stage.

## Task 3: Stage 3 — Verification and handoff

**Goal:** Demonstrate compatibility, review the complete patch and prepare a reviewable branch.
**Success Criteria:** Focused suites pass; touched scope has no new lint/security/type findings; independent review addressed; Backlog notes contain actual evidence and remaining limitations.
**Status:** Not Started

- [ ] From `apps/packages/ui`, run:

```sh
./node_modules/.bin/vitest run src/services/__tests__/service-prompts.test.ts src/services/tldw/domains/__tests__/service-prompts.test.ts src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlayground.phase1-baseline.test.tsx src/services/tldw/__tests__/TldwChat.abort.test.ts src/services/__tests__/tldw-chat.message-sanitization.test.ts --maxWorkers=1
```

- [ ] From the worktree root, run:

```sh
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Prompt_Management/test_service_prompts.py tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py -q
python -m ruff check tldw_Server_API/app/core/Prompt_Management/service_prompts.py tldw_Server_API/tests/Prompt_Management/test_service_prompts.py tldw_Server_API/tests/Prompt_Management/test_service_prompts_api.py
python -m bandit -r tldw_Server_API/app/core/Prompt_Management/service_prompts.py -f json -o /tmp/bandit-writing-continuation.json
```

- [ ] Run existing frontend ESLint config on changed TS/TSX files. Run shared UI typecheck with `NODE_OPTIONS=--max-old-space-size=8192 ./node_modules/.bin/tsc --noEmit --pretty false`; compare against baseline diagnostics rather than claiming a repository-wide clean result. Verify the two English locale trees have matching new entries.
- [ ] Use requesting-code-review for independent review of the full base-to-head diff, emphasizing shared-service cancellation, manual-stop versus scope invalidation, provisional editor ownership and fallback behavior. Address actionable findings test-first.
- [ ] Update TASK-13216 through Backlog MCP/CLI with test counts, lint/Bandit evidence, reviewed scope and known skipped full builds/live browser checks. Mark stage statuses accurately. Keep the design document; remove this task's implementation plan only once all stages are complete, per repository instructions.
- [ ] Remove only temporary dependency symlinks created for this worktree after verifying their targets. Run `git diff --check`, inspect `git status --short`, stage only task files, and commit without bypassing hooks. PR creation/merge remains a separate explicitly authorized finish step.

## Plan self-review

Both registered definitions map to existing call sites. Explicit chat mode is excluded from prompt resolution. Context construction, stopping, provider options and revision actions are preserved. Lifecycle checks are scoped to this generation integration; no new controller framework, transport rewrite, settings UI or persistence scheme is planned.
