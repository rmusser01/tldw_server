# UAT B02: Native Character Auto provider binding

Associated task: `TASK-13260.278.18.83.45`; the parent recorded this distinct live defect before edits. Existing human all-43 repair authorization covers this bounded correction.

## Goal and scope

Manage Models advertises Auto (from model), stores no explicit provider, and serializes the selected Ollama model as `tldw:gemma3:1b`. Both native Character send callers share `sendNativeHistoryCharacter`, which currently rejects that valid selection before generation because it only resolves explicit providers.

Inside the existing owner/auth/settings lease, resolve Auto from a fresh authenticated catalog read. Match the exact canonical model ID and require one known provider across all matching entries. Never use display names, model-name guesses, cached fallback, or a first-match choice between providers. Preserve explicit/provider-qualified selections and native owner, capability, preparation, admission, unsupported-parameter, and settlement checks.

Production scope: `apps/packages/ui/src/hooks/chat/native-history-character-send.ts`. Regression scope: both existing mounted consumer history-selection test files. Existing model-service tests qualify fresh discovery/authentication and cache rejection. No runtime, browser, provider requests, installation, publication, commit, or parent task edits in this subtask.

## ADR check

**ADR required: no.** This restores the existing Auto selection contract using the canonical model service and leaves routing and owner authority unchanged. Governing decisions: [ADR-025](../ADR/025-llm-provider-adapter-routing-and-overrides.md) and [ADR-049](../ADR/049-chat-history-selection-ownership.md). The parent receives this assessment for the associated task notes.

## Stage 1: Diagnose and reproduce

**Goal**: Trace both callers, selection serialization, provider resolution, fresh catalog behavior, and lease boundaries; demonstrate the actual Auto failure in mounted Playground and Sidepanel sends.
**Success Criteria**: On unchanged production code, both Auto regressions fail before the stream receives the exact Ollama/model request. Controls reject ambiguous, missing, default/unknown-provider, and display-name-only catalog matches; lease changes remain fenced.
**Tests**: Mounted `useChatActions.history-selection.test.tsx` and `useMessage.history-selection.test.tsx`; catalog fixtures use exact `gemma3:1b` and `ollama`, and stream fixtures preserve native admission/result acknowledgments.
**Status**: Complete

## Stage 2: Bind Auto to current catalog metadata

**Goal**: Add the smallest shared correction in `sendNativeHistoryCharacter`.
**Success Criteria**: When explicit resolution has no provider, call `tldwModels.getModels(true, { requireFresh: true })` inside the existing snapshot lease; recheck origin/auth/settings/tools after the read; accept only one known provider for the exact canonical ID. Both causal regressions pass while negative controls and explicit selection behavior hold.
**Implementation**:

```ts
const catalog = await tldwModels.getModels(true, { requireFresh: true })
if (!params.originIsCurrent() || !leaseValid()) throw new Error("stale_selection")
const providers = new Set(catalog
  .filter(entry => entry.id === model)
  .map(entry => {
    const binding = parseProviderQualifiedModelSelection(`${entry.provider}:${model}`)
    return binding.modelId === model ? binding.provider : undefined
  }))
if (providers.size === 1) provider = providers.values().next().value
if (!provider) throw new Error("native_history_explicit_model_provider_required")
```

**Tests**: Focused red-to-green tests in both normal existing UI and WebUI configurations, with a 600-second command limit and 300-second case limit; at most three bounded attempts per issue.
**Status**: Complete

## Stage 3: Verify and freeze

**Goal**: Run complete affected suites, both configured TypeScript checks, touched-file lint, and source review; freeze exact scope for the parent’s independent review.
**Success Criteria**: Complete affected suites finish naturally; types pass; no new lint findings; report exact results and limitations. TypeScript-only production change: Bandit is not applicable. Parent retains live UAT/build/integration ownership.
**Tests**: Both mounted history-selection suites plus `TldwModels.test.ts` and `resolve-api-provider.test.ts` in UI/WebUI; `tsc --noEmit --incremental false` in UI/WebUI; touched source/test ESLint using the existing frontend configuration.
**Status**: Complete

## Verification record

On unchanged production, both new mounted Auto regressions failed naturally with the actual `native_history_explicit_model_provider_required` notification and zero provider dispatches. UI focused run: 2 failed, 131 intentionally filtered, exit 1, 2.43 seconds (after improving the initial zero-dispatch assertion to expose the exact error). No current-candidate live B02 acceptance is claimed by this source-level unit.

The first complete affected runs passed 211/211 tests in four files under each normal configuration: UI exit 0 in 3.71 seconds; WebUI exit 0 in 3.64 seconds. Both original causal regressions are included, together with eight catalog rejection controls and four lease-change controls. UI configured TypeScript completed naturally with no diagnostics. Touched-file lint identified three new test casts using `any`; these were replaced with the actual hook parameter type before final verification. Production lint has zero findings.

Final source verification, after that test typing correction:

| Check | Result |
| --- | --- |
| UI complete affected suites (both mounted hooks, model service, provider resolver) | 211 passed / 4 files; natural exit 0; 3.84 seconds; no skips |
| WebUI same complete affected suites with its normal aliases/setup | 211 passed / 4 files; natural exit 0; 3.57 seconds; no skips |
| UI configured `tsc --noEmit --incremental false` | Natural exit 0; zero diagnostics; 45.77 seconds |
| WebUI configured `tsc --noEmit --incremental false` | Natural exit 0; zero diagnostics; 33.96 seconds |
| Touched-file ESLint with existing frontend config | Exit 0; production 0 findings; tests 54 + 105 inherited warnings outside the new blocks; zero new findings |
| Scoped `git diff --check` | Exit 0 |
| Bandit | Not applicable: TypeScript-only production change |

All test commands used the installed Vitest with 300-second test/hook timeouts and a 600-second command alarm; final type commands used the existing configured projects with an 8192 MB Node heap budget and the same command limit. ESLint also emits its repository-root pages-directory context notice; Node emits the existing localStorage experimental notice. No command timed out or needed forced cleanup.

Frozen subtask scope: the one native send production file, additions to both existing mounted hook test files, and this unique plan. No source/test edits remain pending in this unit. Parent-owned independent review, integrated build, runtime refresh, and live B02 UAT remain separate from these source-level results.
