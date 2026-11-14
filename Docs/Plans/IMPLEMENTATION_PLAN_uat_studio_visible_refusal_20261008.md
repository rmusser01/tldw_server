# UAT C02 persistent Execute drawer refusal

Related task: TASK-13260.278.18.83.45. Root recorded/read the new visible-refusal unit via backlog-py before edits; root owns the parent task.

## Actual failure and source ruling

The loaded repaired client dispatches the displayed Ollama/model target and valid v2 execution succeeds. Missing inputs clear output/stats, but actual drawer DOM/screenshot retains no refusal explanation. ExecutePlayground routes failed envelopes to a static notification and clears results without retaining error state. Ant Design JSON validation renders a field error; effective-default provider/model guards only notify. This unit does not replay actual execution, models, runtime, screenshots or logs.

Connected caller/read scope: StudioPromptsTab passes controlled open/selectedPromptId/onClose; project selection resets prompt and closes execution. StudioTabContainer connection/capability gates unmount subcontent. executePrompt forwards to canonical apiSend; request-core formats error envelopes. Existing sanitizeServerErrorMessage adds concise bounded first-line/path/URL/token redaction. Existing Alert supplies real error-role UI. No transport/auth/store/backend/catalog helper change.

Approved design: retain local error state rendered in existing Alert and keep surrounding static notifications. Reuse existing sanitizer for service/thrown errors. Fixed local validation messages become persistent feedback; Form onFinishFailed handles existing field validation. Clear prior error/output/stats on new submission, prompt/open/close lifecycle changes. A local execution generation retires async callbacks on new attempt/prompt/open/close/unmount, including same-prompt reopen, while the unchanged payload is sent to the service. This is UI result ownership, not a substitute for canonical request authentication or permission gates.

ADR check: no new ADR. Reuse existing Alert/error-sanitizer and provider/endpoint contracts, with ADR-025 governing the unchanged provider integration boundary. No durable transport/auth/security ownership, API/storage, dependency or backend rule changes.

## Stage 1: Mounted causal evidence
**Goal**: Show absence of durable refusal and late result publication before production changes.
**Success Criteria**: Actual mounted Form/Drawer/Alert oracle; failure/thrown, stale output, recovery, prompt/close reset/retirement and local validation cases; independent success control.
**Tests**: Normal configured UI/Web runners, outer 600 / case 300, at most 3 issue attempts. Services and notification are external seams; actual Alert is not mocked or replaced by a notification-only oracle. Explicit spy cleanup and supported ByRoleOptions.
**Status**: Complete

## Stage 2: Minimal local retained feedback
**Goal**: Persistent sanitized failure and bounded local submission lifecycle.
**Success Criteria**: Failed results remain visible, successful output/stats/default-target payload remain, late retired callbacks cannot overwrite the current drawer or notify it.
**Tests**: New causal module plus genuinely affected existing mounted Execute error/default/design-system bodies. No unchanged backend/provider/helper broad replay.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Supply exact frozen source/new tests for independent root review and combined client handoff.
**Success Criteria**: Scoped baseline/current lint no new findings; syntax/test formatting pass; complete attempt/coverage limits. Root owns semantic types/build/load/live validation.
**Tests**: TSX only, Bandit N/A. No agent runtime/browser/profile/auth/log/model/config/environment/parent-task/index/commit/publication actions. Prior reviewed units remain frozen.
**Status**: In Progress

## Causal qualification before production

Attempt 1, normal configured UI and Web, unchanged source SHA256 431473f7ad92a4cad6441c10942def0ceff7f854ff11d2c0b6d5aebb67d16fef: each 14 failed / 1 passed, zero pending; UI elapsed 13.710s. Web elapsed was truncated in tool aggregation, so no precise elapsed claim. Failures show missing actual drawer Alert for resolved/thrown/validation refusals, late output after close/prompt change, and late notification after close/unmount. The independent positive output/statistics control passes. No setup/fixture failure. Baseline ESLint source 0 errors / 5 existing no-explicit-any warnings; new test 0 errors / 0 warnings.

The approved repair also resets the mutation observer on prompt/open/close so a retired in-flight request no longer renders as current execution. The underlying canonical request is neither canceled nor resubmitted. Only the local wrapper carries generation; service payload remains unchanged.

## Frozen scoped qualification

Attempt 2, first production green: configured UI 29 / 29 passed in 11.850s and Web 29 / 29 passed in 11.903s, zero failed/pending. Scope: new 15 mounted visible-refusal cases plus genuinely affected default-target 8, execution-errors 5 and design-system 1. Existing backend/catalog/helper tests are unchanged and were not replayed. No fixture/setup failure; no third behavioral attempt required.

Scoped ESLint baseline source 0 errors / 5 no-explicit-any warnings; current 0 errors / 4 inherited warnings, with the existing onError any replaced by inferred Error. New test 0 errors / 0 warnings. Syntax-only TypeScript parse of both touched TSX files has zero diagnostics; diff whitespace check passes. Bandit is N/A for these TSX-only edits; no Python scope changed.

Prettier initially flagged only formatting in the new test's existing mocked getPrompt chain. Formatting after green merely flattens that chain. An initial AST comparison operator included SourceFile.text, so it falsely compared raw whitespace; corrected comparison of parsed node kinds/text against the exact pre-format chain confirms identical AST. Final new-test Prettier/syntax/lint checks pass. This formatting/operator event is not a causal defect or fixture failure and does not justify replaying unchanged behavior.

Frozen SHA256: ExecutePlayground.tsx 57b93cfd1bd2584d39a65672231ff8b53853a5eabe7b4c4d7ee77a08c05df3b4; ExecutePlayground.visible-refusal.test.tsx b0f7171403ce786967d4cf0a4ad2ca147703dc71f98a6ec60d694769ae412824. Existing default-target test 92d093ac2efa37cb1711229db59483b65c86a426ce50511c61524f4b3771b895 and execution-errors test 3aceaaa868c22f94fcb34b30cbe71c40a65228a53765fe04abfb6a277819d0f5 remain unchanged.

Root independent review requested at these frozen hashes; root owns the combined semantic UI/Web type checks, build/load and actual durable-refusal UAT. Mounted external service seams do not establish live HTTP/auth/model or whole 43 acceptance. No runtime/profile/browser/auth/log/model/backend/environment/parent-task/index/commit/publication action was taken.

## Independent root review

Root independently read the complete current ExecutePlayground component, all 15 new mounted test bodies, sanitizer, connected caller and owned plan. Review CLEAR: persistent sanitized Alert, unchanged payload/default/validators, generation callback and lifecycle retirement are sound in the covered scope. No tests or review were replayed. Stage 3 remains In Progress because root combined semantic UI/Web typing, controlled build/load and actual durable-refusal live validation remain open. Production and tests stay frozen at the hashes above.
