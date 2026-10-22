# UAT C02 Prompt Studio required execution variables

Related task: TASK-13260.278.18.83.45. Root recorded the fresh actual C02 failure through backlog-py before edits and owns the parent task. The agent does not edit it.

Actual failure: the Legacy user template `Source:\n{{source}}\n\nWho directs Rowan Observatory? Give full name only.` dispatched inputs `{}` to the actual provider and returned `Not specified.` One populated source control returned the director. This unit prevents missing-template dispatch; it does not replay either live request or assert model quality.

## Source ruling and approved design

ExecutePlayground explicitly extracts only user_prompt variables. The canonical `/prompts/execute` enforces authoritative prompt/project access before execution. PromptExecutor.execute/execute_prompt and TestRunner share `_build_prompt_request`; legacy substitution formerly retained missing tokens literally. Structured assembly already validates required presence, defaults, optional values and literal blocks. Low-level `_build_prompt` partial rendering and optimizer/scorer `_call_llm` instructions are separate existing contracts. Literal legacy system_prompt and every provider/credential/owner fence remain.

Check required legacy user execution tokens using the same exact placeholder pattern at `_build_prompt_request`, before provider dispatch. Carry the existing typed `missing_required_variable` code through failed executor results; map only that code to a fixed bounded400 input envelope at `/execute`. Supplied null, empty, zero and false values retain current presence semantics. Replacement values remain literal with no second template pass.

The actual frontend consumer also required a narrow repair: executePrompt returns canonical apiSend `{ok,status,data,error}` envelopes, and request-core already formats `detail.message`. ExecutePlayground ignored `ok=false` and entered onSuccess. Its async mutation now rejects failed envelopes into the existing notification/output/stats reset path. Successful output, provider/model validation, input parsing and prompt-change resets remain unchanged. Root explicitly approved this necessary consumer scope before frontend edits.

ADR required: no. This repairs existing required-variable and input-error contracts already present in structured assembly/preview, without changing module ownership, auth, request shape, storage, dependencies or generic rendering. Docs/ADR/README.md was searched. ADR-055 governs the unchanged backend-neutral DB implementation used by isolated tests; it does not establish a new validation rule. Existing structured assembly and Prompt Studio input-validation requirements govern this correction.

## Stage 1: Trace and causal regression
**Goal**: Demonstrate missing-input dispatch through actual owned DB/executor and direct canonical endpoint; demonstrate the actual mounted modal's failed-envelope consumer.
**Success Criteria**: Causal failure before source edits, with independent literal valid/owner/structured/transport controls.
**Tests**: Existing activated Python3.12 environment and normal configured pytest plugins, outer600/case300 guards. Actual PromptStudioDatabase fixture and owner guards; only external provider/runtime seams isolated. Actual Form/Drawer/QueryClient in normal UI/Web runners; only external services and notification spies isolated.
**Status**: Complete

## Stage 2: Minimal shared validation and UI consumer
**Goal**: Reject missing user-template inputs before provider execution, expose bounded400 after ownership gates, and route that response through the actual modal error path.
**Success Criteria**: Four supported exact legacy placeholder syntaxes reject absence; v1/v2 source inputs render exactly; structured defaults/optional/literal semantics and low-level partial rendering remain; failed modal output/stats reset and corrected-input recovery work.
**Tests**:20 new backend cases plus genuinely affected credential, structured and execution controls;5 new mounted UI cases plus existing design-system/provider-selection controls.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Supply frozen source/tests for root independent review and controlled combined runtime handoff.
**Success Criteria**: Scoped syntax/lint and baseline/current Bandit have no new findings; independent source review and current combined types completed by root before runtime handoff.
**Tests**: Python3.12.13 for application/tests. Existing primary Python3.11.13/Bandit1.9.3 for static AST-only scanning. Normal UI/Web runners. No install, runtime, browser, model, live-data, Git/index or parent-task actions by agent.
**Status**: In Progress

## Bounded attempt history

### Backend
- Attempt1, unchanged production:8 causal failures /12 controls passed, natural exit1, outer4.162s. Four missing legacy syntaxes and unrelated input crossed `_call_llm`; legacy endpoint returned success; shared TestRunner dispatched unresolved source; structured missing endpoint returned generic502. Valid v1/v2, explicit-value presence, structured defaults/optional/literal, replacement literals, low-level partial rendering and actual foreign-owner controls passed. No fixture/setup failures.
- Attempt2 after minimal correction:44 passed,299 warnings, natural exit0, pytest1.90s /outer3.619s.20 new cases plus affected credential/structured execution and3 actual legacy/structured executor controls. No unrelated response/provider suites replayed.

### Frontend
- Attempt1, unchanged source: all5 cases failed before submission in each runner because the readiness query expected an HTML placeholder attribute. Ant Design Select renders `Default: ollama` as text. No causal-red claim. Corrected only that query.
- Attempt2, unchanged source: both runners reached the recovery case's notification assertion after actual mounted earlier success/stats and failed on resolved400. Standalone success and rejected transport controls passed. The in-memory reporter wrapper expected multiline JSON although Vitest emitted single-line JSON, so full counts and first-two-case diagnostics were not retained; no unobserved counts are claimed and red was not replayed.
- Attempt3 after the minimal source correction: UI9/9 passed, outer4.060s. Web7 passed /2 failed, outer5.065s. Exact diagnostic: standalone success saw3 earlier notification-spy calls, including prior missing-input/fallback calls. The recovery wait could match an older identical notification before its own failure settled, allowing a retry while the mutation was pending. UI sets restoreMocks:true; Web does not. Production error/transport and affected controls passed. The path stopped at the declared limit.
- Reassessment: inspected existing explicit vi.restoreAllMocks teardown in PromptBody.search-pagination.test.tsx and RunDetailDrawer.stream-lifecycle.test.tsx, plus Web setup cleanup. Chose explicit module afterEach restoration, preserving all assertions and production.
- Root then explicitly granted ONE bounded fixture-only correction and failed-cases-only qualification under human all-efforts/all43 authorization. Added the existing-convention afterEach restoration. Only2 failed Web cases ran:2 passed,0 failed, outer1.775s;3 other module cases were filtered and retained from previous qualification. Combined retained evidence: all9 distinct UI and all9 distinct Web cases qualified; no full replay. Production hash stayed unchanged.

## Static qualification and review
- Scoped Python compile() and Ruff passed, no bytecode written. Ruff baseline/current0.
- STATIC Bandit covers both production Python files AND the new module: baseline0/current0 findings, current2308 source lines, exactly19 individually explained pytest B101 skips,0 broad nosec suppressions, no other codes. Reports kept in memory; no proof files.
- Frontend TSX parse diagnostics0 for source/new test; this is syntax-only. New test uses existing frontend Prettier configuration and its check passed.
- Scoped configured ESLint baseline/current comparator matches exactly:0 errors,5 inherited no-explicit-any warnings in ExecutePlayground; new test0 findings. Next's shared-UI invocation prints an inherited missing-pages advisory. No lint/config suppression added.
- Root independently reviewed complete backend changed hunks, executor/shared assembly, owner→credential→executor→bounded400→finally-close, TestRunner and all20 new backend test bodies: no source finding in that covered scope. Frontend independent review and combined current configured UI/Web types are pending; no source/type acceptance carried from older hashes.

## Frozen scope and limitations

Five source/test paths below plus this owned plan. Production and tests are frozen. No edits to prior reviewed units. No full HTTP/Auth, PostgreSQL, actual model/provider, live revised-v2 execution, deployed frontend or whole43 acceptance is claimed. Root owns controlled runtime loading and actual UAT. No Critical/Important/Minor source finding substantiated by agent; inherited lint warnings are reported above.

- `tldw_Server_API/app/core/Prompt_Management/prompt_studio/prompt_executor.py` SHA256 `4b5a042d8280e605dfeca00bf175a013267d5ff5b8d2b205604f6c1c93796046`
- `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_prompts.py` SHA256 `6c6207239bb119a3a3e658019ef23919b9008bf026e6c74767adb1ec5e75a387`
- `tldw_Server_API/tests/prompt_studio/unit/test_prompt_executor_required_variables.py` SHA256 `7346c3c560f29107afdfe4be2f63e0611a34ef2c8a6945f4ce75798d86a9e87c`
- `apps/packages/ui/src/components/Option/Prompt/Studio/Prompts/ExecutePlayground.tsx` SHA256 `304f244b3b7d75ac54391e9e0f50ac55c8d16a8c5e5517b463923af9a23fb03c`
- `apps/packages/ui/src/components/Option/Prompt/Studio/Prompts/__tests__/ExecutePlayground.execution-errors.test.tsx` SHA256 `aa48e4ad5ac6b86aacb89ea58b1988993b4837bee4ae7f3306697e8a07c62040`

- Root corrected final combined configured semantic types: UI natural0, no diagnostics,66.245 seconds outer (finished2026-10-08T22:47:13.773923Z); sequential WebUI natural0, no diagnostics,42.918 seconds outer (finished22:47:56.986458Z). Both use8GB/600second cap/noEmit/nonincremental. All7293 inventoried apps files match before/after; freeze digest13bf4be7bcfd0bfd39be60868cbadcbe78756b28661c26334cdafb97a99fc18c. This qualifies current Writing/C02/C12/A10/Retry source together after five ignored Testing Library option corrections. Root final source reviews CLEAR; combined production build/live UAT remains open.
