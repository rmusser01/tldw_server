# UAT C02 Execute displayed provider and model defaults

Related task: TASK-13260.278.18.83.45. Root appended the new client-only unit through backlog-py before these edits and owns the parent task.

## Source attribution and approved design

The connected `/llm/providers` catalog returns API.default_api as default_provider and provider-specific default_model (configured first model, with canonical override support). Existing Execute helpers normalize that catalog and the modal displays its effective defaults. The submitted payload formerly omitted blank selectors. executePrompt forwards the payload directly; the compatibility request schema/endpoint defaults omitted provider to openai and resolves omitted model from that provider's frozen credential snapshot. Thus a displayed Ollama default can dispatch OpenAI, and a selected Ollama model with blank provider can dispatch the wrong pair. No live/catalog/config/model request is replayed here.

Resolve effective provider and provider-specific model once from submitted form values with the existing helpers. Validate that same target with the existing validators and send it explicitly. Explicit user choices win; a selected provider uses its own default model. If no catalog target exists, preserve optional-field omission and canonical backend compatibility. Required-input validation, the reviewed failed-envelope handler and all backend owner/credential/adapter policies remain.

ADR check: no new ADR. This repairs client adherence to existing catalog/default and endpoint request contracts without changing API shape, backend compatibility defaults, provider routing, endpoints, security, storage or dependencies. ADR-025 (`Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md`) governs the unchanged provider integration boundary; source/catalog behavior and existing Execute helpers determine displayed target resolution. Docs/ADR/README.md and ADR-025 were read; no durable architecture rule changes.

## Stage 1: Mounted causal regressions
**Goal**: Demonstrate displayed target and actual service payload divergence on unchanged production.
**Success Criteria**: Blank, model-only and provider-only cases fail for literal expected payloads; explicit pair/empty catalog controls pass. Real Form/Select/Drawer/QueryClient and browser-like selector interaction, with only service and notification seams isolated.
**Tests**: Normal configured UI/Web runners, outer600/case300 guards, maximum3 bounded attempts. Explicit afterEach spy restoration from the outset; no undefined partial-match assertions.
**Status**: Complete

## Stage 2: Resolve and dispatch one effective target
**Goal**: Validate and send the displayed provider/model using submitted values and existing helpers.
**Success Criteria**: Defaults and explicit choices match dispatch; invalid effective defaults retain existing client refusal; no backend edits. Existing error test changes restricted to4 expected request fields (provider/model in2 assertions).
**Tests**: New mounted default/selection/denial cases plus genuinely affected existing Execute error/design-system/provider helper cases only.
**Status**: Complete

## Stage 3: Scope qualification and freeze
**Goal**: Supply exact frozen client source/tests for independent root review and one combined current type/build handoff.
**Success Criteria**: Scoped baseline/current configured lint has no new findings; syntax/new-test formatting pass. Root owns combined current types, review and runtime.
**Tests**: No agent types, backend, Bandit-on-unchanged-Python, config, runtime, model, environment, dependencies, parent-task or Git/index actions. Client-only scope means Python Bandit is not applicable; prior backend security evidence stays frozen.
**Status**: In Progress

## Qualification evidence and bounded path
- Step1 causal red on unchanged production: UI6 causal failures /2 passing controls, outer6.987s; Web6 causal failures /2 passing controls, outer6.438s. Natural exits1, normal configured runners and outer600/case300 guards. All mounted selector interactions reached independent literal payload/guard assertions; no fixture/setup failures. Explicit pair and no-catalog omission passed.
- Baseline scoped configured ESLint:0 errors;5 inherited no-explicit-any warnings in production; both touched tests0 findings. No suppressions.
- Step2 patch-script verification error: after applying the intended handler patch, a reversible-substring guard also matched the unchanged default_provider fixture key, stopped before writing the prior test or plan, and reported Change exceeded authorized request fields. This is a wrong verification oracle, not a production or test failure; no pytest/Vitest ran. Corrected only that oracle to exact changed-line pairs. Prior test remained its reviewed aa48e4a baseline until the guarded four-field write.
- Minimal production resolves two local values from submitted fields/catalog, runs existing provider/model validators on those values, and sends that exact target. Reviewed error-envelope branch stays unchanged. Prior test now differs only in4 exact expected request lines (provider/model in2 assertions), verified against its frozen baseline. Every result/error/reset/recovery assertion remains unchanged.
- One remaining bounded step3 is green/affected qualification; stop and reassess if it fails. No silent retry.

- Step3 green/affected qualification:17 passed /0 failed /0 filtered in each normal configured runner; UI outer9.412s, Web outer9.445s, natural exits0.8 new mounted default-target cases plus5 genuinely affected error-envelope/recovery cases,1 existing design-system case and3 provider-helper controls. No fixture failures, spy leakage, partial undefined matches or broad old tests replayed.
- Final scoped configured ESLint baseline/current findings comparator matches exactly:0 errors,5 inherited no-explicit-any warnings in production; both touched test modules0 findings. Existing shared-UI Next missing-pages advisory remains. No configuration or suppressions changed.
- All3 touched TSX files parse without syntax diagnostics; both touched test modules pass existing frontend Prettier configuration check; production diff whitespace check passes. Syntax is not semantic type qualification. Root alone will run the combined current configured UI/Web pair after reviewed freeze.
- Python source/tests are unchanged in this client-only unit; Bandit is not applicable to TSX. No backend/security test replay or stale acceptance claim. Existing provider/owner/credential/auth/runtime fences are unchanged. No endpoint/model/config/network/runtime/browser action occurred.
- Source and tests frozen at hashes below. Root independent source/all8 new mounted bodies review and combined current types/build/live actual target validation remain pending. No Critical/Important/Minor finding substantiated in this final agent scope; inherited lint warnings are retained. No whole43, live model, deployed frontend or fullHTTP acceptance claim.

## Frozen scope
- `apps/packages/ui/src/components/Option/Prompt/Studio/Prompts/ExecutePlayground.tsx` SHA256 `431473f7ad92a4cad6441c10942def0ceff7f854ff11d2c0b6d5aebb67d16fef`
- `apps/packages/ui/src/components/Option/Prompt/Studio/Prompts/__tests__/ExecutePlayground.default-target.test.tsx` SHA256 `92d093ac2efa37cb1711229db59483b65c86a426ce50511c61524f4b3771b895`
- `apps/packages/ui/src/components/Option/Prompt/Studio/Prompts/__tests__/ExecutePlayground.execution-errors.test.tsx` SHA256 `3aceaaa868c22f94fcb34b30cbe71c40a65228a53765fe04abfb6a277819d0f5`

## Root type qualification correction
- Root independent default-target source/full8 new mounted bodies/4 prior expected fields review completed CLEAR at the preceding hashes. The first combined current UI type run then reported only5 TS2769 errors: ByRoleOptions does not accept exact. New default-target lines84,90,217,245 and prior execution-errors line55 used that unsupported option. Web types did not run. This is a test API typing failure, not production or behavioral red.
- Read the actual installed @testing-library/dom10.4.1 ByRoleOptions and role/matches implementations: ByRoleOptions has name and no exact property; role.js matches computed accessible names with matches; string matcher uses strict equality. The unrecognized exact property is not read. Removing it preserves exact string matching and every name, assertion, selector interaction and test result.
- Corrected only those5 ignored properties across2 test modules; production hash unchanged. No behavioral replay justified/permitted for this ignored-option removal; retain17 UI/17 Web greens, pending root rerun of current combined configured semantic types. Scoped current test syntax/lint/format checked separately. No app/runtime/backend/config/environment changes.

- Current tests-only static qualification after ignored role-option removal: configured ESLint0 errors/0warnings; TSX syntax0; existing Prettier checks pass. Current test hashes in Frozen scope reflect this correction. Production remains unchanged, no behavioral replay; root combined semantic type rerun pending.

- Root corrected final combined configured semantic types: UI natural0, no diagnostics,66.245 seconds outer (finished2026-10-08T22:47:13.773923Z); sequential WebUI natural0, no diagnostics,42.918 seconds outer (finished22:47:56.986458Z). Both use8GB/600second cap/noEmit/nonincremental. All7293 inventoried apps files match before/after; freeze digest13bf4be7bcfd0bfd39be60868cbadcbe78756b28661c26334cdafb97a99fc18c. This qualifies current Writing/C02/C12/A10/Retry source together after five ignored Testing Library option corrections. Root final source reviews CLEAR; combined production build/live UAT remains open.
