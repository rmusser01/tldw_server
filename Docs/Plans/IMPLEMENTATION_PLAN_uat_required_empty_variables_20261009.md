# C02 required empty variables
Task: TASK-13260.278.18.83.45.
Actual Admin multi-user SQLite WebUI reproduction: negative case #2, prompt #2/v2, saved run #6 called the actual existing Ollama model with an empty source and returned Rowan Observatory. Preserve this failed-acceptance result plus original prompt/revision/case/results.

## Scope and design
Required execution text values must be present and nonblank; zero and false remain supplied values. Legacy exact user-template tokens validate at PromptExecutor's shared request boundary. Canonical structured v1 required variables validate at their existing resolver, so normal Studio, preview and MCP assembly callers agree; optional explicit empties and absent defaults remain. Single-text render/apply is a separate existing contract and is unchanged. Replacement content and low-level partial legacy rendering are unchanged.
Keep typed input errors out of provider sanitization in TestRunner. Strict callers receive the typed error before successful persistence; the owned /test-cases/run endpoint exposes only a bounded400 required-input message after owner/project/case gates. Existing runtime close, credentials and genuine provider-error behavior stay.
The older presence-only required-null/empty test ruling is deliberately superseded by C02's required-empty acceptance; retained zero/false assertions and optional/default controls must prove the narrower boundary. No input content is returned in errors.
ADR assessment: no new durable architecture rule; existing variable validation and ADR055 backend-neutral storage govern. Docs/ADR/README.md searched.

## Stage 1: Causal regression
**Goal**: Reproduce empty execution and misleading input-error classification through real configured DB/executor/runner/endpoint paths.
**Success Criteria**: Red before production changes; legitimate literal/zero/false/optional/default/ownership controls remain.
**Tests**: New required-empty tests, normal Python3.12 pytest plugins,600 outer/300 per-case. Only external adapter/credential seams isolated.
**Status**: Complete

## Stage 2: Shared correction and affected qualification
**Goal**: Reject required empty inputs before provider dispatch and expose actionable bounded input failures.
**Success Criteria**: New cases green; affected required-variable/structured/runner/runtime/endpoint consumers pass; no new touched security findings.
**Tests**: Exact affected suites, compile/Ruff and baseline/current Bandit in memory. Preserve prior WIP and all original unrelated controls.
**Status**: In Progress

## Stage 3: Review and live correction
**Goal**: Independent actual-source review, guarded owned multi-user source loads, and bounded corrected-source negative-case validation with visible UI guidance.
**Success Criteria**: Live400 guidance, no new successful run/provider dispatch; original run #6 and earlier successful v1/v2 provenance retained. Publication and other C02 variants remain open.
**Tests**: Existing Admin tab/model/case, declared correction counters; fixed15-second owned grace/120-second startup, no force/retry/new windows. Actual backend refusal exposed a separate invisible-notification defect. Reuse mutation error in an inline Alert and reset it on Close; qualify the changed UI consumer before one new build/load and correction check.
**Status**: In Progress

## Attempt record
- Initial operator command used an uppercase directory spelling; pytest selected no items/natural4. No causal evidence. Canonical rg inventory uses tests/prompt_studio.
- Actual unchanged-source new31-case run:9 causal failures(6 executor provider-dispatch checks/3 required structured blanks),10 valid-value/default controls passed.12 new authored endpoint/runner/owner cases failed before assertions because create_test_case requires a name; corrected ONLY those three fixture callsites.21failed10passed176warnings/6.33pytest8.027outer/natural1. No endpoint/owner causal-red claim from those12.
- Shared production correction is now authored; next exact affected qualification keeps600/300 guards. Old presence-only null/empty parameters intentionally replaced by new rejection coverage; retained zero/false assertions unchanged.

## Current qualification and limits
- Backend new required-empty paths pass within the completed188 affected cases. Two unrelated full-app preview fixtures fail at startup before assertions; no190-pass claim or repeat of188 passes. Authoring correction25passes plus one fixture-corrected signature case pass separately, not one26-case run. Touched8 baseline/current Bandit0findings/0errors and independent corrected source review clear.
- Corrected API26183 ready with original private config/data. Both initial corrected validation and one observation recovery consumed: no visible refusal or new success history. This is the causal basis for the new inline UI error; no live400/provider-dispatch claim.
- New mounted refusal test fixture errors were corrected from real Studio/installed AntD source without weakening visibility or request assertions. UI two new cases now pass; earlier16 original passes retained. WebUI18pass/one existing optimization5000ms timeout; no complete19green claim. One configured8GiB sequential typecheck recovery per client follows natural default-heap exits134.
- Independent current modal/new test review clear in covered transport/caller/lifecycle scope. It does not qualify in-flight cancellation or live UI. New frontend build/load and actual corrected guidance check remain pending.

## C02 shipped-translation correction
- The one corrected UI live Run ACK278ms then replaced the dialog with the actual Prompts error boundary. No live400/provider/no-new-history acceptance; request was not repeated. The prior frontend graceful shutdown15.001s failed its bound even though old owners later exited naturally0; preserved as an unaccepted timing result. Corrected frontend had normal startup1.058s.
- Real English resources and existing ICU/page boundary added to the mounted refusal tests. Baseline2failed/5.238outer/natural1 explicitly reproduced React invalid-object children from common.error. The earlier mocked translation results remain scoped and did not cover this boundary.
- Minimal correction:26 error-label references in14 Studio callers now use the existing scalar common:error.label; all source equals the pre-unit text after reversing only these literals. No shared resource/i18n, handler, request, selection or validation changes. Current real-mounted tests retain every original request/visibility/close assertion plus boundary and scalar toast checks. ADR assessment unchanged: reuse existing translation/error display contract, no durable architecture decision.
- Next: genuinely affected Studio suites in both configured clients, touched semantic/lint checks and independent new-delta review; only after qualification one new frontend build/load and corrected live check. Existing runtime/build attempts remain consumed and are not reused.

## Localization qualification and corrected live attempt
- UI70 prior passes retained; only4 failed fixture cases corrected and now pass separately (3.120 seconds); WebUI11modules/74passes (45.370 seconds). No combined UI74-run claim. Complete UI/WebUI typechecks0 at unchanged8GiB; scoped lint has no new diagnostics. Both new-delta source reviews clear. Original assertions preserved.
- Declare one new localized-error frontend build (600-second outer), one owned frontend load (15-second grace/120-second startup), and one required-empty case2/prompt2v2 Run in existing Admin tab4. API/profile/config/data and previous attempt outcomes retained; no new window/tab or request replay. Stage2/3 remain In Progress until actual corrected guidance.

## Corrected live outcome
- ONE localization build natural0/94.039s/token0/bundle0/shared591.5KiB; source7411 equal/next-env exact. One normal existing tab4 API-ready navigation preceded owned frontend grace0.015s/startup0.528s, API/config/data unchanged. Current build ApRRboXk8-qKSpGiShV_u.
- Exact saved case2/prompt2v2/Ollama qwen2.5:0.5b Run ACK271ms/refusal278ms: inline Error and Required prompt inputs are missing or empty; dialog/model retained/no Results/no page crash. Close dismissed. Complete six-run history1–6 retained/no new saved success. This narrow SQLite multi-user WebUI missing-variable flow is qualified. Actual HTTP status/provider tracing not observed; no live400/provider-count claim.
- Broader backend full-app preview setup errors, C02 comparisons/optimization/timeouts/concurrent/cancel and other mode/client variants remain open; retain plan and Stage2/3 In Progress. Continue remaining UAT without replaying this consumed case/build/load.
