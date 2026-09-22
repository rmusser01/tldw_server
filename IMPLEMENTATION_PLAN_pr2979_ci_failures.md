# PR2979 CI failure repairs

Task: TASK-13260.278.17. Native/full UAT remains paused.

## Stage 1: Establish causal failures
**Goal**: Read exact GitHub failures and reproduce guard and UI fixture errors.
**Success Criteria**: Each failed gate has a concrete cause recorded without disabling checks.
**Tests**: Guard positive/negative cases; actual Playground integration suite; critical E2E environment inspection.
**Status**: Complete

## Stage 2: Repair the owning contracts
**Goal**: Correct endpoint matching, synchronize test state with the production provider contract, and provision the critical journey requirements.
**Success Criteria**: Existing checks pass with their original behavioral coverage retained.
**Tests**: Focused regressions, workflow contracts and lint; real backend journey checks when repairing CI, without restarting full UAT.
**Status**: Complete

## Stage 3: Verify and publish
**Goal**: Record local and remote evidence in the tracker and PR.
**Success Criteria**: Relevant checks pass, remaining findings remain explicit.
**Tests**: Touched-scope Bandit, diff review and fresh remote CI.
**Status**: In Progress

## Root causes and local verification

UAT419: the guard matched a prefix of complete-v2, and its duplicate test scanned a nonexistent nested tests directory. Actual guard now has endpoint-boundary coverage;2causal failures become7passes. Playground mock omitted required effectiveAssistantState;12causal failures become12passes.

GitHub35692674490 backend startup rejected TEST_MODE=1 outside pytest, while its readiness loop still returned success. All3tier jobs remove this invalid flag and fail closed on process exit/readiness exhaustion. Critical journeys retain all assertions, explicitly disable application API stubs and declare deterministic downstream mode. The existing mock OpenAI server provides three bounded committed source fixtures; it is never represented as real-model acceptance.

Final local workflow/admission147pass; provider/guard13pass; actionlint succeeds. New fixture JSONs are authored test data under the existing mock provider, explicitly staged despite the global JSON ignore. Generated evidence stays excluded. Remote critical checks are still required.

Run35695635379 then exposed authenticated-provider readiness401. The probe omitted Authorization and the synthetic backend key lacked the mock server required sk- prefix. Both now use the same accepted synthetic credential, retaining require_auth. A real FastAPI HTTP regression proves unauthenticated requests fail and the configured backend credential lists the required model. Provider/workflow suites154pass; remote follow-up remains required.

Run35696456100 confirms provider startup succeeds, then backend health returns401 for the full readiness deadline. After removing standalone TEST_MODE, the real protected application health route requires the configured API key. All three tier probes now send X-API-KEY, consistent with the existing live-tier runner. Causal workflow test fails before this repair;154 provider/workflow checks pass after it. Actionlint and Ruff pass. No authentication bypass added.


Run35697113094 first browser pass:36passed,7failed,2flaky,1conditional skip. The second whole-suite pass exhausted the30minute job timeout, overwrote the test-results directory and prevented upload. CI now runs one first-attempt pass with retained traces/JSON, a20minute step deadline and always-on upload of its explicit browser/backend/provider evidence paths. The contract fails before repair;16workflow contracts pass afterward.

Notes failures show the actual Joyride overlay intercepting Save: authenticated fixtures omitted the shared notes-tutorial-shown prerequisite while production auto-starts the tour after1second. ChatPage readiness incorrectly treated every Next dev-tools portal as an error, reloaded in-memory QA handoffs and removed dialogs. Five causal helper failures become7passes after seeding the Notes onboarding prerequisite and making Chat readiness observational. This explains a confirmed harness defect; the exact remote QA navigation failure still requires its retained trace. Character Phase6 toggled a healthy cockpit into focus when sessions were not immediately visible; mobile starts in focus before the mode chip exists. Toggle only when aria-pressed=true before panel assertions. Watchlists removed repeated controls and moved the palette behind Help; journey follows Help and current Updates labels. Preserve all source/persistence/run/inbox assertions. Remote first-attempt rerun is required.

Final first-attempt repair verification:24workflow/provider tests,7frontend helper tests, actionlint/Ruff and frontend TypeScript pass. ESLint3helper warnings predate this change; Bandit additions are test assertions. The conditional Character configuration-recovery case now declares a model and uses the existing downstream scenario-failure mechanism matched only to its exact trigger prompt; actual providerHTTP regression returns404for that case and200for unrelated success. This is controlled inference, not live-provider acceptance. Independent reviewer capacity was unavailable; root reviewed the complete bounded diff.

Actual headless Chromium on local static HTML reproduces the pre-fix readiness damage (handoff resets, recovery dialog removed) and verifies the repaired helper preserves both. No application runtime or nativeUAT launched. Remote frontend shard1 then exposes an existing source-string guard still requiring direct Open Activity navigation; it now requires the real Help/command/Updates path and retains the run/request/Activity/notification contracts.15combined helper/guard tests and scoped lint pass. Critical run35741620414 remains in progress; do not cancel it for this guard-only follow-up.


Run35741620414 completes one first-attempt pass:40passed/6failed/0skips. Retained JSON, traces, backend and provider logs support the next diagnoses. Strict helper enabled offline bypass and Notifications correctly refused unauthenticated lifecycle startup; strict live fixtures now force verification and clear stale bypass. Notes generation correctly failed closed when the CI provider lacked the claim-verifier response. Character SSE provider failures lost their safe classification. The WebUI navigation callback changed each render, retriggering Media permalink synchronization while Chat route loading was pending; causal pre-fix sequence media/chat/media becomes media/chat after callback memoization (18 navigation controls pass). All browser behavior gates stay enabled; remote rerun remains required.

Final follow-up verification:175 combined backend/workflow/provider tests;38 frontend navigation/helper/guard checks;50 Character frontend checks (147 Character backend cases are included in175). Final frontend TypeScript and actionlint1.7.12 pass. Scoped lint retains baseline diagnostics only; Character Bandit has0production findings and test assertions only. Independent navigation, PostgreSQL cleanup, Character framing/responsive controls, and Notes fixture reviews have no actionable findings. Actual static Chromium helper controls fail before both Phase6 repairs and pass afterward. No native/full UAT. Publish this batch, retain completed CI evidence and await fresh first-attempt run before closing419.


Second retained run35745724724:41passed/5failed/0skips/retries. The new boundary failures are repaired without changing original source/persistence/inbox expectations: real inbox Dismiss was clicked by a generic helper; open notification SSE blocked JSON capture; Character error consumption cancelled CDP body retention and its settings button label was stale; Next public router republication invalidated callback identity; legacy redirect awaited an uncompleted development chunk.82combined focused cases and17workflow contracts pass, frontend types clean, lint0errors/8baselinewarnings, Bandit7addedtestassertions only. Existing optimized advanced build/start replaces development compilation in criticalCI and retains both logs; explicit bash propagates pipeline failure. Build/token/budget checks pass577.8KB<600KB. Source review and causal controls complete; final independent reviewer unavailable after identifying pipefail, so remotefirst-attempt confirmation remains required. No native/fullUAT.
