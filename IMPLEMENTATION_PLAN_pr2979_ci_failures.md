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
