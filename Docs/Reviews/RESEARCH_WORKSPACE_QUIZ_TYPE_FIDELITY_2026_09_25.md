# Research Workspace quiz type fidelity validation

TASK-12020.57 makes legacy `question_types` an allowed-type set rather than an
exact count plan. Generated raw types are checked before normalization, claims
verification, or persistence. Explicit `question_plan` still enforces exact
counts. Prompt examples are selected-type and generation-profile specific.

## Verification on the original branch

- Full backend Quizzes suite: 338 passed; focused response/prompt suite:
  50 passed; adjacent question-plan/test-mode suite: 143 passed.
- Prompt tests cover every nonempty type subset. Planned examples use validated
  option and pair counts, including two-option multi-select answer indices.
  Integration tests cover wrong-type rejection before claims/persistence and
  legacy multi-select/matching persistence.
- Real llama.cpp model on 127.0.0.1:9099, backend on 8001, and Next.js WebUI
  on 8080: Research Workspace real-backend Chromium quiz spec passed (1
  executed, 0 unexpected). The spec asserts the request selects MC/TF, every
  persisted question is MC/TF, the claims verdict is grounded, and the quiz
  opens in the native Quiz page. Runner evidence:
  `/tmp/task12020_57_uat_countaware_evidence.json` (local, ephemeral).

## Validation after porting onto current dev

- Final full Quizzes suite: 631 passed, 4 skipped. Focused type-fidelity and
  prompt tests: 64 passed. The first port run exposed a Best-of-Five error-type
  compatibility failure; the targeted regression passed after preserving that
  profile's error contract.
- Ruff, scoped ESLint, test-file Black, Bandit on `quiz_generator.py`, and
  frontend TypeScript typecheck passed.
- Four real-model Chromium attempts reached quiz generation but did not
  validate successful persistence on current dev. With llama.cpp automatic
  reasoning, generation exhausted the 2,000-token output budget and returned
  unparseable or empty content (HTTP 400). With `--reasoning off`, generation
  returned JSON but cited an unselected source twice; strict provenance
  correctly rejected both responses with HTTP 422 before persistence. These
  runs must not be reported as passing UAT. Local evidence:
  `/tmp/task12020_57_dev_uat_nothink_retry_evidence.json`. Follow-up:
  TASK-12020.60.

## TASK-12020.60 follow-up

Raw local-model probes showed that the citation rejection was a representation
mismatch: `source_type="media"` with `source_id="media:59"` rather than the
selected canonical ID `"59"`. The prompt's combined `type:id` contract caused
this ambiguity. Presenting separate JSON field pairs produced six MC/TF
questions with canonical IDs in a controlled non-reasoning probe.

The quiz source contract now shows those separate fields. A qualified ID is
canonicalized only if its source type and remaining ID exactly match a selected
source; exact selected IDs containing colons remain unchanged. Unselected IDs
and wrong source types still fail strict provenance before claims/persistence.
Media citations retain their numeric media reference.

A separate automatic-reasoning probe returned `finish_reason=length`, 2,000
completion tokens, zero content, and 6,672 characters of reasoning. Quiz
generation now reports a specific `max_tokens` exhaustion error before JSON
parsing. For the non-reasoning UAT configuration, use llama.cpp server
`--reasoning off` or `LLAMA_ARG_REASONING=off`; the backend environment variable
`LLAMA_CPP_ENABLE_THINKING=false` did not configure the current server.

The new red/green regressions and focused property/integration suite passed
(78 tests). Review also corrected the prompt instruction to preserve canonical
IDs containing existing colons or prefixes. The first updated real-browser run
passed the citation boundary
but failed in ClaimsEngine with HTTP 500 because two `SourceAuthority` enum
objects were compared by `max()` without a numeric key. The earlier working
branch's authority-ranking fix was not included in the focused current-dev
port; bringing that dependency forward is tracked as TASK-12020.61.
Live evidence remains non-passing:
`/tmp/task12020_60_uat_evidence.json`.

The full Quizzes rerun collected 644 tests but terminated at 68% with filesystem
`OSError` failures and exit code 120 as available disk space fell from 4.9 GB to
467 MB. This is not a passing full-suite result. The final focused run passed
78 tests, including the review regression added after full-suite collection.
Ruff, scoped ESLint, test-file Black, Bandit, frontend typecheck, and
`git diff --check` passed. Full-suite and live browser completion remain blocked.

## TASK-12020.61 authority-ranking follow-up

After requester approval to continue, regressions reproduced the shared enum
ordering defect: seven tests failed and four passed. Multiple evidence sources
caused `TypeError` in the LLM path; the NLI path caught the same error and
returned an unverified result instead of its supported verdict.

Both authority-selection sites now rank by `SourceAuthority.value`, retaining
the enum object and the `SECONDARY` empty-evidence default. Regressions cover
empty, single, repeated, and mixed authorities in both paths, plus property
checks for order independence. The regressions and adjacent engine modes,
status fallback, configuration, and artifact-verification tests passed
(49 tests). Ruff, test-file Black, and Bandit passed. No unrelated predecessor
changes were brought forward. Successful live browser validation still awaits
sufficient free disk space; neither this fix nor the citation changes are
reported as end-to-end validated.

## Final current-dev live validation

After free space was restored to 120 GB, the full Research Workspace Chromium
quiz workflow passed against the real backend and local llama.cpp Gemma model
with `--reasoning off`: one executed, zero skipped, flaky, or unexpected tests.
Generation returned HTTP 200 with six questions and a `grounded` claims verdict.
The browser test confirmed the requested MC/TF types, persisted citations with
canonical selected-media IDs and numeric `media_id`, native Quiz page access,
workspace visibility, and move-to-general without changing the quiz record ID.

Evidence: `/tmp/task12020_61_uat_evidence.json` (`productPassed=true`,
`failureScope=none`) and `/tmp/task12020_61_uat_report.json`. These are local,
ephemeral artifacts. The fresh adjacent claims suite passed 49 tests. Final
Ruff, test-file Black, Bandit on both production modules, scoped ESLint, and
frontend typecheck passed. The validation services were stopped after the test.
The broader Quizzes suite passed: 641 passed, 4 skipped, and 4 warnings in
672 seconds. Log: `/tmp/task12020_61_quizzes_final.log`. The earlier disk-blocked
attempts remain recorded above, but are superseded by this completed run.

## Latest-dev Refresh (2026-09-29)

TASK-12020.63 rebased the three PR commits onto `dev` at `0da68530e8` without
conflicts. `git range-diff` confirms identical patches. The refreshed branch
passed 127 focused quiz, claims, property, and artifact tests; the shard guard,
Ruff, Bandit, scoped ESLint, and frontend typecheck also passed.

The real llama.cpp full-application Chromium quiz workflow passed again:
one executed, zero skipped/flaky/unexpected, with grounded claims, canonical
persisted selected-source citations, requested question types, and native page
access. Evidence: `/tmp/task12020_63_uat_evidence.json` and
`/tmp/task12020_63_uat_report.json` (local, ephemeral). The prior full Quizzes
run remains 641 passed/4 skipped; it was not repeated for this refresh.

ADR check: no new ADR is required because this refresh preserves the existing
generation, provenance, and module-boundary rules.

## Qodo Follow-up (2026-09-29)

TASK-12020.64 addresses all three findings on head `11bf4a133d`. Eight real-DB
regressions reproduced missing tags for legacy and explicit-plan multi-select
and matching questions across both standard and mixed profiles. The shared
normalizer now uses existing tag coercion, preserving ordinary tags while
deduplicating and filtering reserved profile tags. The prompt-shape helper is
documented; new Python regression functions and helpers have type annotations.

The focused plan/profile/prompt/provenance/claims suite passed 126 tests with
four dependency warnings (`/tmp/task12020_64_regressions.log`). Ruff, test-file
Black, and scoped Bandit passed; test Bandit excludes only ordinary assertions
(B101). The prior full-application llama.cpp run is not a fresh tag-specific
browser test: this follow-up verifies tags through actual database persistence,
with external inference stubbed. Hosted final-head CI and re-review remain
pending. ADR required: no; this restores existing metadata behavior without
changing architecture or claims-verification policy.

## Dev6110 Refresh (2026-09-29)

After PR3036 advanced `dev` to `6110d2ae43`, all five PR commits rebased
without conflicts. `git range-diff` confirms identical patches. A fresh run
on Python 3.12.11 (the new supported minimum) passed 145 quiz, plan, profile,
prompt, provenance, authority, and artifact-verification tests with 12 warnings
in 93.72 seconds (`/tmp/task12020_64_dev6110_py312.log`). Ruff, test-file Black,
Bandit on both production modules, whitespace checks, and the updated shard
guard passed (`new_uncovered=0`). Earlier browser and full-Quizzes runs remain
historical evidence, not fresh runs on this base. Final-head hosted CI and
review must complete before protected merge. ADR assessment is unchanged:
no new architectural decision is introduced by this refresh.

## Dev6074 Refresh (2026-09-30)

After PR3053 advanced `dev` to `607431154c` with FastAPI 0.141.1 and nested
route-introspection compatibility changes, all eight PR patches rebased without
conflicts and remained identical by `git range-diff`. A separate temporary
Python 3.12.11 environment using FastAPI 0.141.1 and Starlette 1.7.0 passed
230 quiz, plan, profile, prompt, provenance, authority, artifact, multi-source
generation endpoint, and quiz endpoint integration tests: 2,618 warnings in
205.40 seconds (`/tmp/task12020_64_dev6074_py312.log`). Ruff, test-file Black,
Bandit on both production modules (zero findings/errors), whitespace checks,
and the shard guard (`new_uncovered=0`) passed. No fresh llama.cpp browser run
is claimed. Hosted checks and Qodo review must finish on the refreshed head.

The preceding head's UX smoke job failed its no-flaky-tests gate after an
initial health-probe socket reset; the cockpit test passed on its own retry.
One unchanged-job rerun was requested, but remains queued at publication.
Diagnostics are retained at `/tmp/pr3018-ux-smoke-36636268090`; this is not
reported as a successful UX gate. ADR assessment remains unchanged.

## Full-suite Route Lookup Follow-up (2026-09-30)

On head `82fc1a63bf`, the UX smoke gate passed without weakening its no-flaky
policy. The `gap-verified-4` full-suite shard instead failed one ingestion
capabilities test with `StopIteration` (1,007 passed, 1 skipped). Its direct
`app.routes` lookup missed the included router after the FastAPI upgrade.
The failure reproduced locally before the fix. The test now reuses
`iter_served_routes`, like sibling ingestion tests, and retains the exact
response-model identity assertion on the original route.

All 43 ingestion-policy, sibling API, and route-helper tests passed under
Python 3.12.11/FastAPI 0.141.1: 12 warnings in 8.70 seconds. Ruff, Black on the
changed test function, scoped Bandit excluding test assertions (`B101`), and
whitespace checks passed. No production code or CI gates changed. Final-head
hosted CI and review remain pending; no fresh browser run is claimed.

## Devf3f1 Resource Governor Refresh (2026-09-30)

Rebased onto `f3f1b4fdbe3fe461b371ece30887c5fff8476d9d` after PR3066
changed shared Resource Governor policies, memory/Redis accounting, ingress,
and auth charging. Fourteen preceding patches replayed identically by
`git range-diff`. The ingestion-test patch required one overlap resolution:
the base already uses `iter_served_routes`; its served-route lookup and exact
response-model identity assertion remain intact, alongside our docstring,
type annotations, and module-level helper import.

Fresh validation in the isolated Python 3.12.11/FastAPI 0.141.1 environment:
- 230 quiz, plan, profile, prompt, provenance, authority, artifact, and API
  tests passed: 2,617 warnings in 194.20 seconds
  (`/tmp/pr3018-devf3f1-backend.log`).
- 43 ingestion-policy, sibling API, and route-helper tests passed: 12 warnings
  in 9.57 seconds (`/tmp/pr3018-devf3f1-ingestion.log`).
- 102 Resource Governor safety-net, policy, ingress/cookie-owner, MCP fallback,
  and auth single-charge tests passed: 6 warnings in 3.31 seconds
  (`/tmp/pr3018-devf3f1-rg.log`).

Ruff, four test-file Black checks, Black on the changed ingestion function,
whitespace, and shard coverage (`new_uncovered=0`) passed. Scoped Bandit found
zero findings/errors in both production modules (3,614 LOC) and the ingestion
test (364 LOC, excluding assertion rule `B101`). Shared environments, stashes,
the worktree, and untracked `:memory:.ses` were preserved. No fresh browser UAT
is claimed. Hosted CI and review must refresh on the published head before
protected merge. ADR assessment remains unchanged.

## Separate Verifier Finding

Two live runs with the same model failed closed at claims verification: a quiz
answer that 55 should be flagged invalid was marked refuted although the source
said readings above 50 are invalid. The judge returned `NLI contradiction
(1.00)` for that claim and the API returned 422 without persisting the quiz.
The type-fidelity UAT source now includes an explicit 55 example so this test
does not depend on the judge's numerical inference. This does not fix the
verifier's false positive; it needs separate diagnosis and regression coverage.
Tracked as TASK-12020.59.

Black's check of the full production module also finds pre-existing formatting
differences outside this task, so the module was not wholesale reformatted.
