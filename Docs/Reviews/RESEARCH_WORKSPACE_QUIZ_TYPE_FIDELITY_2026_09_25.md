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

## Separate verifier finding

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
