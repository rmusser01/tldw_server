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
