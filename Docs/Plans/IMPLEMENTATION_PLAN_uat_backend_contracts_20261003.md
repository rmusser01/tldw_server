# UAT Backend Contracts Implementation Plan

Associated task: TASK13260.281.1 (umbrella TASK13260.281). Source-only repair; the coordinator owns Git and PR operations.

**Goal:** Repair S02 backup serialization, A05 plaintext preservation, C01 prompt search, and B02 saved character generation metadata.

**Architecture:** Correct each conversion or validation boundary in place. Preserve character images with explicit base64 encoding and restore bytes on import. Preserve authoritative normalized snapshot sampling while allowing only recognized generation metadata.

**Tech Stack:** Python, SQLite, pytest, FastAPI, Ruff, Bandit.

## Stage 1: Plaintext preservation
**Goal:** Preserve TXT and Markdown reader output without conversion cleanup.
**Success Criteria:** Leading whitespace, tabs, blank lines, CRLF, and final LF survive conversion and unchunked processing.
**Tests:** Parameterized exact-content assertions in `tests/MediaIngestion_NEW/unit/test_plaintext_conversion.py`.
**Status:** Complete

- [x] Write and run failing preservation tests.
- [x] Limit whitespace cleanup to converted document formats.
- [x] Run plaintext conversion, safe-path, and analysis tests.

## Stage 2: Literal prompt search and fallback
**Goal:** Search ordinary names safely through FTS and reach the existing fallback after FTS errors.
**Success Criteria:** Hyphenated names and quotes search successfully; fallback respects requested fields and pagination.
**Tests:** Real SQLite tests in `tests/Prompt_Management/test_prompts_db_v2.py` exercising the FTS path explicitly.
**Status:** Complete

- [x] Reproduce punctuation failure and wrapped FTS-error fallback.
- [x] Quote literal FTS text, catch the database wrapper error, and avoid unfiltered results.
- [x] Run the real SQLite database search tests.

## Stage 3: Saved character generation metadata
**Goal:** Permit recognized generation metadata already captured into snapshot sampling.
**Success Criteria:** History and native fork projection retain authoritative sampling and reject unknown extension effects.
**Tests:** Existing history-context and native-fork unit suites, including normalized sampling and strict unknown-carrier cases.
**Status:** Complete

- [x] Add failing history and fork regressions.
- [x] Accept recognized generation fields without deriving fresh sampling from raw extensions.
- [x] Run projection suites and generation preset tests.

## Stage 4: Truthful image-bearing account backup
**Goal:** Export and restore character image bytes, and fail jobs on unsupported character data.
**Success Criteria:** Complete image-bearing character is listed in the manifest and round-trips; serialization failure yields a failed job and no downloadable archive.
**Tests:** Real database export/import plus unsupported-data job tests in `tests/Chatbooks/test_chatbooks_full_account_export_contract.py`.
**Status:** Complete

- [x] Reproduce missing image-bearing character and misleading completed job.
- [x] Encode/decode image data explicitly and serialize before creating the character file.
- [x] Propagate collection failure to existing job failure handling.
- [x] Run full-account export, import, manifest, and job tests.

## Stage 5: Verification and handoff
**Goal:** Deliver reviewable changes with precise verification scope.
**Success Criteria:** Relevant tests, touched-scope Ruff checks, and Bandit pass; no live UAT claims or dependency changes.
**Tests:** Run focused suites, Ruff, and Bandit from the primary project virtual environment.
**Status:** Complete

- [x] Review the final source diff and run formatting/lint checks on changed code.
- [x] Run Bandit on touched Python scope and resolve new findings.
- [x] Report test counts and limitations to coordinator for publication.

Coordinator verification:342 distinct affected backend cases passed, including two review regressions for embedded author/keyword text after wrapped FTS failure. Final Prompt DB suite128pass. Independent source review is clear; final Bandit six production files0findings/0errors. Ruff has no new findings;19 inherited source/test findings remain. No live UAT, native lifetime or installer acceptance.

Published source was merged with current dev7117 without conflicts; all reviewed repair source paths remained unchanged. Final integrated affected backend run passes353cases across12suites, no failures/skips. Pytest reported pre-existing old temporary-directory cleanup warnings after the passing run; no unrelated cleanup was attempted.
