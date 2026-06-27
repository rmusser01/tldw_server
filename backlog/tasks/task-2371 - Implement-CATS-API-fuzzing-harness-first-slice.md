---
id: TASK-2371
title: Implement CATS API fuzzing harness first slice
status: Done
labels:
- testing
- security
- api
documentation:
- Docs/superpowers/plans/2026-06-27-cats-api-fuzzing-harness-implementation-plan.md
- Docs/superpowers/specs/2026-06-27-cats-api-fuzzing-harness-design.md
modified_files:
- tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py
- Helper_Scripts/cats_fuzz/
- tldw_Server_API/tests/VectorStores/test_vector_stores_openapi_examples.py
- tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_manifest.py
- tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_env.py
- tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cats_cli.py
- tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_summary.py
- tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_runner.py
- tldw_Server_API/tests/Helper_Scripts/test_cats_fuzz_cli.py
- Docs/Development/CATS_Fuzzing.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first local-only CATS API fuzzing harness slice from the approved implementation plan. Scope includes the vector store OpenAPI examples cleanup, importable Helper_Scripts/cats_fuzz modules, focused unit tests, CLI/docs, live contract/public-read verification, and touched-scope Bandit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Vector store OpenAPI query examples are compatible with CATS strict validation.
- [x] #2 Helper_Scripts/cats_fuzz supports manifest blocks, local-only env isolation, OpenAPI export, CATS command construction, summary JSON, uvicorn lifecycle, and CLI execution for contract/public-read.
- [x] #3 Focused pytest coverage passes for the new harness modules and OpenAPI cleanup.
- [x] #4 Live local CATS contract and public-read commands either pass or record actionable tool/API failure summaries.
- [x] #5 Bandit runs on touched executable scope with no unresolved new findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-27: Starting subagent-driven execution from Docs/superpowers/plans/2026-06-27-cats-api-fuzzing-harness-implementation-plan.md in worktree codex/cats-api-fuzzing-harness.

2026-06-27 Task 6: added the CATS fuzzing CLI/module entrypoint, focused CLI parser/orchestration tests, developer documentation, and Makefile cats-fuzz target. Initial red test run failed on missing Helper_Scripts.cats_fuzz.cli as expected; final focused pytest, Black check, Bandit on cli.py, and git diff --check passed.

2026-06-27 Task 7 verification fix: added a regression for `cats --version` banner output before changing `_cats_version()` to prefer `CATS version ...` lines ahead of generic first-line fallback. Red run failed on the banner border being returned; green run passed focused CLI tests. Verification: focused pytest, Black check, Bandit on cli.py, and git diff --check passed.

2026-06-27 Task 7 live verification env fix: added failing regressions for server-safe CATS harness env generation and CLI uvicorn startup env selection. Red run failed on missing `build_server_env`; green run passed focused env/CLI tests after adding `runtime/cats-server.env` without guarded test flags (`TEST_MODE`, `TESTING`, `TLDW_TEST_MODE`) and routing only `start_server()` through that env while OpenAPI export/runtime CATS still use the raw child env.

2026-06-27 Task 7 live verification timeout fix: added failing regressions for CATS subprocess timeouts returning structured results, timeout classification as tool failure, and a >=300s public-read budget. Red run failed on uncaught `TimeoutExpired`, `api` classification, and 120s public-read timeout; green run passed focused cats_cli/manifest tests. Verification: focused pytest, Black check, Bandit on cats_cli.py/manifest.py, and git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first local-only CATS fuzzing harness slice. The harness now exports an isolated OpenAPI contract, constructs CATS contract/runtime commands, starts a sandboxed uvicorn server with generated test-only env files, masks sensitive headers in summaries, writes per-block `summary.json`/stdout/stderr/report artifacts, and exposes CLI/docs/Makefile entrypoints for `contract`, `public-read`, and scaffolded `auth-read` blocks.

Verification completed on 2026-06-27:
- Focused pytest: `57 passed, 6 warnings`.
- Live CATS contract: `/tmp/tldw-cats-public-read-v4/contract/summary.json` has `exit_code: 0`, `failure_class: ok`, CATS `13.8.0`.
- Live CATS public-read: `/tmp/tldw-cats-public-read-v4/public-read/summary.json` has `exit_code: 124`, `failure_class: tool`; stderr records `Command timed out after 300 seconds`, with partial CATS HTML/JUnit reports under `/tmp/tldw-cats-public-read-v4/public-read/cats-report`.
- Final Bandit touched executable scope: `/tmp/bandit_cats_fuzz_final.json` has zero findings.
- `git diff --check` passed.

Known follow-up: the broad first-slice public-read surface starts correctly and records an actionable summary, but it does not complete within the current 300s budget on the full generated tldw OpenAPI. Narrowing fuzzers/paths or splitting public-read into smaller blocks should be handled in a later task before making this a hard CI gate.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
