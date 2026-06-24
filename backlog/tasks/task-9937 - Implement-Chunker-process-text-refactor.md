---
id: TASK-9937
title: Implement Chunker process_text refactor
status: In Progress
created_date: 2026-06-24 22:02
dependencies:
- TASK-9936
labels:
- chunking
- refactor
- implementation
priority: High
modified_files:
- tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py
- tldw_Server_API/app/core/Chunking/chunker.py
- tldw_Server_API/app/core/Chunking/error_policy.py
- tldw_Server_API/app/core/Chunking/option_utils.py
- tldw_Server_API/app/core/Chunking/llm_context.py
- tldw_Server_API/app/core/Chunking/process_text/__init__.py
- tldw_Server_API/app/core/Chunking/process_text/models.py
- tldw_Server_API/app/core/Chunking/process_text/preparation.py
- tldw_Server_API/app/core/Chunking/process_text/options.py
- tldw_Server_API/tests/Chunking/test_process_text_components.py
- backlog/tasks/task-9937 - Implement-Chunker-process-text-refactor.md
updated_date: 2026-06-24 22:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved implementation plan for the behavior-preserving Chunker.process_text refactor, using test-first stages and subagent review gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Characterization and component tests cover the documented process_text behaviors before production logic moves
- [ ] #2 Chunker.process_text delegates to the new internal process_text pipeline without public behavior drift
- [ ] #3 Focused Chunking tests, compileall, diff check, and Bandit verification are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-06-24-chunker-process-text-refactor.md using subagent-driven development: implement each task test-first, run spec and code-quality review gates, then complete final verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 characterization tests added in tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py. Verified with `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py -q` (13 passed, 38 warnings). Bandit touched-scope check run with pytest assert noise excluded: `python -m bandit -r tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py -s B101 -f json -o /tmp/bandit_chunker_process_text_refactor_tests_skip_b101.json` (0 findings). Raw Bandit without B101 exclusion reported only low-severity B101 assert usage in pytest tests.
Task 1 cleanup: loosened hierarchical dispatch characterization to avoid exact whole-kwargs equality while still asserting the instance method call and meaningful forwarded values. Verified with `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py -q` (13 passed, 38 warnings).
Task 2 shared helpers/internal models added: extracted CHUNKER_NONCRITICAL_EXCEPTIONS, _coerce_bool_option, and _LLM_UNSET/llm_override_scope into Chunking helper modules; added process_text internal dataclasses/protocol without importing Chunker; rewired chunker.py imports without changing process_text behavior; added component tests for the extracted helpers and models. Red check: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py -q` exited during collection because `tldw_Server_API.app.core.Chunking.error_policy` did not exist yet; direct import confirmed `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Chunking.error_policy'`. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py -q` (29 passed, 70 warnings); `source .venv/bin/activate && python -m compileall tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/app/core/Chunking/error_policy.py tldw_Server_API/app/core/Chunking/option_utils.py tldw_Server_API/app/core/Chunking/llm_context.py tldw_Server_API/app/core/Chunking/process_text tldw_Server_API/tests/Chunking/test_process_text_components.py` (exit 0); `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/app/core/Chunking/error_policy.py tldw_Server_API/app/core/Chunking/option_utils.py tldw_Server_API/app/core/Chunking/llm_context.py tldw_Server_API/app/core/Chunking/process_text -f json -o /tmp/bandit_chunker_process_text_task2.json` (exit 0, 0 results).
Task 2 spec-review fix: restored `InvalidChunkingMethodError` in the direct `chunker.py` `.exceptions` import while keeping `_CHUNKER_NONCRITICAL_EXCEPTIONS` imported from `error_policy.py`. Added focused component coverage for `Chunker.get_strategy("missing-method")` to preserve the `InvalidChunkingMethodError` factory error path. Red check before production fix: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py::test_chunker_strategy_factory_still_raises_invalid_method_error -q` failed with `NameError: name 'InvalidChunkingMethodError' is not defined`. Verification after fix: focused regression test passed (1 passed, 14 warnings); `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py -q` (30 passed, 72 warnings); Task 2 compileall command exit 0; Bandit touched production Chunking scope exit 0 with 0 results.
Task 3 preparation extraction: added `process_text/preparation.py` with `prepare_frontmatter` and `extract_header`, then wired `Chunker.process_text` to call those helpers while preserving the existing size-enforcement boundary and metric timing. Added direct component tests for default/custom sentinels, disabled parsing, string `"false"` truthiness, tokenizer precedence, legacy header offsets, and malformed leading JSON. Red check: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py -q` failed during collection because `tldw_Server_API.app.core.Chunking.process_text.preparation` did not exist; direct import confirmed `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Chunking.process_text.preparation'`. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py tldw_Server_API/tests/Chunking/test_chunker_v2.py::TestV2Chunker::test_process_text_extracts_frontmatter_with_sentinel tldw_Server_API/tests/Chunking/test_chunker_v2.py::TestV2Chunker::test_process_text_frontmatter_offsets_use_original_text -q` (39 passed, 90 warnings); compileall touched files exit 0; Bandit touched production files exit 0 with 0 results in `/tmp/bandit_chunker_process_text_task3.json`.
Task 3 review fix: split frontmatter preparation into option/control setup and JSON parsing so `chunker_frontmatter_duration_seconds` starts after tokenizer fallback and frontmatter option popping, matching the pre-extraction timing boundary. Added direct `process_text.preparation` import-boundary coverage and a regression test that setup latency is excluded from the frontmatter metric. Red check: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py::test_process_text_frontmatter_metric_excludes_option_setup -q` failed with `AttributeError: module 'tldw_Server_API.app.core.Chunking.chunker' has no attribute '_prepare_frontmatter_options'`. Verification: requested focused suite passed (41 passed, 94 warnings); compileall touched files exit 0; Bandit touched production files exit 0 with 0 results in `/tmp/bandit_chunker_process_text_task3_review_fix.json`.
Task 4 started: extracting Chunker.process_text option resolution into process_text/options.py and adding direct component coverage before production wiring.
Task 4 option resolution extraction: added `process_text/options.py` with `resolve_process_options` and `METHOD_OPTION_EXCLUDES`, wired `Chunker.process_text` to consume `ResolvedProcessOptions`, exported the helper from `process_text.__init__`, and added direct component coverage for max_size validation, negative overlap clamping, language autodetection, method option filtering with tokenizer override preservation, code mode defaults, hierarchy/template multi-level exclusion, and the no-Chunker import boundary. Red check: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chunking/test_process_text_components.py -q` failed during collection because `tldw_Server_API.app.core.Chunking.process_text.options` did not exist; direct import confirmed `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Chunking.process_text.options'`. Verification: requested focused suite passed (53 passed, 118 warnings); compileall touched files exit 0; Bandit touched production files exit 0 with 0 results in `/tmp/bandit_chunker_process_text_task4.json`; `git diff --check` exit 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
