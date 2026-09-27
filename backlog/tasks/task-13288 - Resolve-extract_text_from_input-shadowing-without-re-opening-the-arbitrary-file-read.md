---
id: TASK-13288
title: >-
  Resolve extract_text_from_input shadowing without re-opening the arbitrary
  file read
status: Done
assignee: []
created_date: '2026-09-22 03:56'
updated_date: '2026-09-23 23:20'
labels:
  - llm
  - security
  - bug
dependencies:
  - TASK-2425
references:
  - 'tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py:493'
  - 'tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py:917'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py` defines `extract_text_from_input` **twice** at module scope. The second binding wins at runtime.

- `:493` — richer version. Handles `text` and `content` keys, and for a string input does `os.path.isfile(input_data)` -> `open(input_data).read()`.
- `:917` — weaker version, marked `# noqa: F811`. Wins at runtime. Does not handle `text`/`content`, and returns a file path as literal text rather than reading it.

**Read TASK-2425 before touching this.** That task (Done, 2026-06-24) concluded the summarization arbitrary-file-read finding was "not active through `analyze()` because the file-reading helper is shadowed by a later `extract_text_from_input()` definition."

**So the shadowing is currently load-bearing as an accidental security control.** Naively deleting `:917` to resolve the shadow would re-activate an arbitrary file read reachable from `analyze()`, which 17+ core modules import. Do not do that. This task is to make the intended behaviour explicit rather than leaving a security property resting on definition order plus an inline noqa.

**Separately, the winning copy is buggy.** Only 2 call sites in the whole app pass `input_is_literal_text=True` (`Email_Processing_Lib.py:401`, `api/v1/endpoints/translate.py:109`); everything else reaches `analyze()` at `:698`.
- `analyze(api, {"text": "..."})` and `{"content": "..."}` -> `:917` finds no title/description/transcription/segments -> returns "" -> `:699` returns "Error: Could not extract text content."
- `analyze(api, "123")` / `"true"` / `"null"`: `json.loads` yields a scalar, then `'title' in data` raises TypeError, which is in `_SUMMARIZATION_NONCRITICAL_EXCEPTIONS` (`:97-109`) and is swallowed into a generic error string.

**F811 is not sanctioned policy here** — it appears zero times in `pyproject.toml`, so the inline `# noqa: F811` is a per-line suppression, not a project-wide ignore.

Also in scope: `extract_metadata_and_content` (`:874-907`) and `format_input_with_metadata` (`:909-914`) have zero references anywhere in `app/` or `tests/`. TASK-2425 already noted the metadata helper has no callers.

Source: comprehensive core-module review prompt smoke run, findings LLM_Calls-1 / LLM_Calls-15.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test reproduces the dict-input regression before any production edit: analyze(api, {"text": ...}) currently returns the extraction-error string
- [x] #2 A test pins the security property explicitly: a filesystem path passed through analyze() is NOT read from disk, and that test fails if the file-reading branch is ever restored
- [x] #3 Exactly one module-level definition of extract_text_from_input remains, and the inline # noqa: F811 is gone
- [x] #4 The surviving implementation handles text and content keys, and does not read files from caller-controlled paths
- [x] #5 Scalar JSON input ("123", "true", "null") no longer raises a TypeError that is swallowed into a generic error string
- [x] #6 Dead helpers extract_metadata_and_content and format_input_with_metadata are removed, or retained with a documented reason
- [x] #7 TASK-2425 is cross-referenced and its non-active conclusion is re-verified or explicitly superseded
- [x] #8 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fix commit aec7b5a6cf. New test file tldw_Server_API/tests/LLM_Calls/test_summarization_input_extraction.py (11 tests). RED on ea1cbc6941: 8 failed / 3 passed (single-definition check, text+content dict keys via analyze(), scalar/non-object JSON '123','true','null','"quoted"','[1, 2]'). The two security tests passed on old code only because of the shadowing. GREEN: 11 passed. Mutation check: re-adding an os.path.isfile/open branch to the surviving extractor makes test_filesystem_path_is_not_read and test_extractor_never_opens_files fail (2 failed / 9 passed), so the security property is now pinned by tests, not definition order.

TASK-2425 cross-reference: its 'file read not active through analyze()' conclusion was correct but rested on the second definition shadowing the first. Superseded: the file-reading definition is deleted, the surviving extractor documents that strings are never treated as paths, and analyze()'s docstring no longer advertises 'file path to JSON'. extract_metadata_and_content and format_input_with_metadata had zero references in app/ or tests/ and are deleted (the former also opened caller paths). Unused 'import os' removed.

Suite comparison (LLM_Calls, Translation, Evaluations/unit/test_rag_evaluator.py, Chat/unit/test_authoritative_adapter_translation.py): before 17 failed / 650 passed, after 9 failed / 658 passed; the 9 remaining failures are identical before and after (pre-existing strict_filter/top_k tests), the other 8 before-failures are the new red tests. Bandit -ll on Summarization_General_Lib.py: no findings. Ruff clean on touched files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
One module-level extract_text_from_input remains (no noqa: F811). It never reads files, handles text/content keys, and treats any string that is not a JSON object as literal text, so analyze() no longer errors on {'text': ...}/{'content': ...} or scalar JSON. Dead file-reading helpers removed. Tests pin the no-file-read property explicitly. No known skips.
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
