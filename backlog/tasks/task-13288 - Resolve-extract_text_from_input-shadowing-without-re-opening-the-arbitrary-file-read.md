---
id: TASK-13288
title: >-
  Resolve extract_text_from_input shadowing without re-opening the arbitrary
  file read
status: To Do
assignee: []
created_date: '2026-09-22 03:56'
updated_date: '2026-09-23 15:28'
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
Shipped on fix/summarization-shadowed-file-read.

THE SHADOWING WAS THE CONTROL. extract_text_from_input was defined twice at module scope
(:493 with the file read, :917 without, marked noqa F811). The second won, which is the
only reason TASK-2425 could close the summarization arbitrary-file-read as 'not active
through analyze()'. Deleting the *later* definition as a duplicate -- the obvious tidy-up,
and exactly what the F811 warning invites -- would have re-armed a file read reachable
from analyze(), which 17+ core modules import and which passes caller-supplied input_data
straight through for every caller not setting input_is_literal_text=True (only two sites
in the app do). The shadowed copy is deleted instead, so what already ran is now the only
thing that can run.

SECOND COPY OF THE SAME HAZARD, not previously recorded. extract_metadata_and_content did
os.path.exists(input_data) -> open(input_data) -> json.load(file). It had no callers
anywhere in tldw_Server_API, apps or Helper_Scripts -- dead code holding a live file read.
Deleted, with format_input_with_metadata which only formatted its output. That answers
AC#6's 'removed, or retained with a documented reason': 'it reads any file the caller
names' is not a reason to retain.

TWO DEFECTS IN THE SURVIVOR, both reachable from analyze():
- a dict carrying text under 'text' or 'content' extracted to "", so analyze():699
  returned 'Error: Could not extract text content.' for input that plainly had text. The
  deleted copy handled both keys; kept, checked after the known structures so
  title/description/transcription/segments precedence is unchanged.
- json.loads on "123", "true", "null" yields a scalar and the membership tests raised
  TypeError, which is in _SUMMARIZATION_NONCRITICAL_EXCEPTIONS and was swallowed into a
  generic error. A scalar now comes back as the caller wrote it.

VERIFICATION
- tests/LLM_Calls/test_summarization_input_extraction.py: 21 passed. Pins the security
  property directly rather than via definition order, plus an AST ratchet that fails on a
  second module-scope definition, one that fails if an F811 suppression returns, and one
  that flags os.path.isfile/exists/isdir called on a bare argument name.
- PROBE-THE-FIX, four ways: reintroducing the file read turns the two arbitrary-file-read
  tests red; removing the text/content handling turns 1 red; removing the scalar guard
  turns 7 red; and a reintroduction split across three lines -- which leaves the substring
  count unchanged at 1, all of it docstring prose -- still turns the AST ratchet red. The
  ratchet is an AST walk for exactly that reason: a scan cannot tell code from prose.
- Baseline: tests/LLM_Calls plus tests/Translation/test_translate_service_prompt.py are
  17 failures on clean dev and the identical 17 with this change, byte-identical lists.
  Zero regressions.
- Bandit: clean, 0 issues.
- Ruff: clean on both touched files.

TASK-2425's notes updated to record that its 'not currently active' conclusion is
superseded here.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
