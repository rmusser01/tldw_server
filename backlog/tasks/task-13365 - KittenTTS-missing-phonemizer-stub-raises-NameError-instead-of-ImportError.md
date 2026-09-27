---
id: TASK-13365
title: KittenTTS missing-phonemizer stub raises NameError instead of ImportError
status: Done
assignee: []
created_date: '2026-09-23 20:07'
updated_date: '2026-09-23 20:07'
labels:
  - bug
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TTS/vendors/kittentts_compat.py: the fallback EspeakWrapper methods used the except-bound exc after the block unbound it (ruff F821), so without phonemizer set_library/set_data_path raised NameError rather than the actionable ImportError. Found while removing dead TTS code under TASK-13341.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fallback EspeakWrapper methods raise ImportError naming phonemizer
- [x] #2 Test reloads the module with phonemizer blocked and asserts the ImportError
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in 56ef37d4af: except block captures exc as _phonemizer_error for the closures. test_kittentts_compat.py::test_missing_phonemizer_wrapper_raises_import_error_not_name_error blocks phonemizer via sys.modules, reloads, asserts ImportError for both methods; red on HEAD (NameError), green now; kittentts tests 12 passed. Bandit: n/a beyond variable rename. No docs affected.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The missing-phonemizer stub now raises the intended ImportError; regression test added.
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
