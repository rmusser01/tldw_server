---
id: TASK-13303
title: OCR page image is unlinked before the request that references it
status: Done
assignee: []
created_date: '2026-09-22 04:52'
updated_date: '2026-09-23 19:34'
labels:
  - bug
  - ingestion
  - ocr
  - data-loss
dependencies: []
references:
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/dots_ocr.py:196
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_ocr.py:193
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`dots_ocr.py` and `hunyuan_ocr.py` write the page image to a self-deleting temp file, capture its path, and then issue the request **after** the `with` block has closed and unlinked it:

```python
with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
    f.write(image_bytes)
    f.flush()
    content_image = {"type": "image_url", "image_url": {"url": f.name}}
# <- file unlinked here
...
j = fetch_json(method="POST", url=url, json=data, timeout=timeout)   # path no longer exists
```

Verified: `dots_ocr.py:196` and `hunyuan_ocr.py:193` both use `delete=True`; the POST is outside the block. The sibling `nemotron_parse.py:350` uses `delete=False` and is correct.

**Effect:** the OCR server is handed a path that does not exist. The result is empty text, and because `_ocr_pdf_pages` counts only non-empty pages, the run **reports success with zero content — on every page**. Silent total data loss for these two backends on the path-URL branch.

The code comment on the same line acknowledges the path is probably unreadable by a remote server and says the data-URL mode is the default — so this branch may be rarely exercised, which is consistent with it going unnoticed. That lowers the frequency, not the severity: when it is selected it fails completely and reports success.

Found by the comprehensive core-module review; independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failing test selects the path-URL branch and asserts the file exists at request time
- [ ] #2 Both backends keep the temp file alive across the request (delete=False plus explicit cleanup, matching nemotron_parse.py:350)
- [ ] #3 The temp file is removed after the request on both success and failure paths
- [ ] #4 An empty OCR result no longer counts as success -- zero extracted pages reports failure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Duplicate of TASK-13307 (filed twice during the 2026-09-22 review). Work and status are tracked there.


Notes recorded on dev by the parallel core-review work (merged 2026-09-23):
Fixed on branch ci/review-followup-visibility.
Reproduced first: tldw_Server_API/tests/MediaIngestion_NEW/test_ocr_vllm_path_url_temp_file_lifetime.py asserts the file exists *inside* the stubbed fetch_json -- the only moment that matters. Red on exactly the two named backends (2 failed, 10 passed), with nemotron_parse parametrised as the passing control. Green after the fix: 12 passed.
Fix (AC #2, #3): both backends now mirror nemotron_parse.py -- delete=False, path held in tmp_path, os.unlink in a try/finally around the POST. Four cases are pinned per backend: alive at request time, removed after success, removed when the request raises, and data-URL mode still writes no temp file.
AC #4 declined, with a counterexample rather than as scope-trimming. 'Zero extracted pages reports failure' would fail correct runs. _ocr_pdf_pages only increments ocr_pages inside the as_completed loop (PDF_Processing_Lib.py:1672), and per_page_check skips OCR entirely for pages that already carry text, filling text_by_index from pre_text without entering futures (:1601-:1616). A text-bearing PDF processed with per_page_check=True therefore yields ocr_pages == 0 alongside a full ocr_text -- a successful run. A genuinely blank scan is also a legitimate zero.
The signal the AC was reaching for is empty *text*, not zero pages, and the caller already reports it: PDF_Processing_Lib.py:878 appends the warning 'OCR produced no text' when ocr_text is blank, and analysis_details.ocr.ocr_pages carries the count. So the run was never fully silent; what was silent is now moot, since the temp-file defect that made every page empty is closed. Promoting an all-empty OCR from warning to hard failure is a policy change across every backend and every blank-page PDF, and belongs in its own task if wanted.
Verification: the new file 12 passed; existing OCR tests (test_ocr_backend_dots.py, test_ocr_adapter.py, test_ocr_runtime_auto_selection.py, test_ocr_types.py) 19 passed 1 skipped. ruff clean. Bandit not installed in this environment (python -m bandit -> No module named bandit), so DoD #4 is a documented skip.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed as duplicate of TASK-13307.
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
