---
id: TASK-13303
title: OCR page image is unlinked before the request that references it
status: To Do
assignee: []
created_date: '2026-09-22 04:52'
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

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
