---
id: TASK-13307
title: >-
  OCR temp image is unlinked before the request that names it causing silent
  total data loss
status: In Progress
assignee: []
created_date: '2026-09-22 04:53'
updated_date: '2026-09-22 05:08'
labels:
  - bug
  - ingestion
  - ocr
  - data-loss
dependencies: []
references:
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/dots_ocr.py:195
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_ocr.py:192
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/nemotron_parse.py:350
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
dots_ocr and hunyuan_ocr use NamedTemporaryFile(delete=True), so the file is closed and unlinked at the with-block exit while content_image still holds f.name; the POST that follows names a path that no longer exists.

Triggered by the supported config the code comment at dots_ocr.py:193 explicitly anticipates (e.g. DOTS_VLLM_USE_DATA_URL=0). _ocr_via_vllm returns "" and _ocr_pdf_pages counts only non-empty pages, so the run reports SUCCESS WITH ZERO CONTENT, every page, every document, no error surfaced.

nemotron_parse.py:350-387 and llamacpp_ocr.py do it correctly with delete=False plus a finally unlink. Promote that shape into OCR/runtime_support.py as one image_payload context manager.

Source: synthesis F9
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both backends keep the temp file alive until the request completes
- [ ] #2 Shared image_payload helper in OCR/runtime_support.py used by all vLLM backends
- [ ] #3 Test asserts a non-data-URL run returns non-empty text
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
SCOPE CORRECTION during Stage 0: there is a THIRD site the finding missed - core/Ingestion_Media_Processing/OCR/backends/dolphin_ocr.py:327, identical defect (NamedTemporaryFile(delete=True), file unlinked at the with-exit on :331, POST references f.name at :348). Fixed alongside dots_ocr and hunyuan_ocr.

tesseract_cli.py:27 also uses delete=True and is CORRECT - its subprocess call runs INSIDE the with block. Verified, deliberately left unchanged.

All three fixed with the nemotron_parse.py:350-387 pattern: delete=False, capture tmp_path, unlink in a finally via contextlib.suppress(OSError). Shared helper extraction (OCR/runtime_support.py image_payload) remains open as AC #2.
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
