---
id: TASK-15560
title: Forward contextual chunking options to the PDF processor in /media/add or document
  that PDFs ignore them
status: To Do
labels:
- media
- chunking
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-15486: /media/add's PDF path builds the processor arguments from chunk_options (chunk_method, max_chunk_size, chunk_overlap) and drops the rest, so enable_contextual_chunking and contextual_llm_model never reach PDF processing; document and ebook receive the whole chunk_options. The [pdf] case of test_media_contextual_integration.py asserts only the forwarded chunk_method and says so in a comment. Either forward the contextual options to the PDF processor (and assert them in that test) or document that PDFs ignore them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Contextual chunking options reach the PDF processor and the [pdf] contextual test asserts them, or the docs state that PDFs ignore them
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
