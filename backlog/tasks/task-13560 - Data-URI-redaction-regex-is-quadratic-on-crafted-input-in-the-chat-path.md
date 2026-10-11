---
id: TASK-13560
title: Data-URI redaction regex is quadratic on crafted input in the chat path
status: To Do
labels:
- security
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The pattern (data:image[^,]*,)[^"\s]+ rescans to the end of the text for every data:image that has no comma after it, so a message holding many such markers makes redaction quadratic. Measured on the same pattern: 50,000 repeats (a 500 KB message) took about 10 seconds; 200,000 repeats did not finish in two minutes. It runs on request input an authenticated user controls. Three copies remain: tldw_Server_API/app/core/Chat/chat_service.py (_DATA_URI_RE, used by estimate_tokens_from_json and _sanitize_data_uris), tldw_Server_API/app/api/v1/endpoints/chat.py (the local pattern near line 6321), and tldw_Server_API/app/api/v1/schemas/chat_validators.py (DATA_URI_REDACT_PATTERN). The copy in core/LLM_Calls/usage_accounting.py was fixed in the LLM usage accounting PR 1a by bounding the media-type part: (data:image[^,"\s]{0,128},)[^"\s]+. Found during the PR 1a final review (TASK-13437).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The three remaining patterns bound the media-type part, ideally through one shared constant
- [ ] #2 A test redacts a 500 KB message of repeated data:image markers in under two seconds for each call site
- [ ] #3 Existing data-URI redaction tests still pass
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
