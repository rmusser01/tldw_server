---
id: TASK-13560
title: Data-URI redaction regex is quadratic on crafted input in the chat path
status: Done
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
- [x] #1 The three remaining patterns bound the media-type part, ideally through one shared constant
- [x] #2 A test redacts a 500 KB message of repeated data:image markers in under two seconds for each call site
- [x] #3 Existing data-URI redaction tests still pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed: DATA_URI_REDACT_RE in core/Utils/image_validation.py, the bounded pattern from usage_accounting (whose private copy is removed), is used by chat_service._sanitize_data_uris, the endpoint's _sanitize_json_for_rate_limit and validate_request_size. A 510 KB body of '"url":"data:image' markers: about 4 to 6 s per site before, about 5 ms after. Behaviour changes: the endpoint's queue estimate now also redacts a payload outside a "url" key and in any case, and the size validator an uppercase DATA:IMAGE URI; a header longer than 128 characters or containing whitespace or a quote is left whole (counted in full, so at the size validator a large image with such a header is now too large). Tests: the new file 9 passed (before the fix 6 failed, the 3 guards passed); the redaction tests 131 passed; every test file importing a changed module (189 files plus the new one) 7 failed, 5589 passed, 35 skipped. All 7 also fail on unchanged origin/dev: test_iter_admin_router_specs_populates_expected_specs (billing router moved under /api/v1/admin in 9ba7897af7), the Postgres owner-reuse chat import (500), and five test_iter_minimal_optional_router_specs audio tests, which fail whenever a Claims test ran earlier on the same xdist worker because tests/Claims/conftest.py's session-scoped fixture leaves ROUTES_DISABLE (media, audio, ...) in os.environ. Bandit -ll on the five source files: no issues. Ruff: only the I001 at chat_service.py:10 that origin/dev already has. Left to chat-7 (Docs/superpowers/reviews/chat/2026-09-21-stage2-orchestration-streaming.md): prompt_cost_envelope's own scanner (linear; feeds prompt fingerprints), the two rounding rules, and the redundant outer _sanitize_json_for_rate_limit pass before estimate_tokens_from_json.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Data-URI redaction on the chat path is linear: the chat service, the chat endpoint and the request size validator share one bounded pattern (DATA_URI_REDACT_RE in core/Utils/image_validation.py) with LLM usage accounting. A crafted 510 KB message takes about 5 ms instead of several seconds per call site; a test holds each site under two seconds.
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
