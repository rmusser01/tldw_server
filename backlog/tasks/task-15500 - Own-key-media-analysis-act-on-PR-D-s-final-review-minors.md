---
id: TASK-15500
title: 'Own-key media analysis: act on PR D''s final-review minors'
status: Done
labels:
- byok
- usage-quotas
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-15426 PR D (#3470). Its final review left eight Minor findings (numbered 2-9 there). Three get a fix: a unit test over every entry of media_analysis_provider's other-stage denylist, including claims by the server's ENABLE_INGESTION_CLAIMS default (2); the media ingest job's fire-and-forget embeddings task inheriting the job's admission and the owner's key (4); two test fixtures built at import time (8). Five need no change, each for a stated reason (3, 5, 6, 7, 9).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 media_analysis_provider has a unit test per denylist entry, and one for the server's claims default on, off, and opted out
- [x] #2 The media ingest job's embeddings task runs with no provider key and no admission mark, with a test that fails without the fix
- [x] #3 test_media_llm_quota.py builds no fixture bytes at import time
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Minors 3, 5, 6, 7 and 9 need no change: 3's comment and Usage_Quotas sentence are true, since on media paths rolling_summarize gets no LLM function (only the /chunking endpoints pass one) and raises instead of calling analyze; 5 is the owner's D-2; 6 acts only on bare loops whose default executor is created lazily; 7 is adequate for its file and TASK-15497 covers the class; 9 fails loudly if the environment changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Acted on the eight Minor findings of PR D's final review (#3470). Fixed three: (2) test_media_llm_quota.py pins every entry of media_analysis_provider's other-LLM-stage denylist (10 cases) and claims by the server's ENABLE_INGESTION_CLAIMS default (on, off, opted out); a sabotage of the settings fallback fails exactly [server-default-on] and dropping auto_apply_template fails exactly [auto_apply_template]. (4) The media ingest job's fire-and-forget embeddings task is created under llm_call_context(prechecked=False, provider_credentials=None), so it no longer carries the job's admission or the owner's key; test_the_embeddings_task_carries_no_key_and_no_admission failed before the fix on the key handle and passes after. (8) The WAV and PDF fixtures are built when the test runs, not at import. No change for 3, 5, 6, 7, 9 (see notes). Measured under the network probe: test_media_llm_quota.py 35 passed; the job quota file 8 passed; tests/Docs + lint 520 passed, 1 environment-only error (test_runtime_package_discovery, no setuptools in the local venv); the wider set (14 files) 561 passed; 0 network attempts.
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
