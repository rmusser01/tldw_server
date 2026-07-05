---
id: TASK-12155
title: Stabilize Windows platform simulation tests
status: Done
created_date: 2026-07-04 20:47
labels:
- tests
- stability
priority: medium
modified_files:
- tldw_Server_API/app/core/Ingestion_Media_Processing/Upload_Sink.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/PDF/mineru_adapter.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_filename_and_mime_and_archive.py
- tldw_Server_API/tests/Media_Ingestion_Modification/test_mineru_adapter.py
updated_date: 2026-07-04 22:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix tests that simulate Windows behavior by mutating the real os.name, which leaks into async teardown before monkeypatch cleanup and causes pathlib.WindowsPath construction failures on POSIX.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Windows behavior tests do not mutate process-global os.name
- [x] #2 Affected focused tests pass on POSIX
- [x] #3 Previously order-dependent broad slice no longer fails from pathlib.WindowsPath teardown errors
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added patchable platform helper functions in Upload_Sink and mineru_adapter, then updated Windows simulation tests to patch those helpers instead of mutating the real os.name module singleton. Focused verification: 2 targeted Windows simulation tests -> 2 passed; order-sensitive neighborhood -> 83 passed. Changed-scope slice passed later: 1838 passed, 54 skipped, 1 xfailed, 2 xpassed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed global os.name mutation from Windows platform simulation tests by routing platform checks through local helper functions that tests can patch safely.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused tests pass
- [x] #2 Broad slice rerun confirms original order-dependent failure is resolved or exposes a new unrelated failure
- [x] #3 Backlog task updated with verification results
<!-- DOD:END -->
