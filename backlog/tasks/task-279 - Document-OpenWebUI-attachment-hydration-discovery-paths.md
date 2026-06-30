---
id: TASK-279
title: Document OpenWebUI attachment hydration discovery paths
status: Done
assignee: []
created_date: '2026-05-12 01:12'
updated_date: '2026-05-12 01:16'
labels:
  - docs
  - openwebui
  - chatbooks
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1575'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add user-discoverable documentation for the merged OpenWebUI attachment hydration feature so users can find when and how to restore images/files after OpenWebUI JSON or webui.db import. This is a docs-only follow-up to PR #1575 and should avoid changing runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User-facing docs explain where attachment hydration appears in the WebUI and when to use it after OpenWebUI import.
- [x] #2 Getting-started or index documentation links users to the OpenWebUI import and hydration workflow.
- [x] #3 API documentation remains consistent with the user-facing workflow for previewing and running hydration jobs.
- [x] #4 Documentation tests or focused docs checks verify the new discovery text where applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated README, user guide index, WebUI overview, Chatbook user guide, API overview, published mirrors, and feature status so OpenWebUI attachment hydration is discoverable after import.

Added docs regression assertions in tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py for the new discovery text.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py -q passed with 6 tests.

Verification: git diff --check passed with no output.

Security: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py -f json -o /tmp/bandit_openwebui_docs_discovery.json completed with results: [].
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented the merged OpenWebUI attachment hydration workflow across user-facing discovery surfaces, API overview documentation, published documentation mirrors, and docs regression tests. This keeps the feature discoverable from README/index-style entry points while preserving the detailed Chatbook user guide as the canonical workflow reference.
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
