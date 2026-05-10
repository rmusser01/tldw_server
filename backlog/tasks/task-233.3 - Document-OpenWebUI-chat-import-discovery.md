---
id: TASK-233.3
title: Document OpenWebUI chat import discovery
status: Done
assignee: []
created_date: '2026-05-10 20:20'
updated_date: '2026-05-10 21:02'
labels:
  - documentation
  - chatbooks
  - openwebui
dependencies: []
parent_task_id: TASK-233
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add user-facing documentation for the merged OpenWebUI chat JSON import feature so users can discover what it does, where to find it, what export files are supported, what data is preserved, and how preview/import error handling works.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Documentation mentions OpenWebUI chat import in a discoverable Chatbooks or import-oriented location.
- [x] #2 Documentation explains supported OpenWebUI export shapes and the import path users should follow.
- [x] #3 Documentation lists the key data preserved during import including conversations, messages, roles, timestamps, models, files, metadata, and failed-row reporting.
- [x] #4 Relevant docs are linked or indexed so users browsing Chatbooks/import docs can find the feature.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update user-facing Chatbook docs and navigation to mention OpenWebUI chat JSON import. 2. Refresh the published docs mirror for the touched docs only. 3. Verify docs links/discoverability and record non-code validation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started docs follow-up in isolated worktree codex/openwebui-chat-import-docs. Main checkout is dirty/diverged, so edits are intentionally isolated.

Updated source docs and selected published mirrors for Chatbook/OpenWebUI discoverability. Added a docs regression test for guide, index, API README, API tag, and published docs coverage.

Verification: python -m pytest tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py tldw_Server_API/tests/Docs/test_docs_index_path_hygiene_script.py tldw_Server_API/tests/Docs/test_readme_docs_path_hygiene_script.py tldw_Server_API/tests/Docs/test_top_guides_docs_path_hygiene_script.py -q (6 passed).

Verification: python -m bandit -r tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py -f json -o /tmp/bandit_openwebui_import_docs.json (0 findings).

Verification: git diff --check (clean).

Known skips/blockers: none for this docs-only follow-up.

Review fix pass for PR #1550: addressed Qodo/Gemini/CodeRabbit feedback by correcting import_media/import_embeddings docs, replacing query-string import guidance with multipart form-field guidance, adding source_format/content_selections multipart contract coverage to source and published OpenAPI YAML, and adding README/OpenAPI assertions to the docs regression test.

Review verification: python -m pytest tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py tldw_Server_API/tests/Docs/test_docs_index_path_hygiene_script.py tldw_Server_API/tests/Docs/test_readme_docs_path_hygiene_script.py tldw_Server_API/tests/Docs/test_top_guides_docs_path_hygiene_script.py -q (8 passed).

Review verification: python -m bandit -r tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py -f json -o /tmp/bandit_openwebui_import_docs_review.json (0 findings).

Review verification: git diff --check (clean).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added user-facing and API documentation so the OpenWebUI chat JSON import is discoverable from README, the WebUI user guide, the Chatbook user guide, the user-guide index, the API README, the API tag index, and published docs mirrors. Added a focused docs regression test to keep the guide/index/API discovery paths covered. This is a docs/test-only follow-up; no runtime code changed.

Review follow-up corrected the import option contract docs and OpenAPI multipart schemas raised in PR review, and expanded docs tests to cover README plus the multipart OpenAPI fields.
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
