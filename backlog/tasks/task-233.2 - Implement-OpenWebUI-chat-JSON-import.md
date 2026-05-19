---
id: TASK-233.2
title: Implement OpenWebUI chat JSON import
status: Done
assignee: []
created_date: '2026-05-10 16:52'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-10-openwebui-chat-import-design.md
  - >-
    Docs/superpowers/plans/2026-05-10-openwebui-chat-import-implementation-plan.md
parent_task_id: TASK-233
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatbooks preview/import accept source_format=chatbook by default and source_format=openwebui_json for safe JSON uploads without running ZIP validation.
- [x] #2 OpenWebUI JSON adapter parses standard and legacy exports, previews counts and warnings, and imports valid chats while preserving full message trees with parent_message_id.
- [x] #3 Duplicate handling uses source=openwebui and external_ref with skip by default and rename/import-copy support for intentional copies.
- [x] #4 ChaCha conversation/message metadata preserves OpenWebUI source details under namespaced settings/metadata helpers.
- [x] #5 Existing Chatbook archive preview/import behavior remains compatible and covered by regression tests.
- [x] #6 WebUI Chatbooks import tab supports Chatbook archive and OpenWebUI JSON modes and sends source_format in multipart upload fields.
- [x] #7 Focused backend/frontend tests and Bandit verification are recorded before final completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-05-10-openwebui-chat-import-implementation-plan.md stage by stage. Use TDD for each behavior slice: write failing focused tests, verify red, implement minimal code, verify green, then refactor.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

- Added a Chatbooks OpenWebUI JSON import adapter for standard wrapper exports and legacy chat objects.
- Extended preview/import schemas and endpoints with `source_format=chatbook|openwebui_json`, keeping Chatbook archive behavior as the default.
- Added ChaCha source/external-ref duplicate lookup and persisted OpenWebUI conversation/message metadata under namespaced settings/metadata keys.
- Routed OpenWebUI sync and Jobs imports through JSON-safe path resolution instead of ZIP/archive validation.
- Extended the WebUI Chatbooks import tab with a source selector, JSON preview summary, OpenWebUI-specific conflict options, and multipart `source_format` upload fields.
- Updated user and API docs for supported v1 behavior and out-of-scope direct `webui.db`, admin export, live server import, and attachment hydration.

## Verification

- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_import_adapter.py tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py::test_preview_openwebui_json_source_format_skips_archive_validation tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py::test_import_openwebui_json_source_format_skips_archive_validation tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py -v` - 11 passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB -k "conversation or metadata" -v` - 62 passed, 235 deselected.
- `bun run test src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx src/components/Option/Chatbooks/__tests__/ContentTypePicker.error-state.test.tsx` - 4 passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/core/Chatbooks tldw_Server_API/app/core/DB_Management/chacha -f json -o /tmp/bandit_openwebui_chat_import.json` - 0 findings.
- `git diff --check` - clean.

## Known Skips

- Full `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py` was not rerun because the implementation session previously observed the baseline `test_preview_manifest_version_ok` path hanging. The OpenWebUI preview/import endpoint paths were covered with targeted tests.
- Manual browser smoke was not run; the WebUI source selector and multipart upload contract were covered with focused Vitest tests.

## Final Summary

Implemented v1 OpenWebUI chat JSON import through the existing Chatbooks workflow. The importer previews and imports normal OpenWebUI chat export JSON, preserves valid message branches as parent-linked tldw message trees, detects duplicates via `source=openwebui` and deterministic external refs, supports default skip plus intentional rename copies, preserves OpenWebUI metadata without hydrating attachments, and exposes the workflow in the WebUI import tab and docs.
