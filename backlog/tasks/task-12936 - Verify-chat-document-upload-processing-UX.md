---
id: TASK-12936
title: Verify chat document upload processing UX
status: Done
assignee: []
created_date: '2026-07-09 06:25'
updated_date: '2026-07-09 07:03'
labels:
  - implementation
  - chat
  - frontend
  - testing
dependencies:
  - TASK-12092
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 6 from the chat document upload processing plan: add focused Playwright smoke coverage and run/record integration verification for the WebUI and browser-extension document processing choices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Playwright smoke covers /chat document upload processing choices
- [x] #2 Focused frontend/backend/lint/Bandit verification recorded
<!-- AC:END -->

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

- Added `apps/tldw-frontend/e2e/smoke/playground-document-processing.spec.ts` to cover `/chat` document upload preflight, `Add to chat`, send-time document processing, the immediate processing user turn, the chat-scoped chip, and chat request payload context.
- Fixed the regression exposed by the smoke by rendering `DocumentProcessingTurn` from the fallback `PlaygroundMessage` user-message layout when `metadataExtra.documentProcessing` is present.
- Left unrelated dirty `apps/packages/ui/node_modules/antd` changes untouched.

## Verification

- PASS: `cd apps/tldw-frontend && bunx playwright test e2e/smoke/playground-document-processing.spec.ts --reporter=line`
- PASS: `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/chat-document-processing.test.ts ../packages/ui/src/components/Option/Playground/__tests__/DocumentProcessingChoices.test.tsx ../packages/ui/src/components/Common/Playground/__tests__/DocumentProcessingTurn.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.document-processing.test.tsx ../packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts`
- PASS: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Media/test_document_upload_processing.py -q`
- PASS: `cd apps/tldw-frontend && bun run lint` exited 0 with 177 existing warnings and no errors.
- PASS: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/media/document_upload_processing.py tldw_Server_API/app/api/v1/schemas/document_upload_processing.py -f json -o /tmp/bandit_chat_document_upload_processing.json`; JSON results were empty.
- PASS: `git diff --check`
- INFO: `cd apps/tldw-frontend && bun run test:playground:composer` is not defined in this package.
- BASELINE FAIL: `cd apps/tldw-frontend && bun run e2e:nextgen-composer` passed 54 tests and failed 6 existing composer smoke cases unrelated to touched files: two mobile sidepanel variant-marker misses, two `/chat` 768px variant-marker misses, one preference server-sync assertion, and one CORS-console check in composer variant preview.
- BASELINE FAIL: `cd apps/tldw-frontend && bunx tsc --noEmit --pretty false --project tsconfig.json` still fails on unrelated baseline TypeScript errors outside touched files.
- MANUAL UX NOTE: a dev-server `/chat` browser check was attempted with `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -p 8080`; the in-app browser DOM snapshot API failed, and the fallback MCP Playwright harness could not bridge file-upload buffers in this context. The dedicated Playwright smoke above is the recorded browser-flow verification for the upload UX.
