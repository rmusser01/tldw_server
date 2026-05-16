---
id: TASK-394.5
title: Harden Quick Ingest URL and file input validation
status: Done
assignee: []
created_date: '2026-05-16 00:44'
updated_date: '2026-05-16 03:30'
labels:
  - quick-ingest
  - ux
  - validation
  - task-5
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 5: strengthen URL/text/file input validation, duplicate prevention, unsupported content messaging, and truthful file-size handling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 URL and text validation prevents common invalid submissions with clear recovery copy
- [x] #2 Duplicate or unsupported content is detected or messaged consistently with backend limits
- [x] #3 File-size handling truthfully reflects the implemented browser memory/upload strategy
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 5 after completing TASK-394.4. Scope: normalized URL dedupe, mixed valid/invalid paste summary, file support copy/accept alignment, and truthful client-buffered file-size limits per the approved plan.

Implemented in 87dfd455a. URL queue validation now dedupes with normalizeUrlForDedupe while preserving original submitted/displayed URLs. Mixed valid/invalid URL paste results now show a compact queue summary. File detection, picker accept strings, and user copy now align on supported quick-ingest upload types: PDF, EPUB, DOC/DOCX/TXT/Markdown/HTML/XML/JSON, audio, and video. Browser-buffered uploads now advertise and enforce a current 50 MB limit, with a separate named 500 MB transport-redesign target retained for future work.

Verification: ./node_modules/.bin/vitest run src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx src/services/__tests__/quick-ingest-batch.test.ts --maxWorkers=1 --no-file-parallelism passed with 3 files / 59 tests. git diff --check passed. Focused Playwright constrained-viewport check passed outside the macOS sandbox: TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:18001 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY bunx playwright test e2e/workflows/media-ingest.spec.ts --grep "quick ingest configure options stay reachable" --project=chromium --reporter=line. First sandboxed Playwright launch failed with Chromium MachPort bootstrap_check_in permission denied, then the escalated rerun passed. Bandit skipped because touched files are frontend TypeScript/TSX/JSON only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 complete: Quick Ingest now gives clearer validation for normalized duplicate URLs and mixed valid/invalid URL pastes, aligns supported file-type copy with picker/detection behavior, rejects unsupported local file types earlier, and truthfully presents the current 50 MB browser-buffered upload limit.
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
