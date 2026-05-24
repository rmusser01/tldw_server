---
id: TASK-497
title: Fix chat image upload MIME detection in WebUI and extension
status: Done
labels:
- bug
- webui
- extension
- chat
modified_files:
- apps/packages/ui/src/components/Chat/composer/hooks/useComposerAttachments.ts
- apps/packages/ui/src/components/Chat/composer/__tests__/useComposerAttachments.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate reported WebUI/extension chat image uploads being rejected as the wrong/incorrect file type. Add focused regression coverage and adjust shared attachment handling so valid images with unreliable browser MIME metadata are accepted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed chat image upload rejection for valid images whose browser-provided MIME type is empty or application/octet-stream. Root cause was the shared useComposerAttachments hook trusting File.type and checking the unsupported binary MIME list before image detection. The hook now infers known image MIME types from real filename extensions only when File.type is missing/generic, normalizes the FileReader Data URL to data:image/<type>;base64,..., and still rejects explicit non-image/blocked MIME types normally.

Verification recorded: new regression tests first failed for omitted-MIME and application/octet-stream images, then passed after the implementation. PR review regressions first failed for application/zip named photo.png and formatted " Application/ZIP " MIME, then passed after constraining fallback to generic MIME values and normalizing unsupported checks. Clean PR branch targeted run: bun run test -- src/components/Chat/composer/__tests__/useComposerAttachments.test.tsx src/components/Option/Playground/hooks/__tests__/usePlaygroundAttachments.test.ts -> 2 files passed, 16 tests passed. git diff --check on touched files exited 0. Bandit was run with the project venv against touched files; because the touched files are TS/TSX, Bandit reported parse errors only and zero findings/results. CDP-only browser verification launched a temporary Chrome DevTools endpoint, selected General chat, injected photo.jpg with File.type application/octet-stream into the image input via raw CDP Runtime.evaluate, observed ATTACHMENTS (1), expanded it, saw Attached image with data:image/jpeg src, and saw no incorrect/unsupported file type text. Temporary Chrome was closed via CDP Browser.close.
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
