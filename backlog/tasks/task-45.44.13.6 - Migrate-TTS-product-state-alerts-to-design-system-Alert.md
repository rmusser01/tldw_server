---
id: TASK-45.44.13.6
title: Migrate TTS product-state alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-30 04:53'
labels:
  - design-system
  - webui
  - product-state
  - tts
dependencies: []
references:
  - apps/packages/ui/src/components/Option/TTS/TtsPlaygroundPage.tsx
  - apps/packages/ui/src/components/Option/TTS/VoiceCloningManager.tsx
  - apps/packages/ui/scripts/verify-design-system-product-state.mjs
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45.44.13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the tldw_server WebUI design-system product-state migration by replacing remaining TTS AntD Alert product-state surfaces in TtsPlaygroundPage and VoiceCloningManager with the shared design-system Alert primitive while preserving copy, actions, and layout behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TTS product-state Alert surfaces in TtsPlaygroundPage and VoiceCloningManager render through the shared design-system Alert primitive while preserving existing user-facing copy and actions.
- [x] #2 Focused TTS coverage asserts migrated alerts are inside the design-system Alert marker.
- [x] #3 Direct product-state guard scan over the touched TTS files reports zero findings for the migrated Alert surfaces.
- [x] #4 Verification records focused Vitest coverage, product-state guard status, diff whitespace, TypeScript touched-file status where practical, and Bandit applicability.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- RED: `bunx vitest run src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx` failed on the new ffmpeg and ElevenLabs assertions because the visible text had no `data-ds-component="Alert"` ancestor while the component still used AntD Alert.
- RED: `bunx vitest run src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx` failed on the new disabled-provider and voice-role validation assertions for the same missing design-system Alert marker.
- Migrated TtsPlaygroundPage ffmpeg warning and ElevenLabs setup/load guidance to `@/components/ui/primitives/Alert`, preserving titles, body copy, retry/settings actions, and layout classes.
- Migrated VoiceCloningManager disabled-provider and voice-role validation alerts to the shared design-system Alert primitive while preserving copy and surrounding upload/role flows.
- Removed the five matching TTS Alert baseline exceptions from `design-system-product-state-baseline.json`.
- GREEN: `bunx vitest run src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx` passed 6 tests.
- GREEN: `bunx vitest run src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx` passed 4 tests.
- `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed 54 tests.
- Direct product-state analyzer over `TtsPlaygroundPage.tsx` and `VoiceCloningManager.tsx` returned `[]` for both files.
- `bun run verify:design-system-state` still exits 1 on unrelated current-dev Skills/ScheduledTasks blockers plus one unrelated stale Models baseline row; no TTS blocked or stale findings remain.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still exits 2 on existing repo-wide TypeScript debt; no diagnostics reference the touched TTS component/test files.
- `git diff --check` passed.
- Bandit is not applicable because this slice only touches frontend TypeScript/TSX, JSON baseline data, and Backlog markdown.
- PR: https://github.com/rmusser01/tldw_server/pull/2552
- Review follow-up: Gemini flagged the newly migrated VoiceCloningManager alert title/body strings as hardcoded. Added `useTranslation(["playground"])` and `playground:tts.cloning.*` fallback keys for disabled-provider guidance and voice-role alert titles.
- RED review test: `bunx vitest run src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx` failed 2/4 assertions because translated provider and voice-role alert strings were not rendered.
- GREEN review verification: `bunx vitest run src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx` passed 4 tests; the combined TTS regression command passed 10 tests; `product-state-guard.test.ts` passed 54 tests; direct analyzer over the touched TTS files exited 0; `git diff --check` passed.
- Review verifier status: `bun run verify:design-system-state` still exits 1 only on unrelated Skills/ScheduledTasks blocked findings plus the unrelated stale Models baseline row; `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still exits 2 on existing repo-wide TypeScript debt with no diagnostics in touched VoiceCloningManager files.
- Review follow-up: verified the voice-role Alert body still used hardcoded `voiceRoleError` strings. Localized all five voice-role validation branches with `playground:tts.cloning.voiceRole*` fallback keys before rendering the existing localized warning title.
- RED body-localization test: `bunx vitest run src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx` failed 1/4 because `Translated missing voice` was not rendered.
- GREEN body-localization test: `bunx vitest run src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx` passed 4 tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the remaining TTS playground and voice-cloning product-state Alert callouts from AntD Alert to the shared design-system Alert primitive. Focused tests now assert DS Alert ownership for ffmpeg warning, ElevenLabs setup guidance, disabled provider guidance, and voice-role validation; the five obsolete TTS baseline exceptions were removed. PR review follow-up made the newly migrated VoiceCloningManager alert labels and voice-role validation body strings translation-backed with `playground:tts.cloning.*` fallback keys.
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
