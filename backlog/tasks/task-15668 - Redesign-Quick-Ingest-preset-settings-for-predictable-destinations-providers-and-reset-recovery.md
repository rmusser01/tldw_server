---
id: TASK-15668
title: Redesign Quick Ingest preset settings for predictable destinations, providers
  and reset recovery
status: In Progress
labels:
- webui
- quick-ingest
- accessibility
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the standalone settings review. Make future-run destinations and provider prerequisites predictable and the preset editor usable on phones and by keyboard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Select one named preset, edit one named region and see the current destination above optional controls on a phone. Shared OCR follows the editor.
- [x] #2 Choose Save to Media, Create review drafts or Extract without saving and see provider guidance next to Analysis. Preserve existing stored settings and verify a new ingest uses the saved choices.
- [x] #3 Reset one named preset or all three, see which were reset, and undo without overwriting unrelated changes.
- [ ] #4 Verify settings persistence, keyboard and phone layout, and publish desktop and mobile screenshots with the PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: exercise the existing options flow and capture failing page behaviors, implement the editor on the existing preset and OCR storage keys, then verify the fresh-ingest handoff and publish browser evidence. ADR required: no. This is a presentation change to existing settings, storage, provider and ingest contracts.
Implementation preserves the existing preset map and shared OCR keys. The editor exposes the three destinations, missing-provider guidance, one keyboard-navigable preset region and searchable advanced fields. Reset receipts name the affected presets and offer Undo. Preset writes are ordered, and failures retain a retry path. Tests reproduced the prior missing controls, delayed-write data loss and reset/Undo failures. The final focused run passed 450 tests across 21 suites.
Verified fresh session handoff at the WebUI request boundary using a controlled API. The saved Standard preset sent ephemeral extraction with analysis and the chosen provider. A mobile WebKit run passed keyboard preset changes and reset recovery. The 390px page keeps Destination in the first viewport without horizontal scrolling. Final desktop and phone captures are in Docs/Design/assets/quick-ingest/preset-settings/. Frontend typecheck, scoped lint, English catalog parity, guide parity and diff checks passed. Bandit had no Python source in the changed frontend scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
