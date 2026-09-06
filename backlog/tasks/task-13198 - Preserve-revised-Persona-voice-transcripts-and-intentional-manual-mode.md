---
id: TASK-13198
title: Preserve revised Persona voice transcripts and intentional manual mode
status: Done
assignee: []
created_date: '2026-09-06 00:09'
updated_date: '2026-09-06 01:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Human browser UAT spoke one phrase but Persona submitted repeated fragments to DeepSeek. Manual auto-commit off was also mislabeled as unavailable VAD. Preserve recognizer snapshots and distinguish intentional manual control from failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Revised transcript snapshots replace provisional text and unchanged audio events do not replay already heard words.
- [x] #2 The submitted provider turn and conversation log contain one committed transcript while intentional repeated speech remains intact.
- [x] #3 Auto-commit off remains a usable manual voice mode without a false VAD failure or disabled configuration control.
- [x] #4 Targeted backend and frontend regressions pass and sanitized human UAT evidence records acceptance limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; amend existing Docs/ADR/046-persona-live-conversation-and-voice-runtime.md for the additive transcript snapshot event field. Reason: repair the existing voice streaming contract without new provider or storage authority. 1. Reproduce snapshot rollback and false manual-mode warnings in targeted tests. 2. Preserve unchanged snapshots, send authoritative replacement text and consume it in the existing voice controller; log committed speech once. 3. Keep deliberate manual mode available. 4. Run targeted checks, then coordinate a fresh human browser retest.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved authoritative transcript snapshots on no-result audio chunks, replaced browser provisional revisions including empty corrections, and logged each committed utterance once. Intentional manual mode remains usable without false VAD failure. Initial red/green scope: 125 backend and 165 frontend tests; subsequent recognition fixes under TASK13199/13200/13201 passed the final 134-test Persona/Whisper scope. Bandit zero findings and no added Ruff/ESLint/owned TypeScript findings; existing baseline diagnostics documented. ADR046 and guide updated. Human manual UAT on 31046b8937 now submitted the notebook phrase once, produced the expected DeepSeek reply, and the user confirmed clear Kokoro playback with no added or duplicated words. Receipt: Docs/Reviews/assets/migu-buddy-browser-voice-2026-09-05/whole-turn-human-acceptance.json. Broader direct capture-state/Buddy-state and alternative-provider qualification remain under TASK13195.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
