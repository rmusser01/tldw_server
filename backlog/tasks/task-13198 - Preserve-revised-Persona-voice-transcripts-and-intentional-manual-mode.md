---
id: TASK-13198
title: Preserve revised Persona voice transcripts and intentional manual mode
status: In Progress
assignee: []
created_date: '2026-09-06 00:09'
updated_date: '2026-09-06 00:17'
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
- [ ] #4 Targeted backend and frontend regressions pass and sanitized human UAT evidence records acceptance limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; amend existing Docs/ADR/046-persona-live-conversation-and-voice-runtime.md for the additive transcript snapshot event field. Reason: repair the existing voice streaming contract without new provider or storage authority. 1. Reproduce snapshot rollback and false manual-mode warnings in targeted tests. 2. Preserve unchanged snapshots, send authoritative replacement text and consume it in the existing voice controller; log committed speech once. 3. Keep deliberate manual mode available. 4. Run targeted checks, then coordinate a fresh human browser retest.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed transcript rollback on no-result audio chunks and added authoritative transcript snapshots consumed as replacements in the voice controller. Conversation log now records committed speech once. Explicit auto-commit off no longer emits an unavailable-VAD warning, and status text describes manual Send. Two backend and three frontend regressions failed before repair; final scopes passed 125 Python and 165 frontend tests. Bandit zero findings; Ruff no added findings; ESLint zero errors/12 existing warnings; scoped TypeScript zero owned diagnostics/27 dependency diagnostics. Existing ADR046 amended and user guide updated. Human browser baseline proved audible provider playback but failed transcript integrity; post-fix retest remains pending, so task remains In Progress.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
