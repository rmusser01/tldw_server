---
id: TASK-2355
title: Specify Audio Studio remaining work roadmap
status: Done
documentation:
- Docs/superpowers/specs/2026-06-24-audio-studio-remaining-roadmap-design.md
modified_files:
- Docs/superpowers/specs/2026-06-24-audio-studio-remaining-roadmap-design.md
- backlog/tasks/task-2355 - Specify-Audio-Studio-remaining-work-roadmap.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the accepted overarching roadmap/spec for remaining Audio Studio work, covering creator MVP stabilization, platform hardening, first-class Narration/Podcast/Briefing workflows, timeline editing, and later music/SFX expansion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Roadmap spec documents the remaining Audio Studio phases and priority order.
- [x] #2 Spec keeps Narration, Podcast, and Briefings as first-class spoken-audio workflows.
- [x] #3 Spec covers artifact access, provider capability schema, migration compatibility, render/export, jobs, security, Briefing source authorization, waveform strategy, and ACE-Step/music expansion.
- [x] #4 Spec identifies the recommended next implementation slice and key risks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write `Docs/superpowers/specs/2026-06-24-audio-studio-remaining-roadmap-design.md` from the accepted brainstorming roadmap. Keep the roadmap aligned with the existing Audio Studio design vocabulary and make the next implementation slice explicit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created the roadmap spec from the accepted brainstorming sections and refinement review.
- Kept music and ACE-Step expansion behind creator MVP stabilization and platform hardening.
- Added explicit cross-cutting gates for artifact storage/retention, auth-mode compatibility, versioned provider capabilities, Briefing source authorization, paid-call retry/idempotency rules, waveform strategy, and provider setup docs.
- Applied spec-review fixes: clarified that this roadmap supersedes the prior MVP timing for ACE-Step/music, moved minimum Briefing source authorization into Phase 1, corrected `/audiobook-studio` compatibility wording, and added explicit media response requirements for artifact playback/download.
- Applied second review fixes: clarified that basic timeline volume/mute/fade controls remain part of MVP stabilization, and added untrusted-source prompt handling plus prompt-injection test coverage for Briefing helper generation.
- Folded in approved-review non-blocking suggestions for speech/TTS capability terminology and signed temporary URL constraints.
- Verification: markdown whitespace scans on the spec and task passed; targeted staged `git diff --check` passed after staging. Bandit is not applicable because this task changed documentation and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wrote and reviewed `Docs/superpowers/specs/2026-06-24-audio-studio-remaining-roadmap-design.md`. The roadmap prioritizes Audio Studio creator MVP stabilization plus platform foundation, keeps Narration/Podcast/Briefings first-class, moves ACE-Step/music expansion behind core spoken-workflow reliability, and identifies artifact playback/download as the recommended next implementation slice.
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
