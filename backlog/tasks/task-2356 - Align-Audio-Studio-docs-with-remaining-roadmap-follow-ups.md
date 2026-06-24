---
id: TASK-2356
title: Align Audio Studio docs with remaining roadmap follow-ups
status: Done
documentation:
- Docs/Audio_Studio.md
- Docs/superpowers/specs/2026-06-24-audio-studio-remaining-roadmap-design.md
modified_files:
- Docs/Audio_Studio.md
- Docs/superpowers/specs/2026-06-24-audio-studio-remaining-roadmap-design.md
- backlog/tasks/task-2356 - Align-Audio-Studio-docs-with-remaining-roadmap-follow-ups.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the roadmap review follow-ups by aligning the public Audio Studio docs with deferred music/ACE-Step timing, tightening the next implementation slice split, forcing the artifact access strategy decision into planning, and marking the roadmap accepted for implementation planning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Public Audio Studio docs clarify that Narration, Podcast, and Briefing are the stabilization priority.
- [x] #2 Public Audio Studio docs clarify that Music/ACE-Step support is deferred behind creator MVP stabilization and platform hardening.
- [x] #3 Roadmap next-slice handoff defaults provider capability metadata to a separate follow-up unless code inspection proves it is tiny.
- [x] #4 Roadmap next-slice handoff requires an artifact playback/download access strategy decision before endpoint implementation starts.
- [x] #5 Roadmap status reflects user acceptance for implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Updated `Docs/Audio_Studio.md` to describe Audio Studio as Narration/Podcast/Briefing-first during stabilization, move Music/SFX/ACE-Step language into planned/follow-up framing, and point readers to the accepted remaining-work roadmap.
- Updated `Docs/Audio_Studio.md` to call `/audiobook-studio` a compatibility interstitial/fallback route rather than implying a hard redirect.
- Updated `Docs/superpowers/specs/2026-06-24-audio-studio-remaining-roadmap-design.md` to `Accepted for implementation planning`.
- Tightened the recommended next implementation slice so artifact playback/download is the default next plan, with minimum provider capabilities separate unless code inspection proves it is genuinely tiny.
- Added an explicit access-strategy decision gate for backend streaming vs signed temporary URL vs hybrid before artifact endpoint implementation.
- Verification: targeted `git diff --check` on touched docs/task files exited 0; targeted `rg` confirmation found the accepted status, deferred Music/ACE-Step wording, roadmap alignment section, provider capability split, artifact access strategy gate, and compatibility interstitial wording. Bandit is not applicable because this task changed documentation and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned the public Audio Studio docs and accepted roadmap handoff with the reviewed follow-up findings. The docs now clearly prioritize Narration, Podcast, and Briefing; defer Music/ACE-Step expansion; preserve `/audiobook-studio` as a compatibility interstitial/fallback; and require the artifact playback/download strategy decision before implementation planning proceeds.
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
