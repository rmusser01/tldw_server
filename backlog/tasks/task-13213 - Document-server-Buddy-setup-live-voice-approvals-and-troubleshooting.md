---
id: TASK-13213
title: Document server Buddy setup live voice approvals and troubleshooting
status: Done
created_date: 2026-09-07 08:35
labels:
- documentation
- buddy
updated_date: 2026-09-07 16:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users a single practical Buddy guide with setup, Migu selection, movement, voice, approvals and honest troubleshooting guidance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A discoverable user guide covers the current Buddy workflow and links related setup and voice documentation.
- [x] #2 Instructions distinguish verified behavior from outstanding UAT limitations and use current control labels.
- [x] #3 Relative documentation links and navigation resolve; published mirrors match where applicable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: user documentation of existing behavior, no runtime or contract change.
1. Read current controls, existing guides and UAT records.
2. Write a dedicated guide and link it from the documentation index and relevant feature pages.
3. Validate links/navigation, inspect the rendered Markdown, and record source revision and evidence limitations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added the canonical Persona_Buddy_Guide.md and byte-identical Published mirror, with links from both Persona and WebUI user guides and the MkDocs navigation. Documents configuration, pixel-migu Copy as draft then Activate, dragging, text/voice controls, Send now versus Stop voice, exact-session approvals, and troubleshooting. Explicitly preserves outstanding animation TASK-13202 and repeated-request TASK-13211 limitations. Independent source review found and resolved the missing Copy as draft instruction. Validation: case-sensitive relative links resolve, all three changed canonical/Published pairs match, Markdown layout inspected, git diff --check passes, full MkDocs build passed using the installed git revision plugin's supported serial mode after this machine could not allocate multiprocessing semaphores. Repository build configuration was unchanged. Documentation-only change; no runtime or security boundary change; no new physical microphone test claimed. ADR required: no, documentation of existing behavior.
Bandit skipped: only Markdown and MkDocs navigation YAML changed, with no Python code touched. Known environment workaround (serial docs build) and remaining product UAT limitations are recorded above.
User-requested second review: independently inspect setup and voice controls, reconcile contradictory voice setup guidance in linked Personas guide, and recheck publishing/link behavior. Changes remain within current documentation AC.
Second independent review resolved two onboarding defects: replaced hidden legacy starter picker with Visuals → Buddy builder → Bundled Buddy → pixel-migu → Copy as draft → Continue to activation → Activate; explicitly require local Kokoro/TLDW TTS and document Profiles fields plus Save assistant defaults because Live STT/TTS summaries are read-only. Updated linked Personas guide's stale Start and Whisper-only preparation text to current Start listening and supported Whisper/Parakeet ONNX examples, distinguishing supported selections from physical UAT. Source verified in VisualPackEditor, BuddySourcePicker, BuddyDraftReviewPanel, AssistantDefaultsPanel, AssistantVoiceCard, persona.py and live_stt.py. Final Markdown rendered-link checks pass (4 relative targets per Buddy mirror), all three changed canonical/Published pairs match, and git diff --check passes. Previous full serial MkDocs build remains recorded; no runtime tests or physical microphone UAT repeated for prose changes.
Correction after TASK-13214: the earlier Kokoro-mandatory guidance described an accidental UAT restriction, not the intended product requirement. That runtime restriction is now removed. Both canonical and Published guides document configured TTS provider/browser/gateway, optional model/voice, no silent substitution, and authenticated Chat credentials/policy/budget enforcement at dispatch. Historical Kokoro UAT remains evidence only. Provider-selection verification and remaining limits are in Docs/Reviews/PERSONA_TTS_PROVIDER_CHOICE_2026_09_07.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and linked the server Buddy guide, synchronized its published copy, and verified documentation rendering/build and source accuracy.
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
