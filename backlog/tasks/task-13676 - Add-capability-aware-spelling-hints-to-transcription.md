---
id: TASK-13676
title: Add capability-aware spelling hints to transcription
status: To Do
labels:
- audio
- transcription
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During A05 UAT the user-provided recording was transcribed by the selected Parakeet ONNX route with telescope name Celine instead of the frozen reference Selene. On 2026-10-14 the human explicitly waived this accuracy check for the current UAT and requested a backlog item for spelling-hint options. Preserve the raw failed sample; this waiver is not evidence of improved ASR accuracy. Add explicit model-aware prompt/domain-vocabulary/hotword support where the real recognition engine supports it, with honest unsupported-model guidance. Never implement a fixture-specific spelling replacement. Related UAT record: TASK-13260.278.18.83.45 and Docs/Plans/IMPLEMENTATION_PLAN_uat_a05_ingest_progress_20261013.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Assess supported spelling/context hints for each relevant STT engine and document the capability contract; models lacking support must report that limitation instead of silently accepting ignored hints.
- [ ] #2 Expose supported optional hints in applicable transcription/ingestion interfaces and forward them to the actual recognition engine; absent hints retain current behavior and request/principal/cancellation boundaries.
- [ ] #3 Keep original audio and raw transcript; record the selected hint/model execution provenance for a hinted result, without logging private vocabulary, credentials or audio contents.
- [ ] #4 Qualify no-hint compatibility, supported hint forwarding, unsupported-model rejection, input validation and hint-sensitive cache identity; include a bounded real recognition comparison when a supported installed model is available.
- [ ] #5 Update user guidance and assess governing ADRs before implementation; explicitly distinguish raw model accuracy from user-reviewed transcript corrections.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requested directly by the human on 2026-10-14. Local search found no focused spelling-hints task. ID allocated above highest13675 across local backlog, live dev63438ab0ae664785d64fdd1e94377281458b3711 and55open PRs/56complete backlog trees using read-only API metadata; no fetch/ref/worktree changes. Existing ADR022 media/embedding pipeline and shared native STT execution contract TASK-12985.3 are relevant; a new durable capability rule requires an ADR assessment during design. Current installed onnx-asr0.12.0 RecognizeOptions has no Parakeet spelling-hint option; this observation is scoped to inspected code, not a claim about every engine or future version. Feature not implemented or validated by this tracking task.
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
