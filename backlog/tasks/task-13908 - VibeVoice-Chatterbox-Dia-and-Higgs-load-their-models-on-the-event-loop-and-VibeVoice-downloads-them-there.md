---
id: TASK-13908
title: VibeVoice, Chatterbox, Dia and Higgs load their models on the event loop (and
  VibeVoice downloads them there)
status: To Do
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four local adapters load model weights synchronously on the event loop, so each load stalls every request on the server, including the in-process TTS jobs worker's lease renewal. On dev e06eeec68c: VibeVoice initialize() runs from_pretrained (core/TTS/adapters/vibevoice_adapter.py:416-470) after _download_models (:576), whose huggingface_hub.snapshot_download (:610) is a multi-GB download on the loop, and _reload_model_for_variant (:1341-1359) re-runs initialize() under _model_state_lock; Chatterbox loads lazily from generate() (chatterbox_adapter.py:492, :1055) through _get_model/_get_vc_model -> _load_chatterbox_runtime (:631-651, called at :1368, :1395, :1420, :1449), so the first request stalls every request; Dia (_load_dia_model, dia_adapter.py:257, :280) and Higgs (HiggsAudioServeEngine at higgs_adapter.py:235) have the same shape. TASK-13855/13857 fixed this class for NeuTTS (its load 'blocks the server once per process, including the in-process TTS jobs worker's lease renewal'). TASK-13893 left these four unchanged: a loop-bound load cannot be orphaned or overlap another, so they needed no load lock. Moving a load to a worker thread without the shared lock would bring back what TASK-13893 fixed, so each move must hold tts_model_load_lock (core/TTS/utils.py) through hold_tts_engine. Found by the independent review of the TASK-13893 plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed load (and VibeVoice's snapshot_download) runs off the event loop through hold_tts_engine(tts_model_load_lock(...)), keyed on the adapter's provider_key and model identity
- [ ] #2 Each moved load is a case in tests/TTS/unit/test_tts_engine_lock_cancellation.py's cross-instance matrix (test_a_new_adapter_instance_waits_for_another_instances_load_of_the_same_model), for a plain task cancel and an anyio cancel scope, with stub engines only
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
