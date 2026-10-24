---
id: TASK-13905
title: Kokoro's device probe timeout does not bound the adapter constructor
status: To Do
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
KokoroAdapter.__init__ (core/TTS/adapters/kokoro_adapter.py:222-232 on dev e06eeec68c) runs _probe_devices (a torch import and CUDA/MPS probe) in 'with concurrent.futures.ThreadPoolExecutor(max_workers=1)' and waits future.result(timeout=device_probe_timeout_sec). At the timeout the with-exit calls shutdown(wait=True), which waits for the probe anyway, so the timeout changes only the result (CPU), not the wait. Measured while planning TASK-13893: with a probe that takes 2 s and kokoro_device_probe_timeout_sec 0.1, KokoroAdapter() took 2.00 s and logged 'Device probe timed out; defaulting to CPU'. The registry constructs adapters on the event loop (adapter_registry.py _initialize_adapter and create_adapter_with_overrides), so a hung torch import or device probe blocks the server. The probe is skipped in test mode, so no test sees it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A device probe that outlasts kokoro_device_probe_timeout_sec no longer holds KokoroAdapter() past about that bound, with a test that does not depend on test-mode detection
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
