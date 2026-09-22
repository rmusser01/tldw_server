---
id: TASK-13329
title: Quadratic bytes accumulation when draining audio streams in four TTS adapters
status: In Progress
assignee: []
created_date: '2026-09-22 04:58'
updated_date: '2026-09-22 16:48'
labels:
  - efficiency
  - tts
dependencies: []
references:
  - 'tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py:1100'
  - 'tldw_Server_API/app/core/TTS/waveform_streamer.py:65'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five sites use all_audio = b"" then all_audio += chunk. bytes is immutable, so each iteration reallocates and full-copies: O(N*K) memcpy where N is total encoded bytes and K is chunk count.

MEASURED on 1200-byte chunks: 500 chunks (0.6MB) 2.4ms vs 0.05ms bytearray = 44x; 2000 chunks 47.0ms vs 0.25ms = 186x; 8000 chunks 1087.7ms vs 1.05ms = 1032x. A ~25-min Kokoro/VibeVoice generation lands in the 8000-chunk range: ~1.1s of pure memcpy on the event loop thread, stalling every other in-flight request. Cost is quadratic so it widens with length.

Sites: dia_adapter.py:477, higgs_adapter.py:494, kokoro_adapter.py:1100 and :1118, vibevoice_adapter.py:1120. Five of nine sites in the module already use bytearray correctly (chatterbox x2, waveform_streamer, tts_service_v2, tts_jobs_worker, gateway_execution).

Fix: bytearray() then return bytes(...). No behaviour change, no new test needed.

Source: synthesis F29
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All five sites use bytearray
- [ ] #2 Existing adapter tests still pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE. All five sites converted from bytes to bytearray accumulation:
  dia_adapter.py:480, higgs_adapter.py:497, kokoro_adapter.py:1100 and :1117, vibevoice_adapter.py:1123.
Each carries a comment naming the measured cost (1032x at 8000 chunks, ~1.1s of pure memcpy on the event loop thread).

RETURN TYPE PRESERVED - the one real risk. Every changed return now wraps with bytes(): four plain returns plus kokoro second site which returns a tuple (bytes(all_audio), alignment_payload). A caller doing isinstance(x, bytes) would have broken on a raw bytearray; none can.

Regression, stash-isolated: tests/TTS is 54 failed / 549 passed BOTH with and without the change - identical, so pre-existing and unrelated (the TTS/TTS_NEW split and env gaps the review flagged). No behaviour change, as expected for a pure accumulation fix.
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
