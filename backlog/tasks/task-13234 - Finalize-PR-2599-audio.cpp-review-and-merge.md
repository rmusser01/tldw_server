---
id: TASK-13234
title: Finalize PR 2599 audio.cpp review and merge
status: In Progress
assignee: []
created_date: '2026-09-10 01:19'
updated_date: '2026-09-10 01:20'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2599'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the completed audio.cpp review archived under the original TASK-13231 allocation. Latest dev independently used that ID for VZ work. Track final Qodo feedback, CI and policy gates for PR 2599 without an active ID collision.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rebase reviewed implementation onto latest dev
- [x] #2 Address verified Qodo findings with regression coverage and independent review
- [ ] #3 Merge after final CI and repository human Change summary rationale gate are satisfied
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Rebased the eight PR commits cleanly onto origin/dev 40345571a2cfc8b3a8893545836097d27e4ee86c. Current dev has audio.cpp STT but no equivalent TTS provider, so this remains useful.
Verified and fixed: inactive cloning capability; non-private reference files and cancellation cleanup; WAV bytes incorrectly returned to PCM service chunking; wrong WAV rate/channel metadata; stale managed clients after a child crash; cancelled-startup resource leaks; shared generated config paths; raw upstream error-body exposure; invalid installer backend flags, binary paths, obsolete model manager and package layout; missing PocketTTS default voice. Removed the unrelated Chatterbox alias rewrite. Installer now follows the current Helper_Scripts/TTS_Installers layout.
Existing review feedback: constructor typing/client docstring and blocking file I/O addressed; broad shutdown exception suppression narrowed. The bot claim that asyncio markers violate AGENTS.md is not supported by repository policy or pytest conventions; existing async markers are retained.
The legacy idle_shutdown_seconds now maps to upstream idle_unload_ms, preserving 0. Native model unloading releases model memory without a custom process timer and protects active inference.
Validation: original baseline 34 passed; final focused and adjacent suite 127 passed, 4 existing warnings (6.90s), including real service chunk concatenation. Ruff, compileall, git diff --check passed. Bandit on all touched production Python including registry: zero findings, zero errors. New behavior regressions demonstrated failures before fixes.
Independent review completed twice plus a targeted voice-default check; no remaining actionable blockers.
Real upstream CPU smoke: built fa5aaac9266a98c68f8a5c9fcd1ba6ff65875416 on Apple Silicon, ENGINE_ENABLE_OPENMP=OFF because local OpenMP was unavailable. Downloaded public pocket_tts_english_q8_0 (134051128 bytes) into /tmp. Generated valid 24kHz mono WAV: 59520 frames, 119084 bytes. Killed child, next request restarted it and generated 69120 raw PCM bytes. Initial real synthesis exposed the missing Alba default and passed after correction. /tmp/pr2599-live-smoke.log and /tmp/pr2599-live-speech.wav hold local evidence.
Limitations: CUDA/HIP/Vulkan/Metal were not exercised live; native idle-unload compatibility is tied to the documented current upstream contract. Concurrent port probing can race and is documented as recoverable on startup retry. Full repository CI remains a merge gate; no merge performed. A human-owned Change summary explaining what and why is still required by Docs/superpowers/AI_GENERATED_PR_CHANGE_SUMMARY_POLICY_2026_04_17.md.
The review-specific working plan was completed and removed per AGENTS.md; original PR design/plan remain as historical artifacts.

Follow-up authorized by requester: address newly posted Qodo review 5611102940 before merge. Validate bounded retry after port binding races, require recognized positive health status, and atomically replace installer YAML without truncating existing config on failure. Reopening TASK-13231 for these focused review fixes.

Qodo follow-up 5611102940 addressed: startup retries confirmed bind collisions only within the configured range (including port_probe_max=0), exhausted ranges record backoff, health requires explicit positive status in the client/supervisor/adapter, and installer YAML replacement is atomic with permissions, ownership and symlink-target preservation. Regression red runs: 14 original follow-up failures, one exhausted-port/backoff failure, two ownership failures; all fixed. Final focused/adjacent suite: 147 passed, 4 existing warnings (8.82s). Ruff, compileall and git diff --check pass. Bandit touched production scope has zero findings/errors. Real CPU synthesis and forced-crash restart smoke passed again: 119084-byte 24kHz mono WAV and 69120-byte raw PCM after restart. Independent review found no remaining actionable blockers. Existing inline review threads were replied to and resolved; another Qodo review and fresh CI will run after publishing. Repository human-owned rationale remains required before merge. Follow-up working plan completed locally and removed; task retains verification and pending external merge gates.

Latest rebase: 456eafb7a603449722ba8db806071a5e2aa5e7d6. All ten commits replayed cleanly and git range-diff showed identical patches. The new dev changes only affect VZ guest Go code and its Backlog record. Qodo /agentic_review reported no code changes since its last review after the follow-up push.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed audio.cpp integration remains useful alongside current dev STT support. All verified Qodo findings addressed; 147 focused/adjacent tests pass, static/security checks are clean, and real CPU synthesis plus forced-crash recovery pass. Rebased cleanly onto dev 456eafb7a603449722ba8db806071a5e2aa5e7d6. Final CI and the human-owned rationale in Change summary remain before the authorized merge.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
