## Stage 1: Regression Coverage
**Goal**: Add focused tests for the reviewed TTS defects before production changes.
**Success Criteria**: Tests fail against the current behavior for config redaction/env mapping, VibeVoice generation config propagation, conversion failure surfacing, subprocess timeout handling, and realtime websocket egress validation.
**Tests**: Targeted pytest runs for the new/updated TTS test files.
**Status**: Complete

## Stage 2: VibeVoice State and Async Boundaries
**Goal**: Forward VibeVoice generation configuration in non-streaming mode, serialize shared model state, and move blocking local generation calls off the event loop.
**Success Criteria**: VibeVoice uses request generation config consistently, model reload/generation paths cannot concurrently mutate shared model state, and local blocking generation entry points run through thread offload.
**Tests**: VibeVoice adapter unit tests and compile checks for touched adapters.
**Status**: Complete

## Stage 3: Audio Conversion Hardening
**Goal**: Bound ffmpeg/ffprobe subprocess calls and surface voice-reference conversion failures.
**Success Criteria**: Audio converter subprocesses return timeout failures instead of waiting indefinitely, and strict voice-reference conversion raises instead of silently returning original bytes.
**Tests**: Audio converter and audio utility regression tests.
**Status**: Complete

## Stage 4: Security Policy and Config Redaction
**Goal**: Enforce central egress policy for realtime websocket backend URLs and redact TTS provider secrets by default.
**Success Criteria**: Websocket realtime adapters reject URLs denied by egress policy, `TTSConfig.to_dict()`/YAML export redact API keys unless explicitly requested, and stale non-TTS env mappings are removed.
**Tests**: Realtime adapter and TTS config regression tests.
**Status**: Complete

## Stage 5: Verification and Task Closeout
**Goal**: Run focused tests plus Bandit on touched production TTS scope and record outcomes in TASK-10001.
**Success Criteria**: Focused tests pass, Bandit reports no new actionable findings in touched production code, and task notes/final summary are updated.
**Tests**: `python -m pytest ...`, `python -m bandit -r ...`, `git diff --check`.
**Status**: Complete
