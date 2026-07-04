# Audio Transcription Pipeline Hardening Plan

## Stage 1: Endpoint Canonicalization
**Goal**: Ensure `/audio/transcriptions` either passes a usable canonical WAV path to adapters or returns a clear client error.
**Success Criteria**: Conversion import/conversion failures and unusable conversion outputs do not fall back to the original compressed upload; existing WAV no-op conversion remains valid.
**Tests**: Added endpoint regression coverage for conversion failure, empty conversion output, and non-WAV conversion output.
**Status**: Complete

## Stage 2: Provider Boundary Guards
**Goal**: Keep direct soundfile-backed provider paths from silently accepting compressed inputs that the loader cannot decode.
**Success Criteria**: Canary, Qwen3 ASR, VibeVoice, and Parakeet MLX buffered direct-call paths either canonicalize to WAV or raise clear provider errors without deleting caller-owned files.
**Tests**: Added focused unit tests for Canary, Qwen3 ASR, VibeVoice, Parakeet MLX buffered path conversion behavior, and MLX converted-path base_dir safety.
**Status**: Complete

## Stage 3: In-Memory Sample Rate Normalization
**Goal**: Make NeMo Parakeet and Canary direct NumPy transcription honor non-16 kHz sample rates.
**Success Criteria**: Non-16 kHz NumPy inputs are resampled to 16 kHz before direct model transcription; ONNX/MLX behavior remains unchanged.
**Tests**: Added unit tests that patch model transcribe calls and verify array length normalization for Parakeet and Canary, plus empty-array no-crash coverage.
**Status**: Complete

## Stage 4: Verification and Closeout
**Goal**: Validate the focused bugfix without widening the branch scope.
**Success Criteria**: Focused pytest targets pass, Ruff passes on touched Python files, Bandit runs on touched Python paths, and task notes capture any known skips.
**Tests**: After PR review fixes, focused pytest suite passed (`85 passed, 2 skipped`); high-signal Ruff rules passed; `compileall` passed; `git diff --check` passed; Bandit ran and reported only existing low-severity subprocess findings outside this diff.
**Status**: Complete
