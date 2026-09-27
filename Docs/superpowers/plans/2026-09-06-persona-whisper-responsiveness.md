# Persona Whisper responsiveness — TASK-13208

ADR required: yes, amendment to existing ADR.
ADR path: Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Keep recognition, publication and cleanup ownership bounded while moving
native inference out of the socket receive loop.

## Stage 1: Reproduce blocking

**Goal:** Establish the deferred PR2908 defect on latest dev.
**Success criteria:** A blocked real transcriber method reproduces blocked audio ingestion.
**Tests:** Deterministic gated decoder; baseline one failure and four passes.
**Status:** Complete.

## Stage 2: Preserve recognition and turn ownership

**Goal:** One background decoder, bounded audio, completion-based cadence, retained cleanup.
**Success criteria:** Stop/disconnect remain prompt, retries wait for retired inference,
late results cannot publish, and VAD waits for its exact boundary without losing later audio.
**Tests:** Capacity, timeout, cancellation, reset, revisions, short final turns,
real TestClient WebSocket Stop/disconnect and VAD carry regressions.
**Status:** Complete; VAD boundary/carry correction added after independent review.

## Stage 3: Verify and hand back to physical UAT

**Goal:** Qualify the repair separately from human microphone acceptance.
**Success criteria:** Focused tests, lint, formatting, Bandit and independent review;
real local-model responsiveness receipt; normal Parakeet ONNX CPU preparation.
**Tests:** Synthetic Kokoro-to-Whisper probe without a microphone or external
provider, followed by preparation-only Parakeet/Kokoro probe. TASK-13202 retains
physical floating-Buddy listening/thinking/speaking/idle acceptance.
**Status:** Complete; 165 focused tests, clean touched-file checks and local-model
receipts recorded in Docs/Reviews/MIGU_VOICE_FOLLOWUP_2026_09_06.md. Human acceptance
remains in TASK-13202.
