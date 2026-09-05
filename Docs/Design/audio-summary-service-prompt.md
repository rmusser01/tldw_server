# Audio summarization Service Prompt

Approved scope: synchronous `/api/v1/media/process-audios` only. Track in TASK-13192.

Expose `media.audio.analysis` as one atomic pair of literal `system` and `user`
instructions in the existing shared WebUI/extension Service Prompts editor.
Reuse storage, validation, reset and authenticated-owner lookup. No new settings
store or prompt interpolation.

Before uploads or transcription, resolve missing request parts once for the
authenticated owner. Each explicit request part (including empty text) wins over
the corresponding saved part. Both explicit parts, disabled analysis, and no
provider bypass owner storage. Canonical `api_provider` takes precedence over
legacy `api_name`. The resulting pair stays fixed across files and summary passes.

Without a saved override, use the existing audio prompt-file loader; missing
system instructions fall back to the shared analyzer default, and missing user
instructions remain empty. Settings detail/reset show these effective defaults.
Release the prompt lookup connection on the same worker that opened it, including
when an invalid saved pair is rejected. Reject corruption before saving uploads.

The audio core accepts a narrowly scoped marker for already-resolved instructions
so it does not replace intentional empty values with file defaults. Its existing
default behavior remains unchanged for direct and background callers. Transcriber
configuration, transcript framing, provider credentials, output shape, video and
queued/persisted ingestion remain unchanged.

Verification: real multipart endpoint, owner databases and audio batch/analyzer
flow, with only external transcription/model calls substituted; precedence,
atomic snapshots, deployment defaults/reset, no-read paths, corruption/cleanup,
direct-core compatibility, shared editor tests and Bandit on touched runtime code.
