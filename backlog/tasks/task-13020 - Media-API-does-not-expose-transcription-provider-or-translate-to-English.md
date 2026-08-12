---
id: TASK-13020
title: Media API does not expose transcription provider or translate-to-English
status: To Do
assignee: []
created_date: '2026-08-11'
labels:
  - media
  - api
  - transcription
dependencies: []
---

## Description

The transcription core supports choosing an STT provider and running Whisper's
translate task, but **no media HTTP endpoint lets a client ask for either**. A
caller can select the model, the language, diarization, timestamps and VAD, and
then cannot say which provider should run it or that the output should be
translated to English.

Found from the client side: `tldw_chatbook`'s Library ingest canvas offers
"Transcription provider" and "Translate to English" controls, forwarded them as
form fields, and they were silently discarded because the endpoint does not
declare them. The client now drops them and tells the user they cannot be
honoured — but the capability exists here, so the honest fix is on this side.

**The core already does both:**

- `Audio_Transcription_Lib.transcribe_audio(audio_data, transcription_provider,
  ...)` takes the provider as a positional parameter.
- `stt_provider_adapter` takes `task: str = "transcribe"` and handles
  `"translate"` explicitly (`selected_lang = None if task == "translate" else
  language`, and `target_language="en" if task == "translate" else None`).

**Nothing above that layer carries them.** Checked, in order:

1. `AudioVideoOptions` (`schemas/media_request_models.py`) declares
   `transcription_model`, `transcription_language`, `hotwords`, `diarize`,
   `timestamp_option`, `vad_use` — no provider, no translate.
2. `get_add_media_form` (`API_Deps/media_add_deps.py`) binds each field with an
   explicit `Form(...)` and never reads `request.form()`, so an undeclared field
   cannot arrive by any route.
3. `Audio_Files.process_audio_files(...)` — the ingestion entry point — takes
   `transcription_model`, `transcription_language` and `diarize`, but no
   provider and no task.

So the plumbing is missing across two layers, not one: the request schema and
the ingestion function both need to carry the values down to the adapter that
already understands them.

Confirmed against a live instance: every endpoint under `/api/v1/media/` was
enumerated from `/openapi.json`, and neither `transcription_provider` nor
`translate_to_english` (nor any synonym) appears on `/media/add`,
`/media/ingest/jobs`, or any `process-*` route. `/media/add` is a strict subset
of `/media/ingest/jobs`, so there is no fuller surface a client could use
instead.

Note for whoever picks this up: the endpoint's silent-discard behaviour is what
made this invisible for so long. An undeclared form field produces no error and
a `200`, so a client asking for an unsupported option gets a successful-looking
job that ignored the request. Worth considering whether unknown form fields
should be rejected, or at least reported in the response, independently of this
task.

## Acceptance Criteria

- [ ] A client can select the transcription provider for an audio/video ingest, and the chosen provider is the one that runs
- [ ] A client can request translate-to-English, and the resulting transcript is English for non-English input
- [ ] Both options are accepted on the same endpoints that already accept `transcription_model` (`/media/add` and `/media/ingest/jobs`)
- [ ] An invalid provider value is rejected with a validation error naming the field, rather than ignored
- [ ] The values reach `stt_provider_adapter` — asserted end to end, not only at the schema boundary
