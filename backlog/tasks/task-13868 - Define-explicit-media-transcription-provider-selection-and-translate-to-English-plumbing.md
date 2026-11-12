---
id: TASK-13868
title: Define explicit media transcription provider selection and translate-to-English
  plumbing
status: To Do
labels:
- media
- api
- transcription
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audio/video media ingestion exposes transcription_model, including identifiers that already select several STT backends, but lacks an explicit provider override and translation-task contract. Decide how explicit provider selection composes with model-based routing, then carry the agreed provider and translate-to-English options through AudioVideoOptions, get_add_media_form, the /media/add and /media/ingest/jobs handlers and job payloads, Audio_Files.process_audio_files, and the STT adapter. Preserve existing defaults and validate unsupported provider/model/task combinations. The lower transcription layers already have provider/task capabilities; endpoint-to-adapter wiring and acceptance evidence are still needed. Undeclared-field handling is tracked separately as TASK-13869.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Record explicit provider versus model-routing semantics; a client can select the intended supported provider for audio/video ingestion and that provider runs.
- [ ] #2 A client can request translate-to-English and receives an English transcript for non-English input on a supported provider; unsupported combinations produce a field-specific validation error.
- [ ] #3 Both options are accepted on /media/add and /media/ingest/jobs, including schema, get_add_media_form binding, handlers, job serialization, and Audio_Files ingestion plumbing.
- [ ] #4 An invalid provider value is rejected with a validation error naming the field, rather than ignored.
- [ ] #5 Endpoint tests for both routes assert provider/task forwarding all the way to stt_provider_adapter, not merely schema acceptance; include unchanged-default and invalid-combination cases.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Preserved from PR #2783, commit 3f64149e9974c23facc37c0b15d9eef91d3f297d, originally filed 2026-08-11. The original TASK-13020 and TASK-13021 IDs now identify unrelated Done automation work on dev; those records are unchanged. Original filing: https://github.com/rmusser01/tldw_server/pull/2783 .
Current source review: dev 2b925562e585fe1703219dd4af5eac29fafbce09, 2026-10-20. AudioVideoOptions in tldw_Server_API/app/api/v1/schemas/media_request_models.py and get_add_media_form in tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py have no explicit provider/translation option. The ingestion path is tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py; the lower adapter is stt_provider_adapter.py in that directory. Model identifiers include Parakeet, Canary, and VibeVoice, so lack of a separate provider field does not mean provider selection is impossible. No fresh live API/OpenAPI or provider run was performed. The old live instance, deployment, OpenAPI snapshot and test date were not pinned in the original filing; retain that report as historical, unverified deployment evidence, not current acceptance proof.

ADR check: this preservation requires no new ADR (ADR-002/059); implementation must assess provider integration and public API semantics against ADR-011 and ADR-022.

Historical original description (including then-current client observations; superseded assertions are qualified above):

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
job that ignored the request. That is tracked separately as **TASK-13021** —
it is a property of the media form endpoints as a family, not of this gap.
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
