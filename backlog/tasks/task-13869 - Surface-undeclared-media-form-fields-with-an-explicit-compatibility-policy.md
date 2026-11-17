---
id: TASK-13869
title: Surface undeclared media form fields with an explicit compatibility policy
status: To Do
labels:
- api
- media
- dx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An undeclared form field can be ignored while a media ingest request succeeds, hiding client/server option drift. Declared fields are validated normally: invalid declared values can produce 422. Choose and document reject-versus-report behavior across the media form endpoint family, including /media/add and /media/ingest/jobs and their shared get_add_media_form dependency. Some dedicated forms now inspect request.form() to preserve empty prompts; that does not establish general unknown-field reporting. Logging alone does not make ignored options visible to clients. If rejection is chosen, describe the impact on newer clients talking to older servers and on any client sending fields unknown to the receiving server. Related transcription contract work: TASK-13868.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Record the reject-versus-report-versus-warn decision and compatibility reasoning; the selected behavior must expose undeclared fields to clients rather than only server logs.
- [ ] #2 Sending an undeclared form field no longer produces a response indistinguishable from a request whose fields were all honored.
- [ ] #3 The behavior is consistent across the media form endpoint family, including /media/add and /media/ingest/jobs.
- [ ] #4 Endpoint tests assert undeclared fields are rejected or reported, while declared valid fields still work and declared invalid values retain validation errors.
- [ ] #5 If rejection is chosen, state the impact on clients sending fields unknown to the receiving server, including newer-client/older-server version skew, and give a migration path.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Preserved from PR #2783, commit 3f64149e9974c23facc37c0b15d9eef91d3f297d, originally filed 2026-08-11. The original TASK-13020 and TASK-13021 IDs now identify unrelated Done automation work on dev; those records are unchanged. Original filing: https://github.com/rmusser01/tldw_server/pull/2783 .
Current source review: dev 2b925562e585fe1703219dd4af5eac29fafbce09, 2026-10-20. AudioVideoOptions in tldw_Server_API/app/api/v1/schemas/media_request_models.py and get_add_media_form in tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py have no explicit provider/translation option. The ingestion path is tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py; the lower adapter is stt_provider_adapter.py in that directory. Model identifiers include Parakeet, Canary, and VibeVoice, so lack of a separate provider field does not mean provider selection is impossible. No fresh live API/OpenAPI or provider run was performed. The old live instance, deployment, OpenAPI snapshot and test date were not pinned in the original filing; retain that report as historical, unverified deployment evidence, not current acceptance proof.

ADR check: no new ADR for preservation (ADR-002/059); the future owner must assess the public API compatibility decision before implementation.

Historical original description (the broad opening, request.form claim, and compatibility direction are superseded by the current description above):

A client can send any form field it likes to the media endpoints and get a
`200` back with the field ignored. There is no error, no warning, and nothing
in the response distinguishes "your option was applied" from "your option was
dropped on the floor".

This is not a hypothetical. It hid a real defect in `tldw_chatbook` for months:
nineteen ingest options — OCR language, speaker diarization, timestamps, VAD
filter, PDF engine and more — were being posted under names this API does not
declare. Every submission returned `200`, every job ran, and none of those
settings ever took effect. Nobody could tell, because a successful-looking job
is exactly what a working one looks like.

Demonstrated on a live instance:

    POST /api/v1/media/ingest/jobs -F pdf_parsing_engine=bogus
      -> 422 {"loc":["body","pdf_parsing_engine"],
              "msg":"Input should be 'pymupdf4llm', 'pymupdf' or 'docling'"}

    POST /api/v1/media/ingest/jobs -F pdf_engine=bogus     # undeclared name
      -> 200 {"batch_id":"...","jobs":[{"id":285,"status":"queued"}],"errors":[]}

Same value, same intent, one character of difference in the key. One is
validated; the other is accepted and ignored.

The cause is ordinary FastAPI behaviour rather than a bug in any one handler:
`get_add_media_form` (and its siblings) bind each field with an explicit
`Form(...)` and never read `request.form()`, so anything undeclared is dropped
during parsing. That is a sensible default for a browser form and a poor one for
a versioned API with independent clients, where a field name drifting out of
sync is the normal failure and it currently fails invisibly.

Worth deciding deliberately, because each option trades strictness against
compatibility:

1. **Reject** unknown fields with a 422 naming them. Strictest and catches drift
   immediately, but any client sending a field a newer/older server does not
   know breaks outright — including older clients against a newer server.
2. **Report** them: accept the request, echo the ignored field names in the
   response. Nothing breaks, and a client can surface "these settings were not
   applied". Weaker than rejection, since a client has to look.
3. **Warn** server-side only (log). Cheapest, invisible to clients, and would
   not have surfaced this incident to the people affected by it.

Option 2 is the one that would have caught the `tldw_chatbook` case without a
compatibility cliff, but the choice belongs to whoever owns the API's
compatibility policy. Whatever is chosen should apply to the media form
endpoints as a family, not to one handler.

Related: TASK-13020, the transcription provider/translate gap this same
behaviour concealed.
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
