---
id: TASK-13021
title: Unknown form fields are silently discarded instead of rejected
status: To Do
assignee: []
created_date: '2026-08-11'
labels:
  - api
  - media
  - dx
dependencies: []
---

## Description

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

## Acceptance Criteria

- [ ] A decision is recorded on reject-vs-report-vs-warn, with the compatibility reasoning
- [ ] Sending an undeclared form field to the media endpoints no longer produces a response indistinguishable from one where every field was honoured
- [ ] The chosen behaviour is consistent across the media form endpoints, not added to one handler
- [ ] A test asserts an undeclared field is surfaced (rejected or reported), so the silent path cannot return
- [ ] If rejection is chosen, the compatibility impact on existing clients is stated and a migration path given
