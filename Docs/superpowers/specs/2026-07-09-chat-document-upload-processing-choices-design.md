# Chat Document Upload Processing Choices Design

**Date:** 2026-07-09
**Backlog:** TASK-12091
**Surface:** WebUI `/chat` and browser-extension `/chat`
**Status:** Approved design

## Goal

Improve document upload in chat so users can choose, at upload time, whether
files are added only to the current chat, OCR'd through page images, or ingested
as durable library sources.

The design must make persistence honest:

- **Add to chat** means chat-scoped context, not a library item.
- **OCR pages** means render documents to page images, run OCR, and attach the
  extracted context to this chat.
- **Ingest to library** means durable media/library ingestion, then attach the
  resulting source to the chat.

## Current Context

The existing chat surfaces already have most of the machinery this should reuse:

- `useComposerAttachments` is the shared attachment decision primitive.
- WebUI Playground accepts images and documents.
- Sidepanel chat currently accepts images through the shared attachment path and
  has separate context-file and Quick Ingest wiring.
- `uploadedFiles`, `contextFiles`, `useOCR`, and `fileRetrievalEnabled` already
  exist in chat state.
- The composer queue already models delayed dispatch and should carry the
  processing-aware turn instead of adding a parallel queue.
- Backend PDF processing already accepts OCR options. OCR discovery exists at
  `/api/v1/ocr/backends`.

## UX Flow

When a user drops or selects documents in WebUI `/chat` or extension `/chat`,
the composer shows a compact attachment decision surface with three actions:

1. **Add to chat**
2. **OCR pages**
3. **Ingest to library**

For multiple files, the first choice applies to the whole batch. An **Adjust per
file** control expands file rows so mixed batches can route specific files
differently. Per-file rows are also revealed when capabilities force different
available choices.

Processing starts when the user presses **Send**, not when files are selected.
This lets the user attach files, choose modes, type the prompt, and submit one
intentional request.

After Send, chat creates a pending turn with visible states:

- `waiting for files`
- `processing`
- `ready`
- `failed`
- `sending prompt`

The prompt runs only after required context is ready.

## Modes

### Add to Chat

Use native document parsing where available. The result is attached as
chat-scoped context and is not saved as a library source.

If parsing produces too much content, show truncation explicitly, for example:

`added, 18k tokens, truncated`

Offer a recovery action such as **Use retrieval instead** when direct context is
too large.

### OCR Pages

Render each supported document to page images and run OCR. OCR is available only
when the server reports that the file type can be rendered and OCR is
configured.

The UI must not present OCR as available for a file if the backend cannot render
that file type or no OCR backend is usable. Disabled copy should name the
missing capability, for example:

`OCR unavailable: server cannot render DOCX pages`

Page and file limits must be visible before Send. Oversized files remain
attached but block Send until the user changes mode or removes the file.

### Ingest to Library

Create a durable media/library item. If the user chooses **Ingest to library**,
the system must not silently downgrade to chat-scoped processing. The resulting
library source is attached to the chat after ingestion succeeds.

Retry must be idempotent using a file fingerprint, job id, or existing ingest
dedupe mechanism so repeated retries do not create duplicate media items.

## Minimal State

Do not introduce a large new domain model unless implementation proves it is
needed. Extend the existing composer/file/queue state with the smallest useful
fields:

- `mode`: `add_to_chat`, `ocr_pages`, or `ingest_to_library`
- `capability`: available, unavailable reason, limits
- `status`: pending, processing, ready, failed, cancelled
- `resultRef`: chat context id, OCR context id, media id, job id, or source id
- `error`: user-safe message plus optional diagnostic id

Use existing `contextFiles`, `uploadedFiles`, queue snapshots, and attachment
hooks wherever possible.

## Data Flow

1. User adds files.
2. Client runs lightweight preflight: type, size, supported modes, likely
   token/page impact, and sidepanel handoff requirements.
3. Composer shows batch mode and optional per-file overrides.
4. User presses Send.
5. A processing-aware queued turn is created.
6. Files process by selected mode.
7. Once required context is ready, the prompt runs automatically.
8. The final user message retains file chips that show whether each file was
   chat-scoped, OCR-derived, or ingested to the library.

Sidepanel behavior:

- **Add to chat** can stay inline when size and capability allow.
- **OCR pages** and **Ingest to library** stage files first, then hand off to
  full WebUI `/chat` with a durable draft id if the sidepanel cannot show enough
  status or recovery detail.
- If handoff fails, keep the draft in sidepanel and show **Open full chat
  again** and **Use Add to chat instead**.

## Error Handling

Unsupported mode:

- Disable the mode per file.
- Explain the missing capability in plain language.

Oversized file:

- Keep the file attached.
- Require mode change or removal before Send.

Processing failure:

- Hold the queued turn.
- Offer **Retry**, **Change mode**, **Remove failed**, or, when no ingest intent
  is being violated, **Send with available context**.

Partial batch failure:

- Do not silently drop failed files.
- Show a collapsed summary such as `2 ready, 1 failed`.
- Expand per-file details only for failures or manual adjustment.

Ingest failure:

- No successful library item is assumed.
- Recovery choices are **Retry ingest**, **Remove file**, or **Change to Add to
  chat/OCR**.
- Do not send automatically with available context unless the user explicitly
  changes the failed ingest file to a non-ingest mode or removes it.

Cancel:

- Stop the queued turn and cancel server jobs where possible.
- Keep draft state recoverable unless the user explicitly removes it.
- Staged uploads should be cleaned up or marked recoverable, never ambiguous.

Diagnostics:

- No raw tracebacks in the composer.
- A details action may expose a copyable diagnostic or correlation id.

## Testing

Unit coverage should include:

- attachment mode selection
- batch default and per-file override
- capability gating
- size/page/token limit warnings
- retry/cancel state transitions
- no silent downgrade from ingest

Integration coverage should include:

- WebUI `/chat` creates a pending turn on Send, processes files, then sends the
  prompt after context is ready.
- Sidepanel Add-to-chat works inline.
- Sidepanel OCR/ingest stages files and hands off with a draft id without losing
  attachments.
- Unsupported OCR, oversized file, partial batch failure, ingest retry dedupe,
  and cancel cleanup paths.

UX checks should verify:

- no modal-first routine flow
- no raw traceback
- collapsed batch summary for mixed results
- visible persistence distinction between Add to chat and Ingest to library
- clear status for chat-scoped, OCR-derived, and library-ingested files

## Non-Goals

- Redesigning Quick Ingest.
- Replacing the chat composer architecture.
- Introducing a broad new document-processing framework before existing hooks,
  queue state, and backend endpoints are exhausted.
- Making OCR appear available when backend rendering/OCR capability is unknown.

