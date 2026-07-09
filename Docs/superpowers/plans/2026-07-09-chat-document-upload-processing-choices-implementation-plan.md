# Chat Document Upload Processing Choices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let WebUI `/chat` and browser-extension `/chat` users choose per uploaded document whether to add it only to chat, OCR pages, or ingest it to the durable library.

**Architecture:** Keep the existing composer, upload, and queue primitives. Add a small backend capability/draft seam, then extend frontend attachment state with explicit per-file processing decisions. Processing runs on Send and returns either chat-scoped extracted text or turn-scoped media IDs; Ingest never silently falls back to chat-only handling.

**Tech Stack:** FastAPI, Pydantic, existing media/OCR endpoints, Next.js/React, Zustand store, Vitest, pytest, Playwright smoke coverage.

---

## References

- Spec: `Docs/superpowers/specs/2026-07-09-chat-document-upload-processing-choices-design.md` (`TASK-12091`)
- Implementation Backlog: `TASK-12092`
- Related design context: `Docs/Design/2026-05-11-chat-input-first-ux-design.md`
- Existing frontend seams:
  - `apps/packages/ui/src/hooks/chat/useFileUpload.ts`
  - `apps/packages/ui/src/components/Chat/composer/hooks/useComposerAttachments.ts`
  - `apps/packages/ui/src/components/Option/Playground/AttachmentsSummary.tsx`
  - `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts`
  - `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundQueueManagement.ts`
  - `apps/packages/ui/src/hooks/chat/useChatActions.ts`
  - `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
  - `apps/packages/ui/src/services/tldw/sidepanel-chat-webui-handoff.ts`
- Existing backend seams:
  - `tldw_Server_API/app/api/v1/endpoints/ocr.py`
  - `tldw_Server_API/app/api/v1/endpoints/media/__init__.py`
  - `tldw_Server_API/app/api/v1/endpoints/media/process_documents.py`
  - `tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py`
  - `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py`

## File Structure

Create:

- `tldw_Server_API/app/api/v1/schemas/document_upload_processing.py`
  - Pydantic request/response models for capability preflight and staged chat document drafts.
- `tldw_Server_API/app/api/v1/endpoints/media/document_upload_processing.py`
  - Metadata-only preflight endpoint and short-lived draft endpoints for browser-extension heavy-mode handoff.
- `tldw_Server_API/tests/Media/test_document_upload_processing.py`
  - Unit/integration-style endpoint coverage for capability gating, draft ownership, expiry, and cleanup semantics.
- `apps/packages/ui/src/services/chat-document-processing.ts`
  - Frontend decision types, preflight normalization, no-DB document/PDF processing wrappers, ingest-job wrapper, and send-time preparation helpers.
- `apps/packages/ui/src/services/__tests__/chat-document-processing.test.ts`
  - Pure service tests for mode selection, route selection, no silent ingest downgrade, and request override generation.
- `apps/packages/ui/src/components/Option/Playground/DocumentProcessingChoices.tsx`
  - Compact batch decision surface and per-file override rows.
- `apps/packages/ui/src/components/Option/Playground/__tests__/DocumentProcessingChoices.test.tsx`
  - UX component coverage for batch mode, per-file adjustment, disabled capability copy, and blocked send states.
- `apps/packages/ui/src/components/Common/Playground/DocumentProcessingTurn.tsx`
  - Message-timeline status view for document processing after the user presses Send.
- `apps/packages/ui/src/components/Common/Playground/__tests__/DocumentProcessingTurn.test.tsx`
  - Coverage for waiting, processing, blocked, failed, ready, and sending-prompt states.

Modify:

- `tldw_Server_API/app/api/v1/endpoints/media/__init__.py`
  - Register the new media subrouter.
- `apps/packages/ui/src/db/dexie/types.ts`
  - Add optional processing-decision metadata to `UploadedFile`; keep existing fields backward-compatible.
- `apps/packages/ui/src/hooks/chat/useFileUpload.ts`
  - Stage uploaded document files with default `add_to_chat` decisions and kick off metadata preflight; do not perform durable ingestion here.
- `apps/packages/ui/src/components/Option/Playground/AttachmentsSummary.tsx`
  - Show the selected processing mode/status on uploaded file chips, and preserve the existing large-file warning.
- `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
  - Render the decision surface near the attachment summary and pass processing callbacks into submit/queue hooks.
- `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts`
  - Resolve selected document modes on Send, block invalid batches, and pass processed context/media IDs through `requestOverrides`.
- `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundQueueManagement.ts`
  - Store document-processing request overrides in queued source context so queued sends replay the same file intent.
- `apps/packages/ui/src/hooks/chat/useChatActions.ts`
  - Accept explicit `contextFiles`, `uploadedFiles`, `ragMediaIds`, and `fileRetrievalEnabled` turn overrides to avoid React re-render races.
- `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
  - Upgrade a reserved pending user message by `userMessageId` instead of appending a duplicate after document processing finishes.
- `apps/packages/ui/src/components/Common/Playground/Message.tsx`
  - Render document-processing turn metadata in the timeline.
- `apps/packages/ui/src/components/Common/Playground/PlaygroundUserMessage.tsx`
  - Preserve existing user-message layout while exposing document-processing status metadata.
- `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
  - Reuse the same decision model for sidepanel context-file uploads; keep Add-to-chat inline and hand off OCR/Ingest through server-backed drafts.
- `apps/packages/ui/src/services/tldw/sidepanel-chat-webui-handoff.ts`
  - Add `chatDocumentDraftId`, `ragMediaIds`, and `fileRetrievalEnabled` fields to the fragment payload.
- `apps/packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts`
  - Cover new handoff fields and expiry behavior.
- Locale files only for the new visible strings in English first:
  - `apps/packages/ui/src/assets/locale/en/playground.json`
  - `apps/packages/ui/src/public/_locales/en/playground.json`

## Review Constraints To Preserve

- `Add to chat` is chat-scoped only. If retrieval is offered for large chat-scoped content, it must be chat-scoped retrieval unless the user explicitly switches to `Ingest to library`.
- Sidepanel heavy-mode handoff must have an owner, expiry, retry/read behavior, and cleanup path. URL fragments cannot carry file bytes or large OCR text.
- A blocked processing decision must be explicit. Use `blockedReason` or per-file `processingStatus: "blocked"`; never hide it behind a generic failed state.
- OCR availability must come from backend preflight. Frontend file-extension checks may be used only for optimistic placeholders while waiting for the authoritative response.
- Pressing Send must immediately create a visible user turn with `waiting_for_files`/`processing` state, then update that same message through ready, failed, blocked, and sending-prompt states. Do not silently wait before adding the turn.
- Ingest and mixed ingest batches must pass explicit `contextFiles: []` unless intentionally sending chat-scoped extracted content; never let stale React `contextFiles` route an ingest selection through the document-chat branch.
- Retry and cancel must be idempotent. Ingest retries reuse the same file fingerprint/idempotency key/job where possible, and cancellation aborts local processing plus cancels any active ingest job or batch.
- Direct chat-scoped context must enforce page/token limits, surface truncation or overflow before final send, and offer chat-scoped retrieval separately from durable library ingest.
- Keep the first implementation small: PDF OCR first, native no-DB processing for supported document/PDF types, durable ingest through existing ingest jobs. Do not create a broad document framework.

---

### Task 1: Backend Capability Preflight And Drafts

**Files:**
- Create: `tldw_Server_API/app/api/v1/schemas/document_upload_processing.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/media/document_upload_processing.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/__init__.py`
- Test: `tldw_Server_API/tests/Media/test_document_upload_processing.py`

- [x] **Step 1: Write failing tests for preflight capabilities**

Add tests that call `POST /api/v1/media/document-upload/preflight` with file metadata only.

```python
def test_document_upload_preflight_pdf_ocr_available(client, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.media.document_upload_processing._list_ocr_backends",
        lambda: {"tesseract": {"available": True}},
    )

    response = client.post(
        "/api/v1/media/document-upload/preflight",
        json={
            "files": [
                {
                    "client_id": "file-1",
                    "filename": "scan.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": 1024,
                }
            ]
        },
    )

    assert response.status_code == 200
    item = response.json()["files"][0]
    assert item["client_id"] == "file-1"
    assert item["modes"]["add_to_chat"]["available"] is True
    assert item["modes"]["ocr_pages"]["available"] is True
    assert item["modes"]["ingest_to_library"]["available"] is True
    assert item["default_mode"] == "add_to_chat"
```

Also add tests for:

- DOCX has `add_to_chat` and `ingest_to_library`, but `ocr_pages` is unavailable with a plain reason until a backend renderer exists.
- No OCR backend configured makes PDF `ocr_pages` unavailable with reason `OCR unavailable: no OCR backend configured`.
- Unsupported extensions return all modes unavailable and `default_mode: null`.
- Oversized files remain in the response but are `blocked` with a size-limit reason.
- Client-supplied `page_count` or `estimated_tokens` over server limits marks direct chat/OCR modes blocked with the specific limit in the response.

- [x] **Step 2: Run the backend test and verify it fails**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Media/test_document_upload_processing.py -q
```

Expected: FAIL because the schema and endpoint do not exist.

- [x] **Step 3: Add Pydantic schemas**

Create `document_upload_processing.py` with this shape:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DocumentProcessingMode = Literal["add_to_chat", "ocr_pages", "ingest_to_library"]
DocumentProcessingStatus = Literal["available", "unavailable", "blocked"]


class DocumentUploadPreflightFile(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=128)
    filename: str = Field(..., min_length=1, max_length=512)
    mime_type: str | None = Field(default=None, max_length=255)
    size_bytes: int = Field(..., ge=0)
    page_count: int | None = Field(default=None, ge=0)
    estimated_tokens: int | None = Field(default=None, ge=0)


class DocumentUploadPreflightRequest(BaseModel):
    files: list[DocumentUploadPreflightFile] = Field(default_factory=list, max_length=50)


class DocumentModeCapability(BaseModel):
    available: bool
    status: DocumentProcessingStatus
    reason: str | None = None


class DocumentUploadPreflightItem(BaseModel):
    client_id: str
    filename: str
    media_type: Literal["pdf", "document", "ebook", "unsupported"]
    default_mode: DocumentProcessingMode | None
    modes: dict[DocumentProcessingMode, DocumentModeCapability]
    max_size_bytes: int
    max_pages: int | None
    max_chat_tokens: int
    estimated_pages: int | None = None
    estimated_tokens: int | None = None
    requires_send_time_estimate: bool = False


class DocumentUploadPreflightResponse(BaseModel):
    files: list[DocumentUploadPreflightItem]


class ChatDocumentDraftCreateResponse(BaseModel):
    draft_id: str
    expires_at: str


class ChatDocumentDraftReadResponse(BaseModel):
    draft_id: str
    created_at: str
    expires_at: str
    payload: dict
```

- [x] **Step 4: Add the backend endpoint**

Create `document_upload_processing.py` under media endpoints. Use extension allowlists already present in processing endpoints:

```python
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".docx", ".rtf", ".html", ".htm",
    ".xhtml", ".xml", ".json",
}
SUPPORTED_EBOOK_EXTENSIONS = {".epub"}
SUPPORTED_PDF_EXTENSIONS = {".pdf"}
DEFAULT_MAX_CHAT_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_CHAT_UPLOAD_PAGES = 200
DEFAULT_MAX_DIRECT_CHAT_TOKENS = 24_000
```

Rules:

- `add_to_chat`: available for PDF, document, and ebook types unless oversized.
- `ocr_pages`: available only for PDF when at least one OCR backend is available. Return unavailable for DOCX/HTML/etc with `OCR unavailable: server cannot render <EXT> pages`.
- `ingest_to_library`: available for PDF, document, and ebook types unless oversized.
- `default_mode`: `add_to_chat` when available, else `null`.
- Oversized files return mode capabilities with `status: "blocked"` and a size-limit reason.
- Client-provided `page_count` over `DEFAULT_MAX_CHAT_UPLOAD_PAGES` blocks `ocr_pages` and direct `add_to_chat` with a page-limit reason. Client-provided `estimated_tokens` over `DEFAULT_MAX_DIRECT_CHAT_TOKENS` blocks direct `add_to_chat` with a token-limit reason and leaves `ingest_to_library` available when size/type are allowed.
- When page/token estimates are unknown, return `requires_send_time_estimate: true` so the frontend can do the final extraction-time limit check before dispatching the prompt.

Add short-lived in-memory draft storage only for sidepanel handoff:

- `POST /api/v1/media/document-upload/drafts`
- `GET /api/v1/media/document-upload/drafts/{draft_id}`
- `DELETE /api/v1/media/document-upload/drafts/{draft_id}`

Keep draft payloads bounded to uploaded file metadata, base64 content, selected modes, and user draft text. Include `created_at`, `expires_at`, and owner key derived from the authenticated user dependency. If a draft is expired or owned by another user, reads return 404. Cleanup expired drafts on create/read.

- [x] **Step 5: Register the router**

Add `"document_upload_processing"` to `_MEDIA_ENDPOINT_MODULES` in `tldw_Server_API/app/api/v1/endpoints/media/__init__.py` near the existing process modules.

- [x] **Step 6: Run backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Media/test_document_upload_processing.py -q
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/document_upload_processing.py tldw_Server_API/app/api/v1/endpoints/media/document_upload_processing.py tldw_Server_API/app/api/v1/endpoints/media/__init__.py tldw_Server_API/tests/Media/test_document_upload_processing.py
git commit -m "feat: add chat document upload preflight"
```

---

### Task 2: Frontend Decision Types And Processing Service

**Files:**
- Create: `apps/packages/ui/src/services/chat-document-processing.ts`
- Create: `apps/packages/ui/src/services/__tests__/chat-document-processing.test.ts`
- Modify: `apps/packages/ui/src/db/dexie/types.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/media.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`

- [x] **Step 1: Write failing service tests**

Cover pure behavior before wiring UI:

```ts
it("keeps add-to-chat chat scoped and does not create media ids", async () => {
  const file = makeUploadedFile({
    id: "file-1",
    filename: "notes.md",
    processingMode: "add_to_chat"
  })

  mocks.processDocument.mockResolvedValueOnce({
    content: "parsed notes",
    sourceName: "notes.md"
  })

  const result = await prepareChatDocumentAttachmentsForSend({
    files: [file],
    processDocument: mocks.processDocument,
    processPdf: mocks.processPdf,
    ingestDocument: mocks.ingestDocument
  })

  expect(result.contextFiles[0]).toMatchObject({
    id: "file-1",
    content: "parsed notes",
    processed: true,
    processingMode: "add_to_chat",
    processingStatus: "ready"
  })
  expect(result.requestOverrides?.ragMediaIds).toBeUndefined()
  expect(mocks.ingestDocument).not.toHaveBeenCalled()
})
```

Also test:

- `ocr_pages` calls the PDF/OCR processor with `enable_ocr: true`.
- `ingest_to_library` returns `requestOverrides: { ragMediaIds: [id], fileRetrievalEnabled: true }`.
- Ingest failure does not fall back to add-to-chat.
- Mixed Add/OCR + Ingest batches return `contextFiles: []`, `ragMediaIds`, and `messageForModel` containing the chat-scoped extracted text so `useChatActions` cannot accidentally route through the document-chat branch.
- Preflight response normalization maps backend unavailable reasons exactly into file capabilities.
- File fingerprints and ingest idempotency keys are stable across retry for the same file/session, and a non-terminal existing job is reused instead of creating a duplicate ingest job.
- Direct Add-to-chat overflow returns `processingStatus: "blocked"` with recovery actions including chat-scoped retrieval and explicit Ingest-to-library.
- Cancel aborts local no-DB processing, deletes any server draft, and calls the existing ingest job/batch cancellation endpoint when a job was started.

- [x] **Step 2: Run the frontend service test and verify it fails**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/chat-document-processing.test.ts
```

Expected: FAIL because the service does not exist.

- [x] **Step 3: Add optional metadata to `UploadedFile`**

Modify `apps/packages/ui/src/db/dexie/types.ts`:

```ts
export type DocumentProcessingMode =
  | "add_to_chat"
  | "ocr_pages"
  | "ingest_to_library"

export type DocumentProcessingStatus =
  | "pending"
  | "preflighting"
  | "waiting_for_files"
  | "blocked"
  | "processing"
  | "ready"
  | "failed"
  | "sending_prompt"
  | "cancelled"

export type DocumentProcessingRecoveryAction =
  | "use_chat_scoped_retrieval"
  | "switch_to_ingest"
  | "retry"
  | "remove"
  | "send_available"

export type DocumentModeCapability = {
  available: boolean
  status: "available" | "unavailable" | "blocked"
  reason?: string | null
}

export type DocumentProcessingTurnMetadata = {
  status:
    | "waiting_for_files"
    | "processing"
    | "ready"
    | "blocked"
    | "failed"
    | "sending_prompt"
    | "cancelled"
  files: Array<{
    id: string
    filename: string
    mode: DocumentProcessingMode
    status: DocumentProcessingStatus
    summary?: string | null
    error?: string | null
  }>
  recoveryActions?: DocumentProcessingRecoveryAction[]
}
```

Add optional fields to `UploadedFile`:

```ts
processingMode?: DocumentProcessingMode
processingStatus?: DocumentProcessingStatus
processingCapabilities?: Partial<Record<DocumentProcessingMode, DocumentModeCapability>>
processingResultRef?: {
  kind: "chat_context" | "ocr_context" | "chat_retrieval" | "media" | "job" | "draft"
  id: string | number
} | null
processingError?: string | null
processingSummary?: string | null
processingRecoveryActions?: DocumentProcessingRecoveryAction[]
fileFingerprint?: string
ingestJobId?: string | number | null
ingestBatchId?: string | null
ingestIdempotencyKey?: string | null
pageEstimate?: number | null
tokenEstimate?: number | null
truncatedTokenCount?: number | null
chatScopedRetrieval?: boolean
```

Do not store a `File` object in `UploadedFile`; it is not serializable. Use existing base64 `content` when a later upload body needs to be reconstructed.

- [x] **Step 4: Add media client methods and openapi guard paths**

In `apps/packages/ui/src/services/tldw/domains/media.ts`, add:

```ts
async preflightDocumentUpload(files: Array<{
  client_id: string
  filename: string
  mime_type?: string | null
  size_bytes: number
  page_count?: number | null
  estimated_tokens?: number | null
}>): Promise<DocumentUploadPreflightResponse> {
  return await bgRequest<DocumentUploadPreflightResponse>({
    path: "/api/v1/media/document-upload/preflight",
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: { files }
  })
}
```

Add draft create/read/delete methods that call `/api/v1/media/document-upload/drafts`.

Update `apps/packages/ui/src/services/tldw/openapi-guard.ts` with the new literal paths so background requests are allowed.

- [x] **Step 5: Implement `chat-document-processing.ts`**

Keep this file pure where possible:

```ts
export const DEFAULT_DOCUMENT_PROCESSING_MODE: DocumentProcessingMode =
  "add_to_chat"

export const withDefaultDocumentDecision = (
  file: UploadedFile
): UploadedFile => ({
  ...file,
  processingMode: file.processingMode ?? DEFAULT_DOCUMENT_PROCESSING_MODE,
  processingStatus: file.processingStatus ?? "preflighting"
})
```

Add helpers:

- `normalizeDocumentPreflightResponse(response, files)`
- `setBatchDocumentProcessingMode(files, mode)`
- `setFileDocumentProcessingMode(files, fileId, mode)`
- `hasBlockedDocumentProcessing(files)`
- `computeDocumentFileFingerprint(file)` using Web Crypto SHA-256 over filename, size, MIME type, and base64 content.
- `buildIngestIdempotencyKey({ file, historyId, sessionId })` returning a stable scoped key such as `chat-document-ingest:<history-or-session>:<fingerprint>`.
- `dataUrlToUploadFile(file)` using `fetch(file.content).then(res => res.blob())`
- `processDocumentForChat(file)` routed to `/process-documents`
- `processPdfForChat(file, { enableOcr })` routed to `/process-pdfs`
- `estimateDirectChatTokens(text)` with a conservative local heuristic used only for blocking/truncation copy.
- `buildChatScopedRetrievalContext({ chunks, query, tokenBudget })` using deterministic chunking and simple lexical scoring; this is ephemeral per turn and does not create a library/media source.
- `ingestDocumentToLibrary(file)` routed to `/media/ingest/jobs` with `idempotency_key`; store returned `job_id`/`batch_id` on the file and reuse an existing non-terminal job on retry.
- `cancelPreparedDocumentProcessing(files)` that aborts local processing, deletes server drafts, and calls existing ingest cancel endpoints for active `ingestJobId`/`ingestBatchId`.
- `prepareChatDocumentAttachmentsForSend(input)`

Return shape:

```ts
type PreparedChatDocumentAttachments = {
  contextFiles: UploadedFile[]
  files: UploadedFile[]
  requestOverrides?: {
    contextFiles?: UploadedFile[]
    uploadedFiles?: UploadedFile[]
    ragMediaIds?: number[] | null
    fileRetrievalEnabled?: boolean
    messageForModel?: string
    userMetadataExtra?: Record<string, unknown>
  }
  failedFiles: UploadedFile[]
  blockedFiles: UploadedFile[]
  recoveryActions: DocumentProcessingRecoveryAction[]
  turnMetadata: DocumentProcessingTurnMetadata
}
```

Routing rules in `prepareChatDocumentAttachmentsForSend`:

- All Add/OCR: return extracted `contextFiles`, no `ragMediaIds`.
- All Ingest: return `contextFiles: []`, `ragMediaIds`, and `fileRetrievalEnabled: true`.
- Mixed Add/OCR + Ingest: append extracted chat-scoped snippets into `messageForModel`, return `contextFiles: []`, and include `ragMediaIds`/`fileRetrievalEnabled: true`.
- Direct Add/OCR content over `max_chat_tokens` blocks by default with summary copy like `added, 18k tokens, blocked by 24k token limit`; the `use_chat_scoped_retrieval` recovery switches to the ephemeral chunk scorer and keeps the source chat-scoped.
- Truncation is never silent. If a recovery path chooses truncation or chunk scoring, set `processingSummary` to copy such as `added, 18k tokens, truncated` and include that metadata in the visible turn.

- [x] **Step 6: Run service tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/chat-document-processing.test.ts
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add apps/packages/ui/src/services/chat-document-processing.ts apps/packages/ui/src/services/__tests__/chat-document-processing.test.ts apps/packages/ui/src/db/dexie/types.ts apps/packages/ui/src/services/tldw/domains/media.ts apps/packages/ui/src/services/tldw/openapi-guard.ts
git commit -m "feat: add chat document processing service"
```

---

### Task 3: WebUI Composer Decision Surface

**Files:**
- Create: `apps/packages/ui/src/components/Option/Playground/DocumentProcessingChoices.tsx`
- Create: `apps/packages/ui/src/components/Option/Playground/__tests__/DocumentProcessingChoices.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/AttachmentsSummary.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- Modify: `apps/packages/ui/src/hooks/chat/useFileUpload.ts`
- Modify: English locale files listed above.

- [x] **Step 1: Write failing component tests**

Test the visible behavior:

```tsx
it("applies a batch mode and keeps per-file overrides visible", async () => {
  const onChangeFiles = vi.fn()
  render(
    <DocumentProcessingChoices
      files={[
        makeFile("a.pdf", "add_to_chat"),
        makeFile("b.pdf", "add_to_chat")
      ]}
      onChangeFiles={onChangeFiles}
    />
  )

  await userEvent.click(screen.getByRole("button", { name: /OCR pages/i }))
  expect(onChangeFiles).toHaveBeenCalledWith([
    expect.objectContaining({ filename: "a.pdf", processingMode: "ocr_pages" }),
    expect.objectContaining({ filename: "b.pdf", processingMode: "ocr_pages" })
  ])

  await userEvent.click(screen.getByRole("button", { name: /Adjust per file/i }))
  expect(screen.getByText("a.pdf")).toBeInTheDocument()
  expect(screen.getByText("b.pdf")).toBeInTheDocument()
})
```

Also test:

- Disabled OCR shows backend reason.
- `Ingest to library` copy is visibly distinct from `Add to chat`.
- Choosing `Ingest to library` does not mirror the file into chat-scoped `contextFiles`.
- Blocked files show a blocking state and do not hide the remove action.
- Mixed batch summary collapses to counts like `2 ready, 1 blocked`.

- [x] **Step 2: Run component tests and verify failure**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/DocumentProcessingChoices.test.tsx
```

Expected: FAIL because the component does not exist.

- [x] **Step 3: Update upload staging**

In `useFileUpload.ts`, after the file-size guard and base64 conversion, initialize document metadata:

```ts
const uploadedFile: UploadedFile = withDefaultDocumentDecision({
  id: fileId,
  filename: file.name,
  type: file.type,
  content: source.content,
  size: file.size,
  uploadedAt: Date.now(),
  processed: false
})
```

Call the preflight client after setting local state. When it resolves, merge capabilities into matching file IDs only if the file is still attached. On failure, keep files attached with `processingStatus: "blocked"` and a user-safe preflight error.

- [x] **Step 4: Implement `DocumentProcessingChoices`**

Use native buttons and existing design tokens. The three mode labels are:

- `Add to chat`
- `OCR pages`
- `Ingest to library`

Use icons from `lucide-react`:

- `MessageSquareText` for Add to chat
- `ScanText` for OCR pages
- `LibraryBig` for Ingest to library

Keep routine flow inline; do not add a modal.

- [x] **Step 5: Render the decision surface in `PlaygroundForm.tsx`**

Place it next to the existing attachment summary so users see mode choices immediately after adding documents. Pass:

- `files={uploadedFiles}`
- `onChangeFiles={(next) => { setUploadedFiles(next); setContextFiles(next.filter((file) => file.processingMode !== "ingest_to_library")); }}`
- remove/clear actions remain owned by `AttachmentsSummary`

`contextFiles` here remains a compatibility mirror for chat-scoped attachments only. The authoritative send-time routing still comes from `prepareChatDocumentAttachmentsForSend`; ingest-only sends must later pass `requestOverrides.contextFiles = []`.

- [x] **Step 6: Add compact status to `AttachmentsSummary.tsx`**

For each uploaded file chip, include a short status line:

- `Chat only`
- `OCR`
- `Library ingest`
- `Blocked`
- `Failed`

Keep text small and inside the chip; do not let chips resize toolbar height unpredictably.

- [x] **Step 7: Add English strings**

Add keys under `playground.documentProcessing` in both English locale files. Keep copy short and concrete:

```json
{
  "addToChat": "Add to chat",
  "ocrPages": "OCR pages",
  "ingestToLibrary": "Ingest to library",
  "adjustPerFile": "Adjust per file",
  "chatOnlyHint": "Uses this chat only",
  "libraryHint": "Creates a library source"
}
```

- [x] **Step 8: Run component and existing composer tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/DocumentProcessingChoices.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/AttachmentsSummary.integration.test.tsx ../packages/ui/src/components/Option/Playground/hooks/__tests__/usePlaygroundAttachments.test.ts
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground/DocumentProcessingChoices.tsx apps/packages/ui/src/components/Option/Playground/__tests__/DocumentProcessingChoices.test.tsx apps/packages/ui/src/components/Option/Playground/AttachmentsSummary.tsx apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx apps/packages/ui/src/hooks/chat/useFileUpload.ts apps/packages/ui/src/assets/locale/en/playground.json apps/packages/ui/src/public/_locales/en/playground.json
git commit -m "feat: add chat document processing choices"
```

---

### Task 4: Send-Time Processing And Queue Replay

**Files:**
- Create: `apps/packages/ui/src/components/Common/Playground/DocumentProcessingTurn.tsx`
- Create: `apps/packages/ui/src/components/Common/Playground/__tests__/DocumentProcessingTurn.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts`
- Modify: `apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundQueueManagement.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- Modify: `apps/packages/ui/src/hooks/chat/useCompareSubmit.ts`
- Modify: `apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts`
- Modify: `apps/packages/ui/src/components/Common/Playground/Message.tsx`
- Modify: `apps/packages/ui/src/components/Common/Playground/PlaygroundUserMessage.tsx`
- Test: `apps/packages/ui/src/components/Common/Playground/__tests__/DocumentProcessingTurn.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.document-processing.test.tsx`
- Test: `apps/packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts`

- [ ] **Step 1: Write failing submit tests**

Add a focused PlaygroundForm test or hook test that verifies:

- Pressing Send immediately appends a visible pending user turn with document-processing metadata before `prepareChatDocumentAttachmentsForSend` resolves.
- The pending turn transitions through `waiting_for_files`/`processing` and is updated to `sending_prompt` when dispatch begins.
- The final chat pipeline upgrades the same `userMessageId` instead of appending a duplicate user message.
- The send payload includes explicit `requestOverrides.contextFiles`.
- Send with `ingest_to_library` includes `requestOverrides.ragMediaIds` and `fileRetrievalEnabled: true`.
- Send with `ingest_to_library` includes explicit `requestOverrides.contextFiles: []`.
- Mixed Add/OCR + Ingest sends `messageForModel`, `ragMediaIds`, and `contextFiles: []`.
- Ingest failure keeps the draft and does not call `sendMessage`.
- Direct Add-to-chat token overflow blocks the pending turn, shows recovery actions, and does not call `sendMessage` until the user chooses a recovery.
- Queueing a send while another turn is busy/offline stores replayable uploaded file data and later sends with the same processing modes and request overrides.

Example assertion:

```ts
expect(sendMessage).toHaveBeenCalledWith(
  expect.objectContaining({
    message: "summarize",
    requestOverrides: expect.objectContaining({
      contextFiles: [
        expect.objectContaining({
          filename: "notes.md",
          content: "parsed notes",
          processingMode: "add_to_chat"
        })
      ],
      uploadedFiles: [
        expect.objectContaining({ filename: "notes.md" })
      ]
    })
  })
)
```

- [ ] **Step 2: Run submit tests and verify failure**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.document-processing.test.tsx
```

Expected: FAIL because send-time processing is not wired.

- [ ] **Step 3: Add explicit turn attachment overrides in `useChatActions.ts`**

Extend local `ChatModeOverrides`:

```ts
contextFiles?: UploadedFile[]
uploadedFiles?: UploadedFile[]
```

Resolve turn-scoped files near existing RAG override resolution:

```ts
const turnContextFiles = Array.isArray(requestOverrides?.contextFiles)
  ? requestOverrides.contextFiles
  : contextFiles
const turnUploadedFiles = Array.isArray(requestOverrides?.uploadedFiles)
  ? requestOverrides.uploadedFiles
  : uploadedFiles
```

Use `turnContextFiles` for the `documentChatMode` branch and `turnUploadedFiles` for message metadata. Do the same in compare submit where it reads `contextFiles`.

If `requestOverrides.ragMediaIds` is non-empty and `requestOverrides.contextFiles` is an explicit empty array, do not fall back to closure `contextFiles`. This preserves the user's Ingest decision even if the composer still has staged files.

- [ ] **Step 4: Add the visible processing turn**

Create `DocumentProcessingTurn.tsx` for user-message metadata shaped like:

```ts
type DocumentProcessingTurnMetadata = {
  status:
    | "waiting_for_files"
    | "processing"
    | "ready"
    | "blocked"
    | "failed"
    | "sending_prompt"
    | "cancelled"
  files: Array<{
    id: string
    filename: string
    mode: DocumentProcessingMode
    status: DocumentProcessingStatus
    summary?: string | null
    error?: string | null
  }>
  recoveryActions?: DocumentProcessingRecoveryAction[]
}
```

Render it from `Message.tsx`/`PlaygroundUserMessage.tsx` when `message.metadata.documentProcessing` exists. Keep it compact: status text, file count, failed/blocked rows, and recovery buttons passed in from the playground layer. Do not add a modal.

Canonical metadata field: `message.metadata.documentProcessing`. `requestOverrides.userMetadataExtra.documentProcessing` is only the transport path into `chatModePipeline.ts`; the pipeline must merge it back onto `message.metadata.documentProcessing` for the reserved user message.

- [ ] **Step 5: Wire send-time processing in `usePlaygroundSubmit.ts`**

On Send:

1. Reserve a `userMessageId`.
2. Append a visible pending user message immediately with the user's typed prompt and `metadata.documentProcessing.status = "waiting_for_files"`.
3. Update the same message to `processing` while `prepareChatDocumentAttachmentsForSend` runs.
4. On blocked/failed processing, update the same message to `blocked` or `failed`, leave the form draft/attachments recoverable, and return without dispatching the prompt.
5. On success, update the same message to `sending_prompt`, call dispatch with the reserved `userMessageId`, and only then reset/clear successful attachments.

Before `form.reset()` and `clearUploadedFiles()`, call:

```ts
const preparedDocuments = await prepareChatDocumentAttachmentsForSend({
  files: uploadedFiles,
  prompt: values.message,
  historyId,
  sessionId,
  t,
  notificationApi
})
```

If `preparedDocuments.blockedFiles.length > 0` or required ingest files failed:

- keep the form and attachments intact
- set a field error or notification with a plain message
- return without calling `dispatch`

Merge overrides:

```ts
const mergedRequestOverrides = {
  ...(requestOverrides ?? {}),
  ...(openUIRequestOverrides ?? {}),
  ...(preparedDocuments.requestOverrides ?? {}),
  userMetadataExtra: {
    ...(requestOverrides?.userMetadataExtra ?? {}),
    ...(openUIRequestOverrides?.userMetadataExtra ?? {}),
    ...(preparedDocuments.requestOverrides?.userMetadataExtra ?? {}),
    documentProcessing: preparedDocuments.turnMetadata
  }
}
```

Only reset/clear after preparation succeeds. For ingest-only and mixed sends, `preparedDocuments.requestOverrides.contextFiles` must be `[]`. The pending append and every later update must write `message.metadata.documentProcessing`; the dispatch override carries the same data via `userMetadataExtra.documentProcessing` only so the final pipeline upgrade can preserve it.

- [ ] **Step 6: Upgrade reserved user messages in the chat pipeline**

In `chatModePipeline.ts`, when `userMessageId` is provided and a user message with that ID already exists:

- replace or merge that message's content/metadata instead of appending a new user message
- keep existing created-at ordering
- append the assistant message normally
- merge `requestOverrides.userMetadataExtra.documentProcessing` into the existing message's canonical `metadata.documentProcessing`
- preserve `metadata.documentProcessing` final summary on the user message

Add a regression test that would fail if the transcript contains two user messages after a document-processing send, and assert the surviving user message still has `metadata.documentProcessing.status` plus final per-file summaries.

- [ ] **Step 7: Queue the same processing intent**

In `usePlaygroundQueueManagement.ts`, widen `requestOverrides` from `{ messageForModel?: string }` to `Record<string, unknown>`. Ensure queued items preserve selected processing modes and replay through the same send-time helper when they run.

For new document-upload queued items, store the replayable attachment data in `sourceContext`, not just the mode intent:

- base64 `content`
- filename, MIME type, size, and uploadedAt
- selected `processingMode`
- backend capabilities and blocked reasons
- `fileFingerprint`
- `ingestIdempotencyKey`, `ingestJobId`, and `ingestBatchId` when present
- draft ID when a sidepanel handoff draft is involved

Add a queue replay test that queues while the playground is busy/offline, then later processes and sends the same file decisions without requiring the original browser `File` object.

If a legacy/stale queued item cannot process because files are missing from its source context, mark it blocked with a user-safe reason instead of sending a prompt without files.

- [ ] **Step 8: Run send and queue tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/Common/Playground/__tests__/DocumentProcessingTurn.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.document-processing.test.tsx ../packages/ui/src/components/Chat/composer/__tests__/useComposerQueue.test.tsx ../packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add apps/packages/ui/src/components/Common/Playground/DocumentProcessingTurn.tsx apps/packages/ui/src/components/Common/Playground/__tests__/DocumentProcessingTurn.test.tsx apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundSubmit.ts apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundQueueManagement.ts apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/hooks/chat/useCompareSubmit.ts apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts apps/packages/ui/src/components/Common/Playground/Message.tsx apps/packages/ui/src/components/Common/Playground/PlaygroundUserMessage.tsx apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.document-processing.test.tsx apps/packages/ui/src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts
git commit -m "feat: process chat documents on send"
```

---

### Task 5: Browser Extension `/chat` Sidepanel Handoff

**Files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- Modify: `apps/packages/ui/src/services/tldw/sidepanel-chat-webui-handoff.ts`
- Modify: `apps/packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts`
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx`
- Test: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.document-processing.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx`

- [ ] **Step 1: Write failing handoff tests**

In `sidepanel-chat-webui-handoff.test.ts`, assert new fields survive encode/decode:

```ts
const url = new URL(
  buildSidepanelChatWebUiHandoffUrl({
    payload: {
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      createdAt: Date.now(),
      draft: "summarize",
      chatDocumentDraftId: "draft-123",
      ragMediaIds: [101],
      fileRetrievalEnabled: true
    }
  })
)

const decoded = decodeSidepanelChatWebUiHandoff(getFragmentHandoff(url))
expect(decoded).toMatchObject({
  chatDocumentDraftId: "draft-123",
  ragMediaIds: [101],
  fileRetrievalEnabled: true
})
```

In sidepanel form tests, assert:

- Add to chat stays inline.
- OCR/Ingest creates a server-backed draft before opening WebUI.
- Failed handoff keeps the sidepanel draft and shows retry copy.
- Retrying a failed handoff reuses the same draft while it is unexpired instead of uploading the bytes again.
- Cancelling the sidepanel draft deletes the server draft and cancels any active ingest job/batch.

- [ ] **Step 2: Run sidepanel tests and verify failure**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts ../packages/ui/src/components/Sidepanel/Chat/__tests__/form.document-processing.test.tsx
```

Expected: FAIL because the payload and sidepanel UI are not wired.

- [ ] **Step 3: Extend WebUI handoff payload**

Add optional fields:

```ts
chatDocumentDraftId?: string | null
ragMediaIds?: number[] | null
fileRetrievalEnabled?: boolean
```

Decode `ragMediaIds` defensively: only finite numbers survive. Expiry stays unchanged.

- [ ] **Step 4: Wire sidepanel document choices**

In `form.tsx`, reuse the same service types and render `DocumentProcessingChoices` in the sidepanel attachment area. If the sidepanel needs a tighter layout, wrap the same component with sidepanel-specific spacing; do not fork mode behavior. Sidepanel behavior:

- `add_to_chat`: process inline on Send using `prepareChatDocumentAttachmentsForSend`.
- `ocr_pages`: create a server-backed draft with selected modes, open full WebUI `/chat`, and show sidepanel retry if opening fails.
- `ingest_to_library`: submit ingest with the same file fingerprint/idempotency key when possible, then hand off media IDs; if ingestion must continue in WebUI, stage a draft. Do not downgrade to Add to chat.
- `cancel`: call `cancelPreparedDocumentProcessing`, delete the draft, and keep the sidepanel text draft editable.

Sidepanel blocked/error copy:

- `Open full chat again`
- `Use Add to chat instead`
- `Retry ingest`

- [ ] **Step 5: Import the handoff in WebUI Playground**

In the existing sidepanel handoff import path in `PlaygroundForm`/`Playground.tsx`, if `chatDocumentDraftId` exists:

- fetch the draft from `/api/v1/media/document-upload/drafts/{draft_id}`
- populate `uploadedFiles` with the staged decisions and populate `contextFiles` only with files whose mode is not `ingest_to_library`
- delete the draft only after successful import
- do not delete the draft when import fails for a transient client error
- show a recoverable error if the draft expired, belongs to another user, or cannot be read
- provide a cleanup path that deletes the draft if the user abandons the handoff after a successful read

If `ragMediaIds` exists:

- pass it into the current turn via request overrides or set the visible RAG scoped media state only when the user confirms continuing from the handoff.

- [ ] **Step 6: Run sidepanel and handoff tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts ../packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/form.document-processing.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/form.tsx apps/packages/ui/src/services/tldw/sidepanel-chat-webui-handoff.ts apps/packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.queue.contract.test.tsx apps/packages/ui/src/components/Sidepanel/Chat/__tests__/form.document-processing.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/sidepanel-chat-handoff-import.test.tsx
git commit -m "feat: hand off sidepanel document processing"
```

---

### Task 6: Integration Verification And UX Hardening

**Files:**
- Create: `apps/tldw-frontend/e2e/smoke/playground-document-processing.spec.ts`
- Modify: only files from Tasks 1-5 when the smoke test exposes a concrete regression.

- [ ] **Step 1: Add one Playwright smoke test**

Create a focused smoke that does not require real OCR:

- open `/chat`
- attach a small `.txt` or `.md` fixture
- verify the decision surface appears
- choose `Add to chat`
- stub `/api/v1/media/document-upload/preflight` and no-DB document processing with Playwright route handlers
- send a prompt
- verify a processing user turn appears immediately after Send, then transitions out of `processing`
- verify the message chip says chat-scoped, not library-ingested

Run:

```bash
cd apps/tldw-frontend && bunx playwright test e2e/smoke/playground-document-processing.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend unit tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/chat-document-processing.test.ts ../packages/ui/src/components/Option/Playground/__tests__/DocumentProcessingChoices.test.tsx ../packages/ui/src/components/Common/Playground/__tests__/DocumentProcessingTurn.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.document-processing.test.tsx ../packages/ui/src/services/__tests__/sidepanel-chat-webui-handoff.test.ts
```

Expected: PASS.

- [ ] **Step 3: Run existing composer suite**

Run:

```bash
cd apps/tldw-frontend && bun run test:playground:composer
```

Expected: PASS.

- [ ] **Step 4: Run backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Media/test_document_upload_processing.py -q
```

Expected: PASS.

- [ ] **Step 5: Run lint/type-facing checks**

Run:

```bash
cd apps/tldw-frontend && bun run lint
```

Expected: PASS or only pre-existing unrelated warnings. Fix new issues in touched files.

- [ ] **Step 6: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/media/document_upload_processing.py tldw_Server_API/app/api/v1/schemas/document_upload_processing.py -f json -o /tmp/bandit_chat_document_upload_processing.json
```

Expected: PASS with no new findings in touched code.

- [ ] **Step 7: Manual UX check**

Start the frontend and inspect desktop/mobile widths:

```bash
cd apps/tldw-frontend && bun run dev -- -p 8080
```

Check:

- no modal-first routine flow
- Add to chat and Ingest to library are visually distinct
- disabled OCR names the missing backend capability
- blocked files do not disappear
- mixed batches collapse cleanly
- no text overflow inside chips or buttons
- sidepanel retry actions remain visible

- [ ] **Step 8: Commit final verification/e2e changes**

```bash
git add apps/tldw-frontend/e2e/smoke/playground-document-processing.spec.ts
git commit -m "test: cover chat document processing choices"
```

---

## Completion Checklist

- [ ] `Add to chat` produces chat-scoped context and no durable media item.
- [ ] `OCR pages` is only enabled when backend preflight says OCR/rendering is available.
- [ ] `Ingest to library` creates/uses durable library media IDs and never silently downgrades.
- [ ] Ingest-only and mixed sends explicitly pass `contextFiles: []` unless chat-scoped extracted text is intentionally folded into `messageForModel`.
- [ ] Send starts processing; file selection only stages and preflights.
- [ ] Pressing Send immediately shows a processing user turn; final dispatch upgrades that same user message and does not duplicate it.
- [ ] Queued sends replay the same document decisions.
- [ ] Retry/cancel is idempotent for ingest jobs, local processing, and server-backed drafts.
- [ ] Page/token overflow is blocked or recovered with explicit chat-scoped retrieval/truncation copy; `Use retrieval instead` does not imply durable ingest.
- [ ] Sidepanel heavy modes use owner/expiry/retry/cleanup-safe draft handoff.
- [ ] Partial failures are explicit and recoverable.
- [ ] Focused Vitest, pytest, Playwright smoke, lint, and Bandit checks are recorded in Backlog.
