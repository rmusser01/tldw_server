import type {
  DocumentProcessingCapability,
  DocumentProcessingMode,
  DocumentProcessingRecoveryAction,
  DocumentProcessingTurnMetadata,
  UploadedFile,
} from "@/db/dexie/types"
import { bgRequest, bgUpload } from "@/services/background-proxy"
import i18n from "i18next"
import { extractCompletedIngestJobMediaId } from "@/services/tldw/ingest-job-results"
import {
  extractIngestJobIds,
  pollSingleIngestJob,
} from "@/services/tldw/ingest-jobs-orchestrator"
import {
  getProcessPathForType,
  inferUploadMediaTypeFromFile,
} from "@/services/tldw/media-routing"

export type DocumentUploadPreflightResponse = {
  files: Array<{
    client_id: string
    filename: string
    media_type: "pdf" | "document" | "ebook" | "unsupported"
    default_mode: DocumentProcessingMode | null
    modes: Record<DocumentProcessingMode, DocumentProcessingCapability>
    max_size_bytes: number
    max_pages: number | null
    max_chat_tokens: number
    estimated_pages?: number | null
    estimated_tokens?: number | null
    requires_send_time_estimate?: boolean
  }>
}

export type ProcessedDocumentText = {
  content: string
  sourceName?: string
}

export type IngestDocumentResult = {
  mediaId?: string | number | null
  jobId?: string | number
  batchId?: string
  status?: string
}

type PrepareDependencies = {
  processDocument?: (file: UploadedFile) => Promise<ProcessedDocumentText>
  processPdf?: (
    file: UploadedFile,
    options: { enableOcr: boolean },
  ) => Promise<ProcessedDocumentText>
  ingestDocument?: (file: UploadedFile) => Promise<IngestDocumentResult>
  waitForIngestJob?: (
    jobId: string | number,
  ) => Promise<IngestDocumentResult>
}

export type PrepareChatDocumentAttachmentsInput = PrepareDependencies & {
  files: UploadedFile[]
  historyId?: string
  sessionId?: string
  maxDirectChatTokens?: number
}

export type PreparedChatDocumentAttachments = {
  contextFiles: UploadedFile[]
  failedFiles: UploadedFile[]
  blockedFiles: UploadedFile[]
  recoveryActions: DocumentProcessingRecoveryAction[]
  requestOverrides?: {
    contextFiles: UploadedFile[]
    uploadedFiles: UploadedFile[]
    ragMediaIds?: Array<string | number>
    fileRetrievalEnabled?: boolean
    documentSnippetForModel?: string
    documentProcessing?: DocumentProcessingTurnMetadata
  }
  turnMetadata: DocumentProcessingTurnMetadata
}

const DEFAULT_MAX_DIRECT_CHAT_TOKENS = 24_000
const DIRECT_INGEST_TIMEOUT_MS = 5 * 60 * 1000
const TERMINAL_RETRY_STATUSES = new Set(["failed", "cancelled", "quarantined"])

export const DEFAULT_DOCUMENT_PROCESSING_MODE: DocumentProcessingMode =
  "add_to_chat"

const toText = (value: unknown): string =>
  typeof value === "string" ? value : ""

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}

const firstText = (...values: unknown[]): string => {
  for (const value of values) {
    const text = toText(value).trim()
    if (text) return text
  }
  return ""
}

const extractTextFromProcessResponse = (data: unknown): string => {
  const record = asRecord(data)
  const result = asRecord(record.result)
  const results = Array.isArray(record.results) ? record.results : []
  const firstResult = asRecord(results[0])
  return firstText(
    record.content,
    record.text,
    record.extracted_text,
    result.content,
    result.text,
    firstResult.content,
    firstResult.text,
    firstResult.extracted_text,
  )
}

export const estimateDirectChatTokens = (text: string): number =>
  Math.ceil(String(text || "").length / 4)

const errorMessage = (error: unknown): string =>
  error instanceof Error
    ? error.message
    : String(
        error ||
          i18n.t(
            "playground:documentProcessing.genericFailure",
            "Document processing failed"
          )
      )

const processingModeUnavailableReason = (): string =>
  i18n.t(
    "playground:documentProcessing.modeUnavailable",
    "This processing mode is unavailable."
  )

const unsupportedDocumentReason = (): string => {
  const fallback = "This document type is unsupported."
  return (
    i18n.t("playground:documentProcessing.unsupportedDocument", fallback) ||
    fallback
  )
}

const directChatTooLargeReason = (tokenEstimate: number): string =>
  i18n.t(
    "playground:documentProcessing.directChatTooLarge",
    "Document text is too large for direct chat ({{count}} estimated tokens).",
    { count: tokenEstimate }
  )

const isPdf = (file: UploadedFile): boolean =>
  file.type.toLowerCase().includes("pdf") ||
  file.filename.toLowerCase().endsWith(".pdf")

const uniqueRecoveryActions = (
  files: UploadedFile[],
): DocumentProcessingRecoveryAction[] =>
  Array.from(
    new Set(files.flatMap((file) => file.processingRecoveryActions || [])),
  )

const asUploadFile = async (file: UploadedFile) => {
  const content = String(file.content || "")
  const data = content.startsWith("data:")
    ? new Uint8Array(await (await fetch(content)).arrayBuffer())
    : new TextEncoder().encode(content)
  return {
    name: file.filename || "upload",
    type: file.type || "application/octet-stream",
    data,
  }
}

export const dataUrlToUploadFile = async (
  file: UploadedFile,
): Promise<File> => {
  const content = String(file.content || "")
  const type = file.type || "application/octet-stream"
  const blob = content.startsWith("data:")
    ? await (await fetch(content)).blob()
    : new Blob([content], { type })
  return new File([blob], file.filename || "upload", {
    type: type || blob.type || "application/octet-stream",
  })
}

const applyDocumentProcessingMode = (
  file: UploadedFile,
  mode: DocumentProcessingMode,
): UploadedFile => {
  const capability = file.processingCapabilities?.[mode]
  if (capability && !capability.available) {
    return {
      ...file,
      processingMode: mode,
      processingStatus: "blocked",
      processingBlockedReason:
        capability.reason || processingModeUnavailableReason(),
    }
  }
  return {
    ...file,
    processingMode: mode,
    processingStatus: "pending",
    processingBlockedReason: undefined,
  }
}

export const withDefaultDocumentDecision = (
  file: UploadedFile,
): UploadedFile => ({
  ...file,
  processingMode: file.processingMode ?? DEFAULT_DOCUMENT_PROCESSING_MODE,
  processingStatus: file.processingStatus ?? "preflighting",
})

export const setBatchDocumentProcessingMode = (
  files: UploadedFile[],
  mode: DocumentProcessingMode,
): UploadedFile[] =>
  files.map((file) => applyDocumentProcessingMode(file, mode))

export const setFileDocumentProcessingMode = (
  files: UploadedFile[],
  fileId: string,
  mode: DocumentProcessingMode,
): UploadedFile[] =>
  files.map((file) =>
    file.id === fileId ? applyDocumentProcessingMode(file, mode) : file,
  )

export const hasBlockedDocumentProcessing = (files: UploadedFile[]): boolean =>
  files.some((file) => file.processingStatus === "blocked")

export const buildChatScopedRetrievalContext = ({
  chunks,
  query,
  tokenBudget,
}: {
  chunks: string[]
  query?: string
  tokenBudget: number
}): { content: string; selectedChunks: string[] } => {
  const terms = new Set(
    String(query || "")
      .toLowerCase()
      .split(/\W+/)
      .filter(Boolean),
  )
  const scored = chunks
    .map((chunk, index) => {
      const haystack = chunk.toLowerCase()
      const score = Array.from(terms).reduce(
        (total, term) => total + (haystack.includes(term) ? 1 : 0),
        0,
      )
      return { chunk, index, score }
    })
    .sort((a, b) => b.score - a.score || a.index - b.index)
  const selectedChunks: string[] = []
  let usedTokens = 0
  for (const item of scored) {
    const tokenEstimate = estimateDirectChatTokens(item.chunk)
    if (selectedChunks.length > 0 && usedTokens + tokenEstimate > tokenBudget) {
      continue
    }
    selectedChunks.push(item.chunk)
    usedTokens += tokenEstimate
    if (usedTokens >= tokenBudget) break
  }
  return {
    content: selectedChunks.join("\n\n"),
    selectedChunks,
  }
}

const normalizeJobId = (value: unknown): string | number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim()) return value.trim()
  return undefined
}

const normalizeJobStatus = (value: unknown): string =>
  String(value || "")
    .trim()
    .toLowerCase()

const buildTurnMetadata = (
  files: UploadedFile[],
  ragMediaIds: Array<string | number> = [],
): DocumentProcessingTurnMetadata => {
  const status = files.some((file) => file.processingStatus === "failed")
    ? "failed"
    : files.some((file) => file.processingStatus === "blocked")
      ? "blocked"
      : files.some((file) => file.processingStatus === "processing")
        ? "processing"
        : "ready"

  return {
    status,
    files: files.map((file) => ({
      id: file.id,
      filename: file.filename,
      mode: file.processingMode,
      status: file.processingStatus || "pending",
      summary: file.processingSummary || file.processingBlockedReason,
      error: file.processingError,
    })),
    ragMediaIds: ragMediaIds.length > 0 ? ragMediaIds : undefined,
    fileRetrievalEnabled: ragMediaIds.length > 0 ? true : undefined,
    recoveryActions: uniqueRecoveryActions(files),
  }
}

export const normalizeDocumentPreflightResponse = (
  response: DocumentUploadPreflightResponse,
  files: UploadedFile[],
): UploadedFile[] => {
  const byId = new Map(
    (response.files || []).map((item) => [item.client_id, item]),
  )
  return files.map((file) => {
    const item = byId.get(file.id)
    if (!item) return file
    const mode = file.processingMode || item.default_mode || undefined
    const capability = mode ? item.modes?.[mode] : undefined
    const blocked =
      !mode || !capability?.available || capability.status === "blocked"
    return {
      ...file,
      processingMode: mode,
      processingStatus: blocked ? "blocked" : "pending",
      processingCapabilities: item.modes,
      processingBlockedReason: blocked
        ? capability?.reason || unsupportedDocumentReason()
        : undefined,
      processingPageEstimate: item.estimated_pages ?? null,
      processingTokenEstimate: item.estimated_tokens ?? null,
    }
  })
}

export const computeDocumentFileFingerprint = async (
  file: UploadedFile,
): Promise<string> => {
  const input = [
    file.filename,
    file.type,
    String(file.size),
    String(file.content || ""),
  ].join("\0")
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(input),
  )
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("")
}

export const buildIngestIdempotencyKey = async ({
  file,
  historyId,
  sessionId,
}: {
  file: UploadedFile
  historyId?: string
  sessionId?: string
}): Promise<string> =>
  `chat-document-ingest:${historyId || sessionId || "draft"}:${await computeDocumentFileFingerprint(file)}`

export const processDocumentForChat = async (
  file: UploadedFile,
): Promise<ProcessedDocumentText> => {
  const data = await bgUpload<unknown>({
    path: getProcessPathForType(
      inferUploadMediaTypeFromFile(file.filename, file.type),
    ),
    method: "POST",
    file: await asUploadFile(file),
    fileFieldName: "files",
  })
  return {
    content: extractTextFromProcessResponse(data),
    sourceName: file.filename,
  }
}

export const processPdfForChat = async (
  file: UploadedFile,
  options: { enableOcr: boolean } = { enableOcr: false },
): Promise<ProcessedDocumentText> => {
  const data = await bgUpload<unknown>({
    path: "/api/v1/media/process-pdfs",
    method: "POST",
    fields: { enable_ocr: options.enableOcr },
    file: await asUploadFile(file),
    fileFieldName: "files",
  })
  return {
    content: extractTextFromProcessResponse(data),
    sourceName: file.filename,
  }
}

export const ingestDocumentToLibrary = async (
  file: UploadedFile,
  options: { historyId?: string; sessionId?: string; timeoutMs?: number } = {},
): Promise<IngestDocumentResult> => {
  if (file.ingestJobId != null) {
    const existingJob = await bgRequest<unknown>({
      path: `/api/v1/media/ingest/jobs/${encodeURIComponent(String(file.ingestJobId))}`,
      method: "GET",
    })
    const existingRecord = asRecord(existingJob)
    const status = normalizeJobStatus(existingRecord.status)
    if (!TERMINAL_RETRY_STATUSES.has(status)) {
      return {
        mediaId: extractCompletedIngestJobMediaId(existingJob),
        jobId: file.ingestJobId,
        batchId:
          String(existingRecord.batch_id || file.ingestBatchId || "") ||
          undefined,
        status,
      }
    }
  }

  const idempotencyKey =
    file.ingestIdempotencyKey ||
    (await buildIngestIdempotencyKey({
      file,
      historyId: options.historyId,
      sessionId: options.sessionId,
    }))
  const response = await bgUpload<unknown>({
    path: "/api/v1/media/ingest/jobs",
    method: "POST",
    fields: {
      media_type: inferUploadMediaTypeFromFile(file.filename, file.type),
      idempotency_key: idempotencyKey,
      perform_analysis: true,
      perform_chunking: true,
    },
    file: await asUploadFile(file),
    fileFieldName: "files",
    timeoutMs: options.timeoutMs || DIRECT_INGEST_TIMEOUT_MS,
  })
  const responseRecord = asRecord(response)
  const [jobId] = extractIngestJobIds(response)
  return {
    mediaId: extractCompletedIngestJobMediaId(response),
    jobId: normalizeJobId(jobId),
    batchId: String(responseRecord.batch_id || "") || undefined,
    status: normalizeJobStatus(responseRecord.status),
  }
}

export const waitForIngestDocumentJob = async (
  jobId: string | number,
  options: { timeoutMs?: number; pollIntervalMs?: number } = {},
): Promise<IngestDocumentResult> => {
  const normalizedJobId = Number(jobId)
  if (!Number.isFinite(normalizedJobId) || normalizedJobId <= 0) {
    throw new Error("Ingest job returned an invalid job id.")
  }
  const terminal = await pollSingleIngestJob({
    jobId: Math.trunc(normalizedJobId),
    timeoutMs: options.timeoutMs || DIRECT_INGEST_TIMEOUT_MS,
    pollIntervalMs: options.pollIntervalMs,
    isCancelled: () => false,
    onCancel: () => undefined,
    fetchJob: async (id) => {
      try {
        return {
          ok: true,
          data: await bgRequest<unknown>({
            path: `/api/v1/media/ingest/jobs/${encodeURIComponent(String(id))}`,
            method: "GET",
          }),
        }
      } catch (error) {
        return { ok: false, error: errorMessage(error) }
      }
    },
  })
  if (terminal.terminalStatus !== "completed") {
    throw new Error(
      terminal.error || `Ingest ${terminal.terminalStatus || "failed"}.`,
    )
  }
  const mediaId = extractCompletedIngestJobMediaId(terminal.data)
  if (mediaId == null) {
    throw new Error("Ingest completed without a media id.")
  }
  return {
    mediaId,
    jobId: Math.trunc(normalizedJobId),
    status: "completed",
  }
}

export const prepareChatDocumentAttachmentsForSend = async ({
  files,
  historyId,
  sessionId,
  maxDirectChatTokens = DEFAULT_MAX_DIRECT_CHAT_TOKENS,
  processDocument = processDocumentForChat,
  processPdf = processPdfForChat,
  ingestDocument = (file) =>
    ingestDocumentToLibrary(file, { historyId, sessionId }),
  waitForIngestJob = waitForIngestDocumentJob,
}: PrepareChatDocumentAttachmentsInput): Promise<PreparedChatDocumentAttachments> => {
  const contextFiles: UploadedFile[] = []
  const failedFiles: UploadedFile[] = []
  const blockedFiles: UploadedFile[] = []
  const processedFiles: UploadedFile[] = []
  const ragMediaIds: Array<string | number> = []
  const chatSnippets: string[] = []

  for (const file of files) {
    const mode = file.processingMode || "add_to_chat"
    const capability = file.processingCapabilities?.[mode]
    if (capability && !capability.available) {
      const recoveryActions: DocumentProcessingRecoveryAction[] = []
      if (
        mode !== "add_to_chat" &&
        file.processingCapabilities?.add_to_chat?.available !== false
      ) {
        recoveryActions.push("switch_to_add_to_chat")
      }
      if (
        mode !== "ingest_to_library" &&
        file.processingCapabilities?.ingest_to_library?.available !== false
      ) {
        recoveryActions.push("switch_to_ingest")
      }
      const blockedFile: UploadedFile = {
        ...file,
        processingMode: mode,
        processingStatus: "blocked",
        processingBlockedReason:
          capability.reason || processingModeUnavailableReason(),
        processingRecoveryActions: recoveryActions,
      }
      blockedFiles.push(blockedFile)
      processedFiles.push(blockedFile)
      continue
    }

    try {
      if (mode === "ingest_to_library") {
        let result = await ingestDocument(file)
        if (result.mediaId == null) {
          const status = normalizeJobStatus(result.status)
          if (
            result.jobId != null &&
            status !== "completed" &&
            !TERMINAL_RETRY_STATUSES.has(status)
          ) {
            const pendingResult = result
            const completedResult = await waitForIngestJob(result.jobId)
            result = {
              ...completedResult,
              batchId: pendingResult.batchId ?? completedResult.batchId,
            }
          }
        }
        if (result.mediaId == null) {
          throw new Error("Ingest completed without a media id.")
        }
        ragMediaIds.push(result.mediaId)
        processedFiles.push({
          ...file,
          processingMode: mode,
          processingStatus: "ready",
          processed: true,
          ingestJobId: result.jobId,
          ingestBatchId: result.batchId,
          processingResultRef: result.jobId
            ? { kind: "ingest_job", id: result.jobId }
            : undefined,
        })
        continue
      }

      const result =
        mode === "ocr_pages" || isPdf(file)
          ? await processPdf(file, { enableOcr: mode === "ocr_pages" })
          : await processDocument(file)
      const tokenEstimate = estimateDirectChatTokens(result.content)
      if (tokenEstimate > maxDirectChatTokens) {
        const blockedFile: UploadedFile = {
          ...file,
          processingMode: mode,
          processingStatus: "blocked",
          processingTokenEstimate: tokenEstimate,
          processingBlockedReason: directChatTooLargeReason(tokenEstimate),
          processingRecoveryActions: [
            "use_chat_scoped_retrieval",
            "switch_to_ingest",
          ],
        }
        blockedFiles.push(blockedFile)
        processedFiles.push(blockedFile)
        continue
      }

      const readyFile: UploadedFile = {
        ...file,
        content: result.content,
        processed: true,
        processingMode: mode,
        processingStatus: "ready",
        processingTokenEstimate: tokenEstimate,
        processingResultRef: { kind: "chat_context", id: file.id },
      }
      contextFiles.push(readyFile)
      chatSnippets.push(result.content)
      processedFiles.push(readyFile)
    } catch (error) {
      const failedFile: UploadedFile = {
        ...file,
        processingMode: mode,
        processingStatus: "failed",
        processingError: errorMessage(error),
        processingRecoveryActions: ["retry", "cancel"],
      }
      failedFiles.push(failedFile)
      processedFiles.push(failedFile)
    }
  }

  const recoveryActions = uniqueRecoveryActions([
    ...blockedFiles,
    ...failedFiles,
  ])
  const turnMetadata = buildTurnMetadata(processedFiles, ragMediaIds)
  const hasFailure = failedFiles.length > 0 || blockedFiles.length > 0
  const hasProcessing = processedFiles.some(
    (file) => file.processingStatus === "processing",
  )
  const hasIngest = ragMediaIds.length > 0
  const mixedMessage =
    hasIngest && chatSnippets.length > 0 ? chatSnippets.join("\n\n") : undefined
  const requestOverrides =
    hasFailure || hasProcessing
      ? undefined
      : {
          contextFiles: hasIngest ? [] : contextFiles,
          uploadedFiles: hasIngest ? [] : contextFiles,
          ragMediaIds: hasIngest ? ragMediaIds : undefined,
          fileRetrievalEnabled: hasIngest ? true : undefined,
          documentSnippetForModel: mixedMessage,
          documentProcessing: turnMetadata,
        }

  return {
    contextFiles: hasIngest ? [] : contextFiles,
    failedFiles,
    blockedFiles,
    recoveryActions,
    requestOverrides,
    turnMetadata,
  }
}

export const cancelPreparedDocumentProcessing = async (
  files: UploadedFile[],
  deps: {
    cancelIngestBatch?: (batchId: string) => Promise<void> | void
    cancelIngestJob?: (jobId: string | number) => Promise<void> | void
    deleteDraft?: (draftId: string) => Promise<void> | void
  } = {},
): Promise<void> => {
  const cancelIngestBatch =
    deps.cancelIngestBatch ||
    ((batchId: string) =>
      bgRequest<void>({
        path: `/api/v1/media/ingest/jobs/cancel?batch_id=${encodeURIComponent(batchId)}&reason=user_cancelled`,
        method: "POST",
      }))
  const cancelIngestJob =
    deps.cancelIngestJob ||
    ((jobId: string | number) =>
      bgRequest<void>({
        path: `/api/v1/media/ingest/jobs/${encodeURIComponent(String(jobId))}?reason=user_cancelled`,
        method: "DELETE",
      }))
  const deleteDraft =
    deps.deleteDraft ||
    ((draftId: string) =>
      bgRequest<void>({
        path: `/api/v1/media/document-upload/drafts/${encodeURIComponent(draftId)}`,
        method: "DELETE",
      }))

  const cancelledBatchIds = new Set<string>()
  const cleanup = async (description: string, action: () => Promise<void> | void) => {
    try {
      await action()
    } catch (error) {
      console.warn(`Document processing cleanup failed: ${description}`, error)
    }
  }

  for (const file of files) {
    if (file.ingestBatchId && !cancelledBatchIds.has(file.ingestBatchId)) {
      cancelledBatchIds.add(file.ingestBatchId)
      await cleanup(`batch ${file.ingestBatchId}`, () =>
        cancelIngestBatch(file.ingestBatchId as string)
      )
    }
    if (file.ingestJobId != null) {
      await cleanup(`job ${file.ingestJobId}`, () =>
        cancelIngestJob(file.ingestJobId as string | number)
      )
    }
    const ref = file.processingResultRef
    if (ref?.kind === "draft") {
      await cleanup(`draft ${ref.id}`, () => deleteDraft(String(ref.id)))
    }
    if (file.documentDraftId) {
      await cleanup(`draft ${file.documentDraftId}`, () =>
        deleteDraft(file.documentDraftId as string)
      )
    }
  }
}
