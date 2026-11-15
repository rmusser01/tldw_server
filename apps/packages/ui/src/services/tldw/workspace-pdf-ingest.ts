import type { PersistedQuickIngestTracking } from "@/store/quick-ingest-session"
import { servicePromptAuthorityKey } from "./domains/service-prompts"
import { extractCompletedIngestJobMediaId, extractCompletedIngestJobWarning } from "./ingest-job-results"
import { pollSingleIngestJob, requireSubmittedIngestJobs } from "./ingest-jobs-orchestrator"
import { assertQuickIngestRequestCurrent, scopedQuickIngestRequest, scopedQuickIngestUpload, type QuickIngestRequestContext } from "./quick-ingest-request-scope"

type Context = QuickIngestRequestContext & { workspaceId: string }
export type WorkspacePdfUpload = {
  id: string
  fileName: string
  fileSize: number
  status: "submitting" | "queued" | "processing" | "completed" | "failed" | "interrupted" | "unknown" | "attached"
  tracking?: PersistedQuickIngestTracking
  message?: string
  progressPercent?: number
}
const PREFIX = "tldw:workspace-pdf-ingest:v1:"
const PHASE_MESSAGES: Record<string, string> = {
  process: "Extracting and preparing PDF passages…",
  processing: "Extracting and preparing PDF passages…",
  extract: "Extracting PDF pages…",
  extracting: "Extracting PDF pages…",
  chunk: "Preparing searchable passages…",
  chunking: "Preparing searchable passages…",
  persist: "Saving the PDF and its original file…",
  persisting: "Saving the PDF and its original file…",
  embedding: "Preparing semantic search…",
  queued: "Queued for PDF extraction",
}
const prefixFor = (workspaceId: string) => `${PREFIX}${encodeURIComponent(workspaceId)}:`
const storageKey = (context: Context) => {
  if (!context.requestScope) throw new Error("Sign in before uploading a PDF.")
  return prefixFor(context.workspaceId) + servicePromptAuthorityKey(context.requestScope)
}

/** Receipts contain file labels and existing Jobs identities, never bytes or credentials. */
export const listWorkspacePdfUploads = (context: Context): WorkspacePdfUpload[] => {
  const raw: unknown = JSON.parse(sessionStorage.getItem(storageKey(context)) || "[]")
  return Array.isArray(raw) ? raw.filter((item): item is WorkspacePdfUpload => Boolean(item &&
    typeof item.id === "string" && typeof item.fileName === "string" && typeof item.fileSize === "number" &&
    ["submitting", "queued", "processing", "completed", "failed", "interrupted", "unknown", "attached"].includes(item.status) &&
    (!item.tracking || (item.tracking.mode === "webui-direct" && typeof item.tracking.batchId === "string" &&
      item.tracking.jobIds?.length === 1 && Number.isSafeInteger(item.tracking.jobIds[0]) && item.tracking.jobIds[0] > 0))))
    .map(receipt => ({ ...receipt, message: typeof receipt.message === "string"
      ? PHASE_MESSAGES[receipt.message.trim().toLowerCase()] || receipt.message : undefined })) : []
}

export const hasWorkspacePdfUploads = (workspaceId: string): boolean =>
  Object.keys(sessionStorage).some(key => key.startsWith(prefixFor(workspaceId)))

const save = (context: Context, receipt: WorkspacePdfUpload) => {
  const records = listWorkspacePdfUploads(context).filter(item => item.id !== receipt.id)
  // Keep active receipts: dropping one would permit duplicate submission after reconnecting.
  while (records.length >= 20) {
    const removable = records.findIndex(item => ["attached", "failed"].includes(item.status))
    if (removable < 0) throw new Error("Finish or reconnect to existing PDF uploads before adding more.")
    records.splice(removable, 1)
  }
  sessionStorage.setItem(storageKey(context), JSON.stringify([...records, receipt]))
}

export const markWorkspacePdfAttached = (context: Context, receipt: WorkspacePdfUpload): void => {
  assertQuickIngestRequestCurrent(context)
  save(context, { ...receipt, status: "attached" })
}

/** Submit once, then reconnect using the same ingestion Jobs polling contract as Quick Ingest. */
export const ingestWorkspacePdf = async (options: Context & {
  file?: File
  receipt?: WorkspacePdfUpload
  onProgress?: (receipt: WorkspacePdfUpload) => void
}): Promise<{ mediaId: number; data: unknown; receipt: WorkspacePdfUpload; warning?: string }> => {
  assertQuickIngestRequestCurrent(options)
  let receipt = options.receipt
  if (receipt && !options.file) {
    const saved = listWorkspacePdfUploads(options).find(item => item.id === receipt!.id)
    if (!saved?.tracking || saved.tracking.batchId !== receipt.tracking?.batchId ||
      saved.tracking.jobIds?.[0] !== receipt.tracking?.jobIds?.[0]) {
      throw new Error("This saved upload belongs to a different connection or workspace. Reopen Add Sources to reconnect.")
    }
    receipt = saved
  }
  let data: ArrayBuffer | undefined
  if (options.file) {
    data = await options.file.arrayBuffer()
    const digest = await crypto.subtle.digest("SHA-256", data)
    assertQuickIngestRequestCurrent(options)
    const id = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, "0")).join("")
    receipt = listWorkspacePdfUploads(options).find(item => item.id === id)
    if (receipt?.status === "failed") receipt = undefined
    receipt ??= { id, fileName: options.file.name, fileSize: options.file.size, status: "submitting" }
  }
  if (!receipt) throw new Error("Choose a PDF or reconnect to a saved upload.")
  const update = (patch: Partial<WorkspacePdfUpload>) => {
    receipt = { ...receipt!, ...patch }
    save(options, receipt)
    options.onProgress?.(receipt)
  }
  if (!receipt.tracking?.jobIds?.length) {
    const existing = listWorkspacePdfUploads(options).find(item => item.id === receipt!.id)
    if (existing && existing.status !== "failed") {
      throw new Error("This upload may still be running. Check My Media or ingestion Jobs before submitting it again.")
    }
    if (!options.file || !data) throw new Error("Choose the PDF again to retry its failed upload.")
    update({ status: "submitting", message: "Uploading PDF to the processing queue…" })
    try {
      const submitted = await scopedQuickIngestUpload(options, {
        path: "/api/v1/media/ingest/jobs", method: "POST", fileFieldName: "files",
        file: { name: options.file.name, type: options.file.type, data },
        fields: { media_type: "pdf", keep_original_file: true, perform_chunking: true,
          perform_analysis: false, overwrite: false, generate_embeddings: true, embedding_dispatch_mode: "background" },
      })
      const { batchId, jobIds } = requireSubmittedIngestJobs(submitted)
      update({ status: "queued", message: "Queued for PDF extraction", tracking: {
        mode: "webui-direct", batchId, jobIds, submittedItemIds: [receipt.id], startedAt: Date.now(),
      } })
    } catch (error) {
      const status = typeof error === "object" && error ? (error as { status?: number }).status : undefined
      update({ status: status && status >= 400 && status < 500 && status !== 408 ? "failed" : "unknown",
        message: "Upload acceptance could not be confirmed. Check My Media or ingestion Jobs before uploading again." })
      throw error
    }
  }
  try {
    const terminal = await pollSingleIngestJob({
      jobId: receipt.tracking!.jobIds![0], timeoutMs: 2 * 60 * 60 * 1000, pollIntervalMs: 1500,
      isCancelled: () => Boolean(options.signal?.aborted), onCancel: () => {},
      fetchJob: async id => {
        const job = await scopedQuickIngestRequest<Record<string, unknown>>(options, {
          path: `/api/v1/media/ingest/jobs/${id}`, method: "GET",
        })
        if (job.status !== "completed" && job.status !== "failed" && job.status !== "cancelled" && job.status !== "quarantined") {
          const rawMessage = typeof job.progress_message === "string" ? job.progress_message.trim() : ""
          update({ status: job.status === "queued" ? "queued" : "processing",
            message: PHASE_MESSAGES[rawMessage.toLowerCase()] || rawMessage || (job.status === "queued" ? PHASE_MESSAGES.queued : PHASE_MESSAGES.process),
            progressPercent: typeof job.progress_percent === "number" && Number.isFinite(job.progress_percent) ? Math.max(0, Math.min(100, job.progress_percent)) : undefined })
        }
        return { ok: true, data: job }
      },
    })
    assertQuickIngestRequestCurrent(options)
    if (terminal.terminalStatus !== "completed") {
      update({ status: terminal.terminalStatus === "timeout" ? "interrupted" : "failed", message: terminal.error || "PDF processing failed." })
      throw new Error(terminal.error || "PDF processing failed.")
    }
    const mediaId = Number(extractCompletedIngestJobMediaId(terminal.data))
    if (!Number.isSafeInteger(mediaId) || mediaId <= 0) throw new Error("The job returned no saved media. Check My Media before retrying.")
    const warning = extractCompletedIngestJobWarning(terminal.data)
    update({ status: "completed", message: warning || "PDF saved and prepared.", progressPercent: 100 })
    return { mediaId, data: terminal.data, receipt, warning }
  } catch (error) {
    if (receipt.status !== "failed") save(options, { ...receipt, status: "interrupted" })
    throw error
  }
}
