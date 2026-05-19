import type { PendingClipDraft } from "./pending-draft"
import type {
  WebClipperEnrichmentPayload,
  WebClipperEnrichmentResponse,
  WebClipperEnrichmentType
} from "./types"
import {
  tldwClient,
  type ChatCompletionRequest
} from "@/services/tldw/TldwApiClient"

const MAX_OCR_INLINE = 1500
const MAX_VLM_INLINE = 1000
const MAX_ANALYZE_CONTEXT = 4000
const SCREENSHOT_PLACEHOLDER = "[screenshot captured]"

export const WEB_CLIPPER_PENDING_ANALYZE_STORAGE_KEY =
  "tldw:web-clipper:pendingAnalyze"

export type PendingWebClipAnalyzeRequest = {
  id: string
  clipId: string
  noteId: string
  pageUrl: string
  pageTitle: string
  image: string
  message: string
  requestOverrides: {
    chatMode: "normal" | "vision"
    useOCR: boolean
  }
}

export type WebClipperEnrichmentRunResult = Partial<
  Record<WebClipperEnrichmentType, WebClipperEnrichmentResponse>
>

export type WebClipperPendingEnrichmentState = WebClipperEnrichmentRunResult

const trimText = (value: unknown): string => String(value || "").trim()

const clampInlineSummary = (
  enrichmentType: WebClipperEnrichmentType,
  summary: string
): string => {
  const limit = enrichmentType === "ocr" ? MAX_OCR_INLINE : MAX_VLM_INLINE
  return summary.slice(0, limit).trim()
}

const resolveEnrichmentModel = (selectedModel?: string | null): string => {
  const trimmed = trimText(selectedModel)
  return trimmed || "auto"
}

const getClipContextText = (draft: PendingClipDraft): string => {
  const candidates = [
    trimText(draft.fullExtract),
    trimText(draft.selectionText),
    trimText(draft.visibleBody)
  ]

  for (const candidate of candidates) {
    if (!candidate || candidate === SCREENSHOT_PLACEHOLDER) {
      continue
    }
    return candidate
  }

  return ""
}

const buildStructuredPayload = (
  draft: PendingClipDraft,
  enrichmentType: WebClipperEnrichmentType
): Record<string, unknown> => {
  const clipContext = getClipContextText(draft)
  return {
    source: "extension.web-clipper",
    enrichment_type: enrichmentType,
    page_url: draft.pageUrl,
    page_title: draft.pageTitle,
    screenshot_captured: Boolean(draft.captureMetadata.screenshotDataUrl),
    clip_context: clipContext || null
  }
}

const buildPendingEnrichmentResults = ({
  clipId,
  sourceNoteVersion,
  runOcr,
  runVlm
}: {
  clipId: string
  sourceNoteVersion: number
  runOcr: boolean
  runVlm: boolean
}): WebClipperPendingEnrichmentState => {
  const pendingResults: WebClipperPendingEnrichmentState = {}

  if (runOcr) {
    pendingResults.ocr = {
      clip_id: clipId,
      enrichment_type: "ocr",
      status: "pending",
      source_note_version: sourceNoteVersion,
      inline_applied: false,
      inline_summary: null,
      conflict_reason: null,
      warnings: []
    }
  }

  if (runVlm) {
    pendingResults.vlm = {
      clip_id: clipId,
      enrichment_type: "vlm",
      status: "pending",
      source_note_version: sourceNoteVersion,
      inline_applied: false,
      inline_summary: null,
      conflict_reason: null,
      warnings: []
    }
  }

  return pendingResults
}

const buildEnrichmentSystemPrompt = (
  enrichmentType: WebClipperEnrichmentType
): string =>
  enrichmentType === "ocr"
    ? [
        "You extract text from saved browser clips.",
        "If an image is attached, transcribe the visible text accurately.",
        "If no image is attached, recover the most useful text from the provided clip context.",
        "Return strict JSON with keys inline_summary and structured_payload."
      ].join(" ")
    : [
        "You analyze saved browser clips with an emphasis on visual layout and notable elements.",
        "If an image is attached, describe what is shown and what matters.",
        "If no image is attached, infer the most helpful visual description from the provided clip context.",
        "Return strict JSON with keys inline_summary and structured_payload."
      ].join(" ")

const buildEnrichmentUserPrompt = (
  draft: PendingClipDraft,
  enrichmentType: WebClipperEnrichmentType
): string => {
  const clipContext = getClipContextText(draft)
  const instructions =
    enrichmentType === "ocr"
      ? "Produce inline_summary as the best short extracted-text summary. Put the detailed transcript in structured_payload.transcript."
      : "Produce inline_summary as the best short visual summary. Put richer details in structured_payload.description and structured_payload.notable_elements."

  const parts = [
    instructions,
    `Title: ${draft.pageTitle}`,
    `URL: ${draft.pageUrl}`
  ]

  if (clipContext) {
    parts.push(`Clip context:\n${clipContext}`)
  }

  return parts.join("\n\n")
}

const extractCompletionContent = (payload: unknown): string => {
  if (!isRecord(payload)) return ""

  const choiceList = Array.isArray(payload.choices) ? payload.choices : []
  const firstChoice = choiceList[0]
  if (isRecord(firstChoice) && isRecord(firstChoice.message)) {
    const messageContent = firstChoice.message.content
    if (typeof messageContent === "string") {
      return messageContent.trim()
    }
  }

  if (typeof payload.content === "string") {
    return payload.content.trim()
  }

  if (typeof payload.text === "string") {
    return payload.text.trim()
  }

  return ""
}

const parseStructuredCompletion = (
  rawContent: string
): Record<string, unknown> | null => {
  if (!rawContent) return null
  try {
    const parsed = JSON.parse(rawContent)
    return isRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}

const buildCompletedStructuredPayload = ({
  draft,
  enrichmentType,
  selectedModel,
  rawContent,
  parsedContent
}: {
  draft: PendingClipDraft
  enrichmentType: WebClipperEnrichmentType
  selectedModel?: string | null
  rawContent: string
  parsedContent: Record<string, unknown> | null
}): Record<string, unknown> => {
  const basePayload = buildStructuredPayload(draft, enrichmentType)
  const parsedStructured =
    parsedContent && isRecord(parsedContent.structured_payload)
      ? parsedContent.structured_payload
      : null

  return {
    ...basePayload,
    backend_path: "chat.completions",
    backend_model: resolveEnrichmentModel(selectedModel),
    raw_response: rawContent,
    structured_payload:
      parsedStructured ||
      (enrichmentType === "ocr"
        ? { transcript: rawContent }
        : { description: rawContent })
  }
}

const createFailedEnrichmentResult = ({
  clipId,
  enrichmentType,
  sourceNoteVersion,
  errorMessage
}: {
  clipId: string
  enrichmentType: WebClipperEnrichmentType
  sourceNoteVersion: number
  errorMessage: string
}): WebClipperEnrichmentResponse => ({
  clip_id: clipId,
  enrichment_type: enrichmentType,
  status: "failed",
  source_note_version: sourceNoteVersion,
  inline_applied: false,
  inline_summary: null,
  conflict_reason: null,
  warnings: [errorMessage]
})

const runChatBackedEnrichment = async ({
  draft,
  enrichmentType,
  selectedModel
}: {
  draft: PendingClipDraft
  enrichmentType: WebClipperEnrichmentType
  selectedModel?: string | null
}): Promise<{
  inlineSummary: string
  structuredPayload: Record<string, unknown>
}> => {
  await tldwClient.initialize().catch(() => undefined)

  const request: ChatCompletionRequest = {
    model: resolveEnrichmentModel(selectedModel),
    stream: false,
    save_to_db: false,
    response_format: { type: "json_object" },
    temperature: 0.1,
    max_tokens: 800,
    messages: [
      {
        role: "system",
        content: buildEnrichmentSystemPrompt(enrichmentType)
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: buildEnrichmentUserPrompt(draft, enrichmentType)
          },
          ...(draft.captureMetadata.screenshotDataUrl
            ? [
                {
                  type: "image_url" as const,
                  image_url: {
                    url: draft.captureMetadata.screenshotDataUrl,
                    detail: "high" as const
                  }
                }
              ]
            : [])
        ]
      }
    ]
  }

  const response = await tldwClient.createChatCompletion(request)
  const payload = await response.json().catch(() => null)
  const rawContent = extractCompletionContent(payload)
  if (!rawContent) {
    throw new Error(
      `${enrichmentType === "ocr" ? "OCR" : "Visual analysis"} returned an empty response.`
    )
  }

  const parsedContent = parseStructuredCompletion(rawContent)
  const inlineSummaryCandidate =
    trimText(parsedContent?.inline_summary) ||
    trimText(parsedContent?.summary) ||
    rawContent

  return {
    inlineSummary: clampInlineSummary(enrichmentType, inlineSummaryCandidate),
    structuredPayload: buildCompletedStructuredPayload({
      draft,
      enrichmentType,
      selectedModel,
      rawContent,
      parsedContent
    })
  }
}

const buildCompletedEnrichmentPayload = (
  draft: PendingClipDraft,
  clipId: string,
  enrichmentType: WebClipperEnrichmentType,
  sourceNoteVersion: number,
  selectedModel?: string | null
): Promise<WebClipperEnrichmentPayload> =>
  runChatBackedEnrichment({
    draft,
    enrichmentType,
    selectedModel
  }).then(({ inlineSummary, structuredPayload }) => ({
    clip_id: clipId,
    enrichment_type: enrichmentType,
    status: "complete",
    inline_summary: inlineSummary,
    structured_payload: structuredPayload,
    source_note_version: sourceNoteVersion
  }))

export const persistRequestedWebClipEnrichments = async ({
  draft,
  clipId,
  sourceNoteVersion,
  runOcr,
  runVlm,
  selectedModel
}: {
  draft: PendingClipDraft
  clipId: string
  sourceNoteVersion: number
  runOcr: boolean
  runVlm: boolean
  selectedModel?: string | null
}): Promise<WebClipperEnrichmentRunResult> => {
  const results: WebClipperEnrichmentRunResult = {}
  let currentVersion = sourceNoteVersion

  const requestedTypes: WebClipperEnrichmentType[] = []
  if (runOcr) requestedTypes.push("ocr")
  if (runVlm) requestedTypes.push("vlm")

  for (const enrichmentType of requestedTypes) {
    try {
      const response = await tldwClient.persistWebClipEnrichment(
        clipId,
        await buildCompletedEnrichmentPayload(
          draft,
          clipId,
          enrichmentType,
          currentVersion,
          selectedModel
        )
      )
      results[enrichmentType] = response
      currentVersion = response.source_note_version
    } catch (error) {
      const errorMessage =
        error instanceof Error && error.message.trim()
          ? error.message
          : `${
              enrichmentType === "ocr" ? "OCR" : "Visual analysis"
            } failed.`

      try {
        const failureResponse = await tldwClient.persistWebClipEnrichment(
          clipId,
          {
            clip_id: clipId,
            enrichment_type: enrichmentType,
            status: "failed",
            error: errorMessage,
            structured_payload: {
              ...buildStructuredPayload(draft, enrichmentType),
              backend_path: "chat.completions",
              backend_model: resolveEnrichmentModel(selectedModel),
              error: errorMessage
            },
            source_note_version: currentVersion
          }
        )
        results[enrichmentType] = failureResponse
      } catch {
        results[enrichmentType] = createFailedEnrichmentResult({
          clipId,
          enrichmentType,
          sourceNoteVersion: currentVersion,
          errorMessage
        })
      }
    }
  }

  return results
}

export { buildPendingEnrichmentResults }

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

export const normalizePendingWebClipAnalyzeRequest = (
  raw: unknown
): PendingWebClipAnalyzeRequest | null => {
  if (!isRecord(raw)) return null

  const id = trimText(raw.id)
  const clipId = trimText(raw.clipId)
  const noteId = trimText(raw.noteId)
  const pageUrl = trimText(raw.pageUrl)
  const pageTitle = trimText(raw.pageTitle)
  const image = trimText(raw.image)
  const message = trimText(raw.message)
  const requestOverrides = isRecord(raw.requestOverrides)
    ? raw.requestOverrides
    : null
  const chatMode =
    trimText(requestOverrides?.chatMode) === "vision" ? "vision" : "normal"
  const useOCR = Boolean(requestOverrides?.useOCR)

  if (!id || !clipId || !noteId || !pageUrl || !pageTitle || !message) {
    return null
  }

  return {
    id,
    clipId,
    noteId,
    pageUrl,
    pageTitle,
    image,
    message,
    requestOverrides: {
      chatMode,
      useOCR
    }
  }
}

export const readPendingWebClipAnalyzeRequest =
  (): PendingWebClipAnalyzeRequest | null => {
    if (typeof window === "undefined") return null
    try {
      const raw = window.sessionStorage.getItem(
        WEB_CLIPPER_PENDING_ANALYZE_STORAGE_KEY
      )
      if (!raw) return null
      return normalizePendingWebClipAnalyzeRequest(JSON.parse(raw))
    } catch {
      return null
    }
  }

export const writePendingWebClipAnalyzeRequest = (
  request: PendingWebClipAnalyzeRequest
): void => {
  if (typeof window === "undefined") return
  try {
    window.sessionStorage.setItem(
      WEB_CLIPPER_PENDING_ANALYZE_STORAGE_KEY,
      JSON.stringify(request)
    )
  } catch {
    // ignore storage failures
  }
}

export const clearPendingWebClipAnalyzeRequest = (id?: string): void => {
  if (typeof window === "undefined") return
  if (id) {
    const current = readPendingWebClipAnalyzeRequest()
    if (current && current.id !== id) {
      return
    }
  }
  try {
    window.sessionStorage.removeItem(WEB_CLIPPER_PENDING_ANALYZE_STORAGE_KEY)
  } catch {
    // ignore storage failures
  }
}

export const buildPendingWebClipAnalyzeRequest = ({
  draft,
  clipId,
  noteId,
  useOCR
}: {
  draft: PendingClipDraft
  clipId: string
  noteId: string
  useOCR?: boolean
}): PendingWebClipAnalyzeRequest => {
  const clipContext = getClipContextText(draft).slice(0, MAX_ANALYZE_CONTEXT).trim()
  const messageParts = [
    "Analyze this web clip in the context of the captured page.",
    `Title: ${draft.pageTitle}`,
    `URL: ${draft.pageUrl}`
  ]

  if (clipContext) {
    messageParts.push(`Page text:\n${clipContext}`)
  }

  return {
    id: `web-clipper-analyze:${clipId}:${Date.now()}`,
    clipId,
    noteId,
    pageUrl: draft.pageUrl,
    pageTitle: draft.pageTitle,
    image: draft.captureMetadata.screenshotDataUrl || "",
    message: messageParts.join("\n\n"),
    requestOverrides: {
      chatMode: draft.captureMetadata.screenshotDataUrl ? "vision" : "normal",
      useOCR: Boolean(useOCR)
    }
  }
}
