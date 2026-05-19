import {
  resolveClipCaptureResolution,
  type ClipCaptureInput,
  type ClipCaptureFallbackStep,
  type ClipCaptureType
} from "./content-extract"

export type ClipDraft = {
  clipId: string
  requestedType: ClipCaptureType
  clipType: ClipCaptureType
  pageUrl: string
  pageTitle: string
  visibleBody: string
  fullExtract?: string
  selectionText?: string
  captureMetadata: {
    clipType: ClipCaptureType
    actualType: ClipCaptureType
    fallbackPath: ClipCaptureFallbackStep[]
    screenshotDataUrl?: string
  }
  capturedAt: string
  userVisibleError?: string
}

export type BuildClipDraftInput = {
  requestedType: ClipCaptureType
  pageUrl: string
  pageTitle: string
  clipId?: string
  extracted: {
    selectionText?: string
    articleText?: string
    fullPageText?: string
    screenshotDataUrl?: string
  }
}

const createClipId = (): string =>
  typeof globalThis.crypto?.randomUUID === "function"
    ? globalThis.crypto.randomUUID()
    : `clip-${Date.now()}`

const createCapturedAt = (): string => new Date().toISOString()

const resolveFullExtract = (input: BuildClipDraftInput, visibleBody: string): string | undefined => {
  const fullPageText = input.extracted.fullPageText?.trim()
  if (fullPageText) {
    return fullPageText
  }

  const articleText = input.extracted.articleText?.trim()
  if (articleText) {
    return articleText
  }

  const selectionText = input.extracted.selectionText?.trim()
  if (selectionText) {
    return selectionText
  }

  return visibleBody.trim() || undefined
}

export const buildClipDraft = (input: BuildClipDraftInput): ClipDraft => {
  const capture = resolveClipCaptureResolution({
    requestedType: input.requestedType,
    pageUrl: input.pageUrl,
    pageTitle: input.pageTitle,
    selectionText: input.extracted.selectionText,
    articleText: input.extracted.articleText,
    fullPageText: input.extracted.fullPageText,
    screenshotDataUrl: input.extracted.screenshotDataUrl
  })

  return {
    clipId: input.clipId || createClipId(),
    requestedType: input.requestedType,
    clipType: capture.clipType,
    pageUrl: input.pageUrl,
    pageTitle: input.pageTitle,
    visibleBody: capture.visibleBody,
    fullExtract: resolveFullExtract(input, capture.visibleBody),
    selectionText: input.extracted.selectionText?.trim() || undefined,
    captureMetadata: {
      clipType: capture.clipType,
      actualType: capture.actualType,
      fallbackPath: capture.fallbackPath,
      screenshotDataUrl: capture.screenshotDataUrl
    },
    capturedAt: createCapturedAt(),
    userVisibleError: capture.userVisibleError
  }
}

export const normalizeClipDraft = (
  raw: unknown
): ClipDraft | null => {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null
  const draft = raw as Partial<ClipDraft> & {
    requestedType?: ClipCaptureType
    clipType?: ClipCaptureType
    pageUrl?: string
    pageTitle?: string
    visibleBody?: string
    fullExtract?: string
    captureMetadata?: {
      clipType?: ClipCaptureType
      actualType?: ClipCaptureType
      fallbackPath?: ClipCaptureFallbackStep[]
      screenshotDataUrl?: string
    }
  }
  const clipId = String(draft.clipId || "").trim()
  const requestedType = draft.requestedType
  const clipType = draft.clipType
  const pageUrl = String(draft.pageUrl || "").trim()
  const pageTitle = String(draft.pageTitle || "").trim()
  const visibleBody = String(draft.visibleBody || "").trim()
  const userVisibleError =
    typeof draft.userVisibleError === "string" &&
    draft.userVisibleError.trim()
      ? draft.userVisibleError.trim()
      : undefined
  const fallbackPath = Array.isArray(draft.captureMetadata?.fallbackPath)
    ? draft.captureMetadata?.fallbackPath.filter(
        (value): value is ClipCaptureFallbackStep =>
          value === "blocked" ||
          (typeof value === "string" && value.length > 0)
      )
    : []

  if (
    !clipId ||
    !requestedType ||
    !clipType ||
    !pageUrl ||
    !pageTitle ||
    (!visibleBody && !userVisibleError) ||
    fallbackPath.length === 0
  ) {
    return null
  }

  return {
    clipId,
    requestedType,
    clipType,
    pageUrl,
    pageTitle,
    visibleBody,
    fullExtract:
      typeof draft.fullExtract === "string" && draft.fullExtract.trim()
        ? draft.fullExtract.trim()
        : undefined,
    selectionText:
      typeof draft.selectionText === "string" && draft.selectionText.trim()
        ? draft.selectionText.trim()
        : undefined,
    captureMetadata: {
      clipType: draft.captureMetadata?.clipType || clipType,
      actualType: draft.captureMetadata?.actualType || clipType,
      fallbackPath,
      screenshotDataUrl:
        draft.captureMetadata?.screenshotDataUrl || undefined
    },
    capturedAt:
      typeof draft.capturedAt === "string" && draft.capturedAt.trim()
        ? draft.capturedAt.trim()
        : createCapturedAt(),
    userVisibleError
  }
}

export const buildClipDraftFromCapture = (
  input: BuildClipDraftInput
): ClipDraft => buildClipDraft(input)

export type ClipCaptureInputLike = ClipCaptureInput
