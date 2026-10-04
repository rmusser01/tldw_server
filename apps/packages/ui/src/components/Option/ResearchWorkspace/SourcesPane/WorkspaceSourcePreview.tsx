import React from "react"
import { Button, Modal } from "antd"
import { AlertTriangle, Loader2, RefreshCw } from "lucide-react"
import { useTranslation } from "react-i18next"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { resolveServicePromptScope } from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import type { WorkspaceSourcePreviewResponse } from "@/services/tldw/domains/workspace-api"
import type { WorkspaceSource } from "@/types/workspace"
import { safeExternalUrl } from "@/utils/safe-external-url"
import { parseHttpOrigin } from "@/utils/absolute-url-guard"
import { useWorkspaceStore } from "@/store/workspace"

const SOURCE_PREVIEW_MAX_CHARS = 3000
const SOURCE_PREVIEW_CHUNK_LIMIT = 3

const describePreviewUnavailable = (
  preview: WorkspaceSourcePreviewResponse | null,
  t: (key: string, fallback: string) => string
): string => {
  const reason = preview?.unavailable_reason || preview?.status_reason || ""
  if (reason === "extraction_pending" || preview?.preview_mode === "pending") {
    return t(
      "playground:sources.previewExtractionPending",
      "Text extraction has not completed yet."
    )
  }
  if (
    reason === "media_not_found" ||
    preview?.preview_mode === "missing_media"
  ) {
    return t(
      "playground:sources.previewMediaMissing",
      "Media item is missing or unavailable."
    )
  }
  if (preview?.preview_mode === "failed" || reason.includes("failed")) {
    return t(
      "playground:sources.previewExtractionFailed",
      "Source extraction or indexing failed. Preview content is unavailable."
    )
  }
  return t(
    "playground:sources.previewNoTextAvailable",
    "No captured text is available for this source."
  )
}

type PreviewRequest = {
  workspaceId: string
  sourceId: string
  mediaId: number | null
  generation: number
  controller: AbortController
}

type PreviewLoadState = {
  request: PreviewRequest
  loading: boolean
  error: string | null
  data: WorkspaceSourcePreviewResponse | null
}

type WorkspaceSourcePreviewProps = {
  workspaceId: string | null
  source: WorkspaceSource | null
  onClose: () => void
  title?: string
  statusGuardrailsEnabled?: boolean
  readyStateLabel?: string
  children?: React.ReactNode
}

export const WorkspaceSourcePreview = ({
  workspaceId,
  source: previewSource,
  onClose,
  title,
  statusGuardrailsEnabled = true,
  readyStateLabel = "Ready",
  children
}: WorkspaceSourcePreviewProps) => {
  const { t } = useTranslation(["playground", "common"])
  const capturedWorkspaceId = workspaceId?.trim() || null
  const sourceId = previewSource?.id ?? null
  const mediaId = previewSource?.mediaId ?? null
  const [previewReloadNonce, setPreviewReloadNonce] = React.useState(0)
  const [loadState, setLoadState] = React.useState<PreviewLoadState | null>(
    null
  )
  const [dismissed, setDismissed] = React.useState(false)
  const generation = React.useRef(0)
  const activeRequest = React.useRef<PreviewRequest | null>(null)
  const currentTarget = React.useRef({
    workspaceId: capturedWorkspaceId,
    sourceId,
    mediaId
  })
  const closeCallback = React.useRef(onClose)
  const previousWorkspace = React.useRef(capturedWorkspaceId)
  currentTarget.current = {
    workspaceId: capturedWorkspaceId,
    sourceId,
    mediaId
  }
  closeCallback.current = onClose

  const dismissPreview = React.useCallback(() => {
    generation.current += 1
    activeRequest.current?.controller.abort()
    activeRequest.current = null
    setLoadState(null)
    setDismissed(true)
    closeCallback.current()
  }, [])

  React.useEffect(
    () =>
      watchChatAccountChanges((invalidated) => {
        if (invalidated) dismissPreview()
      }),
    [dismissPreview]
  )

  React.useEffect(() => {
    const workspaceChanged = previousWorkspace.current !== capturedWorkspaceId
    previousWorkspace.current = capturedWorkspaceId
    if (!capturedWorkspaceId || !sourceId) {
      setLoadState(null)
      setDismissed(false)
      return
    }
    if (workspaceChanged) {
      dismissPreview()
      return
    }
    const request: PreviewRequest = {
      workspaceId: capturedWorkspaceId,
      sourceId,
      mediaId,
      generation: ++generation.current,
      controller: new AbortController()
    }
    activeRequest.current = request
    setDismissed(false)
    setLoadState({ request, loading: true, error: null, data: null })
    // Check the captured target before dispatch and every completion, including ABA.
    const matchesRenderedTarget = () =>
      currentTarget.current.workspaceId === request.workspaceId &&
      currentTarget.current.sourceId === request.sourceId &&
      currentTarget.current.mediaId === request.mediaId
    const isCurrent = () => {
      const state = useWorkspaceStore.getState()
      const source = state.sources.find((item) => item.id === request.sourceId)
      return (
        activeRequest.current === request &&
        generation.current === request.generation &&
        !request.controller.signal.aborted &&
        matchesRenderedTarget() &&
        state.workspaceId?.trim() === request.workspaceId &&
        source &&
        (source.mediaId ?? null) === request.mediaId
      )
    }
    // Store subscribers see intervening transitions even when React batches A -> B -> A.
    const stopWorkspace = useWorkspaceStore.subscribe((state, previous) => {
      const source = state.sources.find((item) => item.id === request.sourceId)
      if (
        state.workspaceId !== previous.workspaceId ||
        !source ||
        (source.mediaId ?? null) !== request.mediaId
      ) {
        if (activeRequest.current !== request) return
        if (matchesRenderedTarget()) dismissPreview()
        else request.controller.abort()
      }
    })
    const loadPreview = async () => {
      try {
        if (!isCurrent()) return
        const scope = await resolveServicePromptScope({
          signal: request.controller.signal
        })
        if (!isCurrent()) return
        if (typeof tldwClient.getWorkspaceSourcePreview !== "function") {
          throw new Error("Source preview API is unavailable.")
        }
        const data = await tldwClient.getWorkspaceSourcePreview(
          request.workspaceId,
          request.sourceId,
          {
            max_chars: SOURCE_PREVIEW_MAX_CHARS,
            chunk_limit: SOURCE_PREVIEW_CHUNK_LIMIT
          },
          { requestScope: scope, signal: request.controller.signal }
        )
        if (!isCurrent()) return
        if (
          data.workspace_id !== request.workspaceId ||
          data.source_id !== request.sourceId ||
          (request.mediaId !== null &&
            request.mediaId > 0 &&
            data.media_id !== request.mediaId) ||
          !Array.isArray(data.snippets) ||
          data.snippets.some(
            (snippet) =>
              snippet.source_id !== request.sourceId ||
              snippet.media_id !== data.media_id
          )
        ) {
          throw new Error(
            "Source preview did not match the captured workspace source."
          )
        }
        setLoadState({ request, loading: false, error: null, data })
      } catch (error) {
        if (!isCurrent()) return
        if (isRequestConfigScopeChangedError(error)) {
          dismissPreview()
          return
        }
        setLoadState({
          request,
          loading: false,
          data: null,
          error:
            error instanceof Error
              ? error.message
              : "Source preview could not load."
        })
      }
    }
    void loadPreview()
    return () => {
      stopWorkspace()
      request.controller.abort()
      if (activeRequest.current === request) activeRequest.current = null
    }
  }, [
    capturedWorkspaceId,
    sourceId,
    mediaId,
    previewReloadNonce,
    dismissPreview
  ])

  const loadMatchesTarget = Boolean(
    loadState &&
    activeRequest.current === loadState.request &&
    loadState.request.generation === generation.current &&
    loadState.request.workspaceId === capturedWorkspaceId &&
    loadState.request.sourceId === sourceId &&
    loadState.request.mediaId === mediaId &&
    !loadState.request.controller.signal.aborted
  )

  return (
    <Modal
      open={Boolean(capturedWorkspaceId && previewSource && !dismissed)}
      title={title ?? t("playground:sharedWorkspace.preview", "Source preview")}
      onCancel={dismissPreview}
      footer={null}
      width={680}
    >
      {previewSource &&
        !dismissed &&
        (() => {
          const previewStatus = statusGuardrailsEnabled
            ? previewSource.status || "ready"
            : "ready"
          const previewStatusLabel =
            previewStatus === "processing"
              ? t("playground:sources.statusProcessing", "Processing")
              : previewStatus === "error"
                ? t("playground:sources.statusErrorShort", "Error")
                : t("playground:sources.statusReady", readyStateLabel)
          const previewData = loadMatchesTarget ? loadState?.data : null
          const previewLoading =
            !loadMatchesTarget || Boolean(loadState?.loading)
          const previewError = loadMatchesTarget ? loadState?.error : null
          const sourcePreviewSnippets =
            previewData?.snippets?.filter(
              (snippet) => snippet.kind === "chunk" && snippet.text?.trim()
            ) || []
          const previewTotalChars = previewData?.text_total_chars
          const previewTruncated = Boolean(previewData?.text_truncated)
          const formattedPreviewTotalChars =
            typeof previewTotalChars === "number"
              ? previewTotalChars.toLocaleString()
              : null
          const formattedPreviewShownChars = previewData?.text_preview
            ? previewData.text_preview.length.toLocaleString()
            : null
          const previewCharacterSummary =
            formattedPreviewTotalChars && previewTruncated
              ? t(
                  "playground:sources.previewTruncatedSummary",
                  "Showing first {{shown}} of {{total}} characters.",
                  {
                    shown: formattedPreviewShownChars,
                    total: formattedPreviewTotalChars
                  }
                )
              : formattedPreviewTotalChars
                ? t(
                    "playground:sources.previewFullSummary",
                    "Showing {{total}} characters.",
                    {
                      total: formattedPreviewTotalChars
                    }
                  )
                : null
          const candidateUrl = safeExternalUrl(
            previewData ? previewData.url : previewSource.url
          )
          const previewSafeUrl =
            candidateUrl && /^https?:\/\//i.test(candidateUrl) && parseHttpOrigin(candidateUrl)
              ? candidateUrl
              : null

          return (
            <div className="space-y-4">
              <div className="rounded border border-border bg-surface2/40 p-3">
                <p className="text-sm font-semibold text-text">
                  {previewData ? previewData.title : previewSource.title}
                </p>
                <p className="text-xs capitalize text-text-muted">
                  {previewData ? previewData.source_type : previewSource.type} / {previewStatusLabel}
                </p>
                {previewSafeUrl && (
                  <a
                    href={previewSafeUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-1 inline-block break-all text-xs text-primary hover:underline"
                  >
                    {previewSafeUrl}
                  </a>
                )}
              </div>

              <div className="rounded border border-border bg-surface/50 p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <p className="text-xs font-semibold uppercase text-text-muted">
                    {t(
                      "playground:sources.capturedContent",
                      "Captured content"
                    )}
                  </p>
                  {previewData?.readiness?.citation_ready && (
                    <span className="rounded bg-success/10 px-2 py-0.5 text-[11px] font-medium text-success">
                      {t("playground:sources.citationReady", "Citation ready")}
                    </span>
                  )}
                </div>
                {previewLoading ? (
                  <div
                    role="status"
                    aria-live="polite"
                    aria-atomic="true"
                    className="flex items-center gap-2 text-sm text-text-muted"
                  >
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    {t(
                      "playground:sources.previewLoading",
                      "Loading captured content..."
                    )}
                  </div>
                ) : previewError ? (
                  <div className="space-y-2">
                    <div role="alert" aria-live="assertive" aria-atomic="true" className="space-y-2">
                      <div className="flex items-start gap-2 text-sm text-warning">
                        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                        <span>
                          {t(
                            "playground:sources.previewLoadError",
                            "Source preview could not load."
                          )}
                        </span>
                      </div>
                      <p className="break-words rounded border border-border bg-surface2/40 p-2 font-mono text-xs text-warning">
                        {previewError}
                      </p>
                    </div>
                    <Button
                      size="small"
                      icon={<RefreshCw className="h-3.5 w-3.5" />}
                      onClick={() =>
                        setPreviewReloadNonce((value) => value + 1)
                      }
                    >
                      {t("playground:sources.retryPreview", "Retry preview")}
                    </Button>
                  </div>
                ) : previewData?.content_available &&
                  previewData.text_preview ? (
                  <div className="space-y-2">
                    <p className="max-h-52 overflow-y-auto whitespace-pre-wrap rounded border border-border bg-surface2/40 p-2 text-sm leading-6 text-text">
                      {previewData.text_preview}
                    </p>
                    {previewCharacterSummary && (
                      <p className="text-xs text-text-muted">
                        {previewCharacterSummary}
                      </p>
                    )}
                  </div>
                ) : (
                  <div
                    role="status"
                    aria-live="polite"
                    aria-atomic="true"
                    className="rounded border border-border bg-surface2/40 p-2 text-sm text-text-muted"
                  >
                    {describePreviewUnavailable(previewData, t)}
                  </div>
                )}
              </div>

              <div className="rounded border border-border bg-surface/50 p-3">
                <p className="mb-2 text-xs font-semibold uppercase text-text-muted">
                  {t(
                    "playground:sources.evidenceSnippets",
                    "Evidence snippets"
                  )}
                </p>
                {sourcePreviewSnippets.length === 0 ? (
                  <p className="text-xs text-text-muted">
                    {t(
                      "playground:sources.noEvidenceSnippets",
                      "No chunk evidence is available yet."
                    )}
                  </p>
                ) : (
                  <div className="max-h-48 space-y-2 overflow-y-auto pr-1">
                    {sourcePreviewSnippets.map((snippet) => (
                      <div
                        key={snippet.id}
                        className="rounded border border-border bg-surface2/40 p-2"
                      >
                        <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-text-muted">
                          <span>
                            {snippet.kind === "chunk"
                              ? t("playground:sources.chunkLabel", "Chunk")
                              : t(
                                  "playground:sources.contentExcerptLabel",
                                  "Content excerpt"
                                )}
                            {typeof snippet.chunk_index === "number"
                              ? ` ${snippet.chunk_index}`
                              : ""}
                          </span>
                          {typeof snippet.start_char === "number" &&
                            typeof snippet.end_char === "number" && (
                              <span>
                                {snippet.start_char}-{snippet.end_char}
                              </span>
                            )}
                        </div>
                        <p className="whitespace-pre-wrap text-sm text-text">
                          {snippet.text}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {children}
            </div>
          )
        })()}
    </Modal>
  )
}
