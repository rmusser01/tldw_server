import React from "react"
import { useTranslation } from "react-i18next"
import ClipDestinationFields from "./ClipDestinationFields"
import ClipEnhancementFields from "./ClipEnhancementFields"
import ClipPreview from "./ClipPreview"
import {
  clearPendingClipDraft,
  type PendingClipDraft
} from "@/services/web-clipper/pending-draft"
import {
  buildPendingEnrichmentResults,
  buildPendingWebClipAnalyzeRequest,
  persistRequestedWebClipEnrichments,
  writePendingWebClipAnalyzeRequest,
  type WebClipperEnrichmentRunResult
} from "@/services/web-clipper/enrichment"
import { buildWebClipSaveRuntime } from "@/services/web-clipper/save-runtime"
import type {
  WebClipperDestination,
  WebClipperSaveRequest,
  WebClipperSaveResponse
} from "@/services/web-clipper/types"
import { tldwClient } from "@/services/tldw/TldwApiClient"

type WebClipperPanelProps = {
  draft: PendingClipDraft
  onCancel: () => void
}

type SubmitAction = "save" | "open" | "analyze"

type EnrichmentStatusTone = "success" | "warning" | "error"

type EnrichmentStatusItem = {
  key: string
  label: string
  tone: EnrichmentStatusTone
}

const splitKeywords = (value: string): string[] =>
  value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)

const SCREENSHOT_ATTACHMENT_SLOT = "page-screenshot"
const BASE64_DATA_URL_PATTERN = /^data:([^;,]+);base64,([a-z0-9+/=\s]+)$/i

const buildClipAttachments = (
  draft: PendingClipDraft
): NonNullable<WebClipperSaveRequest["attachments"]> => {
  const screenshotDataUrl = draft.captureMetadata.screenshotDataUrl?.trim()
  if (!screenshotDataUrl) {
    return []
  }

  const matched = screenshotDataUrl.match(BASE64_DATA_URL_PATTERN)
  if (!matched) {
    return []
  }

  const [, mediaType, rawContentBase64] = matched
  const contentBase64 = rawContentBase64.trim()
  if (!contentBase64) {
    return []
  }

  const mediaSubtype = mediaType.split("/")[1] || "bin"
  const fileExtension = mediaSubtype.split("+")[0] || "bin"

  return [
    {
      slot: SCREENSHOT_ATTACHMENT_SLOT,
      file_name: `${SCREENSHOT_ATTACHMENT_SLOT}.${fileExtension}`,
      media_type: mediaType,
      content_base64: contentBase64,
      source_url: draft.pageUrl
    }
  ]
}

const createOpenTargetUrl = (
  destinationMode: WebClipperDestination,
  response: WebClipperSaveResponse
): string | null => {
  const chromeApi = globalThis.chrome
  if (!chromeApi?.runtime?.getURL) {
    return null
  }

  if (response.status === "failed") {
    return null
  }

  if (destinationMode === "workspace" && response.workspace_placement) {
    return chromeApi.runtime.getURL("options.html#/document-workspace")
  }

  return chromeApi.runtime.getURL("options.html#/notes")
}

const parseOptionalFolderId = (value: string): number | null => {
  const trimmedValue = value.trim()
  if (!trimmedValue) return null
  const parsedValue = Number(trimmedValue)
  if (!Number.isInteger(parsedValue) || parsedValue < 1) {
    return null
  }
  return parsedValue
}

const hasSavedCanonicalNote = (response: WebClipperSaveResponse): boolean =>
  response.status !== "failed" && response.note != null

const buildEnrichmentStatusItem = ({
  enrichmentType,
  inlineApplied,
  conflictReason,
  status
}: {
  enrichmentType: "ocr" | "vlm"
  inlineApplied: boolean
  conflictReason?: string | null
  status: string
}): EnrichmentStatusItem => {
  const labelPrefix = enrichmentType === "ocr" ? "OCR" : "Visual analysis"

  if (status === "pending" || status === "running") {
    return {
      key: enrichmentType,
      label: `${labelPrefix} pending`,
      tone: "warning"
    }
  }

  if (status === "failed") {
    return {
      key: enrichmentType,
      label: `${labelPrefix} failed`,
      tone: "error"
    }
  }

  if (!inlineApplied && conflictReason === "source_note_version_mismatch") {
    return {
      key: enrichmentType,
      label: `${labelPrefix} needs refresh`,
      tone: "warning"
    }
  }

  return {
    key: enrichmentType,
    label: `${labelPrefix} complete`,
    tone: "success"
  }
}

const WebClipperPanel = ({ draft, onCancel }: WebClipperPanelProps) => {
  const { t } = useTranslation()
  const [title, setTitle] = React.useState(draft.pageTitle)
  const [comment, setComment] = React.useState("")
  const [tags, setTags] = React.useState("")
  const [folderId, setFolderId] = React.useState("")
  const [destinationMode, setDestinationMode] =
    React.useState<WebClipperDestination>("note")
  const [workspaceId, setWorkspaceId] = React.useState("")
  const [runOcr, setRunOcr] = React.useState(false)
  const [runVlm, setRunVlm] = React.useState(false)
  const [folderValidation, setFolderValidation] =
    React.useState<string | null>(null)
  const [workspaceValidation, setWorkspaceValidation] =
    React.useState<string | null>(null)
  const [submissionError, setSubmissionError] =
    React.useState<string | null>(null)
  const [saveRuntime, setSaveRuntime] = React.useState<ReturnType<
    typeof buildWebClipSaveRuntime
  > | null>(null)
  const [enrichmentResults, setEnrichmentResults] =
    React.useState<WebClipperEnrichmentRunResult | null>(null)
  const [activeAction, setActiveAction] =
    React.useState<SubmitAction | null>(null)
  const [isSaving, setIsSaving] = React.useState(false)
  const isMountedRef = React.useRef(true)
  const activeClipIdRef = React.useRef(draft.clipId)

  React.useEffect(() => {
    activeClipIdRef.current = draft.clipId
    setTitle(draft.pageTitle)
    setComment("")
    setTags("")
    setFolderId("")
    setDestinationMode("note")
    setWorkspaceId("")
    setRunOcr(false)
    setRunVlm(false)
    setFolderValidation(null)
    setWorkspaceValidation(null)
    setSubmissionError(null)
    setSaveRuntime(null)
    setEnrichmentResults(null)
    setActiveAction(null)
    setIsSaving(false)
  }, [draft.clipId, draft.pageTitle])

  React.useEffect(
    () => () => {
      isMountedRef.current = false
    },
    []
  )

  const enrichmentStatuses = React.useMemo<EnrichmentStatusItem[]>(() => {
    if (!enrichmentResults) return []

    const statuses: EnrichmentStatusItem[] = []

    ;(["ocr", "vlm"] as const).forEach((enrichmentType) => {
      const result = enrichmentResults[enrichmentType]
      if (!result) return

      statuses.push(
        buildEnrichmentStatusItem({
          enrichmentType,
          inlineApplied: Boolean(result.inline_applied),
          conflictReason: result.conflict_reason,
          status: result.status
        })
      )
    })

    return statuses
  }, [enrichmentResults])

  const submitSave = async (action: SubmitAction) => {
    const submittedClipId = draft.clipId
    const trimmedWorkspaceId = workspaceId.trim()
    const needsWorkspace =
      destinationMode === "workspace" || destinationMode === "both"
    const parsedFolderId = parseOptionalFolderId(folderId)
    const hasInvalidFolderId = Boolean(folderId.trim()) && parsedFolderId == null

    setSaveRuntime(null)
    setEnrichmentResults(null)
    setFolderValidation(null)
    setWorkspaceValidation(null)
    setSubmissionError(null)

    if (hasInvalidFolderId) {
      setFolderValidation(
        t(
          "sidepanel:clipper.folderValidation",
          "Choose a positive whole-number folder ID or leave it blank."
        )
      )
      return
    }

    if (needsWorkspace && !trimmedWorkspaceId) {
      setWorkspaceValidation(
        t(
          "sidepanel:clipper.workspaceValidation",
          "Choose a workspace before saving to Workspace or Both."
        )
      )
      return
    }

    setActiveAction(action)
    setIsSaving(true)

    const payload: WebClipperSaveRequest = {
      clip_id: draft.clipId,
      clip_type: draft.captureMetadata.actualType,
      source_url: draft.pageUrl,
      source_title: draft.pageTitle,
      destination_mode: destinationMode,
      note: {
        title: title.trim() || draft.pageTitle,
        comment: comment.trim() || null,
        folder_id: parsedFolderId,
        keywords: splitKeywords(tags)
      },
      workspace: needsWorkspace
        ? { workspace_id: trimmedWorkspaceId }
        : null,
      content: {
        visible_body: draft.visibleBody,
        full_extract: draft.fullExtract || draft.visibleBody,
        selected_text: draft.selectionText || null
      },
      attachments: buildClipAttachments(draft),
      enhancements: {
        run_ocr: runOcr,
        run_vlm: runVlm
      },
      capture_metadata: {
        requested_type: draft.requestedType,
        actual_type: draft.captureMetadata.actualType,
        fallback_path: draft.captureMetadata.fallbackPath,
        captured_at: draft.capturedAt
      }
    }

    try {
      await tldwClient.initialize().catch(() => undefined)
      const response = await tldwClient.saveWebClip(payload)
      if (
        !isMountedRef.current ||
        activeClipIdRef.current !== submittedClipId
      ) {
        return
      }
      setSaveRuntime(buildWebClipSaveRuntime(response))

      if (hasSavedCanonicalNote(response)) {
        clearPendingClipDraft(draft.clipId)
      }

      if (hasSavedCanonicalNote(response) && response.note && (runOcr || runVlm)) {
        setEnrichmentResults(
          buildPendingEnrichmentResults({
            clipId: response.clip_id,
            sourceNoteVersion: response.note.version,
            runOcr,
            runVlm
          })
        )

        void persistRequestedWebClipEnrichments({
          draft,
          clipId: response.clip_id,
          sourceNoteVersion: response.note.version,
          runOcr,
          runVlm
        })
          .then((nextEnrichmentResults) => {
            if (
              !isMountedRef.current ||
              activeClipIdRef.current !== submittedClipId
            ) {
              return
            }
            setEnrichmentResults((currentResults) => ({
              ...(currentResults || {}),
              ...nextEnrichmentResults
            }))
          })
          .catch((error) => {
            if (
              !isMountedRef.current ||
              activeClipIdRef.current !== submittedClipId
            ) {
              return
            }
            setSubmissionError(
              error instanceof Error && error.message.trim()
                ? error.message
                : t(
                    "sidepanel:clipper.enrichmentFailed",
                    "The clip was saved, but clip analysis could not be stored."
                  )
            )
          })
      }

      if (action === "analyze" && hasSavedCanonicalNote(response) && response.note) {
        writePendingWebClipAnalyzeRequest(
          buildPendingWebClipAnalyzeRequest({
            draft,
            clipId: response.clip_id,
            noteId: response.note.id,
            useOCR: runOcr
          })
        )
        const navigate = (
          globalThis as typeof globalThis & {
            __tldwNavigate?: (path: string) => void
          }
        ).__tldwNavigate
        if (typeof navigate === "function") {
          navigate("/chat")
        }
      }

      if (action === "open") {
        const url = createOpenTargetUrl(destinationMode, response)
        if (url) {
          globalThis.chrome?.tabs?.create?.({ url })
        }
      }
    } catch (error) {
      if (
        !isMountedRef.current ||
        activeClipIdRef.current !== submittedClipId
      ) {
        return
      }
      setSubmissionError(
        error instanceof Error && error.message.trim()
          ? error.message
          : t("sidepanel:clipper.saveFailed", "The clip could not be saved.")
      )
    } finally {
      if (
        !isMountedRef.current ||
        activeClipIdRef.current !== submittedClipId
      ) {
        return
      }
      setActiveAction(null)
      setIsSaving(false)
    }
  }

  const bannerClasses =
    saveRuntime?.banner.severity === "success"
      ? "border-emerald-200 bg-emerald-50 text-emerald-900"
      : saveRuntime?.banner.severity === "warning"
        ? "border-amber-200 bg-amber-50 text-amber-900"
        : "border-red-200 bg-red-50 text-red-900"

  return (
    <div className="space-y-3">
      <section className="panel-card p-3">
        <div className="grid gap-3">
          <div className="space-y-1">
            <label className="block text-sm font-medium text-text" htmlFor="clip-title">
              {t("sidepanel:clipper.titleLabel", "Title")}
            </label>
            <input
              id="clip-title"
              type="text"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-text"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-sm font-medium text-text" htmlFor="clip-comment">
              {t("sidepanel:clipper.commentLabel", "Comment")}
            </label>
            <textarea
              id="clip-comment"
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              className="min-h-20 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-text"
            />
          </div>

          <div className="space-y-1">
            <label className="block text-sm font-medium text-text" htmlFor="clip-tags">
              {t("sidepanel:clipper.tagsLabel", "Tags")}
            </label>
            <input
              id="clip-tags"
              type="text"
              value={tags}
              onChange={(event) => setTags(event.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-text"
              placeholder={t(
                "sidepanel:clipper.tagsPlaceholder",
                "research, article"
              )}
            />
          </div>
        </div>
      </section>

      <ClipDestinationFields
        destinationMode={destinationMode}
        folderId={folderId}
        folderValidation={folderValidation}
        workspaceId={workspaceId}
        workspaceValidation={workspaceValidation}
        onDestinationChange={(nextValue) => {
          setDestinationMode(nextValue)
          if (nextValue === "note") {
            setWorkspaceValidation(null)
            return
          }
          if (nextValue === "workspace") {
            setFolderId("")
            setFolderValidation(null)
          }
        }}
        onFolderIdChange={(nextValue) => {
          setFolderId(nextValue)
          if (!nextValue.trim()) {
            setFolderValidation(null)
            return
          }
          if (parseOptionalFolderId(nextValue) != null) {
            setFolderValidation(null)
          }
        }}
        onWorkspaceIdChange={(nextValue) => {
          setWorkspaceId(nextValue)
          if (nextValue.trim()) {
            setWorkspaceValidation(null)
          }
        }}
      />

      <ClipEnhancementFields
        runOcr={runOcr}
        runVlm={runVlm}
        onRunOcrChange={setRunOcr}
        onRunVlmChange={setRunVlm}
      />

      <ClipPreview draft={draft} />

      {saveRuntime ? (
        <section className={`rounded-xl border px-3 py-2 ${bannerClasses}`}>
          <p className="text-sm font-semibold">{saveRuntime.banner.title}</p>
          <p className="mt-1 text-sm">{saveRuntime.banner.message}</p>
          {saveRuntime.banner.warnings.length > 0 ? (
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
              {saveRuntime.banner.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}

      {enrichmentStatuses.length > 0 ? (
        <section className="panel-card p-3">
          <ul className="space-y-2">
            {enrichmentStatuses.map((status) => (
              <li
                key={status.key}
                className={
                  status.tone === "success"
                    ? "text-sm font-semibold text-emerald-900"
                    : status.tone === "warning"
                      ? "text-sm font-semibold text-amber-900"
                      : "text-sm font-semibold text-red-900"
                }
              >
                {status.label}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {submissionError ? (
        <section className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-900">
          {submissionError}
        </section>
      ) : null}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={onCancel}
          disabled={isSaving}
          className="rounded-lg border border-border bg-background px-3 py-2 text-sm font-semibold text-text disabled:cursor-not-allowed disabled:opacity-60"
        >
          {t("sidepanel:clipper.cancelLabel", "Cancel")}
        </button>

        <button
          type="button"
          onClick={() => void submitSave("save")}
          disabled={isSaving}
          className="flex-1 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isSaving && activeAction === "save"
            ? t("sidepanel:clipper.savingLabel", "Saving...")
            : t("sidepanel:clipper.saveLabel", "Save clip")}
        </button>

        <button
          type="button"
          onClick={() => void submitSave("analyze")}
          disabled={isSaving}
          className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm font-semibold text-text disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isSaving && activeAction === "analyze"
            ? t("sidepanel:clipper.savingLabel", "Saving...")
            : t("sidepanel:clipper.analyzeNowLabel", "Analyze now")}
        </button>

        <button
          type="button"
          onClick={() => void submitSave("open")}
          disabled={isSaving}
          className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm font-semibold text-text disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isSaving && activeAction === "open"
            ? t("sidepanel:clipper.savingLabel", "Saving...")
            : t("sidepanel:clipper.saveAndOpenLabel", "Save and open")}
        </button>
      </div>
    </div>
  )
}

export default WebClipperPanel
