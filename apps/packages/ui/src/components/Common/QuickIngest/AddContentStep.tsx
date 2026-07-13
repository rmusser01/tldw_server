import React, { useCallback, useMemo, useState } from "react"
import { Button, Input, Tooltip, Typography } from "antd"
import { useTranslation } from "react-i18next"
import {
  AlertTriangle,
  FileText,
  Film,
  Globe,
  Music,
  Image as ImageIcon,
  BookOpen,
  File as FileIcon,
  X,
  Plus,
} from "lucide-react"
import type {
  DetectedMediaType,
  WizardQueueItem,
  QueueItemValidation,
} from "./types"
import { useIngestWizard } from "./IngestWizardContext"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { Alert as DesignSystemAlert, Badge } from "@/components/ui/primitives"
import { FileDropZone } from "./QueueTab/FileDropZone"
import { BatchMetadataPanel } from "./BatchMetadataPanel"
import {
  usePlaylistInspection,
  type PlaylistInspectionCandidate,
} from "./usePlaylistInspection"
import {
  QUICK_INGEST_MAX_FILE_SIZE_LABEL,
  QUICK_INGEST_MAX_FILE_SIZE,
} from "./constants"
import { normalizeUrlForDedupe } from "@/entries/shared/ingest-payloads"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MEDIA_TYPE_ICONS: Record<DetectedMediaType, React.ReactNode> = {
  audio: <Music className="h-4 w-4 text-purple-500" aria-hidden="true" />,
  video: <Film className="h-4 w-4 text-blue-500" aria-hidden="true" />,
  document: <FileText className="h-4 w-4 text-green-500" aria-hidden="true" />,
  pdf: <FileText className="h-4 w-4 text-red-500" aria-hidden="true" />,
  ebook: <BookOpen className="h-4 w-4 text-amber-500" aria-hidden="true" />,
  image: <ImageIcon className="h-4 w-4 text-pink-500" aria-hidden="true" />,
  web: <Globe className="h-4 w-4 text-cyan-500" aria-hidden="true" />,
  unknown: <FileIcon className="h-4 w-4 text-text-muted" aria-hidden="true" />,
}

const ICON_NAME_MAP: Record<DetectedMediaType, string> = {
  audio: "Music",
  video: "Film",
  document: "FileText",
  pdf: "FileText",
  ebook: "BookOpen",
  image: "Image",
  web: "Globe",
  unknown: "File",
}

const hostnameMatches = (hostname: string, allowedHost: string): boolean =>
  hostname === allowedHost || hostname.endsWith(`.${allowedHost}`)

const detectTypeFromExtension = (name: string): DetectedMediaType => {
  const ext = name.split(".").pop()?.toLowerCase() || ""
  if (["mp3", "wav", "ogg", "flac", "m4a", "aac", "wma", "opus"].includes(ext)) return "audio"
  if (["mp4", "mkv", "avi", "mov", "webm", "wmv", "flv", "m4v"].includes(ext)) return "video"
  if (["pdf"].includes(ext)) return "pdf"
  if (["epub"].includes(ext)) return "ebook"
  if (["docx", "txt", "rtf", "md", "markdown", "html", "htm", "xhtml", "xml", "json"].includes(ext)) return "document"
  return "unknown"
}

const detectTypeFromMime = (mimeType: string | undefined): DetectedMediaType => {
  const normalized = String(mimeType || "").trim().toLowerCase()
  if (!normalized) return "unknown"
  if (normalized.startsWith("audio/")) return "audio"
  if (normalized.startsWith("video/")) return "video"
  if (normalized.includes("pdf")) return "pdf"
  if (normalized.includes("epub")) return "ebook"
  if (
    normalized.startsWith("text/") ||
    normalized.includes("markdown") ||
    normalized.includes("html") ||
    normalized.includes("xml") ||
    normalized.includes("json") ||
    normalized.includes("rtf") ||
    normalized.includes("officedocument.wordprocessingml.document")
  ) {
    return "document"
  }
  return "unknown"
}

const detectTypeFromFile = (file: File): DetectedMediaType => {
  const detectedFromExtension = detectTypeFromExtension(file.name)
  return detectedFromExtension !== "unknown"
    ? detectedFromExtension
    : detectTypeFromMime(file.type)
}

export const detectTypeFromUrl = (url: string): DetectedMediaType => {
  try {
    const parsed = new URL(url)
    const pathname = parsed.pathname.toLowerCase()
    const hostname = parsed.hostname.toLowerCase()
    // Check common file extensions in URL path
    const ext = pathname.split(".").pop() || ""
    if (["mp3", "wav", "ogg", "flac", "m4a"].includes(ext)) return "audio"
    if (["mp4", "mkv", "avi", "mov", "webm"].includes(ext)) return "video"
    if (ext === "pdf") return "pdf"
    if (["epub", "mobi"].includes(ext)) return "ebook"
    if (["docx", "txt", "rtf", "md", "markdown", "xml", "json"].includes(ext)) return "document"
    // YouTube and common video platforms
    if (hostnameMatches(hostname, "youtube.com") || hostnameMatches(hostname, "youtu.be")) return "video"
    if (hostnameMatches(hostname, "vimeo.com")) return "video"
    if (hostnameMatches(hostname, "soundcloud.com")) return "audio"
    if (hostnameMatches(hostname, "spotify.com")) return "audio"
    // Default for URLs is web
    return "web"
  } catch {
    return "web"
  }
}

export const detectPlaylistPreflightCandidate = (url: string): boolean => {
  try {
    const parsed = new URL(url)
    const hostname = parsed.hostname.toLowerCase()
    if (!hostnameMatches(hostname, "youtube.com") && !hostnameMatches(hostname, "youtu.be")) {
      return false
    }
    const playlistId = parsed.searchParams.get("list")?.trim()
    return Boolean(playlistId)
  } catch {
    return false
  }
}

const isValidUrl = (raw: string): boolean => {
  const trimmed = raw.trim()
  if (!trimmed) return false
  try {
    const parsed = new URL(trimmed)
    return parsed.protocol === "http:" || parsed.protocol === "https:"
  } catch {
    return false
  }
}

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 B"
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

const validateQueueItem = (
  item: WizardQueueItem,
  existingItems: WizardQueueItem[]
): QueueItemValidation => {
  const errors: string[] = []
  const warnings: string[] = []

  if (item.url) {
    if (!isValidUrl(item.url)) {
      errors.push("Invalid URL format")
    }
    // Check for duplicates
    const dedupeKey = normalizeUrlForDedupe(item.url)
    const isDuplicate = existingItems.some(
      (other) =>
        other.id !== item.id &&
        other.url &&
        normalizeUrlForDedupe(other.url) === dedupeKey
    )
    if (isDuplicate) {
      warnings.push("Already queued")
    }
  }

  if (item.file) {
    if (item.fileSize > QUICK_INGEST_MAX_FILE_SIZE) {
      errors.push(`File exceeds ${QUICK_INGEST_MAX_FILE_SIZE_LABEL} quick-ingest limit`)
    }
    // Check for duplicate files
    const isDuplicate = existingItems.some(
      (other) =>
        other.id !== item.id &&
        other.fileName === item.fileName &&
        other.fileSize === item.fileSize
    )
    if (isDuplicate) {
      warnings.push("Already queued")
    }
  }

  if (item.detectedType === "unknown") {
    errors.push(
      "Unsupported file type. Quick Ingest supports PDF, EPUB, DOCX, TXT/RTF, Markdown, HTML, XML, JSON, audio, and video."
    )
  }

  return {
    valid: errors.length === 0,
    errors: errors.length > 0 ? errors : undefined,
    warnings: warnings.length > 0 ? warnings : undefined,
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

// Warning uses >= (show at boundary); validation uses > (allow exactly at limit)
const LARGE_FILE_WARNING_THRESHOLD = Math.floor(QUICK_INGEST_MAX_FILE_SIZE * 0.8)
const PASSIVE_ALERT_PROPS = {
  role: "status",
  "aria-live": "polite",
} as const

type QuickIngestText = (
  key: string,
  defaultValue: string,
  options?: Record<string, unknown>
) => string

const playlistInspectionCopy = (
  candidate: PlaylistInspectionCandidate,
  qi: QuickIngestText
): { label: string; message: string } => {
  switch (candidate.status) {
    case "ready":
      return {
        label: qi("playlistInspection.readyLabel", "Inspection ready"),
        message: qi(
          "playlistInspection.readyMessage",
          "{{count}} playlist items loaded for review.",
          { count: candidate.items.length }
        ),
      }
    case "unavailable":
      return {
        label: qi("playlistInspection.unavailableLabel", "Inspection unavailable"),
        message: qi(
          "playlistInspection.unavailableMessage",
          "Playlist inspection is unavailable on this server. Update the server or remove this playlist."
        ),
      }
    case "failed":
      return {
        label: qi("playlistInspection.failedLabel", "Inspection failed"),
        message:
          candidate.error?.message ??
          qi("playlistInspection.failedMessage", "Playlist inspection failed. Try again."),
      }
    case "blocked":
      return {
        label: qi("playlistInspection.blockedLabel", "Inspection blocked"),
        message:
          candidate.error?.message ??
          qi(
            "playlistInspection.blockedMessage",
            "Playlist inspection was blocked. Remove it or try again."
          ),
      }
    case "expired":
      return {
        label: qi("playlistInspection.expiredLabel", "Inspection expired"),
        message:
          candidate.error?.message ??
          qi("playlistInspection.expiredMessage", "Playlist inspection expired. Try again."),
      }
    case "cancelled":
      return {
        label: qi("playlistInspection.cancelledLabel", "Inspection cancelled"),
        message:
          candidate.error?.message ??
          qi("playlistInspection.cancelledMessage", "Playlist inspection was cancelled."),
      }
    default:
      return {
        label: qi("playlistInspection.inspectingLabel", "Inspecting playlist"),
        message: qi(
          "playlistInspection.inspectingMessage",
          "Loading playlist details from the server."
        ),
      }
  }
}

const playlistInspectionBadge = (
  status: PlaylistInspectionCandidate["status"]
): "info" | "success" | "warning" | "danger" => {
  if (status === "ready") return "success"
  if (status === "queued" || status === "inspecting") return "info"
  if (status === "failed" || status === "blocked") return "danger"
  return "warning"
}

type AddContentStepProps = {
  isOnlineForIngest?: boolean
  isCheckingConnection?: boolean
  connectionRecoveryMessage?: string
  onRetryConnection?: () => void
  onQuickProcess?: () => void
  quickProcessWarning?: string | null
}

export const AddContentStep: React.FC<AddContentStepProps> = ({
  isOnlineForIngest = true,
  isCheckingConnection = false,
  connectionRecoveryMessage,
  onRetryConnection,
  onQuickProcess,
  quickProcessWarning = null,
}) => {
  const { t } = useTranslation(["option"])
  const { state, setQueueItems, setPlaylistPreflightSeed, goNext } = useIngestWizard()
  const { queueItems, playlistPreflightSeed, firstSourceAddMode } = state

  const [urlInput, setUrlInput] = useState("")
  const [pastedTextInput, setPastedTextInput] = useState("")
  const { capabilities, loading: capabilitiesLoading } = useServerCapabilities()
  const clearPlaylistPreflightSeed = useCallback(
    () => setPlaylistPreflightSeed(null),
    [setPlaylistPreflightSeed]
  )
  const playlistInspection = usePlaylistInspection({
    enabled:
      capabilities === null && capabilitiesLoading
        ? null
        : capabilities?.hasMediaPlaylistIngestV2 === true,
    queueItems,
    seed: playlistPreflightSeed,
    clearSeed: clearPlaylistPreflightSeed,
  })
  const addPlaylistCandidates = playlistInspection.addCandidates
  const shouldShowPastedTextInput = firstSourceAddMode === "paste_text"
  const shouldFocusUrlInput = firstSourceAddMode === "web_url"

  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      options
        ? t(`quickIngest.${key}`, { defaultValue, ...options })
        : t(`quickIngest.${key}`, defaultValue),
    [t]
  )

  // Add files from the drop zone
  const handleFilesAdded = useCallback(
    (files: File[]) => {
      const newItems: WizardQueueItem[] = []
      for (const file of files) {
        const detectedType = detectTypeFromFile(file)
        const item: WizardQueueItem = {
          id: crypto.randomUUID(),
          fileName: file.name,
          file,
          detectedType,
          icon: ICON_NAME_MAP[detectedType],
          fileSize: file.size,
          mimeType: file.type || undefined,
          validation: { valid: true },
        }
        item.validation = validateQueueItem(item, [...queueItems, ...newItems])
        newItems.push(item)
      }
      setQueueItems([...queueItems, ...newItems])
    },
    [queueItems, setQueueItems]
  )

  // Add URLs from the multi-line input
  const handleAddUrls = useCallback(() => {
    const lines = urlInput
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)

    if (lines.length === 0) return

    const playlistCandidates: string[] = []
    const ordinaryLines: string[] = []
    for (const url of lines) {
      if (detectPlaylistPreflightCandidate(url)) {
        playlistCandidates.push(url)
      } else {
        ordinaryLines.push(url)
      }
    }
    const newItems: WizardQueueItem[] = []
    for (const url of ordinaryLines) {
      const detectedType = detectTypeFromUrl(url)
      const item: WizardQueueItem = {
        id: crypto.randomUUID(),
        url,
        detectedType,
        icon: ICON_NAME_MAP[detectedType],
        fileSize: 0,
        validation: { valid: true },
      }
      item.validation = validateQueueItem(item, [...queueItems, ...newItems])
      newItems.push(item)
    }

    if (newItems.length > 0) {
      setQueueItems([...queueItems, ...newItems])
    }
    addPlaylistCandidates(playlistCandidates)
    setUrlInput("")
  }, [addPlaylistCandidates, queueItems, setQueueItems, urlInput])

  const handleAddPastedText = useCallback(() => {
    if (!pastedTextInput.trim()) return

    const file = new File([pastedTextInput], "pasted-text.txt", {
      type: "text/plain"
    })
    const detectedType = detectTypeFromFile(file)
    const item: WizardQueueItem = {
      id: crypto.randomUUID(),
      fileName: file.name,
      file,
      detectedType,
      icon: ICON_NAME_MAP[detectedType],
      fileSize: file.size,
      mimeType: file.type || undefined,
      validation: { valid: true },
    }
    item.validation = validateQueueItem(item, queueItems)
    setQueueItems([...queueItems, item])
    setPastedTextInput("")
  }, [pastedTextInput, queueItems, setQueueItems])

  // Handle Enter key in URL input
  const handleUrlKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault()
        handleAddUrls()
      }
    },
    [handleAddUrls]
  )

  // Remove an item from the queue
  const handleRemoveItem = useCallback(
    (id: string) => {
      setQueueItems(queueItems.filter((item) => item.id !== id))
    },
    [queueItems, setQueueItems]
  )

  // Clear all items
  const handleClearAll = useCallback(() => {
    setQueueItems([])
  }, [setQueueItems])

  const hasItems = queueItems.length > 0
  const selectedItems = useMemo(
    () => queueItems.filter((item) => item.conferenceOverride?.selected !== false),
    [queueItems]
  )
  const validItemCount = useMemo(
    () => selectedItems.filter((item) => item.validation.valid).length,
    [selectedItems]
  )
  const invalidItemCount = selectedItems.length - validItemCount
  const canProceed = validItemCount > 0 && !playlistInspection.hasUnresolvedCandidates
  const canStartProcessing = canProceed && isOnlineForIngest && !isCheckingConnection

  const hasLargeFiles = useMemo(
    () => queueItems.some((item) => item.fileSize >= LARGE_FILE_WARNING_THRESHOLD),
    [queueItems]
  )

  const ffmpegMissing = capabilities?.ffmpegAvailable === false
  const hasAvMediaItems = useMemo(
    () =>
      queueItems.some(
        (item) => item.detectedType === "audio" || item.detectedType === "video"
      ),
    [queueItems]
  )

  return (
    <div className="py-3">
      {/* Drop zone + URL input area */}
      <div className="space-y-3">
        <FileDropZone
          onFilesAdded={handleFilesAdded}
          isOnlineForIngest={isOnlineForIngest}
          autoFocus={firstSourceAddMode === "file_upload"}
        />
        <Typography.Text className="text-[11px] text-text-subtle">
          {qi(
            "fileSizeLimits",
            "Supported: PDF, EPUB, DOCX, TXT/RTF, Markdown, HTML, XML, JSON, audio, video. Max file size: {{maxSize}}.",
            { maxSize: QUICK_INGEST_MAX_FILE_SIZE_LABEL }
          )}
        </Typography.Text>
        <Typography.Text className="block text-xs text-text-muted">
          {qi(
            "wizard.addPurpose",
            "Add URLs or files. Stored items appear in Media; analyzed and chunked items become searchable in Knowledge."
          )}
        </Typography.Text>

        {shouldShowPastedTextInput && (
          <div>
            <Typography.Text className="text-xs text-text-muted">
              {qi("pasteTextTitle", "Paste text:")}
            </Typography.Text>
            <div className="mt-1 flex gap-2">
              <Input.TextArea
                value={pastedTextInput}
                onChange={(e) => setPastedTextInput(e.target.value)}
                placeholder={qi(
                  "pasteTextPlaceholder",
                  "Paste article text, notes, or a short document..."
                )}
                autoSize={{ minRows: 3, maxRows: 6 }}
                autoFocus
                aria-label={qi("pasteTextInputAria", "Pasted text input")}
                className="flex-1"
              />
              <Button
                type="primary"
                onClick={handleAddPastedText}
                disabled={!pastedTextInput.trim()}
                aria-label={qi("addPastedTextAria", "Add pasted text to queue")}
                className="self-end"
              >
                <Plus className="mr-1 h-4 w-4" />
                {qi("addPastedText", "Add text")}
              </Button>
            </div>
          </div>
        )}

        {hasLargeFiles && (
          <DesignSystemAlert
            variant="warning"
            {...PASSIVE_ALERT_PROPS}
            icon={<AlertTriangle className="h-4 w-4" aria-hidden="true" />}
            title={qi(
              "largeFileWarning",
              "Large file -- this browser-buffered upload is close to the {{maxSize}} quick-ingest limit.",
              { maxSize: QUICK_INGEST_MAX_FILE_SIZE_LABEL }
            )}
          />
        )}

        {/* Multi-line URL paste area */}
        <div>
          <Typography.Text className="text-xs text-text-muted">
            {qi("pasteUrlsTitle", "Paste URLs (one per line):")}
          </Typography.Text>
          <div className="mt-1 flex gap-2">
            <Input.TextArea
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={handleUrlKeyDown}
              placeholder={qi(
                "urlsPlaceholder",
                "https://example.com/article\nhttps://youtube.com/watch?v=..."
              )}
              autoSize={{ minRows: 2, maxRows: 4 }}
              autoFocus={shouldFocusUrlInput}
              aria-label={qi("urlsInputAria", "URL input area")}
              className="flex-1"
            />
            <Button
              type="primary"
              onClick={handleAddUrls}
              disabled={!urlInput.trim()}
              aria-label={qi("addUrlsAria", "Add URLs to queue")}
              className="self-end"
            >
              <Plus className="mr-1 h-4 w-4" />
              {qi("addUrls", "Add")}
            </Button>
          </div>
        </div>

        {playlistInspection.candidates.length > 0 && (
          <section
            aria-label={qi("playlistInspection.regionAria", "Playlist inspections")}
            className="space-y-2"
          >
            <Typography.Text className="text-xs font-medium text-text-muted">
              {qi("playlistInspection.title", "PLAYLIST INSPECTION")}
            </Typography.Text>
            {playlistInspection.candidates.map((candidate) => {
              const copy = playlistInspectionCopy(candidate, qi)
              const isActive = candidate.status === "queued" || candidate.status === "inspecting"
              const canRetry =
                ((candidate.status === "failed" || candidate.status === "blocked") &&
                  candidate.error?.retryable === true) ||
                candidate.status === "expired" ||
                candidate.status === "cancelled"
              return (
                <div
                  key={candidate.key}
                  className="rounded-md border border-border bg-surface2/40 px-3 py-2"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{candidate.url}</div>
                      <div
                        className="mt-1 flex flex-wrap items-center gap-2"
                        role="status"
                        aria-live="polite"
                        aria-atomic="true"
                      >
                        <Badge variant={playlistInspectionBadge(candidate.status)} size="sm">
                          {copy.label}
                        </Badge>
                        <Typography.Text className="text-xs text-text-muted">
                          {copy.message}
                        </Typography.Text>
                      </div>
                    </div>
                    <div className="flex flex-shrink-0 items-center gap-1">
                      {isActive && (
                        <Button
                          size="small"
                          onClick={() => playlistInspection.cancelCandidate(candidate.key)}
                          aria-label={qi(
                            "playlistInspection.cancelAria",
                            "Cancel playlist inspection for {{url}}",
                            { url: candidate.url }
                          )}
                        >
                          {qi("playlistInspection.cancel", "Cancel")}
                        </Button>
                      )}
                      {canRetry && (
                        <Button
                          size="small"
                          onClick={() => playlistInspection.retryCandidate(candidate.key)}
                          aria-label={qi(
                            "playlistInspection.retryAria",
                            "Retry playlist inspection for {{url}}",
                            { url: candidate.url }
                          )}
                        >
                          {qi("playlistInspection.retry", "Retry")}
                        </Button>
                      )}
                      {!isActive && (
                        <Button
                          size="small"
                          type="text"
                          onClick={() => playlistInspection.removeCandidate(candidate.key)}
                          aria-label={qi(
                            "playlistInspection.removeAria",
                            "Remove playlist inspection for {{url}}",
                            { url: candidate.url }
                          )}
                        >
                          {qi("playlistInspection.remove", "Remove")}
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
            {playlistInspection.hasTruncatedCandidates && (
              <Typography.Text className="block text-xs text-text-muted">
                {qi("playlistInspection.moreNotLoaded", "More playlist items are not loaded yet.")}
              </Typography.Text>
            )}
            {playlistInspection.sessionDuplicateCount > 0 && (
              <DesignSystemAlert
                variant="warning"
                {...PASSIVE_ALERT_PROPS}
                title={qi(
                  "playlistInspection.sessionDuplicates",
                  "{{count}} staged or inspected items overlap in this session.",
                  { count: playlistInspection.sessionDuplicateCount }
                )}
              />
            )}
          </section>
        )}
      </div>

      {/* FFmpeg missing warning for audio/video items */}
      {ffmpegMissing && hasAvMediaItems && (
        <DesignSystemAlert
          variant="warning"
          {...PASSIVE_ALERT_PROPS}
          icon={<AlertTriangle className="h-4 w-4" aria-hidden="true" />}
          className="mt-3"
          title={qi(
            "ffmpegMissing",
            "FFmpeg is not installed on the server. Audio and video files may fail to process. Other file types (PDF, documents, ebooks) are unaffected."
          )}
        />
      )}

      {!isOnlineForIngest && (
        <DesignSystemAlert
          variant="warning"
          icon={<AlertTriangle className="h-4 w-4" aria-hidden="true" />}
          className="mt-3"
          title={qi("wizard.offline.title", "Server offline")}
          action={
            onRetryConnection
              ? {
                  label: isCheckingConnection
                    ? qi("wizard.offline.checking", "Checking...")
                    : qi("wizard.offline.retry", "Retry connection"),
                  onClick: onRetryConnection,
                  loading: isCheckingConnection,
                  disabled: isCheckingConnection,
                }
              : undefined
          }
        >
          {connectionRecoveryMessage ||
            qi(
              "wizard.offline.description",
              "Reconnect to your tldw server before processing. You can still add URLs and configure queued items."
            )}
        </DesignSystemAlert>
      )}

      {/* Queued items list */}
      {hasItems && (
        <div className="mt-4">
          <div className="flex items-center justify-between">
            <div className="flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-2">
              <Typography.Text className="text-sm font-medium">
                {qi("queueTitle", "QUEUED")}
                <span className="ml-1.5 text-text-muted font-normal">
                  ({queueItems.length}{" "}
                  {queueItems.length === 1
                    ? qi("wizard.item", "item")
                    : qi("wizard.items", "items")}
                  )
                </span>
              </Typography.Text>
              {invalidItemCount > 0 && (
                <Typography.Text className="text-xs text-text-muted">
                  {qi(
                    "queueValiditySummary",
                    "{{valid}} valid / {{invalid}} invalid",
                    {
                      valid: validItemCount,
                      invalid: invalidItemCount,
                    }
                  )}
                </Typography.Text>
              )}
            </div>
            <Button
              size="small"
              type="text"
              danger
              onClick={handleClearAll}
              aria-label={qi("clearAllAria", "Remove all items from queue")}
            >
              {qi("clearAll", "Clear all")}
            </Button>
          </div>

          <div className="mt-2 space-y-1.5">
            {queueItems.map((item) => (
              <div
                key={item.id}
                className={`flex items-center gap-3 rounded-md border px-3 py-2 ${
                  !item.validation.valid
                    ? "border-danger/30 bg-danger/5"
                    : item.validation.warnings?.length
                      ? "border-warn/30 bg-warn/5"
                      : "border-border"
                }`}
              >
                {/* Type icon */}
                <span className="flex-shrink-0">
                  {MEDIA_TYPE_ICONS[item.detectedType]}
                </span>

                {/* Name/URL and metadata */}
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">
                    {item.fileName || item.url || qi("untitledItem", "Untitled")}
                  </div>
                  <div className="flex items-center gap-2 text-[11px] text-text-muted">
                    {item.fileSize > 0 && (
                      <span>{formatFileSize(item.fileSize)}</span>
                    )}
                    {ffmpegMissing &&
                    (item.detectedType === "audio" ||
                      item.detectedType === "video") ? (
                      <Tooltip
                        title={qi(
                          "ffmpegRequiredTooltip",
                          "FFmpeg is not installed on the server -- this file may fail to process"
                        )}
                      >
                        <Badge
                          variant="warning"
                          size="sm"
                          className="!m-0"
                        >
                          <AlertTriangle
                            className="mr-0.5 h-3 w-3"
                            aria-hidden="true"
                          />
                          {item.detectedType.charAt(0).toUpperCase() +
                            item.detectedType.slice(1)}
                        </Badge>
                      </Tooltip>
                    ) : (
                      <Badge
                        variant="info"
                        size="sm"
                        className="!m-0"
                      >
                        {item.detectedType === "web"
                          ? "Web page"
                          : item.detectedType.charAt(0).toUpperCase() +
                            item.detectedType.slice(1)}
                      </Badge>
                    )}
                    {item.detectedType !== "unknown" && (
                      <span className="text-text-subtle">(auto)</span>
                    )}
                  </div>
                  {/* Validation errors/warnings */}
                  {item.validation.errors?.map((err, i) => (
                    <div key={`e-${i}`} className="text-[11px] text-danger mt-0.5">
                      {err}
                    </div>
                  ))}
                  {item.validation.warnings?.map((warn, i) => (
                    <div key={`w-${i}`} className="text-[11px] text-warn mt-0.5">
                      {warn}
                    </div>
                  ))}
                </div>

                {/* Remove button */}
                <button
                  type="button"
                  onClick={() => handleRemoveItem(item.id)}
                  className="flex-shrink-0 rounded p-1 text-text-muted hover:bg-surface2 hover:text-danger transition-colors"
                  aria-label={qi("removeItemAria", "Remove this item from queue")}
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {hasItems && <BatchMetadataPanel />}

      {hasItems && quickProcessWarning && (
        <DesignSystemAlert
          variant="warning"
          role="alert"
          icon={<AlertTriangle className="h-4 w-4" aria-hidden="true" />}
          className="mt-3"
          title={quickProcessWarning}
        />
      )}

      {/* Action buttons */}
      <div className="mt-4 flex items-center justify-end gap-2">
        {(hasItems || playlistInspection.hasUnresolvedCandidates) && onQuickProcess && (
          <Button
            type="primary"
            onClick={onQuickProcess}
            disabled={!canStartProcessing}
          >
            {qi("wizard.useDefaultsProcess", "Use defaults & process")}
          </Button>
        )}
        <Button
          onClick={goNext}
          disabled={!canProceed}
          aria-label={qi("wizard.configureItems", "Configure {{count}} items", {
            count: validItemCount,
          })}
        >
          {qi("wizard.configureItems", "Configure {{count}} items >", {
            count: validItemCount,
          })}
        </Button>
      </div>
    </div>
  )
}

export default AddContentStep
