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
  ConferenceDuplicatePolicy,
  DetectedMediaType,
  WizardQueueItem,
} from "./types"
import { useIngestWizard } from "./IngestWizardContext"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { Alert as DesignSystemAlert, Badge } from "@/components/ui/primitives"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { PlaylistPreflightResult } from "@/services/tldw/playlist-preflight"
import { FileDropZone } from "./QueueTab/FileDropZone"
import { PlaylistPreflightPanel } from "./PlaylistPreflightPanel"
import { BatchMetadataPanel } from "./BatchMetadataPanel"
import {
  QUICK_INGEST_MAX_FILE_SIZE_LABEL,
  QUICK_INGEST_MAX_FILE_SIZE,
} from "./constants"
import { detectTypeFromUrl, createUrlQueueItems, getEligibleQueueItems, getQueueItemExclusionReason, isValidQueueUrl, validateQueueItem } from "./queue-items"
export { detectTypeFromUrl } from "./queue-items"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { buildQuickIngestOpenDetailFromUrl, isQuickIngestPlaylistPreflightDetail } from "@/utils/quick-ingest-open"

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

const isDuplicatePreflightStatus = (status: string | undefined): boolean =>
  status === "duplicate_existing" || status === "duplicate_in_batch"

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 B"
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
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
  const {
    state,
    setQueueItems,
    setPlaylistPreflightSeed,
    setConferenceBatchMetadata,
    goNext,
  } = useIngestWizard()
  const {
    queueItems,
    conferenceBatchMetadata,
    playlistPreflightSeed,
    firstSourceAddMode,
  } = state

  const [urlInput, setUrlInput] = useState("")
  const [pastedTextInput, setPastedTextInput] = useState("")
  const [playlistPreflightUrl, setPlaylistPreflightUrl] = useState("")
  const [playlistPreflight, setPlaylistPreflight] = useState<PlaylistPreflightResult | null>(null)
  const [duplicatePolicy, setDuplicatePolicy] = useState<ConferenceDuplicatePolicy>("skip")
  const [playlistPreflightLoading, setPlaylistPreflightLoading] = useState(false)
  const [playlistPreflightError, setPlaylistPreflightError] = useState<string | null>(null)
  const { capabilities } = useServerCapabilities()
  const seededPlaylistUrlRef = React.useRef<string | null>(null)
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
        item.validation = validateQueueItem(item)
        newItems.push(item)
      }
      setQueueItems([...queueItems, ...newItems])
    },
    [queueItems, setQueueItems]
  )

  // Add URLs from the multi-line input
  const handleAddUrls = useCallback(() => {
    const newItems = createUrlQueueItems(urlInput)
    if (!newItems.length) return

    setQueueItems([...queueItems, ...newItems])
    setUrlInput("")
    setPlaylistPreflight(null)
    setPlaylistPreflightUrl("")
    setDuplicatePolicy("skip")
    setPlaylistPreflightError(null)
  }, [urlInput, queueItems, setQueueItems])

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
    item.validation = validateQueueItem(item)
    setQueueItems([...queueItems, item])
    setPastedTextInput("")
  }, [pastedTextInput, queueItems, setQueueItems])

  const playlistCandidateUrls = useMemo(
    () =>
      urlInput
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .filter(detectPlaylistPreflightCandidate),
    [urlInput]
  )
  const primaryPlaylistCandidateUrl = playlistCandidateUrls[0] || ""
  const shouldOfferPlaylistPreflight =
    Boolean(capabilities?.hasMediaPlaylistPreflight) && Boolean(primaryPlaylistCandidateUrl)

  const handlePreviewPlaylist = useCallback(async () => {
    if (!primaryPlaylistCandidateUrl) return
    setPlaylistPreflightLoading(true)
    setPlaylistPreflightError(null)
    try {
      const result = await tldwClient.preflightPlaylist({
        url: primaryPlaylistCandidateUrl,
        max_items: 100,
        timeoutMs: 60_000
      })
      setPlaylistPreflight(result)
      setPlaylistPreflightUrl(primaryPlaylistCandidateUrl)
      setDuplicatePolicy("skip")
    } catch (error) {
      setPlaylistPreflight(null)
      setPlaylistPreflightUrl(primaryPlaylistCandidateUrl)
      setPlaylistPreflightError(
        error instanceof Error && error.message
          ? error.message
          : "Playlist preview failed."
      )
    } finally {
      setPlaylistPreflightLoading(false)
    }
  }, [primaryPlaylistCandidateUrl])

  React.useEffect(() => {
    if (!isQuickIngestPlaylistPreflightDetail(playlistPreflightSeed)) {
      return
    }

    const seededUrl = playlistPreflightSeed.url.trim()
    if (!seededUrl) return

    seededPlaylistUrlRef.current = seededUrl
    setUrlInput(seededUrl)
    setPlaylistPreflight(null)
    setPlaylistPreflightUrl("")
    setDuplicatePolicy("skip")
    setPlaylistPreflightError(null)
    setPlaylistPreflightSeed(null)
  }, [playlistPreflightSeed, setPlaylistPreflightSeed])

  React.useEffect(() => {
    if (!seededPlaylistUrlRef.current) return
    if (!shouldOfferPlaylistPreflight) return
    if (primaryPlaylistCandidateUrl !== seededPlaylistUrlRef.current) return

    seededPlaylistUrlRef.current = null
    void handlePreviewPlaylist()
  }, [handlePreviewPlaylist, primaryPlaylistCandidateUrl, shouldOfferPlaylistPreflight])

  const handleAddPreflightItems = useCallback(() => {
    if (!playlistPreflight) return
    const newItems: WizardQueueItem[] = []
    const selectedItems = playlistPreflight.items.filter(
      (item) => item.selected && item.sourceUrl
    )
    for (const preflightItem of selectedItems) {
      const detectedType = detectTypeFromUrl(preflightItem.sourceUrl)
      const item: WizardQueueItem = {
        id: crypto.randomUUID(),
        url: preflightItem.sourceUrl,
        detectedType,
        icon: ICON_NAME_MAP[detectedType],
        fileSize: 0,
        validation: { valid: true },
        playlist: {
          playlistId: playlistPreflight.playlistId,
          playlistTitle: playlistPreflight.playlistTitle,
          ordinal: preflightItem.ordinal,
          normalizedSourceId: preflightItem.normalizedSourceId,
          duplicateStatus: preflightItem.duplicateStatus
        },
        conferenceOverride: {
          selected: true,
          ...(isDuplicatePreflightStatus(preflightItem.duplicateStatus)
            ? { duplicatePolicy }
            : {})
        }
      }
      item.validation = validateQueueItem(item)
      newItems.push(item)
    }
    if (newItems.length === 0) return
    setQueueItems([...queueItems, ...newItems])
    setConferenceBatchMetadata({
      collectionName:
        conferenceBatchMetadata?.collectionName ||
        playlistPreflight.playlistTitle ||
        "",
      conferenceName: conferenceBatchMetadata?.conferenceName,
      eventDate: conferenceBatchMetadata?.eventDate,
      eventYear: conferenceBatchMetadata?.eventYear,
      sharedTags: conferenceBatchMetadata?.sharedTags ?? [],
      sourcePlaylistUrl:
        conferenceBatchMetadata?.sourcePlaylistUrl || playlistPreflightUrl,
    })
    setUrlInput((current) =>
      current
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line && line !== playlistPreflightUrl)
        .join("\n")
    )
    setPlaylistPreflight(null)
    setPlaylistPreflightUrl("")
    setDuplicatePolicy("skip")
    setPlaylistPreflightError(null)
  }, [
    conferenceBatchMetadata,
    duplicatePolicy,
    playlistPreflight,
    playlistPreflightUrl,
    queueItems,
    setConferenceBatchMetadata,
    setQueueItems,
  ])

  const handlePreflightItemSelectionChange = useCallback(
    (ordinal: number, selected: boolean) => {
      setPlaylistPreflight((current) => {
        if (!current) return current
        const items = current.items.map((item) =>
          item.ordinal === ordinal ? { ...item, selected } : item
        )
        return {
          ...current,
          selectedCount: items.filter((item) => item.selected && item.sourceUrl).length,
          items
        }
      })
    },
    []
  )

  const handleDuplicatePolicyChange = useCallback(
    (policy: ConferenceDuplicatePolicy) => {
      setDuplicatePolicy(policy)
      setPlaylistPreflight((current) => {
        if (!current) return current
        const items = current.items.map((item) => {
          if (!item.sourceUrl) return { ...item, selected: false }
          if (!isDuplicatePreflightStatus(item.duplicateStatus)) return item
          return { ...item, selected: policy !== "skip" }
        })
        return {
          ...current,
          selectedCount: items.filter((item) => item.selected && item.sourceUrl).length,
          items
        }
      })
    },
    []
  )

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

  const captureMounted = React.useRef(true)
  const queueRef = React.useRef(queueItems)
  queueRef.current = queueItems
  React.useEffect(() => {
    captureMounted.current = true
    return () => {
      captureMounted.current = false
    }
  }, [])
  const [captureError, setCaptureError] = useState<string | null>(null)
  const [capturing, setCapturing] = useState(false)
  const handleCaptureTab = async () => {
    if (!isExtensionRuntime() || capturing) return
    const captureOwner = useQuickIngestSessionStore.getState().authorityKey
    const captureSession = useQuickIngestSessionStore.getState().session?.id
    const isCurrentCapture = () =>
      captureMounted.current &&
      captureOwner === useQuickIngestSessionStore.getState().authorityKey &&
      captureSession === useQuickIngestSessionStore.getState().session?.id
    setCapturing(true)
    setCaptureError(null)
    try {
      const tabsApi =
        (globalThis as typeof globalThis & { browser?: typeof chrome }).browser
          ?.tabs || globalThis.chrome?.tabs
      const tabs = await tabsApi?.query({ active: true, currentWindow: true })
      if (!isCurrentCapture()) return
      const url = tabs?.[0]?.url?.trim() || ""
      if (!isValidQueueUrl(url)) throw new Error("restricted")
      const playlist = buildQuickIngestOpenDetailFromUrl(url)
      setUrlInput(url)
      if (playlist) setPlaylistPreflightSeed(playlist)
      else setQueueItems([...queueRef.current, ...createUrlQueueItems(url)])
    } catch {
      if (isCurrentCapture())
        setCaptureError(
          qi(
            "captureRestricted",
            "This tab cannot be captured. Open an HTTP or HTTPS page, or paste its URL here."
          )
        )
    } finally {
      if (isCurrentCapture()) setCapturing(false)
    }
  }
  const hasItems = queueItems.length > 0
  const selectedItems = useMemo(
    () => getEligibleQueueItems(queueItems),
    [queueItems]
  )
  const validItemCount = selectedItems.length
  const invalidItemCount = queueItems.filter(item => !item.validation.valid).length
  const canProceed = validItemCount > 0
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

        {isExtensionRuntime() && (
          <Button onClick={handleCaptureTab} loading={capturing} disabled={capturing}>
            {qi("captureCurrentTab", "Capture current tab")}
          </Button>
        )}
        {captureError && (
          <div role="alert" className="text-sm text-danger">{captureError}</div>
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

        {shouldOfferPlaylistPreflight && (
          <PlaylistPreflightPanel
            candidateUrl={primaryPlaylistCandidateUrl}
            loading={playlistPreflightLoading}
            error={playlistPreflightError}
            result={
              playlistPreflightUrl === primaryPlaylistCandidateUrl
                ? playlistPreflight
                : null
            }
            onPreview={handlePreviewPlaylist}
            onAddItems={handleAddPreflightItems}
            onItemSelectionChange={handlePreflightItemSelectionChange}
            duplicatePolicy={duplicatePolicy}
            onDuplicatePolicyChange={handleDuplicatePolicyChange}
          />
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
            {queueItems.map((item) => {
              const exclusion = getQueueItemExclusionReason(item, queueItems)
              return (
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
                      {err === "Invalid URL format" ? qi("invalidUrl", "Invalid URL format. Enter one HTTP(S) URL per line. Separate pasted URLs with a comma and a space.") : err}
                    </div>
                  ))}
                  {item.validation.warnings?.map((warn, i) => (
                    <div key={`w-${i}`} className="text-[11px] text-warn mt-0.5">
                      {warn}
                    </div>
                  ))}
                </div>

                {exclusion && (
                  <span className="text-xs text-text-muted">
                    {exclusion === "duplicate"
                      ? qi("queueDuplicateExcluded", "Already queued — excluded")
                      : exclusion === "unselected"
                        ? qi("queueUnselected", "Not selected — excluded")
                        : qi("queueInvalidExcluded", "Invalid — excluded")}
                  </span>
                )}
                {exclusion === "duplicate" && (
                  <Button size="small" onClick={() => setQueueItems(
                    queueItems.map(current => current.id === item.id ? { ...current, processAgain: true } : current)
                  )}>
                    {qi("processAgain", "Process again")}
                  </Button>
                )}
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
              )
            })}
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
        {hasItems && onQuickProcess && (
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
          aria-label={qi("wizard.configureItems", "Configure {count, plural, one {# item} other {# items}}", {
            count: validItemCount,
          })}
        >
          {qi("wizard.configureItems", "Configure {count, plural, one {# item} other {# items}}", {
            count: validItemCount,
          })}
          <span aria-hidden="true"> &gt;</span>
        </Button>
      </div>
    </div>
  )
}

export default AddContentStep
