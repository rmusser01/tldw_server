import { Badge, Alert as DesignSystemAlert } from "@/components/ui/primitives"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { PlaylistPreflightResult } from "@/services/tldw/playlist-preflight"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { buildQuickIngestOpenDetailFromUrl, isQuickIngestPlaylistPreflightDetail, type FirstSourceQuickIngestKind } from "@/utils/quick-ingest-open"

import { Button, Input, Tooltip, Typography } from "antd"
import {
  AlertTriangle,
  BookOpen,
  File as FileIcon,
  FileText,
  Film,
  Globe,
  Image as ImageIcon,
  Music,
  Loader2,
  Plus,
  X
} from "lucide-react"
import React, { useCallback, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"

import { BatchMetadataPanel } from "./BatchMetadataPanel"
import { useIngestWizard } from "./IngestWizardContext"
import { PlaylistPreflightPanel } from "./PlaylistPreflightPanel"
import { FileDropZone } from "./QueueTab/FileDropZone"
import {
  getQuickIngestFileLimit,
  QUICK_INGEST_ACCEPT_STRING,
  QUICK_INGEST_AV_LIMIT,
  QUICK_INGEST_DOCUMENT_LIMIT
} from "./constants"
import {
  canPasteQueueItem,
  createUrlQueueItems,
  detectTypeFromUrl,
  getEligibleQueueItems,
  getQueueItemExclusionReason,
  isValidQueueUrl,
  restoreQueueFile,
  validateQueueItem
} from "./queue-items"
import type {
  ConferenceDuplicatePolicy,
  DetectedMediaType,
  WizardQueueItem,
} from "./types"

export { detectTypeFromUrl } from "./queue-items"

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

const PASSIVE_ALERT_PROPS = {
  role: "status",
  "aria-live": "polite",
} as const

type AddContentStepProps = {
  oneScreen?: boolean
  processingItems?: WizardQueueItem[]
  isOnlineForIngest?: boolean
  isCheckingConnection?: boolean
  connectionRecoveryMessage?: string
  onRetryConnection?: () => void
  onQuickProcess?: () => void
  quickProcessWarning?: string | null
}

export const AddContentStep: React.FC<AddContentStepProps> = ({
  oneScreen = false,
  processingItems,
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

  const [fileRejections, setFileRejections] = useState<string[]>([])
  const [urlInput, setUrlInput] = useState("")
  const [pastedTextInput, setPastedTextInput] = useState("")
  const [playlistPreflightUrl, setPlaylistPreflightUrl] = useState("")
  const [playlistPreflight, setPlaylistPreflight] = useState<PlaylistPreflightResult | null>(null)
  const [duplicatePolicy, setDuplicatePolicy] = useState<ConferenceDuplicatePolicy>("skip")
  const [playlistPreflightLoading, setPlaylistPreflightLoading] = useState(false)
  const [playlistPreflightError, setPlaylistPreflightError] = useState<string | null>(null)
  const { capabilities } = useServerCapabilities()
  const seededPlaylistUrlRef = React.useRef<string | null>(null)
  const [sourceMode, setSourceMode] = useState<FirstSourceQuickIngestKind>(
    () => firstSourceAddMode ?? (queueItems.length && queueItems.every(item => item.url) ? "web_url" : "file_upload")
  )
  React.useEffect(() => {
    if (firstSourceAddMode) setSourceMode(firstSourceAddMode)
  }, [firstSourceAddMode])
  const showFiles = !oneScreen || sourceMode === "file_upload"
  const showLinks = !oneScreen || sourceMode === "web_url"
  const shouldShowPastedTextInput = (oneScreen ? sourceMode : firstSourceAddMode) === "paste_text"
  const repairId = oneScreen && state.currentStep === 1 ? state.processingState.correctionItemIds?.[0] : undefined

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
      setFileRejections([])
      const newItems: WizardQueueItem[] = []
      for (const file of files) {
        const detectedType = detectTypeFromFile(file)
        const item: WizardQueueItem = {
          id: crypto.randomUUID(),
          kind: "file",
          fileOrigin: "upload",
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

    const names = new Set(queueItems.map(item => item.fileName))
    let name = "pasted-text.txt"
    for (let index = 2; names.has(name); index++) name = `pasted-text-${index}.txt`
    const file = new File([pastedTextInput], name, {
      type: "text/plain"
    })
    const detectedType = detectTypeFromFile(file)
    const item: WizardQueueItem = {
      id: crypto.randomUUID(),
      kind: "file",
      fileOrigin: "paste",
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
    setSourceMode("web_url")
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
    () => queueItems.some(({ file }) => file && file.size >= getQuickIngestFileLimit(file).bytes * 0.8),
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
    <div className={oneScreen ? "space-y-5" : "py-3"}>
      {/* Drop zone + URL input area */}
      <div className="space-y-3">
        {oneScreen && (
          <div>
            <h3 className="mb-1 text-base font-semibold text-text">{qi("source.title", "Add sources")}</h3>
            <div role="group" aria-label={qi("source.method", "Source type")} className="flex gap-1 border-b border-border">
              {([
                ["file_upload", qi("source.files", "Files")],
                ["web_url", qi("source.links", "Links")],
                ["paste_text", qi("source.text", "Text")],
              ] as const).map(([mode, label]) => (
                <button
                  key={mode} type="button" aria-pressed={mode === sourceMode}
                  onClick={() => setSourceMode(mode)}
                  className={`min-h-11 border-b-2 px-4 text-sm font-medium focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary ${mode === sourceMode ? "border-primary text-text" : "border-transparent text-text-muted hover:text-text"}`}
                >{label}</button>
              ))}
            </div>
          </div>
        )}
        {showFiles && <>
        <FileDropZone
          singleBrowseControl={oneScreen}
          onFilesAdded={handleFilesAdded}
          onFilesRejected={setFileRejections}
          isOnlineForIngest={isOnlineForIngest}
          autoFocus={!repairId && firstSourceAddMode === "file_upload"}
        />
        {fileRejections.length > 0 && (
          <div role="alert" className="text-sm text-danger">
            {fileRejections.map((reason, index) => <p key={index}>{reason}</p>)}
          </div>
        )}
        <Typography.Text className="text-[11px] text-text-subtle">
          {oneScreen ? qi("source.fileLimits", "Documents up to {{documentLimit}} · Audio and video up to {{avLimit}}", {
            documentLimit: QUICK_INGEST_DOCUMENT_LIMIT.label, avLimit: QUICK_INGEST_AV_LIMIT.label
          }) : qi(
            "fileSizeLimits",
            "Supported: PDF, EPUB, DOCX, TXT/RTF, Markdown, HTML, XML, JSON, audio, and video. Up to {{documentLimit}} for documents and {{avLimit}} for audio and video.",
            { documentLimit: QUICK_INGEST_DOCUMENT_LIMIT.label, avLimit: QUICK_INGEST_AV_LIMIT.label }
          )}
        </Typography.Text>
        </>}
        {!oneScreen && <Typography.Text className="block text-xs text-text-muted">
          {qi(
            "wizard.addReadinessPurpose",
            "Add URLs or files. Stored items appear in Media. Knowledge search availability depends on source readiness; analysis and vector indexing are separate."
          )}
        </Typography.Text>}

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
                autoFocus={!repairId && (!oneScreen || firstSourceAddMode === "paste_text")}
                aria-label={qi("pasteTextInputAria", "Pasted text input")}
                className="flex-1"
              />
              <Button
                type={oneScreen ? "default" : "primary"}
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
              "Large files can take longer to upload."
            )}
          />
        )}

        {isExtensionRuntime() && (
          <Button
            onClick={handleCaptureTab}
            aria-busy={capturing}
            icon={capturing ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> : undefined}
            disabled={capturing}>
            {qi("captureCurrentTab", "Capture current tab")}
          </Button>
        )}
        {captureError && (
          <div role="alert" className="text-sm text-danger">
            {captureError}
          </div>
        )}

        {/* Multi-line URL paste area */}
        {showLinks && <div>
          <Typography.Text className="text-xs text-text-muted">
            {oneScreen ? qi("source.pasteLinks", "Paste links, one per line") : qi("pasteUrlsTitle", "Paste URLs (one per line):")}
          </Typography.Text>
          <div className={oneScreen ? "qi-link-entry mt-1 flex gap-2" : "mt-1 flex gap-2"}>
            <Input.TextArea
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={oneScreen ? undefined : handleUrlKeyDown}
              placeholder={qi(
                "urlsPlaceholder",
                "https://example.com/article\nhttps://youtube.com/watch?v=..."
              )}
              autoSize={{ minRows: 2, maxRows: 4 }}
              autoFocus={!repairId && firstSourceAddMode === "web_url"}
              aria-label={qi("urlsInputAria", "URL input area")}
              className="flex-1"
            />
            <Button
              type={oneScreen ? "default" : "primary"}
              onClick={handleAddUrls}
              disabled={!urlInput.trim()}
              aria-label={qi("addUrlsAria", "Add URLs to queue")}
              className="self-end"
            >
              <Plus className="mr-1 h-4 w-4" />
              {oneScreen ? qi("source.addLinks", "Add links") : qi("addUrls", "Add")}
            </Button>
          </div>
        </div>}

        {showLinks && shouldOfferPlaylistPreflight && (
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
      {oneScreen && hasItems && (
        <SourceQueue
          items={queueItems}
          repairId={repairId}
          processingItems={processingItems}
          setItems={setQueueItems}
          remove={handleRemoveItem}
          clear={handleClearAll}
          qi={qi}
        />
      )}
      {!oneScreen && hasItems && (
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
      {!oneScreen && <div className="mt-4 flex items-center justify-end gap-2">
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
      </div>}
    </div>
  )
}

export default AddContentStep

type SourceQueueProps = {
  items: WizardQueueItem[]
  repairId?: string
  processingItems?: WizardQueueItem[]
  setItems: (items: WizardQueueItem[]) => void
  remove: (id: string) => void
  clear: () => void
  qi: (key: string, fallback: string, values?: Record<string, unknown>) => string
}

function SourceQueue({ items, repairId, processingItems, setItems, remove, clear, qi }: SourceQueueProps) {
  const [editing, setEditing] = useState<{ id: string; value: string } | null>(() => {
    const source = items.find(item => item.id === repairId && !item.validation.valid)
    return source && (source.url || canPasteQueueItem(source)) ? { id: source.id, value: source.url ?? "" } : null
  })
  const submittedIds = processingItems && new Set(processingItems.map(item => item.id))
  const counts = { ready: 0, duplicate: 0, invalid: 0, unselected: 0, outside: 0 }
  const rows = items.map(item => {
    const status = getQueueItemExclusionReason(item, items)
      ?? (submittedIds && !submittedIds.has(item.id) ? "outside" : "ready")
    counts[status]++
    return { item, status }
  })
  const countSummary = [
    qi("source.readyCount", "{{count}} ready", { count: counts.ready }),
    counts.duplicate && qi("source.duplicateCount", "{count, plural, one {# duplicate} other {# duplicates}}", { count: counts.duplicate }),
    counts.invalid && qi("source.invalidCount", "{{count}} invalid", { count: counts.invalid }),
    counts.unselected && qi("source.unselectedCount", "{{count}} unselected", { count: counts.unselected }),
    counts.outside && qi("source.outsideCount", "{{count}} outside this run", { count: counts.outside }),
  ].filter(Boolean).join(" · ")
  const rowLabels = {
    ready: qi("source.ready", "Ready"),
    duplicate: qi("source.duplicate", "Duplicate · skipped"),
    invalid: qi("source.invalid", "Invalid · skipped"),
    unselected: qi("source.unselected", "Unselected · skipped"),
    outside: qi("source.outside", "Outside this run"),
  }
  function replaceItem(updated: WizardQueueItem) {
    setItems(items.map(current => current.id === updated.id ? updated : current))
  }
  function attachFile(item: WizardQueueItem, file: File, origin: "upload" | "paste") {
    const detectedType = detectTypeFromFile(file)
    const mismatch = qi("wizard.results.reattachMismatch", "Choose the original file with the same name and size.")
    replaceItem(restoreQueueFile(item, file, origin, mismatch, { detectedType, icon: ICON_NAME_MAP[detectedType] }))
  }
  function applyEdit(item: WizardQueueItem) {
    if (editing?.id !== item.id) return
    if (item.url) {
      const url = editing.value.trim()
      if (!isValidQueueUrl(url) || url === item.url) return
      const detectedType = detectTypeFromUrl(url)
      const updated = { ...item, url, detectedType, icon: ICON_NAME_MAP[detectedType] }
      replaceItem({ ...updated, validation: validateQueueItem(updated) })
    } else {
      if (!editing.value.trim() || !item.fileName) return
      attachFile(item, new File([editing.value], item.fileName, { type: "text/plain" }), "paste")
    }
    setEditing(null)
  }
  return (
    <section aria-label={qi("source.queue", "Queued sources")}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-1">
        <h3 className="text-base font-semibold text-text">
          {qi("source.count", "{count, plural, one {# source} other {# sources}}", { count: items.length })}
        </h3>
        <span role="status" aria-label={qi("source.counts", "Source counts")} className="text-sm text-text-muted">{countSummary}</span>
      </div>
      <ul className="mt-2 divide-y divide-border border-y border-border">
        {rows.map(({ item, status }) => {
          const name = item.fileName || item.url || qi("untitledItem", "Untitled")
          const isEditing = editing?.id === item.id
          const needsRepair = status === "invalid" && (!repairId || repairId === item.id)
          return (
            <li key={item.id} tabIndex={-1} className="qi-source-row grid min-w-0 items-center gap-x-2 py-2">
              {MEDIA_TYPE_ICONS[item.detectedType]}
              <div className="min-w-0">
                <p title={name} className="break-words text-sm font-medium text-text [overflow-wrap:anywhere]">{name}</p>
                <p className={`text-xs ${status === "invalid" ? "text-danger" : "text-text-muted"}`}>
                  {rowLabels[status]}{item.fileSize > 0 && ` · ${formatFileSize(item.fileSize)}`}
                </p>
                {item.validation.errors?.filter(message => message !== "Invalid URL format").map(message =>
                  <p key={message} role="alert" className="text-xs text-danger">{message}</p>)}
                {item.validation.warnings?.map(message => <p key={message} className="text-xs text-warn">{message}</p>)}
              </div>
              {needsRepair && item.url && !isEditing && (
                <Button className="qi-source-repair" type="link" onClick={() => setEditing({ id: item.id, value: item.url! })}>
                  {qi("source.edit", "Edit link")}
                </Button>
              )}
              {needsRepair && canPasteQueueItem(item) && !isEditing && (
                <Button className="qi-source-repair" type="link" onClick={() => setEditing({ id: item.id, value: "" })}>
                  {qi("source.pasteAgain", "Paste again")}
                </Button>
              )}
              {status === "duplicate" && (
                <Button className="qi-source-repair" type="link" onClick={() => setItems(items.map(current =>
                  current.id === item.id ? { ...current, processAgain: true } : current))}>
                  {qi("source.includeAgain", "Include again")}
                </Button>
              )}
              <button
                type="button" onClick={() => remove(item.id)}
                className="qi-source-remove inline-flex h-11 w-11 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-danger focus-visible:outline"
                aria-label={qi("source.remove", "Remove {{name}}", { name })}
              ><X className="h-4 w-4" aria-hidden="true" /></button>
              {needsRepair && item.fileName && item.fileOrigin !== "paste" && !isEditing && (
                <label className="col-start-2 col-span-2 inline-flex min-h-11 cursor-pointer items-center justify-self-start rounded text-sm text-primary focus-within:outline">
                  {qi("source.chooseFile", "Choose file")}
                  <input type="file" accept={QUICK_INGEST_ACCEPT_STRING} autoFocus={repairId === item.id}
                    aria-label={qi("source.chooseFileFor", "Choose file for {{name}}", { name })}
                    className="sr-only"
                    onChange={event => {
                      const file = event.currentTarget.files?.[0]
                      if (file) {
                        event.currentTarget.closest("li")?.focus()
                        attachFile(item, file, "upload")
                      }
                      event.currentTarget.value = ""
                    }} />
                </label>
              )}
              {editing && isEditing && (
                <form className="col-span-full space-y-1 py-1" onSubmit={event => {
                  event.preventDefault()
                  event.currentTarget.closest("li")?.focus()
                  applyEdit(item)
                }}>
                  <label className="block text-sm">
                    {item.url ? qi("source.correctLink", "Correct link") : qi("source.pasteAgain", "Paste again")}
                    {item.url ? <input type="url" autoFocus value={editing.value}
                      onChange={event => setEditing({ id: item.id, value: event.target.value })}
                      className="mt-1 block w-full rounded border border-border bg-surface px-2 py-2 text-base text-text" />
                      : <textarea autoFocus rows={4} value={editing.value}
                        onChange={event => setEditing({ id: item.id, value: event.target.value })}
                        className="mt-1 block w-full rounded border border-border bg-surface px-2 py-2 text-base text-text" />}
                  </label>
                  {item.url && <p className="text-xs text-text-muted">{qi("source.linkHelp", "Enter a full link starting with http:// or https://.")}</p>}
                  <div className="flex gap-2">
                    <Button htmlType="submit" disabled={item.url ? !isValidQueueUrl(editing.value) || editing.value.trim() === item.url : !editing.value.trim()}>
                      {item.url ? qi("source.apply", "Apply link") : qi("source.applyText", "Apply text")}
                    </Button>
                    <Button type="text" onClick={event => { event.currentTarget.closest("li")?.focus(); setEditing(null) }}>
                      {qi("source.cancelEdit", "Cancel edit")}
                    </Button>
                  </div>
                </form>
              )}
            </li>
          )
        })}
      </ul>
      {items.length > 1 && <Button size="small" type="text" onClick={clear} className="mt-1">
        {qi("clearAll", "Clear all")}
      </Button>}
    </section>
  )
}
