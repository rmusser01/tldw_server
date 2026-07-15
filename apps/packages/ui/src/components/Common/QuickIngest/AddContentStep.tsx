import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button, Input, Tooltip, Typography } from "antd"
import { useVirtualizer } from "@tanstack/react-virtual"
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
  QueueItemValidation,
  WizardQueueItem,
} from "./types"
import { useIngestWizard } from "./IngestWizardContext"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { Alert as DesignSystemAlert, Badge } from "@/components/ui/primitives"
import { FileDropZone } from "./QueueTab/FileDropZone"
import { BatchMetadataPanel } from "./BatchMetadataPanel"
import { usePlaylistInspection } from "./usePlaylistInspection"
import { PlaylistPreflightPanel } from "./PlaylistPreflightPanel"
import {
  QUICK_INGEST_MAX_FILE_SIZE_LABEL,
  QUICK_INGEST_MAX_FILE_SIZE,
} from "./constants"
import { normalizeUrlForDedupe } from "@/entries/shared/ingest-payloads"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  PlaylistIngestPublicError,
  toPlaylistIngestPublicError,
} from "@/services/tldw/playlist-ingest"
import type { PlaylistInspectionCandidate } from "./usePlaylistInspection"

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
  const normalized = String(mimeType || "")
    .trim()
    .toLowerCase()
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

const playlistSourceAliases = (
  normalizedSourceId?: string | null,
  sourceUrl?: string | null
): string[] => {
  const aliases = [
    normalizedSourceId?.trim() || "",
    sourceUrl ? normalizeUrlForDedupe(sourceUrl) : "",
  ]
  return [...new Set(aliases.filter(Boolean))]
}

const queueSourceAliases = (item: WizardQueueItem): string[] => [
  ...playlistSourceAliases(item.playlist?.normalizedSourceId, item.url),
  ...(item.sourceRef?.kind === "direct_url"
    ? playlistSourceAliases(null, item.sourceRef.url)
    : []),
]

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
  const { state, updateQueueItems, setPlaylistPreflightSeed, goNext } = useIngestWizard()
  const { queueItems, playlistPreflightSeed, firstSourceAddMode } = state

  const [urlInput, setUrlInput] = useState("")
  const [pastedTextInput, setPastedTextInput] = useState("")
  const [materializingKeys, setMaterializingKeys] = useState<Set<string>>(() => new Set())
  const [materializationErrors, setMaterializationErrors] = useState<Record<string, string>>({})
  const [pendingMaterializationCommits, setPendingMaterializationCommits] = useState<
    Array<{ candidateKey: string; materializationId: string; occurrenceIds: string[] }>
  >([])
  const [queuePlaylistFilter, setQueuePlaylistFilter] = useState("all")
  const [queueTypeFilter, setQueueTypeFilter] = useState("all")
  const [queueDuplicateFilter, setQueueDuplicateFilter] = useState("all")
  const materializingKeysRef = useRef(new Set<string>())
  const queueListRef = useRef<HTMLDivElement | null>(null)
  const queueRowRefs = useRef(new Map<string, HTMLDivElement>())
  const queueListOwnsFocusRef = useRef(false)
  const activeQueueRowRef = useRef<{ id: string; index: number } | null>(null)
  const [activeQueueId, setActiveQueueId] = useState<string | null>(null)
  const unlockMaterialization = useCallback((candidateKey: string) => {
    materializingKeysRef.current.delete(candidateKey)
    setMaterializingKeys(new Set(materializingKeysRef.current))
  }, [])
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
  useEffect(() => {
    if (pendingMaterializationCommits.length === 0) return
    for (const commit of pendingMaterializationCommits) {
      const committed = commit.occurrenceIds.every((occurrenceId) =>
        queueItems.some(
          (item) =>
            item.id === occurrenceId &&
            item.sourceRef?.kind === "materialized_playlist_item" &&
            item.sourceRef.materializationId === commit.materializationId &&
            item.sourceRef.occurrenceId === occurrenceId
        )
      )
      if (committed) {
        playlistInspection.removeCandidate(commit.candidateKey)
      } else {
        const message = new PlaylistIngestPublicError("invalid_occurrence_selection").message
        setMaterializationErrors((current) => ({
          ...current,
          [commit.candidateKey]: message,
        }))
      }
      unlockMaterialization(commit.candidateKey)
    }
    setPendingMaterializationCommits([])
  }, [pendingMaterializationCommits, playlistInspection, queueItems, unlockMaterialization])
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
      const newItems = files.map((file): WizardQueueItem => {
        const detectedType = detectTypeFromFile(file)
        const id = crypto.randomUUID()
        return {
          id,
          sourceRef: { kind: "file_stub", occurrenceId: id },
          fileName: file.name,
          file,
          detectedType,
          icon: ICON_NAME_MAP[detectedType],
          fileSize: file.size,
          mimeType: file.type || undefined,
          validation: { valid: true },
        }
      })
      updateQueueItems((current) => {
        const validated: WizardQueueItem[] = []
        for (const item of newItems) {
          validated.push({
            ...item,
            validation: validateQueueItem(item, [...current, ...validated]),
          })
        }
        return [...current, ...validated]
      })
    },
    [updateQueueItems]
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
      const id = crypto.randomUUID()
      const item: WizardQueueItem = {
        id,
        sourceRef: { kind: "direct_url", occurrenceId: id, url },
        url,
        detectedType,
        icon: ICON_NAME_MAP[detectedType],
        fileSize: 0,
        validation: { valid: true },
      }
      newItems.push(item)
    }

    if (newItems.length > 0) {
      updateQueueItems((current) => {
        const validated: WizardQueueItem[] = []
        for (const item of newItems) {
          validated.push({
            ...item,
            validation: validateQueueItem(item, [...current, ...validated]),
          })
        }
        return [...current, ...validated]
      })
    }
    addPlaylistCandidates(playlistCandidates)
    setUrlInput("")
  }, [addPlaylistCandidates, updateQueueItems, urlInput])

  const handleAddPastedText = useCallback(() => {
    if (!pastedTextInput.trim()) return

    const file = new File([pastedTextInput], "pasted-text.txt", {
      type: "text/plain",
    })
    const detectedType = detectTypeFromFile(file)
    const id = crypto.randomUUID()
    const item: WizardQueueItem = {
      id,
      sourceRef: { kind: "file_stub", occurrenceId: id },
      fileName: file.name,
      file,
      detectedType,
      icon: ICON_NAME_MAP[detectedType],
      fileSize: file.size,
      mimeType: file.type || undefined,
      validation: { valid: true },
    }
    updateQueueItems((current) => [
      ...current,
      { ...item, validation: validateQueueItem(item, current) },
    ])
    setPastedTextInput("")
  }, [pastedTextInput, updateQueueItems])

  const handleMaterializeCandidate = useCallback(
    async (candidate: PlaylistInspectionCandidate) => {
      if (
        candidate.status !== "ready" ||
        !candidate.preflightId ||
        materializingKeysRef.current.has(candidate.key)
      ) {
        return
      }
      const selectedOccurrenceIds = candidate.items
        .filter(
          (item) =>
            candidate.selectedOccurrenceIds.has(item.occurrenceId) &&
            Boolean(item.sourceUrl) &&
            (item.availability === null || item.availability === "available")
        )
        .sort((left, right) => left.ordinal - right.ordinal)
        .map((item) => item.occurrenceId)
      if (selectedOccurrenceIds.length === 0) return

      materializingKeysRef.current.add(candidate.key)
      setMaterializingKeys(new Set(materializingKeysRef.current))
      setMaterializationErrors((current) => {
        if (!(candidate.key in current)) return current
        const next = { ...current }
        delete next[candidate.key]
        return next
      })
      try {
        const materialization = await tldwClient.materializePlaylistPreflight(
          candidate.preflightId,
          selectedOccurrenceIds
        )
        const expected = new Set(selectedOccurrenceIds)
        const returned = new Set(materialization.items.map((item) => item.occurrenceId))
        if (
          materialization.preflightId !== candidate.preflightId ||
          !materialization.materializationId ||
          materialization.items.length !== selectedOccurrenceIds.length ||
          returned.size !== expected.size ||
          [...expected].some((occurrenceId) => !returned.has(occurrenceId)) ||
          materialization.items.some((item) => !item.sourceUrl)
        ) {
          throw new PlaylistIngestPublicError("invalid_occurrence_selection")
        }
        const existingQueueIds = new Set(queueItems.map((item) => item.id))
        if ([...returned].some((occurrenceId) => existingQueueIds.has(occurrenceId))) {
          throw new PlaylistIngestPublicError("invalid_occurrence_selection")
        }

        const inspectedByOccurrence = new Map(
          candidate.items.map((item) => [item.occurrenceId, item] as const)
        )
        const sortedMaterializedItems = [...materialization.items].sort(
          (left, right) => left.ordinal - right.ordinal
        )
        updateQueueItems((current) => {
          if (
            sortedMaterializedItems.some((item) =>
              current.some((row) => row.id === item.occurrenceId)
            )
          ) {
            return current
          }
          const authoritativeAliases = new Set(current.flatMap(queueSourceAliases))
          const newItems: WizardQueueItem[] = sortedMaterializedItems.map((item) => {
            const inspected = inspectedByOccurrence.get(item.occurrenceId)
            const normalizedSourceId = item.normalizedSourceId?.trim() || null
            const aliases = new Set([
              ...playlistSourceAliases(normalizedSourceId, item.sourceUrl),
              ...playlistSourceAliases(inspected?.normalizedSourceId, inspected?.sourceUrl),
            ])
            const overlapsSelectedOrQueued = [...aliases].some((alias) =>
              authoritativeAliases.has(alias)
            )
            const duplicateTargetWasExcluded = Boolean(
              inspected?.duplicateStatus === "duplicate_in_batch" &&
                inspected.duplicateOfOccurrenceId &&
                !expected.has(inspected.duplicateOfOccurrenceId)
            )
            const duplicateStatus =
              inspected?.duplicateStatus === "duplicate_existing"
                ? "duplicate_existing"
                : overlapsSelectedOrQueued
                  ? "duplicate_in_batch"
                  : inspected?.duplicateStatus === "unknown"
                    ? "unknown"
                    : inspected?.duplicateStatus === "duplicate_in_batch" &&
                        !duplicateTargetWasExcluded
                      ? "duplicate_in_batch"
                      : aliases.size > 0
                        ? "new"
                        : "unknown"
            for (const alias of aliases) authoritativeAliases.add(alias)
            return {
              id: item.occurrenceId,
              kind: "url",
              sourceRef: {
                kind: "materialized_playlist_item",
                materializationId: materialization.materializationId,
                occurrenceId: item.occurrenceId,
              },
              // Display-only cache. Run serialization uses sourceRef exclusively.
              url: item.sourceUrl,
              detectedType: "video",
              icon: ICON_NAME_MAP.video,
              fileSize: 0,
              validation: { valid: true },
              playlist: {
                playlistId: item.displayMetadata.playlistId ?? candidate.summary?.playlistId,
                playlistTitle:
                  item.displayMetadata.playlistTitle ?? candidate.summary?.summary?.playlistTitle,
                ordinal: item.ordinal,
                title: item.displayMetadata.title,
                channelOrUploader: item.displayMetadata.channelOrUploader,
                durationSeconds: item.displayMetadata.durationSeconds,
                normalizedSourceId,
                duplicateStatus,
                sourceUrl: item.sourceUrl,
                materializationExpiresAt: materialization.expiresAt,
              },
              playlistReview: { selected: true },
            }
          })
          return [...current, ...newItems]
        })
        setPendingMaterializationCommits((current) => [
          ...current.filter((commit) => commit.candidateKey !== candidate.key),
          {
            candidateKey: candidate.key,
            materializationId: materialization.materializationId,
            occurrenceIds: sortedMaterializedItems.map((item) => item.occurrenceId),
          },
        ])
      } catch (error) {
        const publicError = toPlaylistIngestPublicError(error)
        setMaterializationErrors((current) => ({
          ...current,
          [candidate.key]: publicError.message,
        }))
        unlockMaterialization(candidate.key)
      }
    },
    [queueItems, unlockMaterialization, updateQueueItems]
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
      updateQueueItems((current) => current.filter((item) => item.id !== id))
    },
    [updateQueueItems]
  )

  // Clear all items
  const handleClearAll = useCallback(() => {
    updateQueueItems(() => [])
  }, [updateQueueItems])

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
    () => queueItems.some((item) => item.detectedType === "audio" || item.detectedType === "video"),
    [queueItems]
  )
  const hasPlaylistItems = useMemo(
    () => queueItems.some((item) => Boolean(item.playlist)),
    [queueItems]
  )
  useEffect(() => {
    if (hasPlaylistItems) return
    setQueuePlaylistFilter("all")
    setQueueTypeFilter("all")
    setQueueDuplicateFilter("all")
  }, [hasPlaylistItems])
  const queuePlaylistOptions = useMemo(
    () =>
      Array.from(
        new Set(
          queueItems
            .map((item) => item.playlist?.playlistTitle?.trim())
            .filter((title): title is string => Boolean(title))
        )
      ).sort((left, right) => left.localeCompare(right)),
    [queueItems]
  )
  const queueTypeOptions = useMemo(
    () =>
      Array.from(new Set(queueItems.map((item) => item.detectedType))).sort((left, right) =>
        left.localeCompare(right)
      ),
    [queueItems]
  )
  const filteredQueueItems = useMemo(
    () =>
      queueItems.filter((item) => {
        if (queuePlaylistFilter !== "all" && item.playlist?.playlistTitle !== queuePlaylistFilter) {
          return false
        }
        if (queueTypeFilter !== "all" && item.detectedType !== queueTypeFilter) {
          return false
        }
        const duplicateStatus = item.playlist?.duplicateStatus ?? "unknown"
        if (queueDuplicateFilter === "duplicates") {
          return (
            duplicateStatus === "duplicate_existing" || duplicateStatus === "duplicate_in_batch"
          )
        }
        if (queueDuplicateFilter !== "all" && duplicateStatus !== queueDuplicateFilter) {
          return false
        }
        return true
      }),
    [queueDuplicateFilter, queueItems, queuePlaylistFilter, queueTypeFilter]
  )
  const queueVirtualizer = useVirtualizer({
    count: filteredQueueItems.length,
    getScrollElement: () => queueListRef.current,
    estimateSize: () => 76,
    overscan: 6,
    getItemKey: (index) => filteredQueueItems[index]?.id ?? index,
    measureElement: (element) => element?.getBoundingClientRect().height || 76,
  })
  const queueVirtualItems = queueVirtualizer.getVirtualItems()
  const restoreQueueRowFocus = useCallback((id: string) => {
    const attempt = (remaining: number) => {
      if (!queueListOwnsFocusRef.current) return
      const row = queueRowRefs.current.get(id)
      if (row) {
        row.focus()
        return
      }
      if (remaining > 0) window.requestAnimationFrame(() => attempt(remaining - 1))
    }
    window.requestAnimationFrame(() => attempt(2))
  }, [])

  useEffect(() => {
    const handleFocusIn = (event: FocusEvent) => {
      const target = event.target
      if (target instanceof Node && queueListRef.current?.contains(target)) return
      queueListOwnsFocusRef.current = false
    }
    document.addEventListener("focusin", handleFocusIn)
    return () => document.removeEventListener("focusin", handleFocusIn)
  }, [])

  useEffect(() => {
    if (filteredQueueItems.length === 0) {
      activeQueueRowRef.current = null
      setActiveQueueId(null)
      return
    }
    const active = activeQueueRowRef.current
    if (!active) {
      const id = filteredQueueItems[0].id
      activeQueueRowRef.current = { id, index: 0 }
      setActiveQueueId(id)
      return
    }
    const currentIndex = filteredQueueItems.findIndex((item) => item.id === active.id)
    if (currentIndex >= 0) {
      active.index = currentIndex
      if (
        queueListOwnsFocusRef.current &&
        !queueRowRefs.current.has(active.id) &&
        queueVirtualItems.length > 0
      ) {
        const nearest = queueVirtualItems.reduce((best, row) =>
          Math.abs(row.index - currentIndex) < Math.abs(best.index - currentIndex) ? row : best
        )
        const nearestItem = filteredQueueItems[nearest.index]
        if (nearestItem) {
          activeQueueRowRef.current = { id: nearestItem.id, index: nearest.index }
          setActiveQueueId(nearestItem.id)
          restoreQueueRowFocus(nearestItem.id)
        }
      }
      return
    }
    const index = Math.min(active.index, filteredQueueItems.length - 1)
    const id = filteredQueueItems[index].id
    activeQueueRowRef.current = { id, index }
    setActiveQueueId(id)
    if (queueListOwnsFocusRef.current) {
      queueVirtualizer.scrollToIndex(index, { align: "auto" })
      restoreQueueRowFocus(id)
    }
  }, [filteredQueueItems, queueVirtualItems, queueVirtualizer, restoreQueueRowFocus])

  const handleQueueRowKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>, index: number) => {
      if (event.target !== event.currentTarget) {
        if (event.key === "Escape") {
          event.preventDefault()
          event.currentTarget.focus()
        }
        return
      }
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return
      event.preventDefault()
      const nextIndex = Math.max(
        0,
        Math.min(filteredQueueItems.length - 1, index + (event.key === "ArrowDown" ? 1 : -1))
      )
      const nextItem = filteredQueueItems[nextIndex]
      if (!nextItem) return
      activeQueueRowRef.current = { id: nextItem.id, index: nextIndex }
      setActiveQueueId(nextItem.id)
      queueVirtualizer.scrollToIndex(nextIndex, { align: "auto" })
      restoreQueueRowFocus(nextItem.id)
    },
    [filteredQueueItems, queueVirtualizer, restoreQueueRowFocus]
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
            {playlistInspection.candidates.map((candidate) => (
              <PlaylistPreflightPanel
                key={candidate.key}
                candidate={candidate}
                qi={qi}
                onCancel={() => playlistInspection.cancelCandidate(candidate.key)}
                onRetry={() => playlistInspection.retryCandidate(candidate.key)}
                onRemove={() => {
                  if (materializingKeysRef.current.has(candidate.key)) return
                  playlistInspection.removeCandidate(candidate.key)
                }}
                onRefresh={() => {
                  if (materializingKeysRef.current.has(candidate.key)) return
                  playlistInspection.refreshCandidate(candidate.key)
                }}
                onAdd={() => void handleMaterializeCandidate(candidate)}
                isAdding={materializingKeys.has(candidate.key)}
                addError={materializationErrors[candidate.key] ?? null}
                onSelectionChange={(occurrenceId, selected) => {
                  if (materializingKeysRef.current.has(candidate.key)) return
                  playlistInspection.setCandidateSelection(candidate.key, occurrenceId, selected)
                }}
                onSelectionBatchChange={(updates) => {
                  if (materializingKeysRef.current.has(candidate.key)) return
                  playlistInspection.setCandidateSelections(candidate.key, updates)
                }}
              />
            ))}
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
                  {qi("queueValiditySummary", "{{valid}} valid / {{invalid}} invalid", {
                    valid: validItemCount,
                    invalid: invalidItemCount,
                  })}
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

          {hasPlaylistItems && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <select
                aria-label="Filter queued items by playlist"
                className="rounded border border-border bg-surface px-2 py-1 text-xs"
                value={queuePlaylistFilter}
                onChange={(event) => setQueuePlaylistFilter(event.target.value)}
              >
                <option value="all">All playlists</option>
                {queuePlaylistOptions.map((title) => (
                  <option key={title} value={title}>
                    {title}
                  </option>
                ))}
              </select>
              <select
                aria-label="Filter queued items by type"
                className="rounded border border-border bg-surface px-2 py-1 text-xs"
                value={queueTypeFilter}
                onChange={(event) => setQueueTypeFilter(event.target.value)}
              >
                <option value="all">All types</option>
                {queueTypeOptions.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
              <select
                aria-label="Filter queued items by duplicate state"
                className="rounded border border-border bg-surface px-2 py-1 text-xs"
                value={queueDuplicateFilter}
                onChange={(event) => setQueueDuplicateFilter(event.target.value)}
              >
                <option value="all">All duplicate states</option>
                <option value="new">New</option>
                <option value="duplicates">Duplicates</option>
                <option value="unknown">Unknown</option>
              </select>
              <span className="text-xs text-text-muted" role="status" aria-live="polite">
                Showing {filteredQueueItems.length} of {queueItems.length} queued items
              </span>
            </div>
          )}

          <div
            ref={queueListRef}
            className="mt-2 max-h-96 overflow-y-auto"
            role="list"
            aria-label="Queued ingest items"
          >
            <div className="relative w-full" style={{ height: queueVirtualizer.getTotalSize() }}>
              {queueVirtualItems.map((virtualRow) => {
                const item = filteredQueueItems[virtualRow.index]
                if (!item) return null
                return (
                  <div
                    key={virtualRow.key}
                    ref={(element) => {
                      queueVirtualizer.measureElement(element)
                      if (element) queueRowRefs.current.set(item.id, element)
                      else queueRowRefs.current.delete(item.id)
                    }}
                    role="listitem"
                    tabIndex={activeQueueId === item.id ? 0 : -1}
                    aria-setsize={filteredQueueItems.length}
                    aria-posinset={virtualRow.index + 1}
                    data-occurrence-id={item.sourceRef?.occurrenceId || item.id}
                    data-index={virtualRow.index}
                    onFocusCapture={() => {
                      queueListOwnsFocusRef.current = true
                      activeQueueRowRef.current = { id: item.id, index: virtualRow.index }
                      setActiveQueueId(item.id)
                    }}
                    onKeyDown={(event) => handleQueueRowKeyDown(event, virtualRow.index)}
                    className={`absolute left-0 top-0 flex w-full items-center gap-3 rounded-md border px-3 py-2 ${
                      !item.validation.valid
                        ? "border-danger/30 bg-danger/5"
                        : item.validation.warnings?.length
                          ? "border-warn/30 bg-warn/5"
                          : "border-border"
                    }`}
                    style={{ transform: `translateY(${virtualRow.start}px)` }}
                  >
                    {/* Type icon */}
                    <span className="flex-shrink-0">
                      {MEDIA_TYPE_ICONS[item.detectedType]}
                    </span>

                    {/* Name/URL and metadata */}
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">
                        {item.playlist?.ordinal && item.playlist?.title
                          ? `${item.playlist.ordinal}. ${item.playlist.title}`
                          : item.fileName || item.url || qi("untitledItem", "Untitled")}
                      </div>
                      <div className="flex items-center gap-2 text-[11px] text-text-muted">
                        {item.playlist?.playlistTitle && (
                          <span>{item.playlist.playlistTitle}</span>
                        )}
                        {item.fileSize > 0 && <span>{formatFileSize(item.fileSize)}</span>}
                        {ffmpegMissing &&
                        (item.detectedType === "audio" || item.detectedType === "video") ? (
                          <Tooltip
                            title={qi(
                              "ffmpegRequiredTooltip",
                              "FFmpeg is not installed on the server -- this file may fail to process"
                            )}
                          >
                            <Badge variant="warning" size="sm" className="!m-0">
                              <AlertTriangle className="mr-0.5 h-3 w-3" aria-hidden="true" />
                              {item.detectedType.charAt(0).toUpperCase() +
                                item.detectedType.slice(1)}
                            </Badge>
                          </Tooltip>
                        ) : (
                          <Badge variant="info" size="sm" className="!m-0">
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
                        <div key={`e-${i}`} className="mt-0.5 text-[11px] text-danger">
                          {err}
                        </div>
                      ))}
                      {item.validation.warnings?.map((warn, i) => (
                        <div key={`w-${i}`} className="mt-0.5 text-[11px] text-warn">
                          {warn}
                        </div>
                      ))}
                      {item.sourceRef?.kind === "materialized_playlist_item" && item.url && (
                        <details className="text-[11px] text-text-muted">
                          <summary>Source details</summary>
                          <span>{item.url}</span>
                        </details>
                      )}
                    </div>

                    {/* Remove button */}
                    <button
                      type="button"
                      onClick={() => handleRemoveItem(item.id)}
                      className="flex-shrink-0 rounded p-1 text-text-muted transition-colors hover:bg-surface2 hover:text-danger"
                      aria-label={qi("removeItemAria", "Remove this item from queue")}
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                )
              })}
            </div>
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
