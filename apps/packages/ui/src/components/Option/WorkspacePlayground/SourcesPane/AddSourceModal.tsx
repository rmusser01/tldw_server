import React from "react"
import { useTranslation } from "react-i18next"
import {
  Modal,
  Tabs,
  Upload,
  Input,
  Button,
  Alert,
  Spin,
  List,
  Checkbox,
  Empty,
  message,
  Progress
} from "antd"
import type { UploadProps } from "antd"
import {
  Upload as UploadIcon,
  Link,
  FileText,
  Search,
  Database,
  X,
  Loader2
} from "lucide-react"
import { useWorkspaceStore } from "@/store/workspace"
import { useMobile } from "@/hooks/useMediaQuery"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { inferUploadMediaTypeFromFile } from "@/services/tldw/media-routing"
import type {
  AddSourceTab,
  WorkspaceSourceStatus,
  WorkspaceSourceType
} from "@/types/workspace"
import {
  buildSourceUploadAccept,
  formatSourceUploadSizeLimit,
  getConfiguredSourceUploadMaxSizeBytes,
  mapSourceIngestionError,
  parseSourceCreatedAt,
  validateSourceUploadFile
} from "./source-ingestion-utils"

const { TextArea } = Input
const { Dragger } = Upload
const EXISTING_MEDIA_CACHE_TTL_MS = 60_000
const ADD_SOURCE_TAB_USAGE_STORAGE_KEY =
  "tldw:workspace-playground:add-source-tab-usage:v1"
const DEFAULT_ADD_SOURCE_TAB_ORDER: AddSourceTab[] = [
  "upload",
  "existing",
  "url",
  "paste",
  "search"
]
const WORKSPACE_RAG_INGEST_FIELDS = {
  perform_chunking: "true",
  generate_embeddings: "true",
  embedding_dispatch_mode: "background"
} as const
const WORKSPACE_UPLOAD_INGEST_FIELDS = {
  overwrite: "false",
  ...WORKSPACE_RAG_INGEST_FIELDS
} as const

let existingMediaCache: { items: any[]; totalCount: number; cachedAt: number } | null =
  null

type AddSourceCandidate = {
  mediaId: number
  title: string
  type: WorkspaceSourceType
  status?: WorkspaceSourceStatus
  statusMessage?: string
  url?: string
  fileSize?: number
  duration?: number
  pageCount?: number
  sourceCreatedAt?: Date
  thumbnailUrl?: string
}

type AddSourceHandler = (
  sources: AddSourceCandidate[],
  options?: {
    closeModal?: boolean
  }
) => Promise<void>

type UploadProgressStatus = "uploading" | "processing" | "error"

type UploadProgressEntry = {
  id: string
  fileName: string
  bytesUploaded: number | null
  totalBytes: number
  status: UploadProgressStatus
  message?: string
}

type AddSourceTabUsage = Record<AddSourceTab, number>

const buildDefaultAddSourceTabUsage = (): AddSourceTabUsage => ({
  upload: 0,
  existing: 0,
  url: 0,
  paste: 0,
  search: 0
})

const normalizeAddSourceTabUsage = (raw: unknown): AddSourceTabUsage => {
  const next = buildDefaultAddSourceTabUsage()
  if (!raw || typeof raw !== "object") {
    return next
  }

  for (const tab of DEFAULT_ADD_SOURCE_TAB_ORDER) {
    const value = (raw as Record<string, unknown>)[tab]
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
      next[tab] = Math.trunc(value)
    }
  }

  return next
}

const readAddSourceTabUsage = (): AddSourceTabUsage => {
  if (typeof window === "undefined") {
    return buildDefaultAddSourceTabUsage()
  }
  try {
    const raw = window.localStorage.getItem(ADD_SOURCE_TAB_USAGE_STORAGE_KEY)
    if (!raw) return buildDefaultAddSourceTabUsage()
    return normalizeAddSourceTabUsage(JSON.parse(raw))
  } catch {
    return buildDefaultAddSourceTabUsage()
  }
}

const persistAddSourceTabUsage = (usage: AddSourceTabUsage) => {
  if (typeof window === "undefined") return
  try {
    window.localStorage.setItem(
      ADD_SOURCE_TAB_USAGE_STORAGE_KEY,
      JSON.stringify(usage)
    )
  } catch {
    // Ignore storage write errors.
  }
}

const isAddSourceTab = (value: string): value is AddSourceTab =>
  DEFAULT_ADD_SOURCE_TAB_ORDER.includes(value as AddSourceTab)


const toMediaId = (value: unknown): number | null => {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return null
  if (parsed <= 0) return null
  return Math.trunc(parsed)
}

const toOptionalNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value) && value >= 0) {
    return value
  }
  if (typeof value === "string") {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed >= 0) {
      return parsed
    }
  }
  return undefined
}

const toOptionalString = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

const extractCandidateMetadata = (candidate: Record<string, unknown>) => ({
  sourceCreatedAt:
    parseSourceCreatedAt(candidate.created_at) ||
    parseSourceCreatedAt(candidate.createdAt) ||
    parseSourceCreatedAt(candidate.created) ||
    parseSourceCreatedAt(candidate.date_added) ||
    parseSourceCreatedAt(candidate.dateAdded) ||
    parseSourceCreatedAt(candidate.ingested_at),
  url:
    toOptionalString(candidate.url) ||
    toOptionalString(candidate.source_url) ||
    toOptionalString(candidate.link),
  fileSize:
    toOptionalNumber(candidate.file_size) ||
    toOptionalNumber(candidate.filesize) ||
    toOptionalNumber(candidate.size),
  duration:
    toOptionalNumber(candidate.duration_seconds) ||
    toOptionalNumber(candidate.duration),
  pageCount:
    toOptionalNumber(candidate.page_count) ||
    toOptionalNumber(candidate.pages),
  thumbnailUrl:
    toOptionalString(candidate.thumbnail_url) ||
    toOptionalString(candidate.thumbnail)
})

const extractMediaFromAddResponse = (
  response: unknown
): {
  mediaId: number | null
  title?: string
  url?: string
  fileSize?: number
  duration?: number
  pageCount?: number
  sourceCreatedAt?: Date
  thumbnailUrl?: string
  mediaType?: string
} => {
  if (!response || typeof response !== "object") {
    return { mediaId: null }
  }
  const root = response as Record<string, unknown>
  const rootTitle =
    typeof root.title === "string" && root.title.trim()
      ? root.title
      : undefined
  const candidates: Array<Record<string, unknown>> = []
  if (Array.isArray(root.results)) {
    for (const item of root.results) {
      if (item && typeof item === "object") {
        candidates.push(item as Record<string, unknown>)
      }
    }
  }
  if (root.result && typeof root.result === "object") {
    candidates.push(root.result as Record<string, unknown>)
  }
  candidates.push(root)

  for (const candidate of candidates) {
    const mediaId = toMediaId(
      candidate.media_id ?? candidate.db_id ?? candidate.id
    )
    if (mediaId == null) {
      continue
    }
    const title =
      typeof candidate.title === "string" && candidate.title.trim()
        ? candidate.title
        : rootTitle
    const metadata = extractCandidateMetadata(candidate)
    return {
      mediaId,
      title,
      ...metadata,
      mediaType: toOptionalString(candidate.type) || toOptionalString(candidate.media_type)
    }
  }
  return {
    mediaId: null,
    title: rootTitle
  }
}

/**
 * UploadTab - Upload files via drag-and-drop
 */
const UploadTab: React.FC<{
  onAddSources: AddSourceHandler
  setProcessing: (p: boolean) => void
  setError: (e: string | null) => void
}> = ({ onAddSources, setProcessing, setError }) => {
  const { t } = useTranslation(["playground", "common"])
  const isMobile = useMobile()
  const draggerContainerRef = React.useRef<HTMLDivElement | null>(null)
  const [uploadProgressEntries, setUploadProgressEntries] = React.useState<
    UploadProgressEntry[]
  >([])
  const activeUploadCountRef = React.useRef(0)
  const uploadSizeLimitBytes = React.useMemo(
    () => getConfiguredSourceUploadMaxSizeBytes(),
    []
  )
  const uploadSizeLimitLabel = React.useMemo(
    () => formatSourceUploadSizeLimit(uploadSizeLimitBytes),
    [uploadSizeLimitBytes]
  )

  const beginProcessing = React.useCallback(() => {
    activeUploadCountRef.current += 1
    setProcessing(true)
  }, [setProcessing])

  const endProcessing = React.useCallback(() => {
    activeUploadCountRef.current = Math.max(0, activeUploadCountRef.current - 1)
    if (activeUploadCountRef.current === 0) {
      setProcessing(false)
    }
  }, [setProcessing])

  const upsertUploadEntry = React.useCallback(
    (
      entryId: string,
      updater: (previous: UploadProgressEntry | null) => UploadProgressEntry
    ) => {
      setUploadProgressEntries((previous) => {
        const index = previous.findIndex((entry) => entry.id === entryId)
        const existing = index >= 0 ? previous[index] : null
        const nextEntry = updater(existing)
        if (index < 0) {
          return [nextEntry, ...previous].slice(0, 8)
        }
        return previous.map((entry, currentIndex) =>
          currentIndex === index ? nextEntry : entry
        )
      })
    },
    []
  )

  const openFilePicker = React.useCallback(() => {
    const fileInput = draggerContainerRef.current?.querySelector(
      "input[type='file']"
    ) as HTMLInputElement | null
    fileInput?.click()
  }, [])

  const handleUpload = React.useCallback(
    async (file: File) => {
      const entryId = `${file.name}-${file.size}-${file.lastModified}`
      beginProcessing()
      setError(null)
      upsertUploadEntry(entryId, () => ({
        id: entryId,
        fileName: file.name,
        bytesUploaded: null,
        totalBytes: file.size,
        status: "uploading"
      }))

      try {
        const uploadMediaType = inferUploadMediaTypeFromFile(file.name, file.type)
        const response = await tldwClient.uploadMedia(file, {
          media_type: uploadMediaType,
          ...WORKSPACE_UPLOAD_INGEST_FIELDS
        })
        const added = extractMediaFromAddResponse(response)

        if (added.mediaId != null) {
          const type = added.mediaType
            ? getSourceTypeFromMediaType(added.mediaType)
            : getSourceTypeFromFile(file)
          await onAddSources([
            {
              mediaId: added.mediaId,
              title: added.title || file.name,
              type,
              status: "processing",
              url: added.url,
              fileSize: added.fileSize,
              duration: added.duration,
              pageCount: added.pageCount,
              sourceCreatedAt: added.sourceCreatedAt,
              thumbnailUrl: added.thumbnailUrl
            }
          ])
          upsertUploadEntry(entryId, (previous) => ({
            id: entryId,
            fileName: file.name,
            bytesUploaded: previous?.totalBytes ?? file.size,
            totalBytes: file.size,
            status: "processing",
            message: t(
              "playground:sources.processingInBackground",
              "Uploaded. Processing in background."
            )
          }))
        } else {
          const friendlyError = t(
            "playground:sources.uploadNoMediaId",
            "Upload completed but no media ID was returned. Please retry."
          )
          setError(friendlyError)
          upsertUploadEntry(entryId, (previous) => ({
            id: entryId,
            fileName: file.name,
            bytesUploaded: previous?.bytesUploaded ?? null,
            totalBytes: file.size,
            status: "error",
            message: friendlyError
          }))
        }
      } catch (err) {
        const friendlyError = mapSourceIngestionError(err)
        setError(friendlyError)
        upsertUploadEntry(entryId, (previous) => ({
          id: entryId,
          fileName: file.name,
          bytesUploaded: previous?.bytesUploaded ?? null,
          totalBytes: file.size,
          status: "error",
          message: friendlyError
        }))
      } finally {
        endProcessing()
      }
    },
    [beginProcessing, endProcessing, onAddSources, setError, t, upsertUploadEntry]
  )

  const uploadProps: UploadProps = {
    name: "file",
    multiple: true,
    showUploadList: false,
    beforeUpload: (file) => {
      const validation = validateSourceUploadFile(file, uploadSizeLimitBytes)
      if (validation.valid === false) {
        const extension = file.name.split(".").pop()?.toLowerCase()
        const friendlyValidationError =
          validation.code === "file_too_large"
            ? t(
                "playground:sources.uploadTooLarge",
                "{{name}} is too large. Maximum size is {{limit}}.",
                {
                  name: validation.fileName,
                  limit: uploadSizeLimitLabel
                }
              )
            : t(
                "playground:sources.uploadUnsupportedType",
                "{{name}} is not a supported file type. Upload PDF, DOCX, text, audio, or video.",
                {
                  name:
                    extension && extension.length > 0
                      ? `${validation.fileName} (.${extension})`
                      : validation.fileName
                }
              )
        setError(friendlyValidationError)
        const entryId = `${file.name}-${file.size}-${file.lastModified}`
        upsertUploadEntry(entryId, () => ({
          id: entryId,
          fileName: file.name,
          bytesUploaded: null,
          totalBytes: file.size,
          status: "error",
          message: friendlyValidationError
        }))
        return Upload.LIST_IGNORE
      }
      void handleUpload(file)
      return Upload.LIST_IGNORE
    },
    accept: buildSourceUploadAccept()
  }

  return (
    <div className="space-y-4">
      <div ref={draggerContainerRef}>
        <Dragger {...uploadProps} className="bg-surface">
          <p className="ant-upload-drag-icon">
            <UploadIcon className="mx-auto h-12 w-12 text-primary" />
          </p>
          <p className="ant-upload-text">
            {isMobile
              ? t("playground:sources.uploadTapText", "Tap to select files")
              : t(
                  "playground:sources.uploadDragText",
                  "Click or drag files to upload"
                )}
          </p>
          <p className="ant-upload-hint text-text-muted">
            {t(
              "playground:sources.uploadHint",
              "Supports PDF, documents, audio, and video files. Max {{limit}} per file.",
              { limit: uploadSizeLimitLabel }
            )}
          </p>
        </Dragger>
      </div>
      {isMobile && (
        <Button
          type="default"
          className="w-full"
          onClick={openFilePicker}
          data-testid="mobile-browse-files-button"
        >
          {t("playground:sources.browseFiles", "Browse files")}
        </Button>
      )}
      {uploadProgressEntries.length > 0 && (
        <div
          className="space-y-2 rounded border border-border bg-surface2/40 p-3"
          data-testid="upload-progress-list"
        >
          {uploadProgressEntries.map((entry) => {
            const hasNumericProgress =
              entry.bytesUploaded != null && entry.totalBytes > 0
            const percent = hasNumericProgress
              ? Math.min(
                  100,
                  Math.max(0, Math.round((entry.bytesUploaded! / entry.totalBytes) * 100))
                )
              : undefined
            return (
              <div
                key={entry.id}
                className="rounded border border-border bg-surface p-2"
                data-testid={`upload-progress-item-${entry.id}`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-medium text-text">
                    {entry.fileName}
                  </span>
                  {entry.status === "uploading" && (
                    <span className="inline-flex items-center gap-1 text-[11px] text-primary">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      {t("playground:sources.uploading", "Uploading")}
                    </span>
                  )}
                  {entry.status === "processing" && (
                    <span className="inline-flex items-center gap-1 text-[11px] text-primary">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      {t("playground:sources.processing", "Processing")}
                    </span>
                  )}
                  {entry.status === "error" && (
                    <span className="inline-flex items-center gap-1 text-[11px] text-error">
                      <X className="h-3 w-3" />
                      {t("common:error", "Error")}
                    </span>
                  )}
                </div>
                {hasNumericProgress && percent != null ? (
                  <Progress
                    percent={percent}
                    size="small"
                    status={entry.status === "error" ? "exception" : "active"}
                    className="mt-1"
                    showInfo={false}
                  />
                ) : (
                  <p className="mt-1 text-[11px] text-text-muted">
                    {entry.status === "uploading"
                      ? t(
                          "playground:sources.uploadProgressFallback",
                          "Uploading…"
                        )
                      : entry.status === "processing"
                        ? t(
                            "playground:sources.uploadQueuedProcessing",
                            "Upload complete. Processing…"
                          )
                        : entry.message ||
                          t(
                            "playground:sources.uploadProgressErrorFallback",
                            "Upload failed."
                          )}
                  </p>
                )}
                {entry.status === "error" && entry.message && (
                  <p className="mt-1 text-[11px] text-error">{entry.message}</p>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

/**
 * UrlTab - Add content from URL
 */
const UrlTab: React.FC<{
  onAddSources: AddSourceHandler
  setProcessing: (p: boolean) => void
  setError: (e: string | null) => void
}> = ({ onAddSources, setProcessing, setError }) => {
  const { t } = useTranslation(["playground", "common"])
  const [inputMode, setInputMode] = React.useState<"single" | "batch">("single")
  const [url, setUrl] = React.useState("")
  const [batchUrls, setBatchUrls] = React.useState("")
  const [batchResults, setBatchResults] = React.useState<
    Array<{
      url: string
      status: "added" | "error"
      message: string
    }>
  >([])

  const parseBatchUrls = React.useCallback((raw: string) => {
    const unique = new Set<string>()
    return raw
      .split(/\r?\n/g)
      .map((line) => line.trim())
      .filter((line) => {
        if (!line) return false
        if (unique.has(line)) return false
        unique.add(line)
        return true
      })
  }, [])

  const buildSourceFromUrlResponse = React.useCallback(
    (rawUrl: string, added: ReturnType<typeof extractMediaFromAddResponse>) => {
      if (added.mediaId == null) return null
      const type = added.mediaType
        ? getSourceTypeFromMediaType(added.mediaType)
        : getSourceTypeFromUrl(rawUrl)
      return {
        mediaId: added.mediaId,
        title: added.title || rawUrl,
        type,
        status: "processing" as const,
        url: added.url || rawUrl,
        fileSize: added.fileSize,
        duration: added.duration,
        pageCount: added.pageCount,
        sourceCreatedAt: added.sourceCreatedAt,
        thumbnailUrl: added.thumbnailUrl
      }
    },
    []
  )

  const handleAddUrl = async () => {
    const singleUrl = url.trim()
    const batchUrlList = parseBatchUrls(batchUrls)
    if (inputMode === "single" && !singleUrl) return
    if (inputMode === "batch" && batchUrlList.length === 0) return

    setProcessing(true)
    setError(null)
    setBatchResults([])

    try {
      if (inputMode === "single") {
        // Use the media/add endpoint with URL
        const response = await tldwClient.addMedia(singleUrl, {
          ...WORKSPACE_RAG_INGEST_FIELDS
        })
        const added = extractMediaFromAddResponse(response)
        const source = buildSourceFromUrlResponse(singleUrl, added)

        if (source) {
          await onAddSources([source])
          setUrl("")
          return
        }

        setError(t("playground:sources.urlError", "Failed to add URL"))
        return
      }

      const addedSources: AddSourceCandidate[] = []
      const results: Array<{
        url: string
        status: "added" | "error"
        message: string
      }> = []

      for (const rawUrl of batchUrlList) {
        try {
          const response = await tldwClient.addMedia(rawUrl, {
            ...WORKSPACE_RAG_INGEST_FIELDS
          })
          const added = extractMediaFromAddResponse(response)
          const source = buildSourceFromUrlResponse(rawUrl, added)
          if (!source) {
            results.push({
              url: rawUrl,
              status: "error",
              message: t(
                "playground:sources.batchResultInvalid",
                "No media ID returned"
              )
            })
            continue
          }

          addedSources.push(source)
          results.push({
            url: rawUrl,
            status: "added",
            message: t("playground:sources.batchUrlAdded", "Added")
          })
        } catch (error) {
          results.push({
            url: rawUrl,
            status: "error",
            message: mapSourceIngestionError(error)
          })
        }
      }

      setBatchResults(results)

      if (addedSources.length > 0) {
        const hasFailures = results.some((entry) => entry.status === "error")
        await onAddSources(addedSources, { closeModal: !hasFailures })
        if (!hasFailures) {
          setBatchUrls("")
        }
      }

      const failedCount = results.filter((entry) => entry.status === "error").length
      if (failedCount > 0) {
        setError(
          t(
            "playground:sources.batchUrlSummary",
            "Added {{added}} of {{total}} URLs. {{failed}} failed.",
            {
              added: addedSources.length,
              total: results.length,
              failed: failedCount
            }
          )
        )
      }
    } catch (err) {
      setError(mapSourceIngestionError(err))
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <div className="mb-2 inline-flex rounded border border-border bg-surface2 p-0.5">
          <button
            type="button"
            onClick={() => setInputMode("single")}
            className={`rounded px-2 py-1 text-xs font-medium ${
              inputMode === "single"
                ? "bg-surface text-text shadow-sm"
                : "text-text-muted hover:text-text"
            }`}
          >
            {t("playground:sources.urlModeSingle", "Single URL")}
          </button>
          <button
            type="button"
            onClick={() => setInputMode("batch")}
            className={`rounded px-2 py-1 text-xs font-medium ${
              inputMode === "batch"
                ? "bg-surface text-text shadow-sm"
                : "text-text-muted hover:text-text"
            }`}
          >
            {t("playground:sources.urlModeBatch", "Batch (one per line)")}
          </button>
        </div>

        {inputMode === "single" ? (
          <>
            <label className="mb-1 block text-sm font-medium text-text">
              {t("playground:sources.urlLabel", "Enter URL")}
            </label>
            <Input
              prefix={<Link className="h-4 w-4 text-text-muted" />}
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onPressEnter={handleAddUrl}
              placeholder={t(
                "playground:sources.urlPlaceholder",
                "https://example.com/article or YouTube URL"
              )}
              size="large"
            />
          </>
        ) : (
          <>
            <label className="mb-1 block text-sm font-medium text-text">
              {t("playground:sources.urlBatchLabel", "Add one URL per line")}
            </label>
            <TextArea
              value={batchUrls}
              onChange={(event) => setBatchUrls(event.target.value)}
              rows={7}
              placeholder={t(
                "playground:sources.urlBatchPlaceholder",
                "https://example.com/article-1\nhttps://example.com/article-2"
              )}
            />
          </>
        )}
      </div>
      <p className="text-xs text-text-muted">
        {t(
          "playground:sources.urlSupportHint",
          "Supports websites, YouTube videos, and direct file links"
        )}
      </p>
      <Button
        type="primary"
        onClick={handleAddUrl}
        disabled={
          inputMode === "single"
            ? !url.trim()
            : parseBatchUrls(batchUrls).length === 0
        }
        className="w-full"
      >
        {inputMode === "single"
          ? t("playground:sources.addUrl", "Add URL")
          : t("playground:sources.addUrlBatch", "Add URLs")}
      </Button>
      {inputMode === "batch" && batchResults.length > 0 && (
        <div className="max-h-40 overflow-y-auto rounded border border-border bg-surface2/30 p-2">
          {batchResults.map((entry) => (
            <div
              key={`${entry.url}-${entry.status}`}
              className="flex items-start justify-between gap-2 border-b border-border/60 py-1 text-xs last:border-b-0"
            >
              <span className="truncate text-text" title={entry.url}>
                {entry.url}
              </span>
              <span
                className={
                  entry.status === "added" ? "text-success" : "text-error"
                }
              >
                {entry.status === "added"
                  ? t("common:added", "Added")
                  : entry.message}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * PasteTab - Paste text content
 */
const PasteTab: React.FC<{
  onAddSources: AddSourceHandler
  setProcessing: (p: boolean) => void
  setError: (e: string | null) => void
}> = ({ onAddSources, setProcessing, setError }) => {
  const { t } = useTranslation(["playground", "common"])
  const [title, setTitle] = React.useState("")
  const [content, setContent] = React.useState("")

  const handleAddText = async () => {
    if (!content.trim()) return

    setProcessing(true)
    setError(null)

    try {
      // Create a text file and upload it
      const blob = new Blob([content], { type: "text/plain" })
      const file = new File([blob], `${title || "Pasted Text"}.txt`, {
        type: "text/plain"
      })

      const uploadMediaType = inferUploadMediaTypeFromFile(file.name, file.type)
      const response = await tldwClient.uploadMedia(file, {
        media_type: uploadMediaType,
        title: title || "Pasted Text",
        ...WORKSPACE_UPLOAD_INGEST_FIELDS
      })
      const added = extractMediaFromAddResponse(response)

      if (added.mediaId != null) {
        await onAddSources([
          {
            mediaId: added.mediaId,
            title: added.title || title || "Pasted Text",
            type: "text",
            status: "processing",
            fileSize: added.fileSize,
            duration: added.duration,
            pageCount: added.pageCount,
            sourceCreatedAt: added.sourceCreatedAt,
            thumbnailUrl: added.thumbnailUrl
          }
        ])
        setTitle("")
        setContent("")
      } else {
        setError(t("playground:sources.pasteError", "Failed to add text"))
      }
    } catch (err) {
      setError(mapSourceIngestionError(err))
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="space-y-4">
      <div>
        <label className="mb-1 block text-sm font-medium text-text">
          {t("playground:sources.titleLabel", "Title (optional)")}
        </label>
        <Input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder={t("playground:sources.titlePlaceholder", "Give your content a title")}
        />
      </div>
      <div>
        <label className="mb-1 block text-sm font-medium text-text">
          {t("playground:sources.contentLabel", "Content")}
        </label>
        <TextArea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder={t(
            "playground:sources.pastePlaceholder",
            "Paste your text content here..."
          )}
          rows={8}
        />
      </div>
      <div className="flex items-center justify-between text-xs text-text-muted">
        <span>
          {t("playground:sources.charCount", "{{count}} characters", {
            count: content.length
          })}
        </span>
      </div>
      <Button
        type="primary"
        onClick={handleAddText}
        disabled={!content.trim()}
        className="w-full"
      >
        {t("playground:sources.addText", "Add Text")}
      </Button>
    </div>
  )
}

/**
 * SearchTab - Web search and add results
 */
const SearchTab: React.FC<{
  onAddSources: AddSourceHandler
  setProcessing: (p: boolean) => void
  setError: (e: string | null) => void
}> = ({ onAddSources, setProcessing, setError }) => {
  const { t } = useTranslation(["playground", "common"])
  const [query, setQuery] = React.useState("")
  const [results, setResults] = React.useState<any[]>([])
  const [selectedResults, setSelectedResults] = React.useState<Set<number>>(
    new Set()
  )
  const [isSearching, setIsSearching] = React.useState(false)

  const getResultUrl = React.useCallback(
    (item: Record<string, unknown>) =>
      (item.url || item.link || item.source_url || "") as string,
    []
  )

  const getResultSnippet = React.useCallback(
    (item: Record<string, unknown>) =>
      (item.snippet ||
        item.content ||
        item.description ||
        item.summary ||
        "") as string,
    []
  )

  const getFaviconUrl = React.useCallback((rawUrl: string): string | null => {
    if (!rawUrl) return null
    try {
      const parsed = new URL(rawUrl)
      if (!parsed.hostname) return null
      return `https://www.google.com/s2/favicons?sz=32&domain=${encodeURIComponent(parsed.hostname)}`
    } catch {
      return null
    }
  }, [])

  const handleSearch = async () => {
    if (!query.trim()) return

    setIsSearching(true)
    setError(null)
    setResults([])
    setSelectedResults(new Set())

    try {
      const response = await tldwClient.webSearch({
        query: query.trim(),
        engine: "searx",
        result_count: 10
      })

      const resultsFromResponse = Array.isArray(
        response?.web_search_results_dict?.results
      )
        ? response.web_search_results_dict.results
        : Array.isArray(response?.results)
          ? response.results
          : []

      setResults(resultsFromResponse)
    } catch (err) {
      setError(mapSourceIngestionError(err))
    } finally {
      setIsSearching(false)
    }
  }

  const handleAddSelected = async () => {
    if (selectedResults.size === 0) return

    setProcessing(true)
      const selectedUrls = results.filter((_, idx) => selectedResults.has(idx))
    const addedSources: AddSourceCandidate[] = []
    const failures: Array<{ url: string; reason: string }> = []

    for (const result of selectedUrls) {
      const resultUrl = getResultUrl(result as Record<string, unknown>) || "unknown url"
      try {
        const response = await tldwClient.addMedia(resultUrl, {
          ...WORKSPACE_RAG_INGEST_FIELDS
        })
        const added = extractMediaFromAddResponse(response)
        if (added.mediaId != null) {
          addedSources.push({
            mediaId: added.mediaId,
            title: added.title || result.title || resultUrl,
            type: "website",
            status: "processing",
            url: added.url || resultUrl,
            fileSize: added.fileSize,
            duration: added.duration,
            pageCount: added.pageCount,
            sourceCreatedAt: added.sourceCreatedAt,
            thumbnailUrl: added.thumbnailUrl
          })
        } else {
          failures.push({
            url: resultUrl,
            reason: t(
              "playground:sources.batchResultInvalid",
              "No media ID returned"
            )
          })
        }
      } catch (error) {
        failures.push({
          url: resultUrl,
          reason: mapSourceIngestionError(error)
        })
      }
    }

    if (addedSources.length > 0) {
      await onAddSources(addedSources)
    }

    if (failures.length > 0) {
      const totalCount = selectedUrls.length
      const addedCount = addedSources.length
      const details = failures
        .slice(0, 2)
        .map((failure) => `${failure.url}: ${failure.reason}`)
        .join("; ")
      const overflowCount = failures.length - 2
      const detailSuffix =
        overflowCount > 0
          ? `${details}; +${overflowCount} more`
          : details

      const summary = t(
        "playground:sources.batchResultSummary",
        "Added {{added}} of {{total}} sources. {{failed}} failed: {{details}}",
        {
          added: addedCount,
          total: totalCount,
          failed: failures.length,
          details: detailSuffix
        }
      )

      if (addedCount > 0) {
        message.warning(summary)
      } else {
        setError(summary)
      }
    }
    setProcessing(false)
  }

  const toggleResult = (idx: number) => {
    const newSelected = new Set(selectedResults)
    if (newSelected.has(idx)) {
      newSelected.delete(idx)
    } else {
      newSelected.add(idx)
    }
    setSelectedResults(newSelected)
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <Input
          prefix={<Search className="h-4 w-4 text-text-muted" />}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onPressEnter={handleSearch}
          placeholder={t("playground:sources.searchPlaceholder", "Search the web...")}
          size="large"
          className="flex-1"
        />
        <Button
          type="primary"
          onClick={handleSearch}
          loading={isSearching}
          disabled={!query.trim()}
        >
          {t("common:search", "Search")}
        </Button>
      </div>

      {results.length > 0 && (
        <>
          <div className="max-h-64 overflow-y-auto rounded border border-border">
            <List
              size="small"
              dataSource={results}
              renderItem={(item, idx) => (
                (() => {
                  const record = item as Record<string, unknown>
                  const resultUrl = getResultUrl(record)
                  const resultSnippet = getResultSnippet(record)
                  const faviconUrl = getFaviconUrl(resultUrl)
                  return (
                    <List.Item
                      className={`cursor-pointer transition hover:bg-surface2 ${
                        selectedResults.has(idx) ? "bg-primary/10" : ""
                      }`}
                      onClick={() => toggleResult(idx)}
                    >
                      <div className="flex items-start gap-2">
                        <Checkbox
                          checked={selectedResults.has(idx)}
                          onChange={() => toggleResult(idx)}
                        />
                        <div className="min-w-0 flex-1">
                          <div className="flex items-start gap-2">
                            {faviconUrl ? (
                              <img
                                src={faviconUrl}
                                alt=""
                                data-testid={`search-result-favicon-${idx}`}
                                className="mt-0.5 h-4 w-4 shrink-0 rounded"
                              />
                            ) : null}
                            <div className="min-w-0 flex-1">
                              <p className="truncate text-sm font-medium text-text">
                                {item.title}
                              </p>
                              <p className="truncate text-xs text-text-muted">
                                {resultUrl}
                              </p>
                              {resultSnippet ? (
                                <p className="mt-0.5 line-clamp-2 text-xs text-text-subtle">
                                  {resultSnippet}
                                </p>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      </div>
                    </List.Item>
                  )
                })()
              )}
            />
          </div>
          <Button
            type="primary"
            onClick={handleAddSelected}
            disabled={selectedResults.size === 0}
            className="w-full"
          >
            {t("playground:sources.addSelected", "Add {{count}} selected", {
              count: selectedResults.size
            })}
          </Button>
        </>
      )}

      {isSearching && (
        <div className="flex justify-center py-8">
          <Spin />
        </div>
      )}
    </div>
  )
}

/**
 * ExistingTab - Pick from already-ingested media
 */
const ExistingTab: React.FC<{
  onAddSources: AddSourceHandler
  setProcessing: (p: boolean) => void
  setError: (e: string | null) => void
}> = ({ onAddSources, setProcessing, setError }) => {
  const { t } = useTranslation(["playground", "common"])
  const [searchQuery, setSearchQuery] = React.useState("")
  const [media, setMedia] = React.useState<any[]>([])
  const [isLoading, setIsLoading] = React.useState(false)
  const [selectedMedia, setSelectedMedia] = React.useState<Set<number>>(
    new Set()
  )
  const [currentPage, setCurrentPage] = React.useState(1)
  const [totalCount, setTotalCount] = React.useState(0)
  const [hasMore, setHasMore] = React.useState(false)

  // Already added source media IDs
  const sources = useWorkspaceStore((s) => s.sources)
  const existingMediaIds = React.useMemo(
    () => new Set(sources.map((s) => s.mediaId)),
    [sources]
  )

  const fetchMediaFromServer = React.useCallback(
    async (
      query?: string,
      options?: { silent?: boolean; page?: number; append?: boolean }
    ) => {
      const shouldShowLoading = !options?.silent
      if (shouldShowLoading) {
        setIsLoading(true)
      }
      setError(null)
      const page = options?.page || 1
      const append = Boolean(options?.append)

      try {
        const trimmedQuery = query?.trim()
        let response
        if (trimmedQuery) {
          response = await tldwClient.searchMedia(
            { query: trimmedQuery },
            { page, results_per_page: 50 }
          )
        } else {
          response = await tldwClient.listMedia({
            page,
            results_per_page: 50,
            include_keywords: true
          })
        }

        if (response?.media || response?.results) {
          const items = response.media || response.results || []
          const total = Number(
            response.total_count ??
              response.total ??
              response.count ??
              response.results_count ??
              response.pagination?.total ??
              (append ? media.length + items.length : items.length)
          )
          const normalizedTotal =
            Number.isFinite(total) && total >= 0
              ? total
              : append
                ? media.length + items.length
                : items.length
          const nextItems = append ? [...media, ...items] : items
          const dedupedItems = Array.from(
            new Map(
              nextItems.map((item: any) => [String(item.media_id || item.id), item])
            ).values()
          )

          setMedia(dedupedItems)
          setTotalCount(normalizedTotal)
          setCurrentPage(page)
          setHasMore(dedupedItems.length < normalizedTotal)

          if (!trimmedQuery && page === 1) {
            existingMediaCache = {
              items: dedupedItems,
              totalCount: normalizedTotal,
              cachedAt: Date.now()
            }
          }
        }
      } catch (err) {
        setError(mapSourceIngestionError(err))
      } finally {
        if (shouldShowLoading) {
          setIsLoading(false)
        }
      }
    },
    [media, setError]
  )

  const loadMedia = React.useCallback(
    async (query?: string) => {
      const trimmedQuery = query?.trim()
      if (!trimmedQuery && existingMediaCache) {
        const cacheIsFresh =
          Date.now() - existingMediaCache.cachedAt < EXISTING_MEDIA_CACHE_TTL_MS
        if (cacheIsFresh) {
          setMedia(existingMediaCache.items)
          setTotalCount(existingMediaCache.totalCount)
          setCurrentPage(1)
          setHasMore(existingMediaCache.items.length < existingMediaCache.totalCount)
          return
        }
      }

      setCurrentPage(1)
      await fetchMediaFromServer(trimmedQuery, { page: 1 })
    },
    [fetchMediaFromServer]
  )

  React.useEffect(() => {
    loadMedia()
  }, [loadMedia])

  const handleSearch = () => {
    setCurrentPage(1)
    void loadMedia(searchQuery)
  }

  const handleLoadMore = () => {
    if (isLoading || !hasMore) return
    void fetchMediaFromServer(searchQuery, {
      page: currentPage + 1,
      append: true
    })
  }

  const handleAddSelected = () => {
    const selectedItems = media.filter(
      (m) => selectedMedia.has(m.media_id || m.id) && !existingMediaIds.has(m.media_id || m.id)
    )

    const newSources = selectedItems.map((m) => ({
      mediaId: m.media_id || m.id,
      title: m.title || m.name || "Untitled",
      type: getSourceTypeFromMediaType(m.type || m.media_type) as WorkspaceSourceType,
      status: "ready" as const,
      url: m.url || m.source_url || undefined,
      fileSize: toOptionalNumber(m.file_size || m.filesize || m.size),
      duration: toOptionalNumber(m.duration_seconds || m.duration),
      pageCount: toOptionalNumber(m.page_count || m.pages),
      sourceCreatedAt:
        parseSourceCreatedAt(m.created_at || m.createdAt || m.date_added) ||
        undefined,
      thumbnailUrl: m.thumbnail_url || m.thumbnail || undefined
    }))

    if (newSources.length > 0) {
      void onAddSources(newSources)
    }
  }

  const toggleMedia = (id: number) => {
    const newSelected = new Set(selectedMedia)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedMedia(newSelected)
  }

  const availableMedia = media.filter(
    (m) => !existingMediaIds.has(m.media_id || m.id)
  )
  const visibleTotalCount =
    totalCount > 0
      ? Math.max(totalCount - existingMediaIds.size, availableMedia.length)
      : availableMedia.length

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <Input
          prefix={<Search className="h-4 w-4 text-text-muted" />}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onPressEnter={handleSearch}
          placeholder={t("playground:sources.searchExisting", "Search your media library...")}
          className="flex-1"
        />
        <Button onClick={handleSearch} loading={isLoading}>
          {t("common:search", "Search")}
        </Button>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-8">
          <Spin />
        </div>
      ) : availableMedia.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={t(
            "playground:sources.noMediaFound",
            "No available media found"
          )}
        />
      ) : (
        <>
          <p className="text-xs text-text-muted">
            {t(
              "playground:sources.libraryCount",
              "Showing {{shown}} of {{total}}",
              {
                shown: availableMedia.length,
                total: visibleTotalCount
              }
            )}
          </p>
          <div className="max-h-64 overflow-y-auto rounded border border-border">
            <List
              size="small"
              dataSource={availableMedia}
              renderItem={(item) => {
                const id = item.media_id || item.id
                return (
                  <List.Item
                    className={`cursor-pointer transition hover:bg-surface2 ${
                      selectedMedia.has(id) ? "bg-primary/10" : ""
                    }`}
                    onClick={() => toggleMedia(id)}
                  >
                    <div className="flex items-start gap-2">
                      <Checkbox
                        checked={selectedMedia.has(id)}
                        onChange={() => toggleMedia(id)}
                      />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-text">
                          {item.title || item.name || "Untitled"}
                        </p>
                        <p className="text-xs text-text-muted capitalize">
                          {item.type || item.media_type || "document"}
                        </p>
                      </div>
                    </div>
                  </List.Item>
                )
              }}
            />
          </div>
          <Button
            type="primary"
            onClick={handleAddSelected}
            disabled={selectedMedia.size === 0}
            className="w-full"
          >
            {t("playground:sources.addSelected", "Add {{count}} selected", {
              count: selectedMedia.size
            })}
          </Button>
          {hasMore && (
            <Button onClick={handleLoadMore} loading={isLoading} className="w-full">
              {t("playground:sources.loadMore", "Load more")}
            </Button>
          )}
        </>
      )}
    </div>
  )
}

// Helper functions
function getSourceTypeFromFile(file: File): WorkspaceSourceType {
  const ext = file.name.split(".").pop()?.toLowerCase() || ""
  const mimeType = file.type.toLowerCase()

  if (ext === "pdf" || mimeType === "application/pdf") return "pdf"
  if (["mp4", "webm", "mkv", "avi", "mov"].includes(ext) || mimeType.startsWith("video/"))
    return "video"
  if (["mp3", "wav", "m4a", "ogg", "flac"].includes(ext) || mimeType.startsWith("audio/"))
    return "audio"
  if (["doc", "docx", "odt", "rtf"].includes(ext)) return "document"
  if (["txt", "md", "markdown"].includes(ext)) return "text"
  if (["html", "htm"].includes(ext)) return "website"

  return "document"
}

function getSourceTypeFromUrl(url: string): WorkspaceSourceType {
  const urlLower = url.toLowerCase()
  if (
    urlLower.includes("youtube.com") ||
    urlLower.includes("youtu.be") ||
    urlLower.includes("vimeo.com")
  ) {
    return "video"
  }
  if (urlLower.endsWith(".pdf")) return "pdf"
  if (urlLower.match(/\.(mp3|wav|m4a|ogg|flac)$/)) return "audio"
  if (urlLower.match(/\.(mp4|webm|mkv|avi|mov)$/)) return "video"

  return "website"
}

function getSourceTypeFromMediaType(mediaType: string): WorkspaceSourceType {
  const type = mediaType?.toLowerCase() || ""
  if (type.includes("pdf")) return "pdf"
  if (type.includes("video")) return "video"
  if (type.includes("audio")) return "audio"
  if (type.includes("website") || type.includes("web") || type.includes("url"))
    return "website"
  if (type.includes("text")) return "text"
  return "document"
}

/**
 * AddSourceModal - Modal for adding sources to workspace
 */
export const AddSourceModal: React.FC = () => {
  const { t } = useTranslation(["playground", "common"])
  const isMobile = useMobile()

  // Store state
  const isOpen = useWorkspaceStore((s) => s.addSourceModalOpen)
  const activeTab = useWorkspaceStore((s) => s.addSourceModalTab)
  const isProcessing = useWorkspaceStore((s) => s.addSourceProcessing)
  const error = useWorkspaceStore((s) => s.addSourceError)

  // Store actions
  const closeModal = useWorkspaceStore((s) => s.closeAddSourceModal)
  const setTab = useWorkspaceStore((s) => s.setAddSourceModalTab)
  const setProcessing = useWorkspaceStore((s) => s.setAddSourceProcessing)
  const setError = useWorkspaceStore((s) => s.setAddSourceError)
  const addSource = useWorkspaceStore((s) => s.addSource)
  const sources = useWorkspaceStore((s) => s.sources)
  const workspaceTag = useWorkspaceStore((s) => s.workspaceTag)
  const [tabUsage, setTabUsage] = React.useState<AddSourceTabUsage>(() =>
    readAddSourceTabUsage()
  )

  React.useEffect(() => {
    if (!isOpen) return
    setTabUsage(readAddSourceTabUsage())
  }, [isOpen])

  React.useEffect(() => {
    persistAddSourceTabUsage(tabUsage)
  }, [tabUsage])

  const handleAddSources: AddSourceHandler = async (
    sourceCandidates,
    options
  ) => {
    // Check for duplicates before adding
    const existingMediaIds = new Set((sources || []).map((s) => s.mediaId))
    const duplicates = sourceCandidates.filter((s) =>
      existingMediaIds.has(s.mediaId)
    )
    const newSources = sourceCandidates.filter(
      (s) => !existingMediaIds.has(s.mediaId)
    )

    if (duplicates.length > 0 && newSources.length === 0) {
      message.warning(
        t(
          "playground:sources.allDuplicates",
          duplicates.length === 1
            ? "This source is already in your workspace"
            : "These sources are already in your workspace"
        )
      )
      return
    }

    if (duplicates.length > 0) {
      message.info(
        t("playground:sources.someDuplicatesSkipped", {
          defaultValue: "{{count}} duplicate source(s) skipped",
          count: duplicates.length
        })
      )
    }

    // Add only new sources to workspace
    for (const source of newSources) {
      addSource(source)

      // Tag media with workspace tag
      if (workspaceTag) {
        try {
          await tldwClient.updateMediaKeywords(source.mediaId, {
            keywords: [workspaceTag],
            mode: "add"
          })
        } catch {
          // Continue even if tagging fails
        }
      }
    }

    if (options?.closeModal ?? true) {
      // Close modal after successful add
      closeModal()
    }
  }

  const tabItems = [
    {
      key: "upload",
      label: (
        <span className="flex items-center gap-1.5">
          <UploadIcon className="h-4 w-4" />
          {t("playground:sources.tabUpload", "Upload")}
        </span>
      ),
      children: (
        <UploadTab
          onAddSources={handleAddSources}
          setProcessing={setProcessing}
          setError={setError}
        />
      )
    },
    {
      key: "existing",
      label: (
        <span className="flex items-center gap-1.5">
          <Database className="h-4 w-4" />
          {t("playground:sources.tabExisting", "My Media")}
        </span>
      ),
      children: (
        <ExistingTab
          onAddSources={handleAddSources}
          setProcessing={setProcessing}
          setError={setError}
        />
      )
    },
    {
      key: "url",
      label: (
        <span className="flex items-center gap-1.5">
          <Link className="h-4 w-4" />
          {t("playground:sources.tabUrl", "URL")}
        </span>
      ),
      children: (
        <UrlTab
          onAddSources={handleAddSources}
          setProcessing={setProcessing}
          setError={setError}
        />
      )
    },
    {
      key: "paste",
      label: (
        <span className="flex items-center gap-1.5">
          <FileText className="h-4 w-4" />
          {t("playground:sources.tabPaste", "Paste")}
        </span>
      ),
      children: (
        <PasteTab
          onAddSources={handleAddSources}
          setProcessing={setProcessing}
          setError={setError}
        />
      )
    },
    {
      key: "search",
      label: (
        <span className="flex items-center gap-1.5">
          <Search className="h-4 w-4" />
          {t("playground:sources.tabSearch", "Search Server")}
        </span>
      ),
      children: (
        <SearchTab
          onAddSources={handleAddSources}
          setProcessing={setProcessing}
          setError={setError}
        />
      )
    }
  ]

  // Use a fixed tab order so tabs don't shift based on usage frequency.
  // Tab usage is still tracked (for potential analytics) but no longer
  // drives the rendered order.
  const orderedTabItems = React.useMemo(() => {
    const itemMap = new Map<AddSourceTab, (typeof tabItems)[number]>(
      tabItems.map((item) => [item.key as AddSourceTab, item])
    )
    return DEFAULT_ADD_SOURCE_TAB_ORDER
      .map((tab) => itemMap.get(tab))
      .filter((item): item is (typeof tabItems)[number] => Boolean(item))
  }, [tabItems])

  const handleTabChange = React.useCallback(
    (key: string) => {
      if (!isAddSourceTab(key)) return
      setTab(key)
      setTabUsage((previous) => ({
        ...previous,
        [key]: previous[key] + 1
      }))
    },
    [setTab]
  )

  return (
    <Modal
      open={isOpen}
      onCancel={closeModal}
      title={t("playground:sources.addSourceTitle", "Add Sources")}
      footer={null}
      width={isMobile ? "100%" : 600}
      style={isMobile ? { top: 0, paddingBottom: 0 } : undefined}
      styles={
        isMobile
          ? {
              body: {
                maxHeight: "70vh",
                overflowY: "auto"
              }
            }
          : undefined
      }
      destroyOnHidden
    >
      <Spin spinning={isProcessing}>
        {error && (
          <Alert
            type="error"
            title={error}
            closable
            onClose={() => setError(null)}
            className="mb-4"
          />
        )}
        <Tabs
          activeKey={activeTab}
          onChange={handleTabChange}
          items={orderedTabItems}
        />
      </Spin>
    </Modal>
  )
}

export default AddSourceModal
