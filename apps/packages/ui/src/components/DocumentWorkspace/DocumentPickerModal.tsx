import React from "react"
import { useTranslation } from "react-i18next"
import {
  Modal,
  Tabs,
  Input,
  Button,
  List,
  Spin,
  Empty,
  Tag,
  Switch,
  Alert
} from "antd"
import { Search, UploadCloud, FileText, BookOpen, PlayCircle } from "lucide-react"
import { tldwClient } from "@/services/tldw"
import { useDebounce } from "@/hooks/useDebounce"
import { useServerOnline } from "@/hooks/useServerOnline"
import { inferUploadMediaTypeFromFile } from "@/services/tldw/media-routing"
import { inferDocumentTypeFromMedia } from "./document-utils"
import type { DocumentType } from "./types"
import { useNavigate } from "react-router-dom"
import { setSetting } from "@/services/settings/registry"
import { LAST_MEDIA_ID_SETTING } from "@/services/settings/ui-settings"
import { useQuickIngestStore } from "@/store/quick-ingest"

export type DocumentPickerTab = "library" | "upload"

interface DocumentPickerModalProps {
  open: boolean
  initialTab?: DocumentPickerTab
  onClose: () => void
  onOpenDocument: (mediaId: number, docTypeHint?: DocumentType | null) => Promise<void>
}

interface MediaListItem {
  id: number
  title?: string
  type?: string
  created_at?: string
  keywords?: string[]
  url?: string
  filename?: string
}

type MediaListResponse = {
  items?: unknown
  media?: unknown
  results?: unknown
  data?: unknown
}

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object") return null
  return value as Record<string, unknown>
}

const asString = (value: unknown): string | undefined => {
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed ? trimmed : undefined
  }
  if (typeof value === "number" && Number.isFinite(value)) return String(value)
  return undefined
}

const asStringArray = (value: unknown): string[] | undefined => {
  if (!Array.isArray(value)) return undefined
  const list = value
    .map((entry) => asString(entry))
    .filter((entry): entry is string => Boolean(entry))
  return list.length > 0 ? list : undefined
}

const getResponseItems = (response: unknown): unknown[] => {
  const record = asRecord(response) as MediaListResponse | null
  if (!record) return []
  const items = record.items ?? record.media ?? record.results ?? record.data ?? []
  return Array.isArray(items) ? items : []
}

const normalizeMediaItems = (response: unknown): MediaListItem[] => {
  const items = getResponseItems(response)
  return items
    .map((item) => {
      const record = asRecord(item)
      if (!record) return null
      const id = Number(record.media_id ?? record.id)
      if (!Number.isFinite(id)) return null
      const normalized: MediaListItem = {
        id,
        title: asString(record.title) ?? asString(record.name),
        type: asString(record.type) ?? asString(record.media_type),
        created_at: asString(record.created_at),
        keywords: asStringArray(record.keywords),
        url: asString(record.url),
        filename: asString(record.filename) ?? asString(record.original_filename)
      }
      return normalized
    })
    .filter((item): item is MediaListItem => item !== null)
}

const isSupportedDocType = (docType: DocumentType | null): docType is DocumentType =>
  docType === "pdf" || docType === "epub"

const getDocType = (item: MediaListItem): DocumentType | null =>
  inferDocumentTypeFromMedia(item.type, item.filename)

const isDocumentCandidate = (item: MediaListItem): boolean =>
  isSupportedDocType(getDocType(item))

export const DocumentPickerModal: React.FC<DocumentPickerModalProps> = ({
  open,
  initialTab = "library",
  onClose,
  onOpenDocument
}) => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const isOnline = useServerOnline()
  const recentlyIngestedDocs = useQuickIngestStore(s => s.recentlyIngestedDocs)

  const [activeTab, setActiveTab] = React.useState<DocumentPickerTab>(initialTab)
  const [searchQuery, setSearchQuery] = React.useState("")
  const debouncedQuery = useDebounce(searchQuery, 300)
  const [mediaItems, setMediaItems] = React.useState<MediaListItem[]>([])
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [openingId, setOpeningId] = React.useState<number | null>(null)
  const [showAllMedia, setShowAllMedia] = React.useState(false)
  const [uploading, setUploading] = React.useState(false)
  const [uploadWarning, setUploadWarning] = React.useState<{
    message: string
    mediaId?: number
  } | null>(null)
  const [isDragOver, setIsDragOver] = React.useState(false)
  const fileInputRef = React.useRef<HTMLInputElement>(null)

  React.useEffect(() => {
    if (open) {
      setActiveTab(initialTab)
      setError(null)
      setUploadWarning(null)
    }
  }, [open, initialTab])

  const loadMedia = React.useCallback(
    async (query?: string) => {
      if (!isOnline) return
      setLoading(true)
      setError(null)
      try {
        let response
        const trimmed = query?.trim()
        if (trimmed) {
            response = await tldwClient.searchMedia(
              {
                query: trimmed,
                ...(showAllMedia ? {} : { media_types: ["pdf", "ebook"] })
              },
              { page: 1, results_per_page: 50 }
            )
        } else {
          response = await tldwClient.listMedia({
            page: 1,
            results_per_page: 50,
            include_keywords: true
          })
        }
        const items = normalizeMediaItems(response)
        const filtered = showAllMedia ? items : items.filter(isDocumentCandidate)
        setMediaItems(filtered)
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : t("option:documentWorkspace.loadMediaError", "Failed to load media library")
        )
      } finally {
        setLoading(false)
      }
    },
    [isOnline, showAllMedia, t]
  )

  React.useEffect(() => {
    if (!open || activeTab !== "library") return
    void loadMedia(debouncedQuery)
  }, [open, activeTab, debouncedQuery, loadMedia, showAllMedia])

  // Recent documents from localStorage
  const RECENT_DOCS_KEY = "document-workspace-recent"
  const MAX_RECENT = 10

  const getRecentDocs = (): Array<{ id: number; title: string; type?: string }> => {
    try {
      const raw = localStorage.getItem(RECENT_DOCS_KEY)
      return raw ? JSON.parse(raw) : []
    } catch {
      return []
    }
  }

  const addToRecent = (item: MediaListItem) => {
    const recent = getRecentDocs().filter((r) => r.id !== item.id)
    recent.unshift({ id: item.id, title: item.title || `Media #${item.id}`, type: item.type })
    localStorage.setItem(RECENT_DOCS_KEY, JSON.stringify(recent.slice(0, MAX_RECENT)))
  }

  const [recentDocs] = React.useState(() => getRecentDocs())

  const handleOpen = async (item: MediaListItem) => {
    if (!item?.id) return
    const docType = getDocType(item)
    if (!isSupportedDocType(docType)) {
      await setSetting(LAST_MEDIA_ID_SETTING, String(item.id))
      navigate("/media-multi")
      onClose()
      return
    }

    addToRecent(item)
    setOpeningId(item.id)
    try {
      await onOpenDocument(item.id, docType)
      onClose()
    } catch (err) {
      console.error("Failed to open document", err)
      onClose()
    } finally {
      setOpeningId(null)
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileSelected = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Validate file type before uploading
    const ext = file.name.toLowerCase().split(".").pop()
    if (ext !== "pdf" && ext !== "epub") {
      setError(t("option:documentWorkspace.unsupportedFileType", "Only PDF and EPUB files are supported."))
      if (event.target) event.target.value = ""
      return
    }

    setUploading(true)
    setError(null)
    setUploadWarning(null)

    try {
      const mediaType = inferUploadMediaTypeFromFile(file.name, file.type)
      const response = await tldwClient.uploadMedia(file, {
        media_type: mediaType,
        keep_original_file: true
      })

      const mediaId = extractMediaId(response)
      if (!mediaId) {
        throw new Error(
          t("option:documentWorkspace.uploadMissingMediaId", "Upload succeeded but no media ID was returned")
        )
      }

      const result = extractUploadResult(response)
      const warnings = extractWarnings(result)
      const hasStorageWarning =
        warnings.some((warning) => warning.includes("original file")) ||
        result?.original_file_stored === false
      if (hasStorageWarning) {
        setUploadWarning({
          message: t(
            "option:documentWorkspace.uploadStorageWarning",
            "Upload finished, but the original file could not be stored. Open in Media to view extracted text, or re-upload after fixing storage."
          ),
          mediaId: Number(mediaId)
        })
        return
      }

      const docType = inferDocumentTypeFromMedia(mediaType, file.name)
      await onOpenDocument(Number(mediaId), docType)
      onClose()
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : t("option:documentWorkspace.uploadFailed", "Upload failed")
      )
    } finally {
      setUploading(false)
      if (event.target) {
        event.target.value = ""
      }
    }
  }

  const libraryPane = () => (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Input
          prefix={<Search className="h-4 w-4 text-text-muted" />}
          placeholder={t(
            "option:documentWorkspace.searchLibrary",
            "Search your media library..."
          )}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          allowClear
        />
        <Button onClick={() => loadMedia(searchQuery)} loading={loading}>
          {t("common:search", "Search")}
        </Button>
      </div>

      <div className="flex items-center justify-between text-xs text-text-muted">
        <span>
          {t(
            "option:documentWorkspace.showAllMedia",
            "Show non-document media"
          )}
        </span>
        <Switch
          checked={showAllMedia}
          onChange={(checked) => setShowAllMedia(checked)}
          size="small"
        />
      </div>

      {/* Recently added from QuickIngest */}
      {recentlyIngestedDocs.length > 0 && !searchQuery && (
        <div className="mb-3">
          <p className="mb-1.5 text-xs font-medium text-text-muted">
            {t("option:documentWorkspace.recentlyIngested", "Recently added")}
          </p>
          <div className="space-y-1">
            {recentlyIngestedDocs.slice(0, 3).map((doc) => (
              <button
                key={doc.id}
                type="button"
                onClick={() => {
                  const normalized = doc.type?.toLowerCase()
                  const resolvedType = normalized === "document" ? "pdf" : normalized === "ebook" ? "epub" : normalized
                  handleOpen({ id: doc.id, type: resolvedType, title: doc.title })
                }}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm hover:bg-surface2 transition-colors"
              >
                <FileText className="h-4 w-4 flex-shrink-0 text-primary" />
                <span className="truncate">{doc.title || `Document #${doc.id}`}</span>
                <span className="ml-auto text-xs text-text-muted">
                  {t("option:documentWorkspace.justIngested", "Just added")}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Recent documents */}
      {!searchQuery && recentDocs.length > 0 && (
        <div className="mb-3">
          <p className="mb-1.5 text-xs font-medium text-text-muted">
            {t("option:documentWorkspace.recentDocuments", "Recently opened")}
          </p>
          <div className="flex flex-wrap gap-2">
            {recentDocs.slice(0, 5).map((doc) => (
              <Button
                key={doc.id}
                size="small"
                icon={<FileText className="h-3.5 w-3.5" />}
                onClick={() => handleOpen({ id: doc.id, title: doc.title, type: doc.type })}
                loading={openingId === doc.id}
              >
                <span className="max-w-[150px] truncate">{doc.title}</span>
              </Button>
            ))}
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-8">
          <Spin />
        </div>
      ) : mediaItems.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            searchQuery
              ? t("option:documentWorkspace.noMediaFound", "No documents match your search")
              : t("option:documentWorkspace.libraryEmpty", "Your library is empty. Upload a document to get started.")
          }
        />
      ) : (
        <div className="max-h-96 overflow-y-auto rounded border border-border">
          <List
            size="small"
            dataSource={mediaItems}
            renderItem={(item) => {
              const docType = getDocType(item)
              const supported = isSupportedDocType(docType)
              const icon = supported
                ? docType === "epub"
                  ? <BookOpen className="h-4 w-4 text-primary" />
                  : <FileText className="h-4 w-4 text-primary" />
                : <PlayCircle className="h-4 w-4 text-text-muted" />

              return (
                <List.Item
                  className="flex items-center justify-between gap-3"
                  actions={[
                    <Button
                      key="open"
                      type={supported ? "primary" : "default"}
                      size="small"
                      onClick={() => handleOpen(item)}
                      loading={openingId === item.id}
                    >
                      {supported
                        ? t("option:documentWorkspace.open", "Open")
                        : t("option:documentWorkspace.openInMedia", "Open in Media")}
                    </Button>
                  ]}
                >
                  <div className="flex min-w-0 flex-1 items-start gap-2">
                    <div className="mt-0.5 shrink-0">{icon}</div>
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium text-text">
                        {item.title || `Media #${item.id}`}
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
                        <span className="capitalize">{item.type || "document"}</span>
                        {item.keywords && item.keywords.length > 0 && (
                          <Tag>{item.keywords.slice(0, 3).join(", ")}</Tag>
                        )}
                      </div>
                    </div>
                  </div>
                </List.Item>
              )
            }}
          />
        </div>
      )}
    </div>
  )

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)

    const file = e.dataTransfer.files?.[0]
    if (!file) return

    // Validate file type
    const ext = file.name.toLowerCase().split(".").pop()
    if (ext !== "pdf" && ext !== "epub") {
      setError(t("option:documentWorkspace.unsupportedFileType", "Only PDF and EPUB files are supported."))
      return
    }

    // Reuse the same upload logic
    const fakeEvent = {
      target: { files: [file], value: "" }
    } as unknown as React.ChangeEvent<HTMLInputElement>
    await handleFileSelected(fakeEvent)
  }

  const uploadPane = () => (
    <div className="space-y-4">
      <div
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`rounded-lg border-2 border-dashed p-6 text-center transition-colors ${
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-border bg-surface2"
        }`}
      >
        <UploadCloud className={`mx-auto mb-3 h-8 w-8 ${isDragOver ? "text-primary animate-bounce" : "text-primary"}`} />
        <p className="text-sm text-text">
          {isDragOver
            ? t("option:documentWorkspace.dropHere", "Drop file here")
            : t("option:documentWorkspace.uploadHint", "Upload a PDF or EPUB file to start reading")}
        </p>
        <p className="text-xs text-text-muted">
          {t(
            "option:documentWorkspace.uploadNote",
            "Files are added to your library and kept for document review."
          )}
        </p>
        <div className="mt-4">
          <Button type="primary" onClick={handleUploadClick} loading={uploading}>
            {t("option:documentWorkspace.chooseFile", "Choose file")}
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.epub"
            className="hidden"
            onChange={handleFileSelected}
          />
        </div>
      </div>
    </div>
  )

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      title={t("option:documentWorkspace.openDocument", "Open document")}
      width={720}
      destroyOnClose
    >
      {!isOnline && (
        <Alert
          type="warning"
          showIcon
          title={t(
            "option:documentWorkspace.serverRequired",
            "Connect to server to use document workspace"
          )}
          className="mb-3"
        />
      )}
      {uploadWarning && (
        <Alert
          type="warning"
          showIcon
          title={uploadWarning.message}
          className="mb-3"
          action={
            uploadWarning.mediaId ? (
              <Button
                size="small"
                onClick={async () => {
                  await setSetting(LAST_MEDIA_ID_SETTING, String(uploadWarning.mediaId))
                  navigate("/media-multi")
                  onClose()
                }}
              >
                {t("option:documentWorkspace.openInMedia", "Open in Media")}
              </Button>
            ) : undefined
          }
        />
      )}
      {error && (
        <Alert
          type="error"
          showIcon
          title={error}
          className="mb-3"
        />
      )}
      <Tabs
        activeKey={activeTab}
        onChange={(key) => setActiveTab(key as DocumentPickerTab)}
        items={[
          {
            key: "library",
            label: t("option:documentWorkspace.fromLibrary", "From library"),
            children: libraryPane()
          },
          {
            key: "upload",
            label: t("option:documentWorkspace.upload", "Upload"),
            children: uploadPane()
          }
        ]}
      />
    </Modal>
  )
}

function extractMediaId(data: unknown): string | number | null {
  const visited = new WeakSet<object>()
  return extractMediaIdInternal(data, visited)
}

function extractMediaIdInternal(
  data: unknown,
  visited: WeakSet<object>
): string | number | null {
  if (!data || typeof data !== "object") return null
  const record = data as Record<string, unknown>
  if (Array.isArray(data)) {
    return data.length > 0 ? extractMediaIdInternal(data[0], visited) : null
  }
  if (visited.has(record)) return null
  visited.add(record)

  if ("results" in record && Array.isArray(record.results) && record.results.length > 0) {
    return extractMediaIdInternal(record.results[0], visited)
  }
  if ("result" in record && record.result) {
    return extractMediaIdInternal(record.result, visited)
  }
  if ("media" in record && record.media) {
    return extractMediaIdInternal(record.media, visited)
  }

  const direct =
    record.id ??
    record.media_id ??
    record.db_id ??
    record.pk ??
    record.uuid
  if (typeof direct === "string" || typeof direct === "number") return direct

  return null
}

function extractUploadResult(data: unknown): Record<string, unknown> | null {
  if (!data || typeof data !== "object") return null
  const record = data as Record<string, unknown>
  if (Array.isArray(record.results) && record.results.length > 0) {
    return record.results[0] as Record<string, unknown>
  }
  if (record.result && typeof record.result === "object") {
    return record.result as Record<string, unknown>
  }
  if (record.media && typeof record.media === "object") {
    return record.media as Record<string, unknown>
  }
  return record
}

function extractWarnings(result: Record<string, unknown> | null): string[] {
  if (!result) return []
  const warnings = result?.warnings
  if (!Array.isArray(warnings)) return []
  return warnings.map((warning) => String(warning || "").toLowerCase())
}

export default DocumentPickerModal
