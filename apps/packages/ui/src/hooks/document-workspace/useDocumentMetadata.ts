import { useQuery } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw"
import { useConnectionStore } from "@/store/connection"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import type { DocumentMetadata, DocumentType } from "@/components/DocumentWorkspace/types"

interface MediaDetailResponse {
  id?: number
  media_id?: number
  uuid?: string
  title?: string
  author?: string
  type?: string
  keywords?: string[]
  metadata?: Record<string, unknown>
  created_at?: string
  last_modified?: string
  file_size?: number
  page_count?: number
  source?: {
    title?: string
    type?: string
    url?: string
  }
  processing?: {
    safe_metadata?: Record<string, unknown>
  }
  content?: {
    metadata?: Record<string, unknown>
  }
}

/**
 * Infer document type from media type string
 */
function inferDocumentType(mediaType?: string): DocumentType {
  if (!mediaType) return "pdf"
  const lower = mediaType.toLowerCase()
  if (lower.includes("epub")) return "epub"
  return "pdf"
}

/**
 * Parse authors from author string (may be comma-separated)
 */
function parseAuthors(author?: string): string[] | undefined {
  if (!author) return undefined
  return author
    .split(/[,;&]/)
    .map((a) => a.trim())
    .filter(Boolean)
}

type MetadataRecord = Record<string, unknown>

function asRecord(value: unknown): MetadataRecord {
  if (!value || typeof value !== "object") return {}
  return value as MetadataRecord
}

function getMetadataValue(
  sources: MetadataRecord[],
  keys: string[]
): unknown {
  for (const source of sources) {
    for (const key of keys) {
      if (key in source) {
        const value = source[key]
        if (value !== undefined && value !== null && String(value).trim() !== "") {
          return value
        }
      }
    }
  }
  return undefined
}

function normalizeString(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed ? trimmed : undefined
  }
  if (typeof value === "number") return String(value)
  return undefined
}

function parseAuthorField(value: unknown): string[] | undefined {
  if (!value) return undefined
  if (Array.isArray(value)) {
    const list = value
      .map((entry) => {
        if (typeof entry === "string" || typeof entry === "number") {
          return String(entry).trim()
        }
        if (entry && typeof entry === "object" && "name" in entry) {
          return String((entry as { name?: unknown }).name ?? "").trim()
        }
        return ""
      })
      .filter(Boolean)
    return list.length > 0 ? list : undefined
  }
  if (typeof value === "string") {
    return parseAuthors(value)
  }
  if (value && typeof value === "object" && "name" in value) {
    const nameValue = String((value as { name?: unknown }).name ?? "").trim()
    return nameValue ? [nameValue] : undefined
  }
  return undefined
}

function parseNumber(value: unknown): number | undefined {
  if (value === undefined || value === null) return undefined
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const cleaned = value.replace(/[^\d.]/g, "")
    if (!cleaned) return undefined
    const parsed = Number(cleaned)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

function parseDate(value: unknown): Date | undefined {
  if (!value) return undefined
  if (value instanceof Date) return value
  if (typeof value === "string" || typeof value === "number") {
    const date = new Date(value)
    if (!Number.isNaN(date.getTime())) return date
  }
  return undefined
}

/**
 * Hook to fetch document metadata from the server.
 *
 * Uses the existing GET /api/v1/media/{media_id} endpoint.
 *
 * @param mediaId - The media ID to fetch metadata for (null to disable query)
 * @returns Query result with metadata, loading state, and error
 */
export function useDocumentMetadata(mediaId: number | null, operation?: QuickIngestOperation) {
  const isConnected = useConnectionStore((s) => s.state.isConnected)
  const mode = useConnectionStore((s) => s.state.mode)
  const isServerAvailable = isConnected && mode !== "demo"

  return useQuery({
    queryKey: ["document-metadata", mediaId, operation?.authorityKey, operation?.authorityRevision],
    queryFn: async ({ signal }): Promise<DocumentMetadata | null> => {
      if (mediaId === null) return null
      operation?.assertCurrent()

      const response: MediaDetailResponse = await tldwClient.getMediaDetails(
        mediaId,
        {
          include_content: false,
          include_versions: false,
          requestScope: operation?.requestScope,
          signal
        }
      )
      operation?.assertCurrent()

      const metadataSources = [
        asRecord(response.processing?.safe_metadata),
        asRecord(response.content?.metadata),
        asRecord(response.metadata),
      ]

      const metadataTitle = normalizeString(
        getMetadataValue(metadataSources, [
          "title",
          "Title",
          "document_title",
          "DocumentTitle",
          "dc:title",
        ])
      )
      const metadataAuthor = parseAuthorField(
        getMetadataValue(metadataSources, [
          "author",
          "Author",
          "authors",
          "Authors",
          "dc:creator",
          "creator",
        ])
      )
      const metadataCreator = normalizeString(
        getMetadataValue(metadataSources, ["creator", "Creator"])
      )
      const metadataProducer = normalizeString(
        getMetadataValue(metadataSources, ["producer", "Producer"])
      )
      const metadataAbstract = normalizeString(
        getMetadataValue(metadataSources, ["abstract", "Abstract"])
      )
      const metadataFileName = normalizeString(
        getMetadataValue(metadataSources, [
          "original_filename",
          "file_name",
          "filename",
          "fileName",
          "FileName",
          "File_Name",
        ])
      )
      const metadataPageCount = parseNumber(
        getMetadataValue(metadataSources, [
          "page_count",
          "pageCount",
          "PageCount",
          "pages",
          "Pages",
        ])
      )
      const metadataFileSize = parseNumber(
        getMetadataValue(metadataSources, [
          "file_size",
          "fileSize",
          "FileSize",
        ])
      )
      const createdDate = parseDate(
        response.created_at ??
          getMetadataValue(metadataSources, [
            "created_at",
            "creation_date",
            "CreationDate",
            "created",
          ])
      )
      const modifiedDate = parseDate(
        response.last_modified ??
          getMetadataValue(metadataSources, [
            "last_modified",
            "mod_date",
            "ModDate",
            "modified",
          ])
      )
      const parsedAuthors = parseAuthors(response.author)
      const authors =
        parsedAuthors && parsedAuthors.length > 0 ? parsedAuthors : metadataAuthor

      return {
        id: response.media_id ?? response.id ?? mediaId,
        uuid: response.uuid,
        title:
          response.source?.title ||
          response.title ||
          metadataTitle ||
          "Untitled",
        authors,
        creator: metadataCreator,
        producer: metadataProducer,
        fileName: metadataFileName,
        abstract: metadataAbstract,
        keywords: response.keywords,
        pageCount: response.page_count ?? metadataPageCount,
        createdDate,
        modifiedDate,
        fileSize: response.file_size ?? metadataFileSize,
        type: inferDocumentType(response.source?.type || response.type)
      }
    },
    enabled: mediaId !== null && isServerAvailable,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    retry: 1,
    refetchOnWindowFocus: false
  })
}
