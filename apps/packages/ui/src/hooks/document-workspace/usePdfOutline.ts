import { readOwnedOutline } from "@/services/tldw/document-outline"
import { useQuery } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw"
import { useConnectionStore } from "@/store/connection"
import type { TocItem, DocumentOutline } from "@/types/document-outline"
import type { MediaNavigationContentResponse, MediaNavigationResponse } from "@/hooks/useMediaNavigation"
import { useDocumentOperation } from "./useDocumentOperation"
import { generatedPdfOutline, isPdfPage, resolvedPdfPage } from "./pdf-navigation-outline"
import { getErrorStatus, shouldRetryDocumentWorkspaceQuery } from "./request-retry"

export interface OutlineEntry {
  level: number
  title: string
  page: number
}

export interface DocumentOutlineResponse {
  media_id: number
  has_outline: boolean
  entries: OutlineEntry[]
  total_pages: number
}

type PdfOutline = DocumentOutline & {
  source: "embedded" | "generated"
  partial: boolean
  totalPages: number
  navigationVersion?: string
}

/**
 * Converts flat outline entries to hierarchical TocItem structure
 */
function buildHierarchy(entries: OutlineEntry[]): TocItem[] {
  if (entries.length === 0) return []

  const items: TocItem[] = []
  const stack: { item: TocItem; level: number }[] = []

  for (const entry of entries) {
    const item: TocItem = {
      title: entry.title,
      page: entry.page,
      level: entry.level,
      children: []
    }

    // Pop items from stack that are at same or higher level
    while (stack.length > 0 && stack[stack.length - 1].level >= entry.level) {
      stack.pop()
    }

    if (stack.length === 0) {
      // Top-level item
      items.push(item)
    } else {
      // Child of the last item in stack
      const parent = stack[stack.length - 1].item
      if (!parent.children) {
        parent.children = []
      }
      parent.children.push(item)
    }

    stack.push({ item, level: entry.level })
  }

  return items
}

/**
 * Hook to fetch document outline/table of contents from the server.
 *
 * @param mediaId - The media ID to fetch outline for (null to disable query)
 * @returns Query result with outline data, loading state, and error
 */
export function usePdfOutline(mediaId: number | null) {
  const operation = useDocumentOperation()
  const isConnected = useConnectionStore((s) => s.state.isConnected)
  const mode = useConnectionStore((s) => s.state.mode)
  const isServerAvailable = isConnected && mode !== "demo"

  const enabled = mediaId !== null && Number.isSafeInteger(mediaId) && mediaId > 0 && isServerAvailable && Boolean(operation)
  const query = useQuery({
    queryKey: ["document-outline", operation?.authorityKey, operation?.authorityRevision, mediaId],
    queryFn: async ({ signal }): Promise<PdfOutline | null> => {
      if (mediaId === null || !operation) return null
      const response = await readOwnedOutline<DocumentOutlineResponse>(mediaId, "outline", operation, signal)
      if (response.has_outline && response.entries.length > 0) {
        const valid = response.entries.filter(entry => isPdfPage(entry.page, response.total_pages) &&
          typeof entry.title === "string" && entry.title.trim() && Number.isSafeInteger(entry.level) && entry.level > 0)
        return { documentId: mediaId, source: "embedded", items: buildHierarchy(valid), partial: valid.length !== response.entries.length, totalPages: response.total_pages }
      }
      const navigation = await readOwnedOutline<MediaNavigationResponse>(mediaId, "navigation?include_generated_fallback=false", operation, signal)
      return { documentId: mediaId, source: "generated", totalPages: response.total_pages,
        navigationVersion: navigation.navigation_version, ...generatedPdfOutline(navigation) }
    },
    enabled,
    staleTime: 10 * 60 * 1000, // Cache for 10 minutes (outlines don't change)
    retry: (count, error) => Boolean(operation?.isCurrent()) &&
      ![401, 403, 412].includes(getErrorStatus(error) ?? 0) && shouldRetryDocumentWorkspaceQuery(count, error, 1),
    refetchOnWindowFocus: false
  })
  const data = enabled && operation?.isCurrent() ? query.data : undefined
  const resolvePage = async (nodeId: string, signal: AbortSignal) => {
    const contains = (items: TocItem[]): boolean => items.some(item =>
      (item.navigationNodeId === nodeId && !item.unavailable) || contains(item.children ?? []))
    if (!operation || mediaId === null || data?.source !== "generated" || !data.navigationVersion || !contains(data.items)) {
      throw new Error("PDF page unavailable")
    }
    const response = await readOwnedOutline<MediaNavigationContentResponse>(mediaId,
      `navigation/${encodeURIComponent(nodeId)}/content?format=auto&include_alternates=false`, operation, signal)
    return resolvedPdfPage(response, nodeId, data.navigationVersion, data.totalPages)
  }
  return { ...query, data, resolvePage, isCurrent: () => enabled && Boolean(operation?.isCurrent()) }
}

/**
 * Hook to get total pages from outline response.
 * Useful when you need page count without the full outline.
 */
export function useDocumentPageCount(mediaId: number | null) {
  const isConnected = useConnectionStore((s) => s.state.isConnected)
  const mode = useConnectionStore((s) => s.state.mode)
  const isServerAvailable = isConnected && mode !== "demo"

  return useQuery({
    queryKey: ["document-page-count", mediaId],
    queryFn: async (): Promise<number> => {
      if (mediaId === null) return 0

      const response = await tldwClient.getDocumentOutline(mediaId)
      return response.total_pages
    },
    enabled: mediaId !== null && isServerAvailable,
    staleTime: 10 * 60 * 1000,
    retry: 1,
    refetchOnWindowFocus: false
  })
}
