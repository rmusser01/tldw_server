import { useQuery } from "@tanstack/react-query"
import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type MediaNavigationFormat = "auto" | "plain" | "markdown" | "html"
export type MediaNavigationTargetType =
  | "page"
  | "char_range"
  | "time_range"
  | "href"

export type MediaNavigationNode = {
  id: string
  parent_id: string | null
  level: number
  title: string
  order: number
  path_label: string | null
  target_type: MediaNavigationTargetType
  target_start: number | null
  target_end: number | null
  target_href: string | null
  source: string
  confidence: number | null
}

export type MediaNavigationResponse = {
  media_id: number
  available: boolean
  navigation_version: string
  source_order_used: string[]
  nodes: MediaNavigationNode[]
  stats: {
    returned_node_count: number
    node_count: number
    max_depth: number
    truncated: boolean
  }
}

export type MediaNavigationContentResponse = {
  media_id: number
  node_id: string
  navigation_version?: string
  resolved_start_char?: number | null
  resolved_end_char?: number | null
  coverage_status?: "exact" | "unavailable"
  page_spans?: Array<{ page_number: number; start_char: number; end_char: number; printed_page_label?: string }>
  title: string
  content_format: MediaNavigationFormat
  available_formats: MediaNavigationFormat[]
  content: string
  alternate_content?: Record<MediaNavigationFormat, string> | null
  target: {
    target_type: MediaNavigationTargetType
    target_start: number | null
    target_end: number | null
    target_href: string | null
  }
}

type UseMediaNavigationOptions = {
  enabled?: boolean
  includeGeneratedFallback?: boolean
}

export const useMediaNavigation = (
  mediaId: string | number | null,
  options: UseMediaNavigationOptions = {}
) => {
  const enabled = Boolean(options.enabled) && mediaId !== null && mediaId !== undefined
  const includeGeneratedFallback = Boolean(options.includeGeneratedFallback)

  return useQuery({
    queryKey: ["media-navigation", String(mediaId || ""), includeGeneratedFallback],
    enabled,
    staleTime: 30_000,
    queryFn: async (): Promise<MediaNavigationResponse> => {
      const query = `include_generated_fallback=${includeGeneratedFallback ? "true" : "false"}`
      const path = `/api/v1/media/${encodeURIComponent(String(mediaId))}/navigation?${query}`
      return await bgRequest<MediaNavigationResponse>({
        path: path as AllowedPath,
        method: "GET"
      })
    }
  })
}

type UseMediaSectionContentOptions = {
  enabled?: boolean
  format?: MediaNavigationFormat
  includeAlternates?: boolean
}

export const useMediaSectionContent = (
  mediaId: string | number | null,
  nodeId: string | null,
  options: UseMediaSectionContentOptions = {}
) => {
  const enabled =
    Boolean(options.enabled) &&
    mediaId !== null &&
    mediaId !== undefined &&
    Boolean(nodeId)
  const format = options.format || "auto"
  const includeAlternates = Boolean(options.includeAlternates)

  return useQuery({
    queryKey: [
      "media-navigation-content",
      String(mediaId || ""),
      String(nodeId || ""),
      format,
      includeAlternates
    ],
    enabled,
    staleTime: 15_000,
    queryFn: async (): Promise<MediaNavigationContentResponse> => {
      const query = `format=${format}&include_alternates=${includeAlternates ? "true" : "false"}`
      const encodedNodeId = encodeURIComponent(String(nodeId || ""))
      const path = `/api/v1/media/${encodeURIComponent(String(mediaId))}/navigation/${encodedNodeId}/content?${query}`
      return await bgRequest<MediaNavigationContentResponse>({
        path: path as AllowedPath,
        method: "GET"
      })
    }
  })
}

