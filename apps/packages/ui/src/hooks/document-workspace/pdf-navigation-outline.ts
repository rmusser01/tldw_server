import type { TocItem } from "@/types/document-outline"
import type { MediaNavigationContentResponse, MediaNavigationResponse } from "@/hooks/useMediaNavigation"
import { buildMediaNavigationTreeIndex } from "@/utils/media-navigation-tree"

/** PDF destinations are physical, one-based pages; never infer printed-page offsets. */
export const isPdfPage = (page: unknown, totalPages: number): page is number =>
  typeof page === "number" && Number.isSafeInteger(page) && page > 0 &&
  Number.isSafeInteger(totalPages) && totalPages > 0 && page <= totalPages

/** Keep unambiguous ancestry; generated page targets may be printed labels, not PDF pages. */
export function generatedPdfOutline(response: MediaNavigationResponse) {
  if (!response.available) return { items: [] as TocItem[], partial: false }
  const nodes = response.nodes
  const counts = new Map<string, number>()
  for (const node of nodes) counts.set(node.id, (counts.get(node.id) ?? 0) + 1)
  const byId = new Map(nodes.filter(node =>
    typeof node.id === "string" && node.id.trim() && !(node.id in Object.prototype) && counts.get(node.id) === 1 &&
    typeof node.title === "string" && node.title.trim() &&
    Number.isSafeInteger(node.level) && node.level > 0 && node.level <= 8 && Number.isFinite(node.order)
  ).map(node => [node.id, node]))
  const unambiguous = nodes.filter(node => {
    const seen = new Set<string>()
    let current = byId.get(node.id)
    if (!current) return false
    while (current) {
      if (seen.has(current.id)) return false
      seen.add(current.id)
      if (current.parent_id === null) return true
      const parent = byId.get(current.parent_id)
      if (!parent || parent.level >= current.level) return false
      current = parent
    }
    return false
  })
  const tree = buildMediaNavigationTreeIndex(unambiguous)
  const convert = (node: typeof nodes[number]): TocItem => {
    const start = node.target_start
    const validStart = typeof start === "number" && Number.isSafeInteger(start) &&
      ((node.target_type === "page" && start > 0) || (node.target_type === "char_range" && start >= 0))
    const validEnd = node.target_end === null || (typeof node.target_end === "number" &&
      Number.isSafeInteger(node.target_end) && typeof start === "number" && node.target_end >= start)
    return { title: node.title, page: 0, level: node.level, navigationNodeId: node.id,
      unavailable: !validStart || !validEnd, children: (tree.childrenByParent[node.id] ?? []).map(convert) }
  }
  return { items: tree.roots.map(convert), partial: nodes.length !== unambiguous.length || response.stats.truncated }
}

/** A page jump requires the exact current node's canonical physical page coverage. */
export function resolvedPdfPage(response: MediaNavigationContentResponse, nodeId: string, version: string, totalPages: number): number {
  const start = response.resolved_start_char, end = response.resolved_end_char
  const spans = response.page_spans
  if (response.node_id !== nodeId || !version || response.navigation_version !== version ||
    response.coverage_status !== "exact" || typeof start !== "number" || typeof end !== "number" ||
    !Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end <= start ||
    !spans?.length || spans[0].start_char !== start) {
    throw new Error("PDF page unavailable")
  }
  // Stored page separators (including blank pages) need not belong to a page span.
  let previousEnd = start, previousPage = 0
  for (const span of spans) {
    if (!isPdfPage(span.page_number, totalPages) || span.page_number <= previousPage ||
      !Number.isSafeInteger(span.start_char) || !Number.isSafeInteger(span.end_char) ||
      span.start_char < previousEnd || span.end_char <= span.start_char || span.end_char > end) {
      throw new Error("PDF page unavailable")
    }
    previousEnd = span.end_char
    previousPage = span.page_number
  }
  return spans[0].page_number
}
