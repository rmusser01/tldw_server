import { KNOWLEDGE_QA_KEYWORD } from "./constants"
import type { SearchHistoryItem } from "./types"
import { getKnowledgeAnswerTrustLabel } from "./trustState"
import { getEvidenceOriginLabel } from "./sourceListUtils"

export type GroupedHistorySections = {
  pinned: SearchHistoryItem[]
  groupedByDate: Map<string, SearchHistoryItem[]>
}

export function isKnowledgeQaHistoryItem(item: SearchHistoryItem): boolean {
  const keywords = Array.isArray(item.keywords) ? item.keywords : []
  return keywords.some(
    (keyword) =>
      String(keyword).trim().toLowerCase() === KNOWLEDGE_QA_KEYWORD.toLowerCase()
  )
}

export function sortHistoryNewestFirst(items: SearchHistoryItem[]): SearchHistoryItem[] {
  return [...items].sort((a, b) => {
    const left = new Date(a.timestamp).getTime()
    const right = new Date(b.timestamp).getTime()
    return right - left
  })
}

export function filterHistoryItems(
  items: SearchHistoryItem[],
  filterText: string
): SearchHistoryItem[] {
  const normalizedFilter = filterText.trim().toLowerCase()
  if (!normalizedFilter) return items

  return items.filter((item) => {
    const query = String(item.query || "").toLowerCase()
    const preview = String(item.answerPreview || "").toLowerCase()
    return query.includes(normalizedFilter) || preview.includes(normalizedFilter)
  })
}

export function groupByDate(
  items: SearchHistoryItem[],
  now: Date = new Date()
): Map<string, SearchHistoryItem[]> {
  const groups = new Map<string, SearchHistoryItem[]>()

  for (const item of items) {
    const date = new Date(item.timestamp)
    const diff = now.getTime() - date.getTime()
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))

    let groupKey: string
    if (days === 0) {
      groupKey = "Today"
    } else if (days === 1) {
      groupKey = "Yesterday"
    } else if (days < 7) {
      groupKey = "This Week"
    } else if (days < 30) {
      groupKey = "This Month"
    } else {
      groupKey = "Older"
    }

    if (!groups.has(groupKey)) {
      groups.set(groupKey, [])
    }
    groups.get(groupKey)!.push(item)
  }

  return groups
}

export function buildGroupedHistorySections(
  items: SearchHistoryItem[],
  now: Date = new Date()
): GroupedHistorySections {
  const sorted = sortHistoryNewestFirst(items)
  const pinned = sorted.filter((item) => item.pinned)
  const unpinned = sorted.filter((item) => !item.pinned)
  return {
    pinned,
    groupedByDate: groupByDate(unpinned, now),
  }
}

export function truncateAnswerPreview(answer: string | null | undefined): string | undefined {
  if (typeof answer !== "string") return undefined
  const normalized = answer.replace(/\s+/g, " ").trim()
  if (!normalized) return undefined
  if (normalized.length <= 120) return normalized
  return `${normalized.slice(0, 117)}...`
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return count === 1 ? singular : plural
}

function countSourceIssues(item: SearchHistoryItem): number {
  const statuses = item.sourceStatus
  if (!statuses || typeof statuses !== "object") return 0
  return Object.values(statuses).filter((entry) => {
    const status = String(entry?.status || "").trim().toLowerCase()
    return status.length > 0 && status !== "searched" && status !== "ready"
  }).length
}

export function buildHistoryStatusLabels(item: SearchHistoryItem): string[] {
  const labels: string[] = []
  if (item.trustState) {
    labels.push(getKnowledgeAnswerTrustLabel(item.trustState))
  }

  if (item.evidenceOrigin && item.evidenceOrigin !== "unknown_origin") {
    labels.push(getEvidenceOriginLabel(item.evidenceOrigin))
  }

  if (typeof item.citationCount === "number" && Number.isFinite(item.citationCount)) {
    const citationCount = Math.max(0, Math.round(item.citationCount))
    labels.push(
      `${citationCount} ${pluralize(citationCount, "citation")}`
    )
  }

  if (item.unsynced && item.trustState !== "unsynced_local_result") {
    labels.push("Unsynced")
  }

  const sourceIssues = countSourceIssues(item)
  if (sourceIssues > 0) {
    labels.push(`${sourceIssues} ${pluralize(sourceIssues, "source issue")}`)
  }

  return labels
}

export function buildHistoryExportMarkdown(
  items: SearchHistoryItem[],
  exportedAt: Date = new Date()
): string {
  const sorted = sortHistoryNewestFirst(items)
  const lines: string[] = [
    "# Knowledge QA History Export",
    "",
    `Exported at: ${exportedAt.toISOString()}`,
    `Total items: ${sorted.length}`,
    "",
  ]

  sorted.forEach((item, index) => {
    lines.push(`## ${index + 1}. ${item.query || "Untitled query"}`)
    lines.push(`- Timestamp: ${item.timestamp}`)
    lines.push(`- Sources: ${item.sourcesCount}`)
    lines.push(`- Has answer: ${item.hasAnswer ? "yes" : "no"}`)
    if (item.preset) {
      lines.push(`- Preset: ${item.preset}`)
    }
    if (item.pinned) {
      lines.push("- Pinned: yes")
    }
    if (item.trustState) {
      lines.push(`- Trust: ${getKnowledgeAnswerTrustLabel(item.trustState)}`)
    }
    if (item.evidenceOrigin && item.evidenceOrigin !== "unknown_origin") {
      lines.push(`- Evidence origin: ${getEvidenceOriginLabel(item.evidenceOrigin)}`)
    }
    if (typeof item.citationCount === "number" && Number.isFinite(item.citationCount)) {
      lines.push(`- Citations: ${Math.max(0, Math.round(item.citationCount))}`)
    }
    if (item.unsynced) {
      lines.push("- Unsynced: yes")
    }
    const sourceIssues = countSourceIssues(item)
    if (sourceIssues > 0) {
      lines.push(`- Source status: ${sourceIssues} ${pluralize(sourceIssues, "source issue")}`)
    }
    if (item.conversationId) {
      lines.push(`- Conversation ID: ${item.conversationId}`)
    }
    if (item.answerPreview) {
      lines.push(`- Answer preview: ${item.answerPreview}`)
    }
    lines.push("")
  })

  return lines.join("\n").trimEnd()
}
