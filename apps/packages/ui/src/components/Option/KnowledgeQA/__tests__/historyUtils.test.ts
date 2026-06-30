import {
  buildHistoryStatusLabels,
  buildGroupedHistorySections,
  buildHistoryExportMarkdown,
  filterHistoryItems,
  isKnowledgeQaHistoryItem,
  truncateAnswerPreview,
} from "../historyUtils"
import type { SearchHistoryItem } from "../types"
import { describe, expect, it } from "vitest"

const makeItem = (overrides: Partial<SearchHistoryItem> = {}): SearchHistoryItem => ({
  id: "history-id",
  query: "Default query",
  timestamp: "2026-02-18T10:00:00.000Z",
  sourcesCount: 2,
  hasAnswer: true,
  keywords: ["__knowledge_QA__"],
  ...overrides,
})

describe("historyUtils", () => {
  it("filters to Knowledge QA-tagged history items", () => {
    const tagged = makeItem()
    const untagged = makeItem({
      id: "h-2",
      keywords: ["other"],
    })

    expect(isKnowledgeQaHistoryItem(tagged)).toBe(true)
    expect(isKnowledgeQaHistoryItem(untagged)).toBe(false)
  })

  it("filters history using query and answer preview text", () => {
    const items = [
      makeItem({ id: "q-1", query: "Find release timeline" }),
      makeItem({
        id: "q-2",
        query: "Different topic",
        answerPreview: "This includes token budget analysis",
      }),
    ]

    expect(filterHistoryItems(items, "timeline")).toHaveLength(1)
    expect(filterHistoryItems(items, "token budget")).toHaveLength(1)
    expect(filterHistoryItems(items, "missing")).toHaveLength(0)
  })

  it("groups pinned items separately and date-buckets unpinned items", () => {
    const now = new Date("2026-02-18T12:00:00.000Z")
    const items = [
      makeItem({
        id: "pinned-item",
        pinned: true,
        timestamp: "2026-02-18T11:00:00.000Z",
      }),
      makeItem({
        id: "today-item",
        pinned: false,
        timestamp: "2026-02-18T09:00:00.000Z",
      }),
      makeItem({
        id: "yesterday-item",
        pinned: false,
        timestamp: "2026-02-17T09:00:00.000Z",
      }),
    ]

    const grouped = buildGroupedHistorySections(items, now)
    expect(grouped.pinned.map((item) => item.id)).toEqual(["pinned-item"])
    expect(grouped.groupedByDate.get("Today")?.map((item) => item.id)).toEqual([
      "today-item",
    ])
    expect(grouped.groupedByDate.get("Yesterday")?.map((item) => item.id)).toEqual([
      "yesterday-item",
    ])
  })

  it("truncates long answer previews while preserving short values", () => {
    expect(truncateAnswerPreview("short answer")).toBe("short answer")

    const longAnswer = "x".repeat(180)
    const preview = truncateAnswerPreview(longAnswer)
    expect(preview).toBeDefined()
    expect(preview!.length).toBe(120)
    expect(preview!.endsWith("...")).toBe(true)
  })

  it("builds markdown export including pin and preview metadata", () => {
    const markdown = buildHistoryExportMarkdown([
      makeItem({
        id: "h-export",
        query: "What changed in stage 3?",
        pinned: true,
        answerPreview: "Stage 3 adds pinning and filters.",
        trustState: "uncited_degraded_answer",
        evidenceOrigin: "local_library",
        citationCount: 0,
        unsynced: true,
        sourceStatus: {
          media_db: { status: "unavailable", count: 0, reason: "index offline" },
        },
      }),
    ])

    expect(markdown).toContain("# Knowledge QA History Export")
    expect(markdown).toContain("Pinned: yes")
    expect(markdown).toContain("Trust: Uncited answer")
    expect(markdown).toContain("Evidence origin: Local library")
    expect(markdown).toContain("Citations: 0")
    expect(markdown).toContain("Unsynced: yes")
    expect(markdown).toContain("Source status: 1 source issue")
    expect(markdown).toContain("Answer preview: Stage 3 adds pinning and filters.")
  })

  it("builds compact status labels for history and recent-session rows", () => {
    expect(
      buildHistoryStatusLabels(
        makeItem({
          trustState: "cited_answer",
          evidenceOrigin: "local_library",
          citationCount: 2,
        })
      )
    ).toEqual(["Cited answer", "Local library", "2 citations"])

    expect(
      buildHistoryStatusLabels(
        makeItem({
          trustState: "failed_search",
          sourceStatus: {
            media_db: { status: "error", count: 0 },
            notes: { status: "searched", count: 3 },
          },
        })
      )
    ).toEqual(["Failed search", "1 source issue"])

    expect(
      buildHistoryStatusLabels(
        makeItem({
          trustState: "unknown_trust",
          unsynced: true,
        })
      )
    ).toEqual(["Trust unknown", "Unsynced"])
  })
})
