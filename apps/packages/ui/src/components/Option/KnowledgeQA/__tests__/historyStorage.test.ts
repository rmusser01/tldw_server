import { describe, expect, it, vi } from "vitest"
import { persistKnowledgeQaHistory } from "../historyStorage"
import type { SearchHistoryItem } from "../types"

const makeHistory = (count: number): SearchHistoryItem[] =>
  Array.from({ length: count }).map((_, index) => ({
    id: `h-${index + 1}`,
    query: `query-${index + 1}`,
    timestamp: new Date().toISOString(),
    sourcesCount: 1,
    hasAnswer: true,
  }))

describe("persistKnowledgeQaHistory", () => {
  it("persists full history when storage write succeeds", () => {
    const history = makeHistory(5)
    const writer = vi.fn()

    const result = persistKnowledgeQaHistory(history, writer)

    expect(writer).toHaveBeenCalledTimes(1)
    expect(result.wasTrimmed).toBe(false)
    expect(result.storedHistory).toHaveLength(5)
  })

  it("persists Knowledge QA trust and evidence metadata", () => {
    const baseHistoryItem = makeHistory(1)[0]!
    const history: SearchHistoryItem[] = [
      {
        ...baseHistoryItem,
        trustState: "uncited_degraded_answer",
        trustReasonCodes: ["missing_inspectable_evidence"],
        evidenceOrigin: "local_library",
        citationCount: 0,
        unsynced: true,
        sourceStatus: {
          media_db: {
            status: "unavailable",
            count: 0,
            reason: "index offline",
          },
        },
      },
    ]
    const writer = vi.fn()

    persistKnowledgeQaHistory(history, writer)

    const serialized = writer.mock.calls[0]?.[0]
    expect(JSON.parse(serialized)).toMatchObject([
      {
        trustState: "uncited_degraded_answer",
        trustReasonCodes: ["missing_inspectable_evidence"],
        evidenceOrigin: "local_library",
        citationCount: 0,
        unsynced: true,
        sourceStatus: {
          media_db: {
            status: "unavailable",
            count: 0,
            reason: "index offline",
          },
        },
      },
    ])
  })

  it("trims oldest items and retries when storage quota is exceeded", () => {
    const history = makeHistory(25)
    const writer = vi
      .fn()
      .mockImplementationOnce(() => {
        const error = new Error("quota exceeded")
        ;(error as Error & { name: string }).name = "QuotaExceededError"
        throw error
      })
      .mockImplementationOnce(() => undefined)

    const result = persistKnowledgeQaHistory(history, writer)

    expect(writer).toHaveBeenCalledTimes(2)
    expect(result.wasTrimmed).toBe(true)
    expect(result.storedHistory).toHaveLength(15)
    expect(result.storedHistory[0]?.id).toBe("h-1")
    expect(result.storedHistory[result.storedHistory.length - 1]?.id).toBe("h-15")
  })
})
