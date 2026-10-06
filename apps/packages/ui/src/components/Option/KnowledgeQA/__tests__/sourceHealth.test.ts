import { describe, expect, it, vi } from "vitest"

const designSystemMocks = vi.hoisted(() => ({
  READY_STATE_LABEL: "Registry Ready",
  UNAVAILABLE_STATE_LABEL: "Registry Unavailable",
}))

vi.mock("@/design-system", () => ({
  READY_STATE_LABEL: designSystemMocks.READY_STATE_LABEL,
  UNAVAILABLE_STATE_LABEL: designSystemMocks.UNAVAILABLE_STATE_LABEL,
}))

import {
  buildSourceHealthSummary,
  getSourceHealthStatusLabel,
  normalizeKnowledgeSourceHealth,
} from "../sourceHealth"

describe("Knowledge QA source health normalization", () => {
  it("normalizes partial backend payloads without colliding with search source status", () => {
    const normalized = normalizeKnowledgeSourceHealth({
      sources: [
        {
          source_id: "media_db",
          label: "Documents & Media",
          available: true,
          searchable: true,
          index_status: "ready",
          embedding_status: "not_applicable",
          disabled_reason: null,
        },
      ],
    })

    expect(normalized.bySource.media_db?.indexStatus).toBe("ready")
    expect(normalized.bySource.media_db?.embeddingStatus).toBe("not_applicable")
    expect(normalized.bySource.media_db).not.toHaveProperty("status")
  })

  it("builds a compact summary", () => {
    const normalized = normalizeKnowledgeSourceHealth({
      sources: [
        {
          source_id: "media_db",
          label: "Documents & Media",
          available: true,
          searchable: true,
          index_status: "ready",
          embedding_status: "unknown",
          disabled_reason: null,
        },
        {
          source_id: "prompts",
          label: "Prompts",
          available: false,
          searchable: false,
          index_status: "unavailable",
          embedding_status: "unavailable",
          disabled_reason: "no_retriever_configured",
        },
      ],
    })

    expect(buildSourceHealthSummary(normalized)).toBe(
      "Available services: 1 of 2 · Personal items: unknown"
    )
  })

  it("drops unknown source IDs and normalizes unknown statuses safely", () => {
    const normalized = normalizeKnowledgeSourceHealth({
      sources: [
        {
          source_id: "generated_test_artifacts",
          label: "Generated",
          available: true,
          searchable: true,
          index_status: "ready",
          embedding_status: "ready",
        },
        {
          source_id: "notes",
          label: "Notes",
          available: true,
          searchable: false,
          index_status: "surprising",
          embedding_status: "unexpected",
        },
      ],
    })

    expect(normalized.sources).toHaveLength(1)
    expect(normalized.bySource.notes?.indexStatus).toBe("unknown")
    expect(normalized.bySource.notes?.embeddingStatus).toBe("unknown")
    expect(getSourceHealthStatusLabel(normalized.bySource.notes)).toBe("Unknown")
  })

  it("uses design-system registry labels for canonical ready and unavailable statuses", () => {
    const normalized = normalizeKnowledgeSourceHealth({
      sources: [
        {
          source_id: "media_db",
          label: "Documents & Media",
          available: true,
          searchable: true,
          index_status: "ready",
          embedding_status: "ready",
        },
        {
          source_id: "notes",
          label: "Notes",
          available: true,
          searchable: false,
          index_status: "ready",
          embedding_status: "missing",
        },
        {
          source_id: "prompts",
          label: "Prompts",
          available: false,
          searchable: false,
          index_status: "unavailable",
          embedding_status: "unavailable",
        },
      ],
    })

    expect(getSourceHealthStatusLabel(normalized.bySource.media_db)).toBe(
      "Registry Ready"
    )
    expect(getSourceHealthStatusLabel(normalized.bySource.notes)).toBe(
      "Registry Unavailable"
    )
    expect(getSourceHealthStatusLabel(normalized.bySource.prompts)).toBe(
      "Registry Unavailable"
    )
  })
})

it("uses stored totals without inventing vector readiness", () => {
  const state = normalizeKnowledgeSourceHealth(
    {
      sources: [
        {
          source_id: "media_db",
          available: true,
          searchable: true,
          index_status: "ready",
          item_count: null,
          indexed_count: null,
          embedding_status: "unknown"
        },
        {
          source_id: "notes",
          available: true,
          searchable: true,
          index_status: "ready",
          item_count: null,
          indexed_count: null,
          embedding_status: "unknown"
        }
      ]
    },
    { media_db: { pagination: { total: 0 } }, notes: { total: 0 } }
  )
  expect(state.bySource.media_db?.itemCount).toBe(0)
  expect(state.bySource.notes?.itemCount).toBe(0)
  expect(state.bySource.media_db?.indexedCount).toBeNull()
  expect(state.bySource.media_db?.embeddingStatus).toBe("unknown")
  expect(buildSourceHealthSummary(state)).toBe(
    "Available services: 2 of 2 · Stored personal items: 0 · Searchable personal items: 0"
  )
})

it("does not label stored content searchable when indexed counts are unknown", () => {
  const state = normalizeKnowledgeSourceHealth(
    {
      sources: ["media_db", "notes"].map((source_id) => ({
        source_id,
        available: true,
        searchable: true,
        index_status: "ready",
        indexed_count: null
      }))
    },
    { media_db: { pagination: { total: 1 } }, notes: { total: 0 } }
  )
  expect(buildSourceHealthSummary(state)).toBe(
    "Available services: 2 of 2 · Stored personal items: 1 · Searchable personal items: unknown"
  )
})
