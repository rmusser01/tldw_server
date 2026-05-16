import { beforeEach, describe, expect, it, vi } from "vitest"

const designSystemMocks = vi.hoisted(() => ({
  getDesignSystemStateLabel: vi.fn(
    (key: string, fallback: string) =>
      ({
        ready: "Registry Ready",
        unavailable: "Registry Unavailable",
      })[key] ?? fallback
  ),
}))

vi.mock("@/design-system", () => ({
  getDesignSystemStateLabel: designSystemMocks.getDesignSystemStateLabel,
}))

import {
  buildSourceHealthSummary,
  getSourceHealthStatusLabel,
  normalizeKnowledgeSourceHealth,
} from "../sourceHealth"

describe("Knowledge QA source health normalization", () => {
  beforeEach(() => {
    designSystemMocks.getDesignSystemStateLabel.mockClear()
  })

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

    expect(buildSourceHealthSummary(normalized)).toBe("Sources ready: 1 of 2")
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
    expect(designSystemMocks.getDesignSystemStateLabel).toHaveBeenCalledWith(
      "ready",
      ""
    )
    expect(designSystemMocks.getDesignSystemStateLabel).toHaveBeenCalledWith(
      "unavailable",
      ""
    )
  })
})
