import { describe, expect, it } from "vitest"

import { validateKnowledgeResultScope } from "../scopeValidation"

describe("validateKnowledgeResultScope", () => {
  it("flags excluded exact media sources", () => {
    const result = validateKnowledgeResultScope({
      selectedSources: ["media_db"],
      selectedMediaIds: [42],
      selectedNoteIds: [],
      webFallbackEnabled: false,
      results: [
        {
          id: "out-of-scope",
          metadata: { source_type: "media_db", source_id: "99" },
        },
      ],
    })

    expect(result.violations).toEqual([
      {
        index: 0,
        sourceId: "99",
        sourceType: "media_db",
        reason: "excluded_source",
      },
    ])
    expect(result.acceptedResults).toEqual([])
  })

  it("allows selected notes, enabled web fallback evidence, and explicit scope broadening", () => {
    const scoped = validateKnowledgeResultScope({
      selectedSources: ["notes"],
      selectedMediaIds: [],
      selectedNoteIds: ["note-a"],
      webFallbackEnabled: true,
      results: [
        {
          id: "note-a-result",
          metadata: { source_type: "notes", source_id: "note-a" },
        },
        {
          id: "web-result",
          metadata: {
            source_type: "web",
            source_id: "https://example.com",
            evidence_origin: "web_fallback",
          },
        },
        {
          id: "workspace-broadened",
          metadata: {
            source_type: "media_db",
            source_id: "77",
            scope_broadened_reason: "scope_broadened_by_workspace",
          },
        },
      ],
    })

    expect(scoped.violations).toEqual([])
    expect(scoped.acceptedResults).toHaveLength(3)
  })

  it("flags unselected source categories even when exact filters are empty", () => {
    const result = validateKnowledgeResultScope({
      selectedSources: ["notes"],
      selectedMediaIds: [],
      selectedNoteIds: [],
      webFallbackEnabled: false,
      results: [
        {
          id: "media-result",
          metadata: { source_type: "media_db", source_id: "42" },
        },
      ],
    })

    expect(result.violations).toEqual([
      {
        index: 0,
        sourceId: "42",
        sourceType: "media_db",
        reason: "excluded_source",
      },
    ])
  })

  it("treats content source types as media library results for exact-id checks", () => {
    const result = validateKnowledgeResultScope({
      selectedSources: ["media_db"],
      selectedMediaIds: [42],
      selectedNoteIds: [],
      webFallbackEnabled: false,
      results: [
        {
          id: "selected-pdf",
          metadata: { source_type: "pdf", source_id: "42" },
        },
        {
          id: "unselected-video",
          metadata: { source_type: "video", source_id: "99" },
        },
      ],
    })

    expect(result.acceptedResults).toEqual([
      expect.objectContaining({ id: "selected-pdf" }),
    ])
    expect(result.violations).toEqual([
      {
        index: 1,
        sourceId: "99",
        sourceType: "video",
        reason: "excluded_source",
      },
    ])
  })

  it("flags web fallback results when fallback is disabled", () => {
    const result = validateKnowledgeResultScope({
      selectedSources: [],
      selectedMediaIds: [],
      selectedNoteIds: [],
      webFallbackEnabled: false,
      results: [
        {
          id: "web-result",
          metadata: {
            source_type: "web",
            source_id: "https://example.com",
            evidence_origin: "web_fallback",
          },
        },
      ],
    })

    expect(result.violations).toEqual([
      {
        index: 0,
        sourceId: "https://example.com",
        sourceType: "web",
        reason: "excluded_source",
      },
    ])
  })
})
