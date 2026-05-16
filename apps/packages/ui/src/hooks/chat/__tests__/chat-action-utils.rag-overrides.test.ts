import { describe, expect, it } from "vitest"

import {
  resolveCompareModelSelection,
  resolveTurnFileRetrievalEnabled,
  resolveTurnRagMediaIds,
  shouldUseRagForTurn
} from "../chat-action-utils"

describe("turn-level RAG media overrides", () => {
  it("preserves an explicit empty override", () => {
    const resolved = resolveTurnRagMediaIds({
      requestOverrides: { ragMediaIds: [] },
      ragMediaIds: [1, 2]
    })

    expect(resolved).toEqual([])
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: null,
        fileRetrievalEnabled: true,
        ragMediaIds: resolved
      })
    ).toBe(false)
  })

  it("uses an explicit non-empty override", () => {
    const resolved = resolveTurnRagMediaIds({
      requestOverrides: { ragMediaIds: [7, 9] },
      ragMediaIds: [1, 2]
    })

    expect(resolved).toEqual([7, 9])
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: null,
        fileRetrievalEnabled: true,
        ragMediaIds: resolved
      })
    ).toBe(true)
  })

  it("uses an explicit file retrieval override for the turn", () => {
    const fileRetrievalEnabled = resolveTurnFileRetrievalEnabled({
      requestOverrides: { fileRetrievalEnabled: true },
      fileRetrievalEnabled: false
    })

    expect(fileRetrievalEnabled).toBe(true)
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: null,
        fileRetrievalEnabled,
        ragMediaIds: [7, 9]
      })
    ).toBe(true)
  })

  it("falls back to inherited media ids when no override is present", () => {
    const resolved = resolveTurnRagMediaIds({
      requestOverrides: {},
      ragMediaIds: [3]
    })

    expect(resolved).toEqual([3])
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: null,
        fileRetrievalEnabled: true,
        ragMediaIds: resolved
      })
    ).toBe(true)
  })

  it("falls back to inherited media ids when an optional override is undefined", () => {
    const resolved = resolveTurnRagMediaIds({
      requestOverrides: { ragMediaIds: undefined },
      ragMediaIds: [3]
    })

    expect(resolved).toEqual([3])
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: null,
        fileRetrievalEnabled: true,
        ragMediaIds: resolved
      })
    ).toBe(true)
  })

  it("uses RAG when selectedKnowledge is set without media ids", () => {
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: { id: "kb-1" },
        fileRetrievalEnabled: false,
        ragMediaIds: []
      })
    ).toBe(true)
  })

  it("does not use media-id RAG when file retrieval is disabled", () => {
    expect(
      shouldUseRagForTurn({
        selectedKnowledge: null,
        fileRetrievalEnabled: false,
        ragMediaIds: [5]
      })
    ).toBe(false)
  })
})

describe("compare model selection", () => {
  it("uses a provider-qualified key for compare branch identity while keeping the API model bare", () => {
    expect(
      resolveCompareModelSelection("openrouter:anthropic/claude-3.5-sonnet")
    ).toEqual({
      selectedModel: "anthropic/claude-3.5-sonnet",
      historyModelKey: "openrouter:anthropic/claude-3.5-sonnet",
      provider: "openrouter"
    })
  })

  it("keeps unqualified compare selections unchanged", () => {
    expect(resolveCompareModelSelection("gpt-4o")).toEqual({
      selectedModel: "gpt-4o",
      historyModelKey: "gpt-4o",
      provider: undefined
    })
  })
})
