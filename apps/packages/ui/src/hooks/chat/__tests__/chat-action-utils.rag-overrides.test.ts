import { describe, expect, it } from "vitest"

import {
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
