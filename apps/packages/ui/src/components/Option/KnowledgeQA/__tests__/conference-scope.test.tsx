import { describe, expect, it } from "vitest"
import {
  DEFAULT_RAG_SETTINGS,
  buildRagSearchRequest,
} from "@/services/rag/unified-rag"
import { buildConferenceCollectionKnowledgeQaOptions } from "../conference-scope"

describe("conference collection Knowledge QA scope", () => {
  it("builds backend-owned collection scoped RAG options", () => {
    const scope = buildConferenceCollectionKnowledgeQaOptions(44)
    const request = buildRagSearchRequest({
      ...DEFAULT_RAG_SETTINGS,
      ...scope,
      query: "What themes repeat across the conference?",
    })

    expect(scope).toEqual({
      collection_id: 44,
      sources: ["media_db"],
    })
    expect(request.options.collection_id).toBe(44)
    expect(request.options.sources).toEqual(["media_db"])
    expect(request.options.include_media_ids).toBeUndefined()
  })

  it("rejects missing or invalid collection identifiers", () => {
    expect(() => buildConferenceCollectionKnowledgeQaOptions(0)).toThrow(
      "collection_id_required"
    )
    expect(() => buildConferenceCollectionKnowledgeQaOptions("not-a-number")).toThrow(
      "collection_id_required"
    )
    expect(() => buildConferenceCollectionKnowledgeQaOptions("44abc")).toThrow(
      "collection_id_required"
    )
  })
})
