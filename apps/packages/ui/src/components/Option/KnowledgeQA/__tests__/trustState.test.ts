import { describe, expect, it } from "vitest"
import { normalizeKnowledgeAnswerTrust } from "../trustState"

describe("normalizeKnowledgeAnswerTrust", () => {
  it("fails closed for older payloads without trust metadata", () => {
    expect(
      normalizeKnowledgeAnswerTrust({
        answer: "Answer",
        results: [],
        citations: [],
      }).state
    ).toBe("unknown_trust")
  })

  it("marks answer text without valid citations as degraded", () => {
    expect(
      normalizeKnowledgeAnswerTrust({
        answer: "Answer without citations",
        results: [{ id: "source-1", content: "Evidence" }],
        citations: [],
        hasRequiredMetadata: true,
      }).state
    ).toBe("uncited_degraded_answer")
  })

  it("preserves unsynced local result over cited answer", () => {
    expect(
      normalizeKnowledgeAnswerTrust({
        answer: "Answer [1]",
        results: [{ id: "source-1", excerpt: "Evidence" }],
        citations: [{ index: 1, documentId: "source-1" }],
        hasRequiredMetadata: true,
        syncFailed: true,
      }).state
    ).toBe("unsynced_local_result")
  })

  it("preserves failed search over other evidence", () => {
    expect(
      normalizeKnowledgeAnswerTrust({
        answer: "Answer [1]",
        results: [{ id: "source-1", excerpt: "Evidence" }],
        citations: [{ index: 1, documentId: "source-1" }],
        hasRequiredMetadata: true,
        transportFailed: true,
      }).state
    ).toBe("failed_search")
  })

  it("uses backend trust metadata over local heuristics", () => {
    const trust = normalizeKnowledgeAnswerTrust({
      answer: "Answer [1]",
      results: [{ id: "source-1", content: "Evidence" }],
      citations: [{ index: 1, documentId: "source-1" }],
      hasRequiredMetadata: true,
      backendTrust: {
        state: "no_answer_insufficient_evidence",
        reason_codes: ["missing_inspectable_evidence"],
        evidence_origin: "web_fallback",
      },
    })

    expect(trust.state).toBe("no_answer_insufficient_evidence")
    expect(trust.reasonCodes).toEqual(["missing_inspectable_evidence"])
    expect(trust.evidenceOrigin).toBe("web_fallback")
  })

  it("keeps transport and sync failures above backend trust metadata", () => {
    expect(
      normalizeKnowledgeAnswerTrust({
        answer: "Answer [1]",
        results: [{ id: "source-1", content: "Evidence" }],
        citations: [{ index: 1, documentId: "source-1" }],
        hasRequiredMetadata: true,
        transportFailed: true,
        backendTrust: {
          state: "cited_answer",
          reason_codes: [],
          evidence_origin: "local_library",
        },
      }).state
    ).toBe("failed_search")

    expect(
      normalizeKnowledgeAnswerTrust({
        answer: "Answer [1]",
        results: [{ id: "source-1", content: "Evidence" }],
        citations: [{ index: 1, documentId: "source-1" }],
        hasRequiredMetadata: true,
        syncFailed: true,
        backendTrust: {
          state: "cited_answer",
          reason_codes: [],
          evidence_origin: "local_library",
        },
      }).state
    ).toBe("unsynced_local_result")
  })

  it("separates no results from insufficient evidence with weak matches", () => {
    expect(
      normalizeKnowledgeAnswerTrust({
        answer: null,
        results: [],
        citations: [],
        hasRequiredMetadata: true,
      }).state
    ).toBe("no_results")

    expect(
      normalizeKnowledgeAnswerTrust({
        answer: null,
        results: [{ id: "source-1", score: 0.1 }],
        citations: [],
        hasRequiredMetadata: true,
        weakEvidence: true,
      }).state
    ).toBe("no_answer_insufficient_evidence")
  })

  it("marks answer text with citations as cited", () => {
    expect(
      normalizeKnowledgeAnswerTrust({
        answer: "Answer [1]",
        results: [{ id: "source-1", content: "Evidence" }],
        citations: [{ index: 1, documentId: "source-1" }],
        hasRequiredMetadata: true,
      }).state
    ).toBe("cited_answer")
  })
})
