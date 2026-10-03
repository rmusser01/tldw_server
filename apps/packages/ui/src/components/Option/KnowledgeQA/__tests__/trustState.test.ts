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
  it.each([
    {
      name: "no visible citation",
      citations: [],
      results: [{ id: "source-1", content: "Evidence" }],
      state: "uncited_degraded_answer",
      reason: "missing_citations",
    },
    {
      name: "citation outside returned scope",
      citations: [{ index: 1, documentId: "hidden-source" }],
      results: [{ id: "source-1", content: "Evidence" }],
      state: "uncited_degraded_answer",
      reason: "citation_source_not_returned",
    },
    {
      name: "missing excerpt",
      citations: [{ index: 1, documentId: "source-1" }],
      results: [{ id: "source-1" }],
      state: "no_answer_insufficient_evidence",
      reason: "missing_inspectable_evidence",
    },
    {
      name: "unavailable source",
      citations: [{ index: 1, documentId: "source-1" }],
      results: [
        { id: "source-1", content: "Old evidence", sourceStatus: "deleted" },
      ],
      state: "no_answer_insufficient_evidence",
      reason: "missing_inspectable_evidence",
    },
    {
      name: "source with unavailable reason",
      citations: [{ index: 1, documentId: "source-1" }],
      results: [
        {
          id: "source-1",
          excerpt: "Old evidence",
          metadata: { unavailable_reason: "permission_denied" },
        },
      ],
      state: "no_answer_insufficient_evidence",
      reason: "missing_inspectable_evidence",
    },
  ])(
    "qualifies backend cited trust when $name",
    ({ citations, results, state, reason }) => {
      expect(
        normalizeKnowledgeAnswerTrust({
          answer: "Answer [1]",
          results,
          citations,
          backendTrust: {
            state: "cited_answer",
            reason_codes: ["web_fallback_used"],
            evidence_origin: "mixed",
          },
        })
      ).toEqual({
        state,
        reasonCodes: ["web_fallback_used", reason],
        evidenceOrigin: "mixed",
      })
    }
  )

  it.each([
    "uncited_degraded_answer",
    "no_answer_insufficient_evidence",
    "no_results",
    "failed_search",
    "unsynced_local_result",
    "unknown_trust",
  ] as const)(
    "preserves authoritative %s even with inspectable citations",
    (state) => {
      expect(
        normalizeKnowledgeAnswerTrust({
          answer: "Answer [1]",
          results: [{ id: "source-1", content: "Evidence" }],
          citations: [{ index: 1, documentId: "source-1" }],
          hasRequiredMetadata: true,
          backendTrust: {
            state,
            reason_codes: ["server_qualification"],
            evidence_origin: "web_fallback",
          },
        })
      ).toEqual({
        state,
        reasonCodes: ["server_qualification"],
        evidenceOrigin: "web_fallback",
      })
    }
  )

  it("retains backend cited trust with an inspectable returned answer citation", () => {
    expect(
      normalizeKnowledgeAnswerTrust({
        answer: "Answer [1]",
        results: [{ id: "source-1", excerpt: "Evidence" }],
        citations: [{ index: 1, documentId: "source-1" }],
        backendTrust: {
          state: "cited_answer",
          reason_codes: ["web_fallback_used"],
          evidence_origin: "web_fallback",
        },
      })
    ).toEqual({
      state: "cited_answer",
      reasonCodes: ["web_fallback_used"],
      evidenceOrigin: "web_fallback",
    })
  })
})
