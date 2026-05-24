import { describe, expect, it } from "vitest"
import {
  buildKnowledgeQaSeedNote,
  buildKnowledgeQaWorkspacePrefill,
} from "../research-workspace-prefill"

describe("research-workspace-prefill", () => {
  it("builds a normalized Knowledge QA prefill payload", () => {
    const payload = buildKnowledgeQaWorkspacePrefill({
      threadId: "thread-1",
      query: "Compare findings",
      answer: "Answer body",
      citations: [1, 2, 2],
      results: [
        {
          id: "101",
          metadata: {
            title: "Quarterly Report",
            source_type: "pdf",
            page_number: 3,
          },
        },
        {
          id: "abc",
          metadata: {
            source: "web source",
            source_type: "website",
            url: "https://example.com",
          },
        },
      ],
    })

    expect(payload.kind).toBe("knowledge_qa_thread")
    expect(payload.citations).toEqual([1, 2])
    expect(payload.sources[0]).toEqual(
      expect.objectContaining({
        mediaId: 101,
        type: "pdf",
        citationIndex: 1,
      })
    )
    expect(payload.sources[1]).toEqual(
      expect.objectContaining({
        mediaId: null,
        type: "website",
      })
    )
  })

  it("formats seed note content with question, answer, and source list", () => {
    const payload = buildKnowledgeQaWorkspacePrefill({
      threadId: "thread-1",
      query: "When did the policy change?",
      answer: "It changed in 2024.",
      citations: [1],
      results: [
        {
          id: "44",
          metadata: {
            title: "Policy memo",
            page_number: 12,
            url: "https://example.com/policy",
          },
        },
      ],
    })

    const note = buildKnowledgeQaSeedNote(payload)
    expect(note).toContain("Imported from Knowledge QA")
    expect(note).toContain("Question: When did the policy change?")
    expect(note).toContain("It changed in 2024.")
    expect(note).toContain("[1] Policy memo (p. 12) - https://example.com/policy")
  })
})
