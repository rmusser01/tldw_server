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
  it("retains note UUID, both chunks, web reference, scope and answer qualifications", () => {
    const payload = buildKnowledgeQaWorkspacePrefill({
      threadId: "thread-1",
      query: "What supports this claim?",
      answer: "Draft",
      citations: [1, 2, 3],
      results: [
        {
          id: "chunk-1",
          content: "Supporting excerpt",
          metadata: {
            source_type: "notes",
            title: "Field note",
            note_id: "note-uuid",
          },
        },
        {
          id: "chunk-2",
          content: "Second excerpt",
          metadata: {
            source_type: "notes",
            title: "Field note",
            note_id: "note-uuid",
          },
        },
        {
          id: "71",
          content: "Web excerpt",
          metadata: { source_type: "web", url: "https://example.com/evidence" },
        },
      ],
      answerTrustState: "uncited_degraded_answer",
      answerEvidenceOrigin: "local_library",
      answerTrustReasonCodes: ["missing_citations"],
      scope: {
        sources: ["notes"],
        include_media_ids: [],
        include_note_ids: ["note-uuid"],
      },
    })
    const note = buildKnowledgeQaSeedNote(payload)
    for (const text of [
      "Supporting excerpt",
      "Second excerpt",
      "note-uuid",
      "https://example.com/evidence",
      "uncited_degraded_answer",
      "missing_citations",
      "thread-1",
    ])
      expect(note).toContain(text)
    expect(payload.sources.map((source) => source.mediaId)).toEqual([
      null,
      null,
      null,
    ])
    expect(payload.sources[0].originalId).toBe("note-uuid")
    expect(payload.sources[0].url).toBe("/notes?source_ref_id=note-uuid")
    expect(payload.scope?.include_note_ids).toEqual(["note-uuid"])
  })
  it("labels direct reviewed-media selections without implying a generated QA answer", () => {
    const payload = buildKnowledgeQaWorkspacePrefill({
      threadId: null,
      query: "",
      answer: null,
      citations: [],
      results: [{ metadata: { media_id: 7, title: "Reviewed paper" } }],
    })
    const note = buildKnowledgeQaSeedNote(payload)
    expect(note).toContain("Imported reviewed sources")
    expect(note).not.toContain("Knowledge QA")
    expect(note).not.toContain("Answer:")
    expect(note).toContain("Reviewed paper")
  })
})
