import { describe, expect, it } from "vitest"
import {
  stripKnowledgeNoteProvenance,
  readKnowledgeNoteProvenance,
  retainKnowledgeNoteProvenance,
} from "../knowledge-note-provenance"
const marker = (value: unknown) =>
  `<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(value))} -->`
const original = {
  origin: "knowledge_qa",
  trust_state: "uncited_degraded_answer",
  thread_id: "owned-thread",
}
describe("canonical Knowledge content provenance", () => {
  it("retains the original marker through a replaced editor body without duplicating it", () => {
    const content = retainKnowledgeNoteProvenance("Edited body", original)
    expect(retainKnowledgeNoteProvenance(content, original)).toBe(content)
    expect(readKnowledgeNoteProvenance(content)).toEqual(original)
  })
  it.each([
    "<!-- tldw-knowledge:v1:%bad -->",
    marker({ ...original, trust_state: "invented" }),
    marker({ ...original, thread_id: "x".repeat(513) }),
    marker({
      ...original,
      research: {
        workspace_id: "w",
        import_id: "i",
        sources: [{ mediaId: -1, evidence: {} }],
      },
    }),
    `<!-- tldw-knowledge:v1:${"x".repeat(1_000_001)} -->`,
  ])(
    "does not infer provenance from invalid or oversized marker",
    (content) => {
      expect(readKnowledgeNoteProvenance(content)).toBeNull()
    },
  )
  it("strips unrecognized metadata rather than persisting credentials or arbitrary properties", () => {
    const content = retainKnowledgeNoteProvenance("Body", {
      ...original,
      apiKey: "never-persist",
    })
    expect(content).not.toContain("never-persist")
  })
})

it("hides only recognized provenance while preserving user comments and text", () => {
  expect(
    stripKnowledgeNoteProvenance(
      retainKnowledgeNoteProvenance(
        "Body\n<!-- ordinary comment -->",
        original,
      ),
    ),
  ).toBe("Body\n<!-- ordinary comment -->")
  expect(
    stripKnowledgeNoteProvenance("Body\n<!-- tldw-knowledge:v1:invalid -->"),
  ).toBe("Body\n<!-- tldw-knowledge:v1:invalid -->")
})

it("keeps malformed provenance-like user comments when saving recognized original provenance", () => {
  expect(retainKnowledgeNoteProvenance("Body\n<!-- tldw-knowledge:v1:invalid -->", original)).toContain("<!-- tldw-knowledge:v1:invalid -->")
})
