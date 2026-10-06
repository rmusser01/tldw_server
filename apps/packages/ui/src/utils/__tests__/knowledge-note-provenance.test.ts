import { describe, expect, it } from "vitest"
import {
  stripKnowledgeNoteProvenance,
  validateKnowledgeNoteProvenance,
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
  expect(
    retainKnowledgeNoteProvenance(
      "Body\n<!-- tldw-knowledge:v1:invalid -->",
      original,
    ),
  ).toContain("<!-- tldw-knowledge:v1:invalid -->")
})

it.each([
  ["", true],
  [null, true],
  [[], true],
  [["topic"], true],
  ["topic", true],
  ["x".repeat(513), false],
  [[""], false],
  [{ arbitrary: "value" }, false],
])(
  "validates bounded keyword filter %j (valid=%s)",
  (keyword_filter, valid) => {
    const provenance = validateKnowledgeNoteProvenance({
      ...original,
      research: {
        workspace_id: "workspace-a",
        import_id: "current-import",
        sources: [
          {
            mediaId: 7,
            evidence: {
              importId: "current-import",
              threadId: "thread-a",
              snapshot: false,
              sources: [],
              scope: { keyword_filter, collection_id: null },
            },
          },
        ],
      },
    })
    expect(provenance !== null).toBe(valid)
  },
)

it.each([4, 0, -1, 1.5, "4"])(
  "retains only valid original note revisions (%j)",
  (originalVersion) => {
    const provenance = validateKnowledgeNoteProvenance({
      origin: "knowledge_qa",
      research: {
        workspace_id: "workspace-a",
        import_id: "import-a",
        sources: [
          {
            mediaId: 101,
            evidence: {
              importId: "import-a",
              threadId: null,
              snapshot: true,
              sources: [
                {
                  originalId: "note-uuid",
                  originalVersion,
                  excerpt: "Retrieved evidence",
                  mediaId: null,
                  title: "Field note",
                  type: "text",
                  sourceType: "notes",
                },
              ],
            },
          },
        ],
      },
    })
    if (originalVersion !== 4) expect(provenance).toBeNull()
    else
      expect(provenance?.research?.sources[0].evidence.sources[0]).toEqual(
        expect.objectContaining({ originalVersion: 4 }),
      )
  },
)
