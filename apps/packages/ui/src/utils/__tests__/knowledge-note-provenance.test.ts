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

const active = {
  knowledge_provenance_state: "active",
  knowledge_provenance_version: 3,
  knowledge_provenance_hash: `sha256:${"a".repeat(64)}`,
  knowledge_provenance: original,
}
it("uses canonical history over a changed portable marker", () => {
  const text = retainKnowledgeNoteProvenance(marker({ origin: "reviewed_sources" }), active)
  expect(readKnowledgeNoteProvenance(text)).toEqual(original)
})
it("suppresses a removed history marker even if nested metadata still retains it", () => {
  expect(retainKnowledgeNoteProvenance(marker(original), {
    ...active, knowledge_provenance_state: "deleted", knowledge_provenance: null,
    metadata: { knowledge_provenance: original },
  })).toBe("")
})
it("does not replace a malformed active head with editable marker evidence", () => {
  expect(readKnowledgeNoteProvenance(retainKnowledgeNoteProvenance(marker(original), {
    ...active, knowledge_provenance: { origin: "invalid" },
  }))).toBeNull()
})
it.each([undefined, "unsupported", "future-state"])("keeps portable compatibility for %s", state => {
  expect(readKnowledgeNoteProvenance(retainKnowledgeNoteProvenance(marker(original), {
    knowledge_provenance_state: state,
  }))).toEqual(original)
})
it("retains direct question, reasons, nullable scope and original excerpts exactly", () => {
  const value = { ...original, question: "What happened?", trust_reason_codes: ["missing_citations"],
    scope: { collection_id: null, keyword_filter: "" }, sources: [{
      originalId: "note-a", mediaId: null, title: "Note", type: "text", sourceType: "notes",
      excerpt: "original excerpt", url: "", originalVersion: null,
    }] }
  expect(validateKnowledgeNoteProvenance(value)).toEqual(value)
})
it.each([
  { ...original, arbitrary: "forbidden" },
  { ...original, scope: { unknown: true } },
  { ...original, sources: [{ originalId: null, mediaId: null, title: "T", type: "text", sourceType: null, excerpt: "", secret: "no" }] },
  { ...original, thread_id: "😀".repeat(257) },
  { ...original, scope: { sources: null } },
  { ...original, question: "\ud800" },
  { ...original, sources: Array.from({ length: 12 }, () => ({ originalId: null, mediaId: null, title: "T", type: "text", sourceType: null, excerpt: "😀".repeat(45000) })) },
])("rejects strict or portable contract violation", value => {
  expect(validateKnowledgeNoteProvenance(value)).toBeNull()
})
it("explicit capture references coexist with original references using strict v1 fields", async () => {
  const { appendCapturedNoteProvenance } =
    await import("../knowledge-note-provenance")
  const originalSource = {
    originalId: "result",
    excerpt: "original excerpt",
    mediaId: null,
    title: "Result",
    type: "website" as const,
    sourceType: "web"
  }
  const head = {
    knowledge_provenance_state: "active",
    knowledge_provenance_version: 2,
    knowledge_provenance: { origin: "knowledge_qa", sources: [originalSource] }
  }
  const source = {
    id: "web-clipper:clip",
    mediaId: 71,
    title: "Article",
    type: "website" as const,
    addedAt: new Date(),
    webCapture: {
      clipId: "clip",
      requestedUrl: "https://example.org",
      capturedAt: "2026-10-07T00:00:00Z",
      contentSha256: "hash",
      refreshOf: null,
      mediaId: 71,
      versionNumber: 9,
      versionUuid: "version"
    }
  }
  const value = appendCapturedNoteProvenance(head, [source])
  expect(value?.sources).toEqual([
    originalSource,
    expect.objectContaining({
      originalVersion: 9,
      snapshotMediaId: 71,
      sourceType: "server_article"
    })
  ])
  expect(validateKnowledgeNoteProvenance(value)).toEqual(value)
  expect(
    appendCapturedNoteProvenance(
      { ...head, knowledge_provenance_state: "deleted" },
      [source]
    )
  ).toBeNull()
  expect(
    appendCapturedNoteProvenance(
      { ...head, knowledge_provenance_state: "unsupported" },
      [source]
    )
  ).toBeNull()
})
