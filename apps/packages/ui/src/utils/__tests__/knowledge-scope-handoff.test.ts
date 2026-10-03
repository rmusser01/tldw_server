import { describe, expect, it } from "vitest"
import {
  buildKnowledgeMediaScopePath,
  parseKnowledgeMediaScope,
  buildKnowledgeNoteScopePath,
  parseKnowledgeScope
} from "../knowledge-scope-handoff"
describe("Knowledge media scope handoff", () => {
  it("round-trips the exact added media set", () => {
    const path = buildKnowledgeMediaScopePath([3, 7, 3])
    expect(path).toBe("/knowledge?media_ids=3%2C7")
    expect(
      parseKnowledgeMediaScope(new URL(path, "https://local.test").search),
    ).toEqual({ mediaIds: [3, 7], invalid: false })
  })
  it("returns null only when no transfer was requested", () =>
    expect(parseKnowledgeMediaScope("?other=3")).toBeNull())
  it.each([
    "",
    "invalid",
    "1,0",
    "-1",
    "1.5",
    "1e2",
    "9007199254740992",
    "1,,2",
    "1&media_ids=2",
  ])(
    "does not treat invalid transfer scope %s as the whole library",
    (value) => {
      expect(parseKnowledgeMediaScope(`?media_ids=${value}`)).toEqual({
        mediaIds: [],
        invalid: true,
      })
    },
  )
  it.each(
    [[], [0], [1.5], [Number.MAX_SAFE_INTEGER + 1], [3, NaN]].map((ids) => ({
      ids,
    })),
  )("emits explicit invalid scope for unusable IDs $ids", ({ ids }) => {
    expect(buildKnowledgeMediaScopePath(ids)).toBe("/knowledge?media_ids=")
  })
})

const noteId = "12345678-1234-4234-8234-123456789abc"
describe("Knowledge captured-note scope", () => {
  it("round-trips a canonical note without a media scope", () => {
    const path = buildKnowledgeNoteScopePath([noteId])
    expect(
      parseKnowledgeScope(new URL(path, "https://local.test").search)
    ).toEqual({
      mediaIds: [],
      noteIds: [noteId],
      invalid: false
    })
  })
  it.each([
    "",
    "note-1",
    `${noteId}&note_ids=${noteId}`,
    `${noteId}&media_ids=`,
    `invalid&media_ids=3`
  ])("never widens malformed or mixed scope %s", (value) => {
    expect(parseKnowledgeScope(`?note_ids=${value}`)).toEqual({
      mediaIds: [],
      noteIds: [],
      invalid: true
    })
  })
  it("keeps both valid explicit scopes", () => {
    expect(parseKnowledgeScope(`?note_ids=${noteId}&media_ids=3`)).toEqual({
      mediaIds: [3],
      noteIds: [noteId],
      invalid: false
    })
  })
})
