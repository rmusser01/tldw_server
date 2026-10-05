import { describe, expect, it } from "vitest"
import {
  buildKnowledgeMediaScopePath,
  parseKnowledgeMediaScope,
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
