import { describe, expect, it } from "vitest"
import { dedupeTags, hasTag, tagIdentity } from "@/utils/tag-identity"

describe("tag identity", () => {
  it("folds case and NFC composition, and trims", () => {
    expect(tagIdentity("  Machine Learning ")).toBe("machine learning")
    expect(tagIdentity("Éclair")).toBe(tagIdentity("éclair"))
  })

  it("dedupes case variants and keeps the first-seen casing", () => {
    expect(dedupeTags(["Machine Learning", "machine learning", " NLP ", "", "nlp"])).toEqual([
      "Machine Learning",
      "NLP"
    ])
  })

  it("finds a tag in any case", () => {
    expect(hasTag(["Foo", "Bar"], "FOO")).toBe(true)
    expect(hasTag(["Foo", "Bar"], "baz")).toBe(false)
  })
})
