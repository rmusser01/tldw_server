import { describe, expect, it } from "vitest"

import { normalizeGeneratedCards } from "../shared"

describe("normalizeGeneratedCards", () => {
  it.each([null, undefined, {}, "not a card list"])("rejects malformed top-level output: %s", value => {
    expect(normalizeGeneratedCards(value)).toEqual([])
  })

  it("keeps only complete editable cards from partial output", () => {
    const drafts = normalizeGeneratedCards([
      null, {}, { front: "Question only" }, { back: "Answer only" },
      { front: "  Complete question  ", back: "  Complete answer  ", tags: ["biology"] }
    ])
    expect(drafts).toHaveLength(1)
    expect(drafts[0]).toMatchObject({ front: "Complete question", back: "Complete answer", tags: ["biology"] })
  })

  it("falls back to model_type for legacy generation_type and preserves true_false", () => {
    const drafts = normalizeGeneratedCards([
      {
        front: "What powers the cell?",
        back: "ATP",
        model_type: "basic_reverse"
      },
      {
        front: "True or false: ATP powers the cell.",
        back: "True.",
        model_type: "basic",
        generation_type: "true_false"
      }
    ])

    expect(drafts.map((draft) => draft.generation_type)).toEqual([
      "basic_reverse",
      "true_false"
    ])
    expect(drafts.map((draft) => draft.model_type)).toEqual(["basic_reverse", "basic"])
  })
})
