import { describe, expect, it } from "vitest"
import { getTutorialById } from "../registry"

describe("Flashcards tutorial registration", () => {
  it("uses current transfer locale keys for the transfer step", () => {
    const tutorial = getTutorialById("flashcards-basics")
    expect(tutorial).toBeDefined()

    const transferStep = tutorial!.steps.find((step) =>
      step.target.includes("flashcards-import-format")
    )

    expect(transferStep).toMatchObject({
      titleKey: "tutorials:flashcards.basics.transferTitle",
      contentKey: "tutorials:flashcards.basics.transferContent",
      titleFallback: "Transfer Tab",
      contentFallback: "Use Transfer for CSV/JSON/APKG import and export workflows."
    })
  })
})
