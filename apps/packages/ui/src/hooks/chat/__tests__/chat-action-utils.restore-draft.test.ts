import { describe, expect, it } from "vitest"

import {
  chatSubmitFailed,
  chatSubmitSkipped,
  chatSubmitSubmitted,
  shouldRestoreDraftAfterSubmit
} from "../chat-action-utils"

// CM-02: a send that didn't complete puts its text back in the composer,
// unless its question stays in the transcript.
describe("shouldRestoreDraftAfterSubmit", () => {
  it("restores the draft of a skipped send", () => {
    expect(shouldRestoreDraftAfterSubmit(chatSubmitSkipped("Request cancelled"))).toBe(true)
  })

  it("keeps the composer empty when the question stays in the transcript", () => {
    expect(
      shouldRestoreDraftAfterSubmit(chatSubmitSkipped("Request cancelled", { questionKept: true }))
    ).toBe(false)
  })

  it("restores only refusals that ask for it", () => {
    expect(shouldRestoreDraftAfterSubmit({ ...chatSubmitFailed("refused"), restoreDraft: true })).toBe(true)
    expect(shouldRestoreDraftAfterSubmit(chatSubmitFailed("provider down"))).toBe(false)
    expect(shouldRestoreDraftAfterSubmit(chatSubmitSubmitted())).toBe(false)
  })
})
