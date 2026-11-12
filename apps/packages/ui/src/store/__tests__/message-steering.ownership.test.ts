import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { useStoreMessageOption } from "@/store/option"

describe("single-response steering intent", () => {
  let original: ReturnType<typeof useStoreMessageOption.getState>
  beforeEach(() => {
    original = useStoreMessageOption.getState()
  })
  afterEach(() => {
    useStoreMessageOption.setState(original)
  })

  it("consumes the unchanged original intent", () => {
    useStoreMessageOption.getState().setMessageSteeringMode("continue_as_user")
    const revision = useStoreMessageOption.getState().messageSteeringRevision
    useStoreMessageOption
      .getState()
      .clearMessageSteering({ expectedRevision: revision })
    expect(useStoreMessageOption.getState()).toMatchObject({
      messageSteeringMode: "none",
      messageSteeringForceNarrate: false,
      messageSteeringRevision: revision + 1
    })
  })

  it.each(["mode", "narrate", "same value", "edit back"] as const)(
    "preserves later %s intent when an old response finishes",
    (kind) => {
      useStoreMessageOption
        .getState()
        .setMessageSteeringMode("continue_as_user")
      const revision = useStoreMessageOption.getState().messageSteeringRevision
      if (kind === "narrate")
        useStoreMessageOption.getState().setMessageSteeringForceNarrate(true)
      else if (kind === "same value")
        useStoreMessageOption
          .getState()
          .setMessageSteeringMode("continue_as_user")
      else {
        useStoreMessageOption
          .getState()
          .setMessageSteeringMode("impersonate_user")
        if (kind === "edit back")
          useStoreMessageOption
            .getState()
            .setMessageSteeringMode("continue_as_user")
      }
      const replacement = useStoreMessageOption.getState()
      useStoreMessageOption
        .getState()
        .clearMessageSteering({ expectedRevision: revision })
      expect(useStoreMessageOption.getState()).toBe(replacement)
    }
  )

  it("explicit Clear retires old intent even after selecting the same flags again", () => {
    useStoreMessageOption.getState().setMessageSteeringMode("continue_as_user")
    const revision = useStoreMessageOption.getState().messageSteeringRevision
    useStoreMessageOption.getState().clearMessageSteering()
    useStoreMessageOption.getState().setMessageSteeringMode("continue_as_user")
    const replacement = useStoreMessageOption.getState()
    useStoreMessageOption
      .getState()
      .clearMessageSteering({ expectedRevision: revision })
    expect(useStoreMessageOption.getState()).toBe(replacement)
  })
})
