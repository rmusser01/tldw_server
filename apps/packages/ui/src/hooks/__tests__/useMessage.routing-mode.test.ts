import { describe, expect, it } from "vitest"

import { resolveUseMessageSendMode } from "@/hooks/useMessage.routing"

describe("useMessage routing mode resolution", () => {
  it("prefers tracked character over overlay", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "tracked_character",
        hasEffectiveAssistant: true
      })
    ).toBe("tracked_character")
  })

  it("prefers tracked persona over overlay", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "tracked_persona",
        hasEffectiveAssistant: true
      })
    ).toBe("tracked_persona")
  })

  it("keeps plain mode when no effective assistant exists", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "plain",
        hasEffectiveAssistant: false
      })
    ).toBe("plain")
  })

  it("requires explicit tracked intent before routing a draft character through tracked chat", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "plain",
        hasEffectiveAssistant: true,
        draftAssistantKind: "character"
      })
    ).toBe("plain")
  })

  it("requires explicit tracked intent before routing a draft persona through tracked chat", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "plain",
        hasEffectiveAssistant: true,
        draftAssistantKind: "persona"
      })
    ).toBe("plain")
  })

  it("routes tracked-character startup when the draft explicitly requests tracked mode", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "plain",
        hasEffectiveAssistant: true,
        draftAssistantKind: "character",
        draftAssistantSelectionMode: "tracked"
      })
    ).toBe("tracked_character")
  })

  it("routes tracked-persona startup when the draft explicitly requests tracked mode", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "plain",
        hasEffectiveAssistant: true,
        draftAssistantKind: "persona",
        draftAssistantSelectionMode: "tracked"
      })
    ).toBe("tracked_persona")
  })

  it("uses overlay routing when overlay state is active", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "overlay",
        hasEffectiveAssistant: true,
        draftAssistantKind: "persona"
      })
    ).toBe("overlay")
  })

  it("keeps overlay routing when only draft overlay intent exists before settings sync", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "plain",
        hasEffectiveAssistant: true,
        draftAssistantKind: "character",
        draftAssistantSelectionMode: "overlay"
      })
    ).toBe("overlay")
  })
})
