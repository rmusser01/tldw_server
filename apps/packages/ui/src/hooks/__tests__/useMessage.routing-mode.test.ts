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

  it("preserves tracked-character startup for a plain chat with a character draft selection", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "plain",
        hasEffectiveAssistant: true,
        draftAssistantKind: "character"
      })
    ).toBe("tracked_character")
  })

  it("preserves tracked-persona startup for a plain chat with a persona draft selection", () => {
    expect(
      resolveUseMessageSendMode({
        effectiveMode: "plain",
        hasEffectiveAssistant: true,
        draftAssistantKind: "persona"
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
})
