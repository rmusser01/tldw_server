import { describe, expect, it, vi } from "vitest"

import { hydrateTrackedCharacterForSend } from "../tracked-character-hydration"

describe("hydrateTrackedCharacterForSend", () => {
  it("replaces a placeholder character name with server metadata", async () => {
    const loadCharacter = vi.fn(async () => ({
      id: 27,
      name: "Captain Mira",
      system_prompt: "Stay in character."
    }))

    const result = await hydrateTrackedCharacterForSend(
      { id: "27", name: "Assistant", localOnly: true },
      loadCharacter
    )

    expect(loadCharacter).toHaveBeenCalledWith("27")
    expect(result).toEqual({
      id: "27",
      name: "Captain Mira",
      system_prompt: "Stay in character.",
      localOnly: true
    })
  })

  it("keeps an authoritative local character without another request", async () => {
    const loadCharacter = vi.fn()
    const character = { id: "27", name: "Captain Mira" }

    await expect(
      hydrateTrackedCharacterForSend(character, loadCharacter)
    ).resolves.toBe(character)
    expect(loadCharacter).not.toHaveBeenCalled()
  })

  it("falls back to the candidate when hydration fails", async () => {
    const candidate = { id: "27", name: "Assistant" }
    const loadCharacter = vi.fn(async () => {
      throw new Error("catalog unavailable")
    })

    await expect(
      hydrateTrackedCharacterForSend(candidate, loadCharacter)
    ).resolves.toBe(candidate)
  })
})
