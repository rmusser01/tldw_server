import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getPersonaProfile: vi.fn(async () => null),
  getCharacter: vi.fn(async () => null)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getPersonaProfile: mocks.getPersonaProfile,
    getCharacter: mocks.getCharacter
  }
}))

import { resolveAssistantOverlaySnapshot } from "../assistant-overlay"

describe("resolveAssistantOverlaySnapshot", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses full persona profile detail instead of catalog summary data", async () => {
    mocks.getPersonaProfile.mockResolvedValue({
      id: "persona-1",
      name: "Research Guide",
      avatar_url: "https://example.com/persona-full.png",
      system_prompt: "Persona full prompt"
    })

    await expect(
      resolveAssistantOverlaySnapshot({
        kind: "persona",
        id: "persona-1",
        name: "Summary Name",
        avatar_url: "https://example.com/persona-summary.png",
        system_prompt: "Persona summary prompt"
      })
    ).resolves.toEqual(
      expect.objectContaining({
        kind: "persona",
        id: "persona-1",
        name: "Research Guide",
        avatar_url: "https://example.com/persona-full.png",
        system_prompt_snapshot: "Persona full prompt"
      })
    )
  })

  it("refreshes character detail when summary data is missing prompt material", async () => {
    mocks.getCharacter.mockResolvedValue({
      id: "char-2",
      name: "Beta",
      avatar_url: "https://example.com/beta-full.png",
      system_prompt: "Character fetched prompt"
    })

    await expect(
      resolveAssistantOverlaySnapshot({
        kind: "character",
        id: "char-2",
        name: "Beta"
      })
    ).resolves.toEqual(
      expect.objectContaining({
        kind: "character",
        id: "char-2",
        name: "Beta",
        avatar_url: "https://example.com/beta-full.png",
        system_prompt_snapshot: "Character fetched prompt"
      })
    )

    expect(mocks.getCharacter).toHaveBeenCalledWith("char-2", {
      forceRefresh: true
    })
  })

  it("uses full character detail instead of summary payload data at apply time", async () => {
    mocks.getCharacter.mockResolvedValue({
      id: "char-1",
      name: "Alpha Full",
      avatar_url: "https://example.com/alpha-full.png",
      system_prompt: "Alpha full prompt"
    })

    await expect(
      resolveAssistantOverlaySnapshot({
        kind: "character",
        id: "char-1",
        name: "Alpha Summary",
        avatar_url: "https://example.com/alpha-summary.png",
        system_prompt: "Alpha summary prompt"
      })
    ).resolves.toEqual(
      expect.objectContaining({
        kind: "character",
        id: "char-1",
        name: "Alpha Full",
        avatar_url: "https://example.com/alpha-full.png",
        system_prompt_snapshot: "Alpha full prompt"
      })
    )

    expect(mocks.getCharacter).toHaveBeenCalledWith("char-1", {
      forceRefresh: true
    })
  })

  it("stores the normalized snapshot payload shape", async () => {
    mocks.getCharacter.mockResolvedValue({
      id: "char-1",
      name: "Alpha Full",
      avatar_url: "https://example.com/alpha.png",
      system_prompt: "Alpha full prompt"
    })

    const result = await resolveAssistantOverlaySnapshot({
      kind: "character",
      id: "char-1",
      name: "Alpha Summary"
    })

    expect(result).toEqual(
      expect.objectContaining({
        kind: "character",
        id: "char-1",
        name: "Alpha Full",
        avatar_url: "https://example.com/alpha.png",
        system_prompt_snapshot: "Alpha full prompt"
      })
    )
    expect(typeof result.updatedAt).toBe("string")
  })

  it("preserves prior snapshot content even if source records later change", async () => {
    mocks.getPersonaProfile.mockResolvedValueOnce({
      id: "persona-9",
      name: "Planner",
      avatar_url: "https://example.com/planner-v1.png",
      system_prompt: "Prompt v1"
    })
    mocks.getPersonaProfile.mockResolvedValueOnce({
      id: "persona-9",
      name: "Planner Updated",
      avatar_url: "https://example.com/planner-v2.png",
      system_prompt: "Prompt v2"
    })

    const first = await resolveAssistantOverlaySnapshot({
      kind: "persona",
      id: "persona-9",
      name: "Planner"
    })
    const second = await resolveAssistantOverlaySnapshot({
      kind: "persona",
      id: "persona-9",
      name: "Planner"
    })

    expect(first.system_prompt_snapshot).toBe("Prompt v1")
    expect(second.system_prompt_snapshot).toBe("Prompt v2")
    expect(first.system_prompt_snapshot).not.toBe(second.system_prompt_snapshot)
  })

  it("falls back to persona summary data when full detail lookup fails", async () => {
    mocks.getPersonaProfile.mockRejectedValueOnce(new Error("persona detail unavailable"))

    await expect(
      resolveAssistantOverlaySnapshot({
        kind: "persona",
        id: "persona-2",
        name: "Summary Persona",
        avatar_url: "https://example.com/persona-summary.png",
        system_prompt: "Persona summary prompt"
      })
    ).resolves.toEqual(
      expect.objectContaining({
        kind: "persona",
        id: "persona-2",
        name: "Summary Persona",
        avatar_url: "https://example.com/persona-summary.png",
        system_prompt_snapshot: "Persona summary prompt"
      })
    )
  })
})
