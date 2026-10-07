import { beforeEach, describe, expect, it, vi } from "vitest"
import { createDefaultActorSettings } from "@/types/actor"

const io = vi.hoisted(() => ({
  persistent: true,
  values: new Map<string, unknown>(),
  failures: new Set<string>(),
  get: vi.fn(),
  set: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get hasPersistentBackend() {
      return io.persistent
    },
    get: io.get,
    set: io.set
  })
}))
import {
  getActorSettingsForChat,
  getActorSettingsForChatWithCharacterFallback,
  getActorProfileForCharacter,
  saveActorSettingsForChat
} from "../actor-settings"

const source = { historyId: "local-one", serverChatId: "server-one" }
const inherited = {
  ...createDefaultActorSettings(),
  notes: "Keep the inherited scene."
}

describe("scene read availability", () => {
  beforeEach(() => {
    io.persistent = true
    io.values.clear()
    io.failures.clear()
    io.set.mockReset().mockResolvedValue(undefined)
    io.get.mockReset().mockImplementation(async (key: string) => {
      if (io.failures.has(key)) throw new Error("storage read failed")
      return io.values.get(key)
    })
    vi.spyOn(console, "error").mockImplementation(() => undefined)
  })

  it("rejects an unread chat scene instead of certifying a neutral scene", async () => {
    io.failures.add("actorSettings:server:server-one")
    await expect(getActorSettingsForChat(source)).rejects.toThrow(
      "actor_settings_unavailable"
    )
  })

  it("does not replace an unread chat scene with a readable character profile", async () => {
    io.failures.add("actorSettings:server:server-one")
    io.values.set("actorProfile:character-one", inherited)
    await expect(
      getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "character-one"
      })
    ).rejects.toThrow("actor_settings_unavailable")
    expect(io.get).not.toHaveBeenCalledWith("actorProfile:character-one")
  })

  it("rejects an unread inherited profile after known chat absence", async () => {
    io.failures.add("actorProfile:character-one")
    await expect(
      getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "character-one"
      })
    ).rejects.toThrow("actor_settings_unavailable")
  })

  it("does not certify absence from a memory-only backend", async () => {
    io.persistent = false
    await expect(
      getActorSettingsForChatWithCharacterFallback(source)
    ).rejects.toThrow("actor_settings_unavailable")
    expect(io.get).not.toHaveBeenCalled()
  })

  it("keeps known absence usable with existing neutral defaults", async () => {
    expect(await getActorSettingsForChatWithCharacterFallback(source)).toEqual(
      createDefaultActorSettings()
    )
  })

  it("inherits a readable character profile only after known chat absence", async () => {
    io.values.set("actorProfile:character-one", inherited)
    expect(
      await getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "character-one"
      })
    ).toEqual(inherited)
  })

  it("keeps a stored chat scene ahead of its character profile", async () => {
    const scene = { ...inherited, notes: "This conversation's scene." }
    io.values.set("actorSettings:server:server-one", scene)
    io.values.set("actorProfile:character-one", inherited)
    expect(
      await getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "character-one"
      })
    ).toEqual(scene)
    expect(io.get).not.toHaveBeenCalledWith("actorProfile:character-one")
  })

  it("does not acknowledge a scene save to memory-only storage", async () => {
    io.persistent = false
    expect(
      await saveActorSettingsForChat({ ...source, settings: inherited })
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("acknowledges the exact durable scene payload", async () => {
    expect(
      await saveActorSettingsForChat({ ...source, settings: inherited })
    ).toBe(true)
    expect(io.set).toHaveBeenCalledWith(
      "actorSettings:server:server-one",
      inherited
    )
  })

  it("retains a rejected scene save as a failed acknowledgement", async () => {
    io.set.mockRejectedValueOnce(new Error("write unavailable"))
    expect(
      await saveActorSettingsForChat({ ...source, settings: inherited })
    ).toBe(false)
  })

  it("retains the nullable advisory profile contract", async () => {
    io.failures.add("actorProfile:character-one")
    expect(await getActorProfileForCharacter("character-one")).toBeNull()
  })
})
