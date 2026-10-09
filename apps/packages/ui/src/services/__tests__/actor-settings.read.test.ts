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
  saveActorSettingsForChat,
  saveActorProfileForCharacter
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
  it("acknowledges a durable character default only after the exact write", async () => {
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        settings: inherited
      })
    ).toBe(true)
    expect(io.set).toHaveBeenCalledExactlyOnceWith(
      "actorProfile:character-one",
      inherited
    )
  })

  it("refuses a character default save to a memory-only backend", async () => {
    io.persistent = false
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        settings: inherited
      })
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("returns false for a rejected character default write", async () => {
    io.set.mockRejectedValueOnce(new Error("write unavailable"))
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        settings: inherited
      })
    ).toBe(false)
  })

  it("refuses an invalid default before writing persistent storage", async () => {
    const invalid = {
      ...inherited,
      chatRole: "invalid"
    } as unknown as typeof inherited
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        settings: invalid
      })
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("refuses a retired default source before any write", async () => {
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        settings: inherited,
        validate_lease: () => false
      })
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("does not acknowledge an issued default write after its source retires", async () => {
    let current = true
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.set.mockReturnValueOnce(held)
    const pending = saveActorProfileForCharacter({
      characterId: "character-one",
      settings: inherited,
      validate_lease: () => current
    })
    await vi.waitFor(() => expect(io.set).toHaveBeenCalledOnce())
    current = false
    release()
    expect(await pending).toBe(false)
    expect(io.set).toHaveBeenCalledOnce()
  })

  it("strictly refuses a profile read whose initiating source is already retired", async () => {
    await expect(
      getActorProfileForCharacter("character-one", {
        throwOnError: true,
        validate_lease: () => false
      })
    ).rejects.toThrow("actor_settings_unavailable")
    expect(io.get).not.toHaveBeenCalled()
  })

  it("strictly refuses a held profile read after its initiating source retires", async () => {
    let current = true
    let release!: (value: typeof inherited) => void
    const held = new Promise<typeof inherited>((resolve) => {
      release = resolve
    })
    io.get.mockReturnValueOnce(held)
    const pending = getActorProfileForCharacter("character-one", {
      throwOnError: true,
      validate_lease: () => current
    })
    await vi.waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    current = false
    release(inherited)
    await expect(pending).rejects.toThrow("actor_settings_unavailable")
  })

  it.each([
    { ...inherited, chatRole: "invalid" },
    { ...inherited, version: 0 }
  ])(
    "strictly refuses an invalid saved profile without replacing or deleting it (%j)",
    async (invalid) => {
      io.values.set("actorProfile:character-one", invalid)
      await expect(
        getActorProfileForCharacter("character-one", { throwOnError: true })
      ).rejects.toThrow("actor_settings_unavailable")
      expect(io.values.get("actorProfile:character-one")).toEqual(invalid)
    }
  )

  it("does not acknowledge a profile write after its persistent backend disappears", async () => {
    io.set.mockImplementationOnce(async () => {
      io.persistent = false
    })
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        settings: inherited
      })
    ).toBe(false)
    expect(io.set).toHaveBeenCalledOnce()
  })

  it("strictly refuses a profile read after its persistent backend disappears", async () => {
    io.get.mockImplementationOnce(async () => {
      io.persistent = false
      return inherited
    })
    await expect(
      getActorProfileForCharacter("character-one", { throwOnError: true })
    ).rejects.toThrow("actor_settings_unavailable")
  })

  it.each([false, 0, "", null])(
    "strictly rejects a present malformed falsy profile without deleting it (%j)",
    async (invalid) => {
      io.values.set("actorProfile:character-one", invalid)
      await expect(
        getActorProfileForCharacter("character-one", { throwOnError: true })
      ).rejects.toThrow("actor_settings_unavailable")
      expect(io.values.has("actorProfile:character-one")).toBe(true)
      expect(io.values.get("actorProfile:character-one")).toEqual(invalid)
      expect(io.set).not.toHaveBeenCalled()
    }
  )
})
