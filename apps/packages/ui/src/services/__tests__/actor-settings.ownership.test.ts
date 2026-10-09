import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createDefaultActorSettings } from "@/types/actor"
import { installSettingsLocks } from "./chat-settings-fixture"

const io = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  localPersistent: true,
  legacyPersistent: true,
  get: vi.fn(),
  set: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: (options?: { area?: string }) => ({
    get hasPersistentBackend() {
      return options?.area === "local"
        ? io.localPersistent
        : io.legacyPersistent
    },
    get: io.get,
    set: io.set
  })
}))
import {
  getActorSettingsForChat,
  getActorSettingsForChatWithCharacterFallback,
  getActorSceneForChat,
  saveActorSettingsForChat,
  getActorProfileForCharacter,
  saveActorProfileForCharacter
} from "../actor-settings"

const accountA = "a".repeat(64),
  accountB = "b".repeat(64)
const sceneA = {
  ...createDefaultActorSettings(),
  notes: "Original account and conversation scene."
}
const sceneB = {
  ...createDefaultActorSettings(),
  notes: "Independent account and conversation scene."
}
// Owner/lease qualification is a declared fixture boundary; these tests exercise actual storage routing/CAS.
const target = (parts: unknown[], accountKey: string | null = accountA) => ({
  key: JSON.stringify(parts),
  accountKey,
  isCurrent: () => true,
  profileIsCurrent: () => accountKey !== null
})
const native = target(["native", accountA, "global", null, "collision"])
const source = { historyId: null, serverChatId: "collision", target: native }
const sceneKey = (value: typeof native) => `actorSettings:owned:${value.key}`
const profile = (accountKey: string) => ({
  key: JSON.stringify([accountKey, "card-one"]),
  isCurrent: () => true
})
const profileA = profile(accountA),
  profileB = profile(accountB)

const isolated = [
  {
    label: "fresh address",
    historyId: null,
    serverChatId: null,
    a: target(["fresh", accountA, "global", null, "fresh-a"]),
    b: target(["fresh", accountA, "global", null, "fresh-b"])
  },
  {
    label: "native account",
    historyId: null,
    serverChatId: "collision",
    a: native,
    b: target(["native", accountB, "global", null, "collision"], accountB)
  },
  {
    label: "native workspace",
    historyId: null,
    serverChatId: "collision",
    a: target(["native", accountA, "workspace", "workspace-a", "collision"]),
    b: target(["native", accountA, "workspace", "workspace-b", "collision"])
  },
  {
    label: "genuine local profile",
    historyId: "collision",
    serverChatId: null,
    a: target(["local", "profile-a", "collision"], null),
    b: target(["local", "profile-b", "collision"], null)
  }
]

describe("qualified scene and character-default records", () => {
  beforeEach(() => {
    installSettingsLocks()
    io.localPersistent = true
    io.legacyPersistent = true
    io.values.clear()
    io.get
      .mockReset()
      .mockImplementation(async (key: string) => io.values.get(key))
    io.set
      .mockReset()
      .mockImplementation(async (key: string, value: unknown) => {
        io.values.set(key, value)
      })
    vi.spyOn(console, "error").mockImplementation(() => undefined)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it.each(isolated)(
    "isolates $label across scene Save and read",
    async ({ historyId, serverChatId, a, b }) => {
      const first = { historyId, serverChatId, target: a },
        second = { historyId, serverChatId, target: b }
      expect(
        await saveActorSettingsForChat({
          ...first,
          settings: sceneA,
          expectedRecord: null
        })
      ).toBe(true)
      expect(
        await saveActorSettingsForChat({
          ...second,
          settings: sceneB,
          expectedRecord: null
        })
      ).toBe(true)
      expect(await getActorSettingsForChat(first)).toEqual(sceneA)
      expect(await getActorSettingsForChat(second)).toEqual(sceneB)
    }
  )

  it("retains an ambiguous bare conversation scene without adopting or rewriting it", async () => {
    const legacyKey = "actorSettings:server:collision"
    io.values.set(legacyKey, sceneA)
    expect(await getActorSettingsForChat(source)).toEqual(
      createDefaultActorSettings()
    )
    expect(io.values.get(legacyKey)).toEqual(sceneA)
    expect(io.get).not.toHaveBeenCalledWith(legacyKey)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("isolates the same character card's defaults by captured account", async () => {
    expect(
      await saveActorProfileForCharacter({
        characterId: "card-one",
        target: profileA,
        settings: sceneA
      })
    ).toBe(true)
    expect(
      await saveActorProfileForCharacter({
        characterId: "card-one",
        target: profileB,
        settings: sceneB
      })
    ).toBe(true)
    const first = { throwOnError: true, target: profileA },
      second = { throwOnError: true, target: profileB }
    expect(await getActorProfileForCharacter("card-one", first)).toEqual(sceneA)
    expect(await getActorProfileForCharacter("card-one", second)).toEqual(
      sceneB
    )
  })

  it("retains an ambiguous bare character default without calling it this account's default", async () => {
    const legacyKey = "actorProfile:card-one"
    io.values.set(legacyKey, sceneA)
    const options = { throwOnError: true, target: profileB }
    expect(await getActorProfileForCharacter("card-one", options)).toBeNull()
    expect(io.values.get(legacyKey)).toEqual(sceneA)
    expect(io.get).not.toHaveBeenCalledWith(legacyKey)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("refuses a scene Save without an original record or certified absence", async () => {
    expect(
      await saveActorSettingsForChat({ ...source, settings: sceneA })
    ).toBe(false)
    expect(io.get).not.toHaveBeenCalled()
    expect(io.set).not.toHaveBeenCalled()
  })

  it("preserves a competing scene instead of refreshing the original absence at Save time", async () => {
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneB,
        expectedRecord: null
      })
    ).toBe(true)
    const before = structuredClone(io.values.get(sceneKey(native)))
    io.set.mockClear()
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneA,
        expectedRecord: null
      })
    ).toBe(false)
    expect(io.values.get(sceneKey(native))).toEqual(before)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("preserves a newer scene when the original hydrated record has changed", async () => {
    const original = { value: sceneA, updatedAt: 1 }
    const newer = { value: sceneB, updatedAt: 2 }
    io.values.set(sceneKey(native), newer)
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneA,
        expectedRecord: original
      })
    ).toBe(false)
    expect(io.values.get(sceneKey(native))).toEqual(newer)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("writes a replacement timestamp above the original even when the clock moves backward", async () => {
    const original = { value: sceneA, updatedAt: 100 }
    io.values.set(sceneKey(native), original)
    vi.spyOn(Date, "now").mockReturnValue(1)
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneB,
        expectedRecord: original
      })
    ).toBe(true)
    expect(io.values.get(sceneKey(native))).toEqual({
      value: sceneB,
      updatedAt: 101
    })
  })

  it.each(["read", "save"])(
    "refuses a retired scene target before %s IO",
    async (kind) => {
      const retired = {
        ...source,
        target: { ...native, isCurrent: () => false }
      }
      if (kind === "read")
        await expect(getActorSettingsForChat(retired)).rejects.toThrow(
          "actor_settings_unavailable"
        )
      else
        expect(
          await saveActorSettingsForChat({
            ...retired,
            settings: sceneA,
            expectedRecord: null
          })
        ).toBe(false)
      expect(io.get).not.toHaveBeenCalled()
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  it("refuses a held scene read after its original qualified target retires", async () => {
    let current = true,
      release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.get.mockImplementationOnce(async () => {
      await held
      return undefined
    })
    const pending = getActorSettingsForChat({
      ...source,
      target: { ...native, isCurrent: () => current }
    })
    await vi.waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    current = false
    release()
    await expect(pending).rejects.toThrow("actor_settings_unavailable")
    expect(io.set).not.toHaveBeenCalled()
  })

  it("refuses conditional scene Save without cross-context locks", async () => {
    vi.stubGlobal(
      "navigator",
      Object.create(window.navigator, { locks: { value: undefined } })
    )
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneA,
        expectedRecord: null
      })
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it.each([false, { ...sceneA, version: 0 }])(
    "preserves an invalid qualified scene without certifying neutral defaults (%j)",
    async (value) => {
      const stored = { value, updatedAt: 1 }
      io.values.set(sceneKey(native), stored)
      await expect(getActorSettingsForChat(source)).rejects.toThrow(
        "actor_settings_unavailable"
      )
      expect(io.values.get(sceneKey(native))).toEqual(stored)
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  it("retains raw scene metadata separately from the normalized editor proposal", async () => {
    const raw = {
      ...sceneA,
      appendable: undefined,
      aspects: [{ ...sceneA.aspects[0], key: " User Role " }]
    }
    const original = { value: raw, updatedAt: 1 }
    io.values.set(sceneKey(native), original)
    const read = await getActorSceneForChat({ target: native })
    expect(read.originalRecord).toEqual(original)
    expect(read.settings.appendable).toBe(false)
    expect(read.settings.aspects[0].key).not.toBe(raw.aspects[0].key)
    read.settings.notes = "Only staged here."
    expect(io.values.get(sceneKey(native))).toEqual(original)
  })
  it("keeps original scene absence as Save expectation when inheriting a profile", async () => {
    io.values.set(`actorProfile:owned:${profileA.key}`, {
      value: sceneA,
      updatedAt: 1
    })
    const read = await getActorSceneForChat({
      target: native,
      characterId: "card-one"
    })
    expect(read.origin).toBe("profile")
    expect(read.originalRecord).toBeNull()
    expect(read.settings).toEqual(sceneA)
    expect(
      await saveActorSettingsForChat({
        ...source,
        settings: sceneB,
        expectedRecord: read.originalRecord
      })
    ).toBe(true)
    expect(io.values.get(`actorProfile:owned:${profileA.key}`)).toEqual({
      value: sceneA,
      updatedAt: 1
    })
  })
  it("keeps certified genuine-local absence editable with unverified optional defaults", async () => {
    const local = target(["local", "profile-a", "collision"], null)
    const read = await getActorSceneForChat({
      target: local,
      characterId: "card-one"
    })
    expect(read.originalRecord).toBeNull()
    expect(read.profileUnavailable).toBe(true)
    expect(read.origin).toBe("default")
    expect(
      await saveActorSettingsForChat({
        historyId: "collision",
        serverChatId: null,
        target: local,
        settings: sceneA,
        expectedRecord: read.originalRecord
      })
    ).toBe(true)
    expect(io.get).not.toHaveBeenCalledWith(
      `actorProfile:owned:${profileA.key}`
    )
  })
  it("reports unavailable optional defaults separately from certified absence", async () => {
    io.get.mockImplementation(async (key: string) => {
      if (key.startsWith("actorProfile:")) throw new Error("unavailable")
      return io.values.get(key)
    })
    const read = await getActorSceneForChat({
      target: native,
      characterId: "card-one"
    })
    expect(read.originalRecord).toBeNull()
    expect(read.profileUnavailable).toBe(true)
    expect(io.set).not.toHaveBeenCalled()
  })
  it("refuses a held read after editor retirement while the destination stays valid", async () => {
    let operationCurrent = true,
      release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.get.mockImplementationOnce(async () => {
      await held
      return undefined
    })
    const pending = getActorSceneForChat({
      target: native,
      validate_lease: () => operationCurrent
    })
    await vi.waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    operationCurrent = false
    release()
    await expect(pending).rejects.toThrow("actor_settings_unavailable")
    expect(native.isCurrent()).toBe(true)
  })
  it("freezes the scene proposal before a held conditional write", async () => {
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.get.mockImplementationOnce(async () => {
      await held
      return undefined
    })
    const proposal = structuredClone(sceneA)
    const pending = saveActorSettingsForChat({
      ...source,
      settings: proposal,
      expectedRecord: null
    })
    await vi.waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    proposal.notes = "Changed while pending."
    release()
    expect(await pending).toBe(true)
    expect(io.values.get(sceneKey(native))).toMatchObject({
      value: { notes: sceneA.notes }
    })
  })
  it("refuses success after a held write loses its original destination", async () => {
    let current = true,
      release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    io.set.mockImplementationOnce(async (key: string, value: unknown) => {
      io.values.set(key, value)
      await held
    })
    const pending = saveActorSettingsForChat({
      ...source,
      target: { ...native, isCurrent: () => current },
      settings: sceneA,
      expectedRecord: null
    })
    await vi.waitFor(() => expect(io.set).toHaveBeenCalledOnce())
    current = false
    release()
    expect(await pending).toBe(false)
    expect(io.values.get(sceneKey(native))).toMatchObject({
      value: { notes: sceneA.notes }
    })
  })
  it("retains an invalid qualified character default instead of replacing it on explicit Save", async () => {
    const raw = { value: { ...sceneA, version: 0 }, updatedAt: 1 }
    io.values.set(`actorProfile:owned:${profileA.key}`, raw)
    expect(
      await saveActorProfileForCharacter({
        characterId: "card-one",
        target: profileA,
        settings: sceneB
      })
    ).toBe(false)
    expect(io.values.get(`actorProfile:owned:${profileA.key}`)).toEqual(raw)
    expect(io.set).not.toHaveBeenCalled()
  })

  it.each(["scene", "profile"])(
    "keeps qualified %s Save independent of the unrelated legacy storage area",
    async (kind) => {
      io.legacyPersistent = false
      const saved =
        kind === "scene"
          ? await saveActorSettingsForChat({
              ...source,
              settings: sceneA,
              expectedRecord: null
            })
          : await saveActorProfileForCharacter({
              characterId: "card-one",
              target: profileA,
              settings: sceneA
            })
      expect(saved).toBe(true)
    }
  )

  it.each([
    { label: "false", original: false },
    { label: "zero", original: 0 },
    { label: "empty string", original: "" },
    { label: "empty array", original: [] },
    {
      label: "array record",
      original: Object.assign([], { value: sceneA, updatedAt: 1 })
    },
    {
      label: "invalid date timestamp",
      original: { value: sceneA, updatedAt: Number.MAX_VALUE }
    }
  ])(
    "refuses present $label as a scene CAS expectation before IO",
    async ({ original }) => {
      if (original && typeof original === "object")
        io.values.set(sceneKey(native), original)
      const before = io.values.get(sceneKey(native))
      // Runtime/storage boundary inputs must fail closed despite the static record type.
      const params = {
        ...source,
        settings: sceneB,
        expectedRecord: original as never
      }
      expect(await saveActorSettingsForChat(params)).toBe(false)
      expect(io.values.get(sceneKey(native))).toEqual(before)
      expect(io.get).not.toHaveBeenCalled()
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  it.each(["unavailable", "malformed", "retired"])(
    "fails a qualified profile read closed without an extra strict flag (%s)",
    async (kind) => {
      const options = {
        target:
          kind === "retired"
            ? { ...profileA, isCurrent: () => false }
            : profileA
      }
      if (kind === "unavailable")
        io.get.mockRejectedValueOnce(new Error("unavailable"))
      if (kind === "malformed")
        io.values.set(`actorProfile:owned:${profileA.key}`, {
          value: false,
          updatedAt: 1
        })
      await expect(
        getActorProfileForCharacter("card-one", options)
      ).rejects.toThrow("actor_settings_unavailable")
      expect(io.set).not.toHaveBeenCalled()
    }
  )

  it("refuses an array wrapper masquerading as an owned scene record", async () => {
    const stored = Object.assign([], { value: sceneA, updatedAt: 1 })
    io.values.set(sceneKey(native), stored)
    await expect(getActorSettingsForChat(source)).rejects.toThrow(
      "actor_settings_unavailable"
    )
    expect(io.values.get(sceneKey(native))).toEqual(stored)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("keeps the settings-only fallback wrapper from hiding unavailable inherited defaults", async () => {
    io.get.mockImplementation(async (key: string) => {
      if (key.startsWith("actorProfile:")) throw new Error("unavailable")
      return io.values.get(key)
    })
    await expect(
      getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "card-one"
      })
    ).rejects.toThrow("actor_settings_unavailable")
    expect(io.set).not.toHaveBeenCalled()
  })
  it.each(["scene", "profile"])(
    "preserves a %s record when its monotonic replacement would exceed the valid Date range",
    async (kind) => {
      const original = { value: sceneA, updatedAt: 8_640_000_000_000_000 }
      const key =
        kind === "scene"
          ? sceneKey(native)
          : `actorProfile:owned:${profileA.key}`
      io.values.set(key, original)
      const saved =
        kind === "scene"
          ? await saveActorSettingsForChat({
              ...source,
              settings: sceneB,
              expectedRecord: original
            })
          : await saveActorProfileForCharacter({
              characterId: "card-one",
              target: profileA,
              settings: sceneB
            })
      expect(saved).toBe(false)
      expect(io.values.get(key)).toEqual(original)
      expect(io.set).not.toHaveBeenCalled()
    }
  )
})
