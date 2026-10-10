import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  ACTOR_SETTINGS_VERSION,
  createDefaultActorSettings,
  type ActorSettings
} from "@/types/actor"
import { installSettingsLocks } from "./chat-settings-fixture"
import {
  captureActorProfileTarget,
  captureActorSettingsTarget
} from "@/utils/actor-settings-target"

const io = vi.hoisted(() => ({
  persistent: true,
  values: new Map<string, unknown>(),
  failures: new Set<string>(),
  get: vi.fn<(key: string) => Promise<unknown>>(),
  set: vi.fn<(key: string, value: unknown) => Promise<void>>()
}))
// Only persistent IO is doubled. The service, qualified target helpers, bucket
// validation/CAS and serialized Web Locks are the actual production path.
vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    get hasPersistentBackend() {
      return io.persistent
    },
    get: io.get,
    set: io.set
  })
}))
import {
  getActorSceneForChat,
  getActorSettingsForChat,
  getActorSettingsForChatWithCharacterFallback,
  getActorProfileForCharacter,
  saveActorSettingsForChat,
  saveActorProfileForCharacter
} from "../actor-settings"

const requestScope = {
  config: {
    serverUrl: "https://actor-read-owner.example",
    authMode: "single-user" as const
  },
  userId: 1
}
// This unit declares the account/owner/view boundary rather than simulating an
// H1 load. The actual qualifier retains this exact original native address.
const sceneTarget = captureActorSettingsTarget({
  address: {
    kind: "native",
    id: "server-one",
    ownerKey: "read-owner",
    scope: { type: "global" }
  },
  owner: {
    kind: "native",
    conversation_id: "server-one",
    owner_key: "read-owner",
    request_scope: requestScope,
    scope: { type: "global" },
    validate_lease: () => true
  },
  view: {
    view_session_id: "read-view",
    owner_key: "read-owner",
    conversation_id: "server-one",
    interpretation: { kind: "parent_graph_v1" },
    cursor: { kind: "empty" },
    selection_revision: 1
  },
  requestScope,
  accountIsCurrent: () => true,
  isCurrent: () => true
})!
const profileTarget = captureActorProfileTarget(
  sceneTarget.accountKey,
  "character-one",
  sceneTarget.profileIsCurrent
)!
const source = {
  historyId: "local-one",
  serverChatId: "server-one",
  target: sceneTarget
}
const sceneKey = `actorSettings:owned:${sceneTarget.key}`
const profileKey = `actorProfile:owned:${profileTarget.key}`
const record = (value: unknown) => ({
  value: structuredClone(value),
  updatedAt: 41
})
const inherited = {
  ...createDefaultActorSettings(),
  notes: "Keep the inherited scene."
}

const invalidSceneRecords: Array<{ label: string; raw: unknown }> = [
  { label: "false", raw: false },
  { label: "zero", raw: 0 },
  { label: "empty string", raw: "" },
  { label: "null", raw: null },
  { label: "empty object", raw: {} },
  { label: "unsupported role", raw: { ...inherited, chatRole: "unsupported" } },
  {
    label: "null aspect value",
    raw: { ...inherited, aspects: [{ ...inherited.aspects[0], value: null }] }
  },
  { label: "legacy version", raw: { ...inherited, version: 0 } },
  {
    label: "future version",
    raw: { ...inherited, version: ACTOR_SETTINGS_VERSION + 1 }
  },
  {
    label: "array masquerading as a record",
    raw: Object.assign([], inherited)
  },
  {
    label: "sparse aspects",
    raw: {
      ...inherited,
      aspects: new Array<(typeof inherited.aspects)[number]>(1)
    }
  },
  {
    label: "invalid lorebook reference",
    raw: {
      ...inherited,
      aspects: [{ ...inherited.aspects[0], lorebookId: 42 }]
    }
  },
  {
    label: "invalid entry reference",
    raw: { ...inherited, aspects: [{ ...inherited.aspects[0], entryId: null }] }
  }
]

describe("scene read availability", () => {
  beforeEach(() => {
    installSettingsLocks()
    io.persistent = true
    io.values.clear()
    io.failures.clear()
    io.set.mockReset().mockImplementation(async (key: string, value: unknown) => {
      io.values.set(key, structuredClone(value))
    })
    io.get.mockReset().mockImplementation(async (key: string) => {
      if (io.failures.has(key)) throw new Error("storage read failed")
      return io.values.get(key)
    })
    vi.spyOn(console, "error").mockImplementation(() => undefined)
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it.each(
    invalidSceneRecords.flatMap((record) =>
      [false, true].map((fallback) => ({ ...record, fallback }))
    )
  )(
    "retains $label and refuses scene profile fallback=$fallback",
    async ({ raw, fallback }) => {
      const key = sceneKey
      io.values.set(key, record(raw))
      io.values.set(profileKey, record(inherited))
      await expect(
        fallback
          ? getActorSettingsForChatWithCharacterFallback({
              ...source,
              characterId: "character-one"
            })
          : getActorSettingsForChat(source)
      ).rejects.toThrow("actor_settings_unavailable")
      expect(io.values.has(key)).toBe(true)
      expect(io.values.get(key)).toEqual(record(raw))
      expect(io.set).not.toHaveBeenCalled()
      expect(io.get).not.toHaveBeenCalledWith(profileKey)
    }
  )

  it.each(
    [false, true].flatMap((fallback) =>
      [undefined, inherited].map((raw) => ({ fallback, raw }))
    )
  )(
    "does not certify a scene read after durable backend loss, fallback=$fallback, raw=$raw",
    async ({ fallback, raw }) => {
      io.get.mockImplementationOnce(async () => {
        io.persistent = false
        return raw === undefined ? undefined : record(raw)
      })
      await expect(
        fallback
          ? getActorSettingsForChatWithCharacterFallback({
              ...source,
              characterId: "character-one"
            })
          : getActorSettingsForChat(source)
      ).rejects.toThrow("actor_settings_unavailable")
      expect(io.get).toHaveBeenCalledOnce()
    }
  )

  it.each(
    invalidSceneRecords.flatMap((record) =>
      ["scene", "profile"].map((kind) => ({ ...record, kind }))
    )
  )("rejects $label before $kind Save storage IO", async ({ raw, kind }) => {
    const payload = raw as ActorSettings
    expect(
      await (kind === "scene"
        ? saveActorSettingsForChat({
            ...source,
            expectedRecord: null,
            settings: payload
          })
        : saveActorProfileForCharacter({
            characterId: "character-one",
            target: profileTarget,
            settings: payload
          }))
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it.each(["scene", "profile"])(
    "does not acknowledge a held %s write after durable backend loss",
    async (kind) => {
      let release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      io.set.mockImplementationOnce(async (key: string, value: unknown) => {
        io.values.set(key, structuredClone(value))
        await held
      })
      const pending =
        kind === "scene"
          ? saveActorSettingsForChat({
              ...source,
              expectedRecord: null,
              settings: inherited
            })
          : saveActorProfileForCharacter({
              characterId: "character-one",
              target: profileTarget,
              settings: inherited
            })
      await vi.waitFor(() => expect(io.set).toHaveBeenCalledOnce())
      const issuedKey = kind === "scene" ? sceneKey : profileKey
      expect(io.set.mock.calls[0][0]).toBe(issuedKey)
      expect(io.values.get(issuedKey)).toMatchObject({ value: inherited })
      io.persistent = false
      release()
      expect(await pending).toBe(false)
      expect(io.values.get(issuedKey)).toMatchObject({ value: inherited })
    }
  )

  it("keeps a supported minimal scene and normalizes only missing optional appendable", async () => {
    const minimal = {
      version: ACTOR_SETTINGS_VERSION,
      isEnabled: false,
      aspects: [],
      notes: "Supported minimal scene.",
      chatPosition: "before",
      chatDepth: 0,
      chatRole: "system"
    }
    io.values.set(sceneKey, record(minimal))
    const read = await getActorSceneForChat({ target: sceneTarget })
    expect(read.originalRecord).toEqual(record(minimal))
    expect(read.settings).toEqual({
      ...minimal,
      appendable: false
    })
    expect(io.values.get(sceneKey)).toEqual(record(minimal))
    expect(io.set).not.toHaveBeenCalled()
  })

  it("uses the original raw record for CAS after normalizing an editable minimal scene", async () => {
    const raw = {
      version: ACTOR_SETTINGS_VERSION,
      isEnabled: false,
      aspects: [],
      notes: "Original minimal scene.",
      chatPosition: "before",
      chatDepth: 0,
      chatRole: "system"
    }
    const original = record(raw)
    io.values.set(sceneKey, original)
    const read = await getActorSceneForChat({ target: sceneTarget })
    const proposal = { ...read.settings, notes: "Only this intended edit." }
    expect(read.originalRecord).toEqual(original)
    expect(read.settings.appendable).toBe(false)
    expect(io.values.get(sceneKey)).toEqual(original)
    io.get.mockClear()
    vi.spyOn(Date, "now").mockReturnValue(42)
    expect(
      await saveActorSettingsForChat({
        ...source,
        expectedRecord: read.originalRecord,
        settings: proposal
      })
    ).toBe(true)
    expect(io.get).toHaveBeenCalledExactlyOnceWith(sceneKey)
    expect(io.set).toHaveBeenCalledExactlyOnceWith(sceneKey, {
      value: proposal,
      updatedAt: 42
    })
    expect(io.values.get(sceneKey)).toEqual({ value: proposal, updatedAt: 42 })
    expect(original).toEqual(record(raw))
  })

  it("keeps supported aspect-key normalization without rewriting the stored scene", async () => {
    const stored = {
      ...inherited,
      aspects: [{ ...inherited.aspects[0], key: "  Role now  " }]
    }
    io.values.set(sceneKey, record(stored))
    const read = await getActorSceneForChat({ target: sceneTarget })
    expect(read.originalRecord).toEqual(record(stored))
    expect(read.settings.aspects[0].key).toBe("user_role_now")
    expect(io.values.get(sceneKey)).toEqual(record(stored))
    expect(io.set).not.toHaveBeenCalled()
  })

  it("rejects an unread chat scene instead of certifying a neutral scene", async () => {
    io.failures.add(sceneKey)
    await expect(getActorSettingsForChat(source)).rejects.toThrow(
      "actor_settings_unavailable"
    )
  })

  it("does not replace an unread chat scene with a readable character profile", async () => {
    io.failures.add(sceneKey)
    io.values.set(profileKey, record(inherited))
    await expect(
      getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "character-one"
      })
    ).rejects.toThrow("actor_settings_unavailable")
    expect(io.get).not.toHaveBeenCalledWith(profileKey)
  })

  it("rejects an unread inherited profile after known chat absence", async () => {
    io.failures.add(profileKey)
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
    io.values.set(profileKey, record(inherited))
    expect(
      await getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "character-one"
      })
    ).toEqual(inherited)
  })

  it("keeps a stored chat scene ahead of its character profile", async () => {
    const scene = { ...inherited, notes: "This conversation's scene." }
    io.values.set(sceneKey, record(scene))
    io.values.set(profileKey, record(inherited))
    expect(
      await getActorSettingsForChatWithCharacterFallback({
        ...source,
        characterId: "character-one"
      })
    ).toEqual(scene)
    expect(io.get).not.toHaveBeenCalledWith(profileKey)
  })

  it("does not acknowledge a scene save to memory-only storage", async () => {
    io.persistent = false
    expect(
      await saveActorSettingsForChat({
        ...source,
        expectedRecord: null,
        settings: inherited
      })
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("acknowledges the exact durable scene payload and original absence CAS", async () => {
    vi.spyOn(Date, "now").mockReturnValue(42)
    expect(
      await saveActorSettingsForChat({
        ...source,
        expectedRecord: null,
        settings: inherited
      })
    ).toBe(true)
    expect(io.get).toHaveBeenCalledExactlyOnceWith(sceneKey)
    expect(io.set).toHaveBeenCalledExactlyOnceWith(sceneKey, {
      value: inherited,
      updatedAt: 42
    })
    expect(io.values.get(sceneKey)).toEqual({ value: inherited, updatedAt: 42 })
  })

  it("retains a rejected scene save as a failed acknowledgement", async () => {
    io.set.mockRejectedValueOnce(new Error("write unavailable"))
    expect(
      await saveActorSettingsForChat({
        ...source,
        expectedRecord: null,
        settings: inherited
      })
    ).toBe(false)
  })

  it("fails an unread qualified advisory profile closed without requiring a strict flag", async () => {
    io.failures.add(profileKey)
    await expect(
      getActorProfileForCharacter("character-one", { target: profileTarget })
    ).rejects.toThrow("actor_settings_unavailable")
    expect(io.get).toHaveBeenCalledExactlyOnceWith(profileKey)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("retains null for a known absent qualified character default", async () => {
    expect(
      await getActorProfileForCharacter("character-one", {
        target: profileTarget
      })
    ).toBeNull()
    expect(io.get).toHaveBeenCalledExactlyOnceWith(profileKey)
    expect(io.set).not.toHaveBeenCalled()
  })
  it("acknowledges a durable character default only after the exact write", async () => {
    vi.spyOn(Date, "now").mockReturnValue(42)
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        target: profileTarget,
        settings: inherited
      })
    ).toBe(true)
    expect(io.set).toHaveBeenCalledExactlyOnceWith(profileKey, {
      value: inherited,
      updatedAt: 42
    })
    expect(io.values.get(profileKey)).toEqual({ value: inherited, updatedAt: 42 })
  })

  it("refuses a character default save to a memory-only backend", async () => {
    io.persistent = false
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        target: profileTarget,
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
        target: profileTarget,
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
        target: profileTarget,
        settings: invalid
      })
    ).toBe(false)
    expect(io.set).not.toHaveBeenCalled()
  })

  it("refuses a retired default source before any write", async () => {
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        target: profileTarget,
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
    io.set.mockImplementationOnce(async (key: string, value: unknown) => {
      io.values.set(key, structuredClone(value))
      await held
    })
    const pending = saveActorProfileForCharacter({
      characterId: "character-one",
      target: profileTarget,
      settings: inherited,
      validate_lease: () => current
    })
    await vi.waitFor(() => expect(io.set).toHaveBeenCalledOnce())
    expect(io.set.mock.calls[0][0]).toBe(profileKey)
    expect(io.values.get(profileKey)).toMatchObject({ value: inherited })
    current = false
    release()
    expect(await pending).toBe(false)
    expect(io.set).toHaveBeenCalledOnce()
    expect(io.values.get(profileKey)).toMatchObject({ value: inherited })
  })

  it("strictly refuses a profile read whose initiating source is already retired", async () => {
    await expect(
      getActorProfileForCharacter("character-one", {
        target: profileTarget,
        throwOnError: true,
        validate_lease: () => false
      })
    ).rejects.toThrow("actor_settings_unavailable")
    expect(io.get).not.toHaveBeenCalled()
  })

  it("strictly refuses a held profile read after its initiating source retires", async () => {
    let current = true
    let release!: (value: ReturnType<typeof record>) => void
    const held = new Promise<ReturnType<typeof record>>((resolve) => {
      release = resolve
    })
    io.get.mockReturnValueOnce(held)
    const pending = getActorProfileForCharacter("character-one", {
      target: profileTarget,
      throwOnError: true,
      validate_lease: () => current
    })
    await vi.waitFor(() => expect(io.get).toHaveBeenCalledOnce())
    current = false
    release(record(inherited))
    await expect(pending).rejects.toThrow("actor_settings_unavailable")
  })

  it.each([
    { ...inherited, chatRole: "invalid" },
    { ...inherited, version: 0 }
  ])(
    "strictly refuses an invalid saved profile without replacing or deleting it (%j)",
    async (invalid) => {
      io.values.set(profileKey, record(invalid))
      await expect(
        getActorProfileForCharacter("character-one", {
          target: profileTarget,
          throwOnError: true
        })
      ).rejects.toThrow("actor_settings_unavailable")
      expect(io.values.get(profileKey)).toEqual(record(invalid))
    }
  )

  it("does not acknowledge a profile write after its persistent backend disappears", async () => {
    io.set.mockImplementationOnce(async (key: string, value: unknown) => {
      io.values.set(key, structuredClone(value))
      io.persistent = false
    })
    expect(
      await saveActorProfileForCharacter({
        characterId: "character-one",
        target: profileTarget,
        settings: inherited
      })
    ).toBe(false)
    expect(io.set).toHaveBeenCalledOnce()
  })

  it("strictly refuses a profile read after its persistent backend disappears", async () => {
    io.get.mockImplementationOnce(async () => {
      io.persistent = false
      return record(inherited)
    })
    await expect(
      getActorProfileForCharacter("character-one", {
        target: profileTarget,
        throwOnError: true
      })
    ).rejects.toThrow("actor_settings_unavailable")
  })

  it.each([false, 0, "", null])(
    "strictly rejects a present malformed falsy profile without deleting it (%j)",
    async (invalid) => {
      io.values.set(profileKey, record(invalid))
      await expect(
        getActorProfileForCharacter("character-one", {
          target: profileTarget,
          throwOnError: true
        })
      ).rejects.toThrow("actor_settings_unavailable")
      expect(io.values.has(profileKey)).toBe(true)
      expect(io.values.get(profileKey)).toEqual(record(invalid))
      expect(io.set).not.toHaveBeenCalled()
    }
  )
})
