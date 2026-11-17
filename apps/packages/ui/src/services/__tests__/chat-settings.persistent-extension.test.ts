import { afterEach, beforeEach, expect, it, vi } from "vitest"
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
let memory: any
beforeEach(async () => {
  vi.resetModules()
  const fixture = await import("@/hooks/chat/__tests__/local-history-fixture")
  memory = fixture.memory
  vi.doMock("@/db/dexie/schema", () => ({ db: memory }))
  await fixture.seedLocalFork(false)
  vi.stubGlobal("browser", undefined)
  vi.stubGlobal("chrome", undefined)
})
afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})
it.each([false, true])(
  "rejects installed Plasmo without its captured persistent client, initialized=%s",
  async (initialized) => {
    const guard = initialized
      ? { initialized: true, revision: "existing", pending: [] }
      : undefined
    if (guard)
      await memory.chatHistories.update("history-1", {
        local_settings_guard: guard,
      })
    const { withPlainLocalForkSettings } = await import("../chat-settings")
    await expect(
      withPlainLocalForkSettings("history-1", async () => "child-write"),
    ).rejects.toThrow("fork_chat_settings_unavailable")
    expect(
      (await memory.chatHistories.get("history-1")).local_settings_guard,
    ).toEqual(guard)
  },
)
it.each([false, true])(
  "uses canonical local data without a sync client, initialized=%s",
  async (initialized) => {
    const key = "chatSettings:local:history-1"
    const values: Record<string, unknown> = { [key]: "null" }
    vi.stubGlobal("browser", {
      storage: {
        local: {
          get: async (keys: string[]) =>
            Object.fromEntries(keys.map((key) => [key, values[key]])),
          set: async (patch: object) => {
            Object.assign(values, patch)
          },
        },
      },
    })
    if (initialized)
      await memory.chatHistories.update("history-1", {
        local_settings_guard: {
          initialized: true,
          revision: "existing",
          pending: [],
        },
      })
    const { withPlainLocalForkSettings, saveChatSettingsForKey } =
      await import("../chat-settings")
    expect(
      await withPlainLocalForkSettings("history-1", async (validate) => {
        validate(await memory.chatHistories.get("history-1"))
        return "eligible"
      }),
    ).toBe("eligible")
    expect(
      await saveChatSettingsForKey("local:history-1", {
        schemaVersion: 2,
        updatedAt: "2026-09-17T00:00:00Z",
      }),
    ).toBe(true)
    expect(JSON.parse(values[key] as string)).toMatchObject({
      schemaVersion: 2,
    })
  },
)

it("keeps an absent local conversation read nonblocking without creating or erasing settings", async () => {
  const values = { "chatSettings:local:missing": JSON.stringify({ schemaVersion: 2, updatedAt: "2026-11-17T00:00:00Z", authorNote: "Retain this orphaned record" }) }
  const before = structuredClone(values)
  vi.stubGlobal("browser", { storage: { local: {
    get: async () => values,
    set: async (patch: object) => { Object.assign(values, patch) }
  } } })
  const error = vi.spyOn(console, "error").mockImplementation(() => undefined)
  const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined)
  const { getChatSettingsForKey } = await import("../chat-settings")

  expect(await getChatSettingsForKey("local:missing")).toBeNull()
  expect(error).not.toHaveBeenCalled()
  expect(warning).toHaveBeenCalledOnce()
  expect(await memory.chatHistories.get("missing")).toBeUndefined()
  expect(values).toEqual(before)
})

it("still refuses writes and strict fork reads for an absent local conversation", async () => {
  const values: Record<string, unknown> = {}
  vi.stubGlobal("browser", { storage: { local: {
    get: async () => values,
    set: async (patch: object) => { Object.assign(values, patch) }
  } } })
  vi.spyOn(console, "error").mockImplementation(() => undefined)
  const { saveChatSettingsForKey, withPlainLocalForkSettings } = await import("../chat-settings")

  await expect(withPlainLocalForkSettings("missing", async () => "must not fork")).rejects.toThrow("fork_chat_settings_unavailable")
  expect(await saveChatSettingsForKey("local:missing", { schemaVersion: 2, updatedAt: "2026-11-17T00:00:00Z" })).toBe(false)
  expect(await memory.chatHistories.get("missing")).toBeUndefined()
  expect(values).toEqual({})
})

it.each([
  ["local:history-1", "storage failed"],
  ["scratch", "missing_conversation"]
])("retains error diagnostics for %s storage faults", async (chatKey, message) => {
  await memory.chatHistories.update("history-1", {
    local_settings_guard: { initialized: true, revision: "existing", pending: [] }
  })
  const backingStorage = {
    get: async () => { throw new Error(message) },
    set: async () => undefined
  }
  vi.stubGlobal("browser", { storage: { local: backingStorage, sync: backingStorage } })
  const error = vi.spyOn(console, "error").mockImplementation(() => undefined)
  const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined)
  const { getChatSettingsForKey } = await import("../chat-settings")

  expect(await getChatSettingsForKey(chatKey)).toBeNull()
  expect(error).toHaveBeenCalledWith("Failed to load chat settings", expect.any(Error))
  expect(warning).not.toHaveBeenCalled()
})
