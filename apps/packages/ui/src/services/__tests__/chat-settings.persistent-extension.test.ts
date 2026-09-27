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
