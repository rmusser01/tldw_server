import { beforeEach, describe, expect, it, vi } from "vitest"

const storageState = vi.hoisted(() => {
  const store = new Map<string, unknown>()
  return {
    store,
    get: vi.fn(async (key: string) => store.get(key)),
    set: vi.fn(async (key: string, value: unknown) => {
      store.set(key, value)
    }),
    remove: vi.fn(async (key: string) => {
      store.delete(key)
    }),
    initialize: vi.fn(async () => undefined),
    getChatSettings: vi.fn(),
    updateChatSettings: vi.fn()
  }
})

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: storageState.get,
    set: storageState.set,
    remove: storageState.remove
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => storageState.initialize(...args),
    getChatSettings: (...args: unknown[]) => storageState.getChatSettings(...args),
    updateChatSettings: (...args: unknown[]) =>
      storageState.updateChatSettings(...args)
  }
}))

import {
  applyChatSettingsPatch,
  getChatSettingsStorageKey,
  normalizeChatSettingsRecord,
  resolveChatSettingsKey
} from "@/services/chat-settings"

const buildOverlay = (overrides: Record<string, unknown> = {}) => ({
  kind: "persona",
  id: "persona-7",
  name: "Planner",
  avatar_url: "https://example.com/avatar.png",
  system_prompt_snapshot: "You are concise and structured.",
  updatedAt: "2026-05-22T18:00:00.000Z",
  ...overrides
})

describe("chat settings assistant overlay", () => {
  beforeEach(() => {
    storageState.store.clear()
    vi.clearAllMocks()
  })

  it("accepts assistantOverlay.system_prompt_snapshot during normalization", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-05-22T18:00:00.000Z",
      assistantOverlay: buildOverlay()
    })

    expect(settings?.assistantOverlay).toEqual(
      expect.objectContaining({
        system_prompt_snapshot: "You are concise and structured."
      })
    )
  })

  it("rejects malformed assistantOverlay payloads during normalization", () => {
    const settings = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-05-22T18:00:00.000Z",
      assistantOverlay: buildOverlay({
        kind: "invalid-kind",
        system_prompt_snapshot: { text: "bad" }
      })
    })

    expect(settings?.assistantOverlay).toBeUndefined()
  })

  it("rejects oversized assistantOverlay id and name during normalization", () => {
    const oversized = "x".repeat(20_001)
    const tooLongId = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-05-22T18:00:00.000Z",
      assistantOverlay: buildOverlay({
        id: oversized
      })
    })
    const tooLongName = normalizeChatSettingsRecord({
      schemaVersion: 2,
      updatedAt: "2026-05-22T18:00:00.000Z",
      assistantOverlay: buildOverlay({
        name: oversized
      })
    })

    expect(tooLongId?.assistantOverlay).toBeUndefined()
    expect(tooLongName?.assistantOverlay).toBeUndefined()
  })

  it("persists assistantOverlay locally before a server chat id exists", async () => {
    const scratch = await applyChatSettingsPatch({
      historyId: null,
      serverChatId: null,
      patch: {
        assistantOverlay: buildOverlay({
          id: "persona-scratch"
        })
      }
    })
    const local = await applyChatSettingsPatch({
      historyId: "history-overlay-1",
      serverChatId: null,
      patch: {
        assistantOverlay: buildOverlay({
          id: "persona-local"
        })
      }
    })

    expect(scratch?.assistantOverlay?.id).toBe("persona-scratch")
    expect(local?.assistantOverlay?.id).toBe("persona-local")
    expect(
      storageState.store.get(
        getChatSettingsStorageKey(
          resolveChatSettingsKey({ historyId: null, serverChatId: null })
        )
      )
    ).toMatchObject({
      assistantOverlay: expect.objectContaining({ id: "persona-scratch" })
    })
    expect(
      storageState.store.get(
        getChatSettingsStorageKey(
          resolveChatSettingsKey({
            historyId: "history-overlay-1",
            serverChatId: null
          })
        )
      )
    ).toMatchObject({
      assistantOverlay: expect.objectContaining({ id: "persona-local" })
    })
  })

  it("merges a partial local overlay patch into the existing valid overlay", async () => {
    const historyId = "history-overlay-preserve"
    const storageKey = getChatSettingsStorageKey(
      resolveChatSettingsKey({
        historyId,
        serverChatId: null
      })
    )
    storageState.store.set(storageKey, {
      schemaVersion: 2,
      updatedAt: "2026-05-22T18:00:00.000Z",
      assistantOverlay: buildOverlay({
        id: "persona-existing",
        name: "Existing Overlay"
      })
    })

    const next = await applyChatSettingsPatch({
      historyId,
      serverChatId: null,
      patch: {
        assistantOverlay: {
          name: "Renamed Only"
        } as never
      }
    })

    expect(next?.assistantOverlay).toEqual(
      expect.objectContaining({
        id: "persona-existing",
        name: "Renamed Only"
      })
    )
    expect(storageState.store.get(storageKey)).toMatchObject({
      assistantOverlay: expect.objectContaining({
        id: "persona-existing",
        name: "Renamed Only"
      })
    })
  })

  it("preserves an existing valid local overlay when a merged overlay patch remains invalid", async () => {
    const historyId = "history-overlay-invalid-merge"
    const storageKey = getChatSettingsStorageKey(
      resolveChatSettingsKey({
        historyId,
        serverChatId: null
      })
    )
    storageState.store.set(storageKey, {
      schemaVersion: 2,
      updatedAt: "2026-05-22T18:00:00.000Z",
      assistantOverlay: buildOverlay({
        id: "persona-existing",
        name: "Existing Overlay"
      })
    })

    const next = await applyChatSettingsPatch({
      historyId,
      serverChatId: null,
      patch: {
        assistantOverlay: {
          name: "x".repeat(20_001)
        } as never
      }
    })

    expect(next?.assistantOverlay).toEqual(
      expect.objectContaining({
        id: "persona-existing",
        name: "Existing Overlay"
      })
    )
    expect(storageState.store.get(storageKey)).toMatchObject({
      assistantOverlay: expect.objectContaining({
        id: "persona-existing",
        name: "Existing Overlay"
      })
    })
  })

  it("allows assistantOverlay to be cleared explicitly with null", async () => {
    const historyId = "history-overlay-clear"

    await applyChatSettingsPatch({
      historyId,
      serverChatId: null,
      patch: {
        assistantOverlay: buildOverlay({
          id: "persona-clear"
        })
      }
    })

    const next = await applyChatSettingsPatch({
      historyId,
      serverChatId: null,
      patch: {
        assistantOverlay: null
      }
    })

    expect(next?.assistantOverlay).toBeNull()
    expect(
      storageState.store.get(
        getChatSettingsStorageKey(
          resolveChatSettingsKey({
            historyId,
            serverChatId: null
          })
        )
      )
    ).toMatchObject({
      assistantOverlay: null
    })
  })
})
