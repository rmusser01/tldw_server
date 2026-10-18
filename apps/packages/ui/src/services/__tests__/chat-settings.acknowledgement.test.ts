import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatSettingsRecord } from "@/types/chat-session-settings"

const state = vi.hoisted(() => ({ records: new Map<string, unknown>(), readError: false, writeError: false }))
const transport = vi.hoisted(() => ({ initialize: vi.fn(), getChatSettings: vi.fn(), updateChatSettings: vi.fn() }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({
  hasPersistentBackend: true,
  get: async (key: string) => { if (state.readError) throw new Error("device read failed"); return state.records.get(key) },
  set: async (key: string, value: unknown) => { if (state.writeError) throw new Error("device write failed"); state.records.set(key, value) },
  remove: async (key: string) => { state.records.delete(key) }
}) }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: transport }))
import { applyChatSettingsPatch, getChatSettingsStorageKey, syncChatSettingsForServerChat, resolveChatSettingsKey } from "../chat-settings"
import { settingsTestOwner, installSettingsLocks } from "./chat-settings-fixture"

const initial: ChatSettingsRecord = { schemaVersion: 2, updatedAt: "2026-10-08T00:00:00.000Z", authorNote: "Keep this source" }
const params = { historyId: null, serverChatId: "chat-1", ...settingsTestOwner }
const key = getChatSettingsStorageKey(resolveChatSettingsKey(params))

beforeEach(() => {
  state.records.clear(); state.records.set(key, { value: initial, updatedAt: 1 }); state.readError = false; state.writeError = false
  installSettingsLocks()
  vi.resetAllMocks()
  transport.initialize.mockResolvedValue(undefined)
  transport.getChatSettings.mockResolvedValue({ conversation_id: "chat-1", settings: {} })
  transport.updateChatSettings.mockImplementation(async (id: string, settings: unknown) => ({ conversation_id: id, settings }))
})

afterEach(() => vi.unstubAllGlobals())

describe("ordinary settings acknowledgement boundaries", () => {
  it.each([401,403,404,503,"timeout"])("does not overwrite or remove source settings after failed GET %s", async code => {
    const error = Object.assign(new Error("lookup failed"), { status: code })
    transport.getChatSettings.mockRejectedValueOnce(error)
    await expect(syncChatSettingsForServerChat(params)).rejects.toBe(error)
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
    expect(state.records.get(key)).toMatchObject({ value: initial })
  })

  it("treats a matching successful empty GET as known absence", async () => {
    expect(await syncChatSettingsForServerChat(params)).toBeNull()
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
    expect(state.records.get(key)).toMatchObject({ value: null })
  })

  it("does not treat failed initialization as permission to write", async () => {
    transport.initialize.mockRejectedValueOnce(new Error("not initialized"))
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).rejects.toThrow("not initialized")
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
    expect(state.records.get(key)).toMatchObject({ value: initial })
  })

  it("preserves an unreadable device source instead of replacing it with defaults", async () => {
    state.readError = true
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).rejects.toThrow("device read failed")
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
    expect(state.records.get(key)).toMatchObject({ value: initial })
  })

  it("does not return a saved record after a rejected PUT", async () => {
    transport.updateChatSettings.mockRejectedValueOnce(new Error("write failed"))
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).rejects.toMatchObject({ message: "chat_settings_server_unconfirmed", cause: expect.objectContaining({ message: "write failed" }) })
    expect(state.records.get(key)).toMatchObject({ value: initial })
  })

  it.each([
    { settings: initial },
    { conversation_id: "other-chat", settings: initial },
    { conversation_id: "chat-1", settings: null },
    { conversation_id: "chat-1", settings: [] }
  ])("requires matching destination and canonical settings in PUT receipt %j", async response => {
    transport.updateChatSettings.mockResolvedValueOnce(response)
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).rejects.toThrow("chat_settings_ack_invalid")
    expect(state.records.get(key)).toMatchObject({ value: initial })
  })

  it("uses the server canonical response instead of an optimistic local projection", async () => {
    const canonical = { ...initial, authorNote: "Submitted", updatedAt: "2026-10-08T01:00:00.000Z" }
    transport.updateChatSettings.mockResolvedValueOnce({ conversation_id: "chat-1", settings: canonical })
    expect(await applyChatSettingsPatch({ ...params, patch: { authorNote: "Submitted" } })).toEqual(canonical)
    expect(state.records.get(key)).toMatchObject({ value: canonical })
  })

  it("sends only edited keys and retains unrelated canonical server settings", async () => {
    state.records.set(key, { value: { ...initial, greetingEnabled: true, conversationContext: { world_book_ids: [1] } }, updatedAt: 1 })
    const canonical = { ...initial, greetingEnabled: false, conversationContext: { world_book_ids: [2] }, authorNote: "New" }
    transport.updateChatSettings.mockImplementationOnce(async (_id: string, patch: Record<string, unknown>) => {
      expect(patch).not.toHaveProperty("greetingEnabled")
      expect(patch).not.toHaveProperty("conversationContext")
      return { conversation_id: "chat-1", settings: canonical }
    })
    expect(await applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).toEqual(canonical)
    expect(state.records.get(key)).toMatchObject({ value: canonical })
  })

  it.each([
    { settings: {} },
    { conversation_id: "other-chat", settings: {} },
    { conversation_id: "chat-1", settings: null },
    { conversation_id: "chat-1", settings: [] }
  ])("rejects malformed GET before any write %j", async response => {
    transport.getChatSettings.mockResolvedValueOnce(response)
    await expect(syncChatSettingsForServerChat(params)).rejects.toThrow("chat_settings_ack_invalid")
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
    expect(state.records.get(key)).toMatchObject({ value: initial })
  })

  it("retains the canonical server acknowledgement when its device save fails", async () => {
    const canonical = { ...initial, authorNote: "Submitted" }
    transport.updateChatSettings.mockImplementationOnce(async () => {
      state.writeError = true
      return { conversation_id: "chat-1", settings: canonical }
    })
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "Submitted" } })).rejects.toMatchObject({ message: "chat_settings_device_save_failed", serverSettings: canonical })
    expect(state.records.get(key)).toMatchObject({ value: initial })
  })
})
