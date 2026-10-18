// @vitest-environment jsdom
import { renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"

const storage = vi.hoisted(() => ({ watch: vi.fn(), unwatch: vi.fn(), instance: null as unknown }))
vi.mock("@/services/chat-settings", async original => ({
  ...await original<typeof import("@/services/chat-settings")>(),
  chatSettingsStorageForKey: () => ({ get: vi.fn(), watch: storage.watch, unwatch: storage.unwatch })
}))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: (options: { instance: unknown }) => {
  storage.instance = options.instance
  return [undefined, vi.fn()]
} }))
vi.mock("../useHistorySelection", () => ({ useHistorySelectionContext: () => ({
  settingsMode: () => "fork", fence: () => () => true, getCurrent: () => ({ owner: null })
}) }))
import { useChatSettingsRecord } from "../useChatSettingsRecord"

beforeEach(() => { storage.watch.mockReset(); storage.unwatch.mockReset(); storage.instance = null })
it("presentation watcher retains the native callback context and unregisters the exact mapped callbacks", () => {
  const hook = renderHook(() => useChatSettingsRecord({ historyId: null, serverChatId: null }))
  try {
    const instance = storage.instance as { watch: (callbacks: Record<string, (...args: unknown[]) => void>) => void; unwatch: (callbacks: Record<string, (...args: unknown[]) => void>) => void }
    const receiver = vi.fn()
    const key = "chatSettings:h1-settings-presentation"
    const callbacks = { [key]: receiver }
    instance.watch(callbacks)
    const mapped = storage.watch.mock.lastCall![0]
    const context = "native-local:" + key
    mapped[key]({ oldValue: "old", newValue: { schemaVersion: 2, authorNote: "Keep this note", updatedAt: "2026-10-13T00:00:00.000Z" } }, context)
    expect(receiver).toHaveBeenCalledWith(expect.objectContaining({ oldValue: "old", newValue: expect.objectContaining({ storageKey: key, settings: expect.objectContaining({ authorNote: "Keep this note" }) }) }), context)
    instance.unwatch(callbacks)
    expect(storage.unwatch).toHaveBeenCalledWith(mapped)
  } finally { hook.unmount() }
})
