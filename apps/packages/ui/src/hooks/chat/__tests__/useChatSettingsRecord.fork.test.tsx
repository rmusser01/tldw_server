import { act, renderHook } from "@testing-library/react"
import { expect, it, vi } from "vitest"

import { useChatSettingsRecord } from "../useChatSettingsRecord"

const mocks = vi.hoisted(() => ({
  mode: "pending",
  read: vi.fn(),
  write: vi.fn(),
  scopedWrite: vi.fn(),
  keys: [] as string[]
}))
vi.mock("../useHistorySelection", () => ({
  useHistorySelectionContext: () => ({
    settingsMode: () => mocks.mode,
    forkSettings: { authorNote: "verified server" },
    updateForkSettings: mocks.scopedWrite
  })
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: ({ key }: any) => {
    mocks.keys.push(key)
    return [{ authorNote: "poisoned cache" }]
  }
}))
vi.mock("@/services/chat-settings", () => ({
  applyChatSettingsPatch: mocks.write,
  chatSettingsStorageForKey: () => ({}),
  getChatSettingsForKey: mocks.read,
  getChatSettingsStorageKey: (key: string) => key,
  normalizeChatSettingsRecord: (value: any) => value,
  resolveChatSettingsKey: () => "server:child"
}))

it("held native verification never subscribes/imports server-ID cache, then explicit fork edits use scoped owner", async () => {
  const hook = renderHook(() =>
    useChatSettingsRecord({ historyId: null, serverChatId: "child" })
  )
  expect(hook.result.current.settings).toBeNull()
  expect(mocks.keys).not.toContain("server:child")
  expect(mocks.read).not.toHaveBeenCalled()
  await expect(
    hook.result.current.updateSettings({ authorNote: "too early" })
  ).rejects.toThrow("fork_settings_owner_unavailable")
  mocks.mode = "fork"
  hook.rerender()
  expect(hook.result.current.settings).toEqual({
    authorNote: "verified server"
  })
  await act(async () => {
    await hook.result.current.updateSettings({ authorNote: "explicit" })
  })
  expect(mocks.scopedWrite).toHaveBeenCalledWith({ authorNote: "explicit" })
  expect(mocks.write).not.toHaveBeenCalled()
})
