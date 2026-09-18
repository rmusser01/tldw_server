import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"

import { useChatSettingsRecord } from "../useChatSettingsRecord"

const mocks = vi.hoisted(() => ({
  mode: "pending",
  controller: null as any,
  candidate: vi.fn(async () => null as any),
  read: vi.fn(),
  write: vi.fn(),
  scopedWrite: vi.fn(),
  keys: [] as string[]
}))
vi.mock("../useHistorySelection", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../useHistorySelection")>()
  return {
    ...actual,
    useHistorySelectionContext: () =>
      mocks.controller ?? {
        settingsMode: () => mocks.mode,
        forkSettings: { authorNote: "verified server" },
        updateForkSettings: mocks.scopedWrite
      }
  }
})
vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: async () => "profile",
  loadHistoryBookmark: async () => null,
  saveHistoryBookmark: async () => {},
  loadHistoryTurnRecoveries: async () => []
}))
vi.mock("@/db/dexie/fork-operations", () => ({
  findForkCandidate: () => mocks.candidate(),
  loadForkOperations: async () => []
}))
vi.mock("@/services/chat-history-selection", () => ({
  captureHistorySnapshot: async (owner: any, view: any) => ({
    status: "legacy_review_required",
    code: "legacy_review_required",
    view,
    snapshot: {
      version: 1,
      owner_key: owner.owner_key,
      conversation_id: owner.conversation_id,
      nodes: [],
      source_digest: "source",
      storage_context_digest: "storage",
      fences: {},
      interpretation_status: { kind: "legacy_review_required" }
    }
  }),
  readNativeForkSettings: async () => ({ authorNote: "verified server" }),
  updateNativeForkSettings: (...args: any[]) => mocks.scopedWrite(...args)
}))
import { useHistorySelection } from "../useHistorySelection"
beforeEach(() => {
  mocks.controller = null
  mocks.mode = "pending"
  mocks.keys = []
  mocks.read.mockClear()
  mocks.write.mockClear()
  mocks.scopedWrite.mockClear()
  mocks.candidate.mockReset().mockResolvedValue(null)
})
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

it.each(["ordinary", "fork"])(
  "real qualified legacy controller keeps %s settings editable without ancestry confirmation",
  async (kind) => {
    const controller = renderHook(() => useHistorySelection())
    if (kind === "fork")
      mocks.candidate.mockResolvedValue({ candidate_child_id: "child" })
    const owner: any = {
      kind: "native",
      owner_key: "owner",
      conversation_id: "child",
      validate_lease: () => true
    }
    await act(async () => {
      await controller.result.current.open(owner)
    })
    mocks.controller = controller.result.current
    const settings = renderHook(() =>
      useChatSettingsRecord({ historyId: null, serverChatId: "child" })
    )
    expect(controller.result.current.status).toBe("legacy_review_required")
    expect(settings.result.current.settings).toEqual({
      authorNote: kind === "fork" ? "verified server" : "poisoned cache"
    })
    mocks.scopedWrite.mockResolvedValue({ authorNote: "explicit" })
    await act(async () => {
      await settings.result.current.updateSettings({ authorNote: "explicit" })
    })
    if (kind === "fork") {
      expect(mocks.write).not.toHaveBeenCalled()
      expect(mocks.keys).not.toContain("server:child")
      expect(mocks.scopedWrite).toHaveBeenCalledWith(
        owner,
        { authorNote: "explicit" },
        expect.any(AbortSignal)
      )
    } else {
      expect(mocks.write).toHaveBeenCalledWith({
        historyId: null,
        serverChatId: "child",
        patch: { authorNote: "explicit" }
      })
      expect(mocks.read).toHaveBeenCalledWith("server:child")
    }
    expect(controller.result.current.status).toBe("legacy_review_required")
  }
)
