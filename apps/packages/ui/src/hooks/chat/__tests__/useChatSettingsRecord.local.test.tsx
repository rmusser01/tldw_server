import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { memory, settings, seedLocalFork } from "./local-history-fixture"
const unguardedWrite = vi.hoisted(() => vi.fn())
vi.mock("@/db/dexie/schema", async () => ({
  db: (await import("./local-history-fixture")).memory
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
vi.mock("@/utils/safe-storage", async () => {
  const { settings } = await import("./local-history-fixture")
  return {
    createSafeStorage: (options: any) => ({
      hasPersistentBackend: true,
      get: async (key: string) => settings.get((options?.area ?? "sync") + key),
      set: async (key: string, value: any) => {
        settings.set((options?.area ?? "sync") + key, value)
      }
    })
  }
})
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [undefined, unguardedWrite]
}))
import { useChatSettingsRecord } from "../useChatSettingsRecord"
beforeEach(async () => {
  settings.clear()
  unguardedWrite.mockClear()
  await seedLocalFork(false)
})
it("migrates the local hook's legacy settings and applies patches through exactly one guarded writer", async () => {
  const key = "chatSettings:local:history-1"
  const legacy = {
    schemaVersion: 2,
    updatedAt: "2026-09-17T00:00:00.000Z",
    assistantOverlay: {
      kind: "persona",
      id: "p",
      name: "Guide",
      updatedAt: "2026-09-17T00:00:00.000Z"
    }
  }
  settings.set("sync" + key, legacy)
  const hook = renderHook(() =>
    useChatSettingsRecord({ historyId: "history-1", serverChatId: null })
  )
  await waitFor(() => expect(settings.get("local" + key)).toEqual(legacy))
  await act(async () => {
    await hook.result.current.updateSettings({ authorNote: "new note" })
  })
  expect(settings.get("local" + key)).toMatchObject({
    authorNote: "new note",
    assistantOverlay: legacy.assistantOverlay
  })
  expect(settings.get("sync" + key)).toEqual(legacy)
  expect(unguardedWrite).not.toHaveBeenCalled()
  expect(
    (await memory.chatHistories.get("history-1")).local_settings_guard.pending
  ).toEqual([])
})
