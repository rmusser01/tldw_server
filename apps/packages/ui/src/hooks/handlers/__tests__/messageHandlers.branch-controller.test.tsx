import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import {
  memory,
  settings,
  seedLocalFork
} from "@/hooks/chat/__tests__/local-history-fixture"
vi.mock("@/db/dexie/schema", async () => ({
  db: (await import("@/hooks/chat/__tests__/local-history-fixture")).memory
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
vi.mock("@/utils/safe-storage", async () => {
  const { settings } = await import(
    "@/hooks/chat/__tests__/local-history-fixture"
  )
  return {
    createSafeStorage: (options: any) => ({
      get: async (key: string) => settings.get((options?.area ?? "sync") + key),
      set: async (key: string, value: any) => {
        settings.set((options?.area ?? "sync") + key, value)
      }
    })
  }
})
import { useHistorySelection } from "@/hooks/chat/useHistorySelection"
import { createSelectedForkAction } from "@/hooks/chat/chat-action-utils"
import { createBranchMessage } from "../messageHandlers"
beforeEach(async () => {
  settings.clear()
  await seedLocalFork(false)
})
it("navigation while the real child load waits preserves commit without adopting or cleaning the new view", async () => {
  const controller = renderHook(() => useHistorySelection())
  await act(async () => {
    await controller.result.current.loadConversation({ historyId: "history-1" })
  })
  const originalGet = memory.chatHistories.get
  let release!: () => void
  let held = false
  vi.spyOn(memory.chatHistories, "get").mockImplementation(async (id: any) => {
    if (id !== "history-1" && !held) {
      held = true
      await new Promise<void>((resolve) => {
        release = resolve
      })
    }
    return originalGet(id)
  })
  const options = {
    historyId: "history-1",
    historySelection: controller.result.current,
    captureViewFence: controller.result.current.fence,
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setHistoryId: vi.fn(),
    notification: { error: vi.fn() } as any,
    onOpened: vi.fn()
  }
  let result: any
  await act(async () => {
    const pending = createSelectedForkAction(
      controller.result.current,
      createBranchMessage(options),
      "history-1"
    )("a1")
    await vi.waitFor(() => expect(held).toBe(true))
    await controller.result.current.loadConversation({ historyId: "history-1" })
    release()
    result = await pending
  })
  expect(result.state).toBe("committed")
  expect(controller.result.current.getCurrent().owner).toMatchObject({
    conversation_id: "history-1"
  })
  expect(options.setMessages).not.toHaveBeenCalled()
  expect(options.setHistoryId).not.toHaveBeenCalled()
  expect(options.onOpened).not.toHaveBeenCalled()
})
it("copy without opening does not adopt or clean up a view", async () => {
  const controller = renderHook(() => useHistorySelection())
  await act(async () => {
    await controller.result.current.loadConversation({ historyId: "history-1" })
  })
  const setHistoryId = vi.fn()
  const result = await createSelectedForkAction(
    controller.result.current,
    createBranchMessage({
      historyId: "history-1",
      captureViewFence: controller.result.current.fence,
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setHistoryId,
      notification: { error: vi.fn() } as any
    }),
    "history-1"
  )("a1")
  expect(result.state).toBe("committed")
  expect(controller.result.current.getCurrent().owner).toMatchObject({
    conversation_id: "history-1"
  })
  expect(setHistoryId).not.toHaveBeenCalled()
})
