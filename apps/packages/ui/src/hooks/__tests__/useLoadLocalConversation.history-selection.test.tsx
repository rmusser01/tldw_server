import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
const mocks = vi.hoisted(() => ({
  history: vi.fn(),
  details: vi.fn(),
  files: vi.fn(),
  prompt: vi.fn()
}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {
    getChatHistory = mocks.history
    getHistoryInfo = mocks.details
  }
}))
vi.mock("@/db/dexie/helpers", () => ({
  formatToMessage: (rows: any[]) => rows,
  formatToChatHistory: (rows: any[]) => rows,
  getPromptById: mocks.prompt,
  getSessionFiles: mocks.files
}))
vi.mock("@/services/model-settings", () => ({
  lastUsedChatModelEnabled: async () => false
}))
vi.mock("@/utils/update-page-title", () => ({ updatePageTitle: vi.fn() }))
vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: () => null
}))
import { useLoadLocalConversation } from "../useLoadLocalConversation"
const deferred = () => {
  let resolve!: (value: any) => void
  const promise = new Promise((r) => {
    resolve = r
  })
  return { promise, resolve }
}
beforeEach(() => {
  mocks.details.mockResolvedValue({})
  mocks.files.mockResolvedValue([])
})
it("does not install an older conversation after a new navigation finishes", async () => {
  const old = deferred()
  mocks.history.mockImplementation((id: string) =>
    id === "old" ? old.promise : Promise.resolve([{ id }])
  )
  const { result } = renderHook(() => {
    const [rows, setRows] = React.useState<any[]>([])
    const load = useLoadLocalConversation(
      {
        setServerChatId: vi.fn(),
        setHistoryId: vi.fn(),
        setHistory: vi.fn(),
        setMessages: setRows,
        setSelectedModel: vi.fn(),
        setSelectedSystemPrompt: vi.fn(),
        setSystemPrompt: vi.fn(),
        setContextFiles: vi.fn()
      },
      { t: (key) => key, errorLogPrefix: "load", errorDefaultMessage: "failed" }
    )
    return { rows, load }
  })
  let pending!: Promise<void>
  act(() => {
    pending = result.current.load("old")
  })
  await act(async () => {
    await result.current.load("new")
  })
  await act(async () => {
    old.resolve([{ id: "old" }])
    await pending
  })
  expect(result.current.rows).toEqual([{ id: "new" }])
})
