import { describe, expect, it, vi } from "vitest"
import type { ChatHistory, Message } from "@/store/option"
const writes = vi.hoisted(() => vi.fn(async () => undefined))
vi.mock("@/db/dexie/helpers", () => ({ updateMessageById: writes }))
import { createEditMessage } from "../messageHandlers"

const fixture = () => {
  const generationInfo = {
    mode: "rag",
    grounded: false,
    reason: "selected_source_retrieval_failed"
  }
  const messages: Message[] = [
    {
      id: "diagnostic-user",
      isBot: false,
      name: "You",
      message: "Source question",
      sources: [],
      images: [],
      generationInfo
    },
    {
      id: "diagnostic-answer",
      parentMessageId: "diagnostic-user",
      isBot: true,
      name: "Assistant",
      message: "Diagnostic",
      sources: [],
      generationInfo
    },
    {
      id: "real-user",
      isBot: false,
      name: "You",
      message: "Real question",
      sources: [],
      images: []
    },
    {
      id: "real-answer",
      parentMessageId: "real-user",
      isBot: true,
      name: "Assistant",
      message: "Real answer",
      sources: []
    }
  ]
  const history: ChatHistory = [
    { role: "user", content: "Real question" },
    { role: "assistant", content: "Real answer" }
  ]
  const setHistory = vi.fn()
  const onSubmit = vi.fn()
  return {
    history,
    setHistory,
    onSubmit,
    edit: createEditMessage({
      messages,
      history,
      setMessages: vi.fn(),
      setHistory,
      historyId: "local",
      validateBeforeSubmitFn: () => true,
      onSubmit
    })
  }
}

describe("editing visible rows after a local-only diagnostic", () => {
  it("refuses an unsupported resend before editing or dispatching the genuine row", async () => {
    writes.mockClear()
    const { edit, onSubmit, setHistory } = fixture()
    await expect(edit(2, "Edited question", true, true)).rejects.toThrow(
      "unsupported_history_edit_and_send:real-user"
    )
    expect(writes).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
  })
  it("updates the genuine prompt row at its eligible index", async () => {
    const { edit, setHistory } = fixture()
    await edit(2, "Edited question", true, false)
    expect(
      setHistory.mock.calls[0][0].map(
        ({ role, content }: ChatHistory[number]) => ({ role, content })
      )
    ).toEqual([
      { role: "user", content: "Edited question" },
      { role: "assistant", content: "Real answer" }
    ])
  })
  it("keeps a local diagnostic edit out of the genuine prompt row", async () => {
    const { edit, setHistory, history } = fixture()
    await edit(0, "Edited diagnostic question", true, false)
    expect(
      setHistory.mock.calls[0][0].map(
        ({ role, content }: ChatHistory[number]) => ({ role, content })
      )
    ).toEqual(history)
  })
})

describe("current edit outcomes", () => {
  it("acknowledges the mutation after it finishes and updates the current view", async () => {
    let resolve!: () => void
    const held = new Promise<void>((accept) => {
      resolve = accept
    })
    const setMessages = vi.fn()
    const setHistory = vi.fn()
    const edit = createEditMessage({
      messages: [{ id: "u", name: "You", isBot: false, message: "Before" }],
      history: [],
      historyId: "chat",
      setMessages,
      setHistory,
      validateBeforeSubmitFn: () => true,
      onSubmit: vi.fn(),
      mutate: () => held
    })
    const attempt = edit(0, "After", true, false)
    expect(setMessages).not.toHaveBeenCalled()
    resolve()
    expect(await attempt).toEqual({ status: "submitted" })
    expect(setMessages).toHaveBeenCalledWith([
      expect.objectContaining({ id: "u", message: "After" })
    ])
  })
  it("returns a skipped result when the view retires during a held mutation", async () => {
    let resolve!: () => void
    const held = new Promise<void>((accept) => {
      resolve = accept
    })
    let current = true
    const setMessages = vi.fn()
    const setHistory = vi.fn()
    const edit = createEditMessage({
      messages: [{ id: "u", name: "You", isBot: false, message: "Before" }],
      history: [],
      historyId: "chat",
      setMessages,
      setHistory,
      validateBeforeSubmitFn: () => true,
      onSubmit: vi.fn(),
      mutate: () => held,
      captureViewFence: () => () => current
    })
    const attempt = edit(0, "After", true, false)
    current = false
    resolve()
    expect(await attempt).toEqual({
      status: "skipped",
      reason: "stale_selection"
    })
    expect(setMessages).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
  })
})

it("a retained edit cannot adopt a newer rendered view before writing", async () => {
  let epoch = 1
  const mutate = vi.fn(async () => {})
  const captureViewFence = () => {
    const token = epoch
    return () => epoch === token
  }
  const edit = createEditMessage({
    messages: [{ id: "same", name: "You", isBot: false, message: "Original" }],
    history: [],
    historyId: "same-chat",
    setMessages: vi.fn(),
    setHistory: vi.fn(),
    validateBeforeSubmitFn: () => true,
    onSubmit: vi.fn(),
    mutate,
    captureViewFence
  })
  epoch++
  expect(await edit(0, "Stale authored text", true, false)).toEqual({
    status: "skipped",
    reason: "stale_selection"
  })
  expect(mutate).not.toHaveBeenCalled()
})

it("an acknowledged edit preserves a row appended during the write", async () => {
  let release!: () => void
  const held = new Promise<void>((accept) => {
    release = accept
  })
  const original = [
    { id: "edited", name: "You", isBot: false, message: "Original" }
  ]
  let live = original
  const setMessages = vi.fn((next) => {
    live = next
  })
  const setHistory = vi.fn()
  const edit = createEditMessage({
    messages: original,
    getMessages: () => live,
    history: [],
    historyId: "chat",
    setMessages,
    setHistory,
    validateBeforeSubmitFn: () => true,
    onSubmit: vi.fn(),
    mutate: () => held
  })
  const attempt = edit(0, "Saved edit", true, false)
  live = [
    ...live,
    { id: "new", name: "You", isBot: false, message: "Newer appended row" }
  ]
  release()
  expect(await attempt).toEqual({ status: "submitted" })
  expect(live.map((row) => row.message)).toEqual([
    "Saved edit",
    "Newer appended row"
  ])
  expect(
    setHistory.mock.calls[0][0].map((row: ChatHistory[number]) => row.content)
  ).toEqual(["Saved edit", "Newer appended row"])
})

it("retirement during message publication prevents obsolete history publication", async () => {
  let current = true
  const setHistory = vi.fn()
  const edit = createEditMessage({
    messages: [{ id: "u", name: "You", isBot: false, message: "Original" }],
    history: [],
    historyId: "chat",
    setMessages: () => {
      current = false
    },
    setHistory,
    validateBeforeSubmitFn: () => true,
    onSubmit: vi.fn(),
    mutate: async () => {},
    captureViewFence: () => () => current
  })
  expect(await edit(0, "Saved edit", true, false)).toEqual({
    status: "skipped",
    reason: "stale_selection"
  })
  expect(setHistory).not.toHaveBeenCalled()
})
