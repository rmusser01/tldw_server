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

it("native edit-resend submits a detached owned intent without editing or replacing originals", async () => {
  let release!: () => void
  const held = new Promise<void>((done) => {
    release = done
  })
  const original = [
    { id: "user", name: "You", isBot: false, message: "Original  \n" }
  ]
  const state = {
    status: "ready",
    owner: {
      kind: "native",
      owner_key: "owner",
      conversation_id: "native-chat"
    },
    view: {
      owner_key: "owner",
      conversation_id: "native-chat",
      selection_revision: 4
    },
    capture: {
      status: "captured",
      snapshot: {
        nodes: [{ id: "user", role: "user", revision: "canonical-revision" }]
      }
    }
  }
  const setMessages = vi.fn()
  const setHistory = vi.fn()
  const mutate = vi.fn(async () => {})
  const isEditorCurrent = () => true
  const onSubmit = vi.fn(async (_input) => {
    await held
    return { status: "submitted" }
  })
  const edit = createEditMessage({
    messages: original,
    getMessages: () => original,
    history: [],
    historyId: "native-mirror",
    setMessages,
    setHistory,
    mutate,
    onSubmit,
    validateBeforeSubmitFn: () => true,
    historySelection: { getCurrent: () => state }
  } as Parameters<typeof createEditMessage>[0])
  const attempt = (edit as (...args: unknown[]) => Promise<unknown>)(
    0,
    "Edited  \n",
    true,
    true,
    isEditorCurrent
  )
  void attempt.catch(() => {})
  try {
    expect(onSubmit).toHaveBeenCalledOnce()
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Edited  \n",
        historyEditResend: expect.objectContaining({
          owner_key: "owner",
          conversation_id: "native-chat",
          message_id: "user",
          message_revision: "canonical-revision",
          isCurrent: expect.any(Function),
          isOriginCurrent: expect.any(Function)
        })
      })
    )
    expect(onSubmit.mock.calls[0][0].historyEditResend.isCurrent()).toBe(true)
    expect(onSubmit.mock.calls[0][0].historyEditResend.isOriginCurrent()).toBe(
      true
    )
    expect(mutate).not.toHaveBeenCalled()
    expect(setMessages).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
    expect(original[0].message).toBe("Original  \n")
  } finally {
    release()
    await attempt.catch(() => {})
  }
  expect(await attempt).toEqual({ status: "submitted" })
})

it.each(["images", "documents", "metadata"])(
  "native edit rejects displayed %s before discarding payload or dispatching",
  async (kind) => {
    const target = {
      id: "user",
      name: "You",
      isBot: false,
      message: "Original",
      ...(kind === "images"
        ? { images: ["first", "second"] }
        : kind === "documents"
          ? { documents: [{}] }
          : { metadataExtra: { tool_calls: [] } })
    }
    const state = {
      status: "ready",
      owner: { kind: "native" },
      view: {
        owner_key: "owner",
        conversation_id: "chat",
        selection_revision: 1
      },
      capture: {
        status: "captured",
        snapshot: { nodes: [{ id: "user", role: "user", revision: "u1" }] }
      }
    }
    const onSubmit = vi.fn(),
      mutate = vi.fn(),
      setMessages = vi.fn(),
      setHistory = vi.fn()
    const edit = createEditMessage({
      messages: [target],
      history: [],
      historyId: "mirror",
      onSubmit,
      mutate,
      setMessages,
      setHistory,
      validateBeforeSubmitFn: () => true,
      historySelection: { getCurrent: () => state }
    } as Parameters<typeof createEditMessage>[0])
    await expect(edit(0, "Edited", true, true, () => true)).rejects.toThrow(
      "unsupported_message_payload"
    )
    expect(onSubmit).not.toHaveBeenCalled()
    expect(mutate).not.toHaveBeenCalled()
    expect(setMessages).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
  }
)

it("native edit intent retires its editor and origin when the draft source is replaced", async () => {
  let sourceCurrent = true
  const target = { id: "u", name: "You", isBot: false, message: "Original" }
  const state = {
    status: "ready",
    owner: { kind: "native" },
    view: {
      owner_key: "owner",
      conversation_id: "chat",
      selection_revision: 1
    },
    capture: {
      status: "captured",
      snapshot: { nodes: [{ id: "u", role: "user", revision: "u1" }] }
    }
  }
  const onSubmit = vi.fn(async () => ({ status: "submitted" as const }))
  const edit = createEditMessage({
    messages: [target],
    history: [],
    historyId: "mirror",
    onSubmit,
    setMessages: vi.fn(),
    setHistory: vi.fn(),
    validateBeforeSubmitFn: () => true,
    captureSourceFence: () => () => sourceCurrent,
    captureViewFence: () => () => true,
    historySelection: { getCurrent: () => state }
  } as Parameters<typeof createEditMessage>[0])
  await edit(0, "Edited", true, true, () => true)
  const intent = (onSubmit.mock.calls[0] as any)[0].historyEditResend
  sourceCurrent = false
  expect(intent.isCurrent()).toBe(false)
  expect(intent.isOriginCurrent()).toBe(false)
  expect(await edit(0, "Newer", true, true, () => true)).toMatchObject({
    status: "skipped",
    reason: "stale_selection"
  })
  expect(onSubmit).toHaveBeenCalledOnce()
})


it("native character edit admits matching backend sender-role metadata without replacing originals", async () => {
  const target = {
    id: "u",
    name: "You",
    isBot: false,
    message: "Original",
    metadataExtra: { sender_role: "user" }
  }
  const state = {
    status: "ready",
    owner: { kind: "native" },
    view: {
      owner_key: "owner",
      conversation_id: "chat",
      selection_revision: 1
    },
    capture: {
      status: "captured",
      snapshot: { nodes: [{ id: "u", role: "user", revision: "u1" }] }
    }
  }
  const onSubmit = vi.fn(async () => ({ status: "submitted" as const })),
    setMessages = vi.fn(),
    setHistory = vi.fn()
  const edit = createEditMessage({
    messages: [target],
    history: [],
    historyId: "mirror",
    onSubmit,
    setMessages,
    setHistory,
    validateBeforeSubmitFn: () => true,
    historySelection: { getCurrent: () => state }
  } as Parameters<typeof createEditMessage>[0])
  await expect(edit(0, "Edited", true, true, () => true)).resolves.toEqual({
    status: "submitted"
  })
  expect(onSubmit).toHaveBeenCalledOnce()
  expect(target.message).toBe("Original")
  expect(setMessages).not.toHaveBeenCalled()
  expect(setHistory).not.toHaveBeenCalled()
})


it.each(["conflict", "unsupported"])(
  "edit notifications use safe recovery for %s",
  async (kind) => {
    const notify = { error: vi.fn() }
    const failure = Object.assign(
      new Error("/api/v1/messages/private-id secret-token-value"),
      { status: 409 }
    )
    const setMessages = vi.fn(),
      setHistory = vi.fn(),
      onSubmit = vi.fn()
    const edit = createEditMessage({
      messages: [{ id: "u", isBot: false, name: "You", message: "Original" }],
      history: [],
      historyId: "chat",
      setMessages,
      setHistory,
      onSubmit,
      validateBeforeSubmitFn: () => true,
      mutate: async () => {
        throw failure
      },
      notification: notify as any
    })
    await expect(
      edit(0, "Retained edit", true, kind === "unsupported")
    ).rejects.toThrow()
    expect(notify.error).toHaveBeenCalledOnce()
    expect(notify.error.mock.calls[0][0].description).toMatch(
      kind === "conflict"
        ? /saved message has changed/i
        : /does not support this edit action/i
    )
    expect(notify.error.mock.calls[0][0].description).not.toMatch(
      /unsupported_|private-id|secret-token|api\/v1/
    )
    expect(setMessages).not.toHaveBeenCalled()
    expect(setHistory).not.toHaveBeenCalled()
    expect(onSubmit).not.toHaveBeenCalled()
  }
)
