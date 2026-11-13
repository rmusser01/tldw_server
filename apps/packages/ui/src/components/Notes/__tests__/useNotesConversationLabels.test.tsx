import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useNotesConversationLabels } from "../hooks/useNotesConversationLabels"

// NL-04 review: chat-title lookups belong to one account and stop when it changes.

const { mockGetChat, mockLoadSnapshot } = vi.hoisted(() => ({
  mockGetChat: vi.fn(),
  mockLoadSnapshot: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: mockGetChat
  }
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: mockLoadSnapshot
}))

const snapshotFor = (account: string) => ({
  requestScope: { account },
  scopeSignal: new AbortController().signal,
  scopeInvalidatedSignal: new AbortController().signal,
  release: vi.fn()
})

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((res) => {
    resolve = res
  })
  return { promise, resolve }
}

type Props = { authorityScope: string | null; conversationIds: string[] }

const renderLabels = (initialProps: Props) =>
  renderHook((props: Props) => useNotesConversationLabels({ ...props, isOnline: true }), { initialProps })

describe("useNotesConversationLabels", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    let account = "A"
    mockLoadSnapshot.mockImplementation(async () => snapshotFor(account))
    ;(globalThis as { __setAccount?: (next: string) => void }).__setAccount = (next) => {
      account = next
    }
  })

  it("asks for each chat with the account's request scope and an abort signal", async () => {
    mockGetChat.mockResolvedValue({ id: "c1", title: "Research chat" })
    const { result } = renderLabels({ authorityScope: "scope-A", conversationIds: ["c1"] })

    await waitFor(() => expect(result.current.conversationLabelById).toEqual({ c1: "Research chat" }))
    const [id, options] = mockGetChat.mock.calls[0]
    expect(id).toBe("c1")
    expect(options.requestScope).toEqual({ account: "A" })
    expect(options.signal).toBeInstanceOf(AbortSignal)
  })

  it("aborts lookups in flight when the account changes, and shows no title from the old account", async () => {
    const oldLookup = deferred<{ id: string; title: string }>()
    const signals: AbortSignal[] = []
    mockGetChat.mockImplementation((_id: string, options: { signal: AbortSignal }) => {
      signals.push(options.signal)
      return signals.length === 1 ? oldLookup.promise : new Promise(() => undefined)
    })
    const { result, rerender } = renderLabels({ authorityScope: "scope-A", conversationIds: ["c1"] })
    await waitFor(() => expect(signals).toHaveLength(1))

    ;(globalThis as { __setAccount?: (next: string) => void }).__setAccount?.("B")
    rerender({ authorityScope: "scope-B", conversationIds: ["c1"] })

    expect(signals[0].aborted).toBe(true)
    await act(async () => {
      oldLookup.resolve({ id: "c1", title: "Account A chat" })
    })
    expect(result.current.conversationLabelById).toEqual({})
  })

  it("drops titles it already had when the account changes", async () => {
    mockGetChat.mockResolvedValueOnce({ id: "c1", title: "Account A chat" })
    const { result, rerender } = renderLabels({ authorityScope: "scope-A", conversationIds: ["c1"] })
    await waitFor(() => expect(result.current.conversationLabelById).toEqual({ c1: "Account A chat" }))

    mockGetChat.mockImplementation(() => new Promise(() => undefined))
    rerender({ authorityScope: "scope-B", conversationIds: ["c1"] })
    expect(result.current.conversationLabelById).toEqual({})
  })

  it("forgets a chat that was missing in the old account", async () => {
    mockGetChat.mockRejectedValueOnce(Object.assign(new Error("Chat session c1 not found"), { status: 404 }))
    const { result, rerender } = renderLabels({ authorityScope: "scope-A", conversationIds: ["c1"] })
    await waitFor(() => expect(mockGetChat).toHaveBeenCalledTimes(1))

    mockGetChat.mockResolvedValue({ id: "c1", title: "Account B chat" })
    rerender({ authorityScope: "scope-B", conversationIds: ["c1"] })
    await waitFor(() => expect(result.current.conversationLabelById).toEqual({ c1: "Account B chat" }))
  })

  it("aborts lookups in flight on unmount", async () => {
    const signals: AbortSignal[] = []
    mockGetChat.mockImplementation((_id: string, options: { signal: AbortSignal }) => {
      signals.push(options.signal)
      return new Promise(() => undefined)
    })
    const { unmount } = renderLabels({ authorityScope: "scope-A", conversationIds: ["c1"] })
    await waitFor(() => expect(signals).toHaveLength(1))
    unmount()
    expect(signals[0].aborted).toBe(true)
  })

  it("keeps a title the page learned some other way", async () => {
    mockGetChat.mockImplementation(() => new Promise(() => undefined))
    const { result } = renderLabels({ authorityScope: "scope-A", conversationIds: [] })
    act(() => result.current.rememberConversationLabel("c9", "Opened chat"))
    expect(result.current.conversationLabelById).toEqual({ c9: "Opened chat" })
  })
})
