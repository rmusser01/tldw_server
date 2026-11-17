import "./knowledgeQaAuthorityFixture"
import React from "react"
import { act, fireEvent, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"
import { SearchBar } from "../SearchBar"

const ragSearchMock = vi.fn()
const messageOpenMock = vi.fn()
const trackMetricMock = vi.fn()
const createChatMock = vi.fn()
const deleteChatMock = vi.fn()
const addChatMessageMock = vi.fn()
const searchCharactersMock = vi.fn()
const listCharactersMock = vi.fn()

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [undefined],
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: messageOpenMock,
  }),
}))

vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: (...args: unknown[]) => trackMetricMock(...args),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    fetchWithAuth: vi.fn().mockResolvedValue({
      ok: false,
      json: async () => [],
      text: async () => "",
    }),
    ragSearch: (...args: unknown[]) => ragSearchMock(...args),
    createChat: (...args: unknown[]) => createChatMock(...args),
    deleteChat: (...args: unknown[]) => deleteChatMock(...args),
    addChatMessage: (...args: unknown[]) => addChatMessageMock(...args),
    searchCharacters: (...args: unknown[]) => searchCharactersMock(...args),
    listCharacters: (...args: unknown[]) => listCharactersMock(...args),
    getChat: vi.fn().mockResolvedValue({ version: 1 }),
  },
}))

// A context consumer that re-renders only when the provider's context value
// identity changes — i.e. once per reducer-driven state update. Render count
// is therefore a direct observation of reducer update count.
const consumerStats = vi.hoisted(() => ({ renders: 0 }))
let latestQuery = ""
function ContextConsumer() {
  const context = useKnowledgeQA()
  consumerStats.renders += 1
  latestQuery = context.query
  return null
}

const settle = async (ms: number) => {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, ms))
  })
}

const typeIntoSearch = (input: HTMLInputElement, text: string) => {
  for (const char of text) {
    act(() => {
      fireEvent.change(input, { target: { value: input.value + char } })
    })
  }
}

describe("KnowledgeQA query debounce (WP-08)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    consumerStats.renders = 0
    latestQuery = ""
    trackMetricMock.mockResolvedValue(undefined)
    createChatMock.mockResolvedValue({ id: "thread-default", version: 1 })
    deleteChatMock.mockResolvedValue(undefined)
    addChatMessageMock.mockResolvedValue({ id: "msg-default" })
    searchCharactersMock.mockResolvedValue([
      { id: 7, name: "Helpful AI Assistant" },
    ])
    listCharactersMock.mockResolvedValue([
      { id: 7, name: "Helpful AI Assistant" },
    ])
    ragSearchMock.mockResolvedValue({
      results: [],
      answer: null,
      citations: [],
    })
  })

  it("10 rapid query keystrokes produce at most one reducer update after the debounce settles", async () => {
    const { container } = render(
      <KnowledgeQAProvider>
        <SearchBar autoFocus={false} />
        <ContextConsumer />
      </KnowledgeQAProvider>
    )

    const input = container.querySelector(
      "#knowledge-search-input"
    ) as HTMLInputElement
    expect(input).toBeTruthy()
    await waitFor(() => expect(consumerStats.renders).toBeGreaterThanOrEqual(1))

    // Let mount-time provider effects (source health, hydration checks) drain
    // so the render counter is quiet before typing starts.
    await settle(400)
    const baseline = consumerStats.renders

    typeIntoSearch(input, "0123456789")

    // Local input state reflects every keystroke immediately...
    expect(input.value).toBe("0123456789")
    // ...without a single reducer sync during the debounce window.
    expect(consumerStats.renders).toBe(baseline)

    await settle(350)

    // Exactly one trailing reducer update carries the final query.
    expect(consumerStats.renders).toBeLessThanOrEqual(baseline + 1)
    expect(latestQuery).toBe("0123456789")
  })

  it("submitting (Enter/Ask) does not wait for the debounce: the full query is dispatched and searched immediately", async () => {
    const { container } = render(
      <KnowledgeQAProvider>
        <SearchBar autoFocus={false} />
        <ContextConsumer />
      </KnowledgeQAProvider>
    )

    const input = container.querySelector(
      "#knowledge-search-input"
    ) as HTMLInputElement
    await waitFor(() => expect(consumerStats.renders).toBeGreaterThanOrEqual(1))
    await settle(400)

    typeIntoSearch(input, "full query")

    // Submit within the 250ms debounce window: the submit path must flush the
    // reducer sync and pass the query text explicitly to search() so the
    // request carries the complete query, not the stale reducer value.
    await act(async () => {
      fireEvent.submit(input.closest("form") as HTMLFormElement)
    })

    await waitFor(() => expect(ragSearchMock).toHaveBeenCalled())
    expect(ragSearchMock.mock.calls[0][0]).toBe("full query")
    // The flush dispatched the query into the reducer without waiting for
    // the debounce timer.
    expect(latestQuery).toBe("full query")
  })
})
