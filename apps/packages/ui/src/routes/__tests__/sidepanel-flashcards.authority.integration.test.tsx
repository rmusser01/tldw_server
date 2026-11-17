import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import SidepanelFlashcards from "../sidepanel-flashcards"

const state = vi.hoisted(() => ({
  user: 1,
  request: vi.fn(),
  executeScript: vi.fn(),
  tabsQuery: vi.fn()
}))
vi.mock(
  "@plasmohq/storage",
  async () =>
    import("../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))
vi.mock("@/utils/browser-runtime", () => ({ isExtensionRuntime: () => true }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    ensureConfigForRequest: async () =>
      JSON.parse(window.localStorage.getItem("tldwConfig") || "null")
  }
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: async () => ({ id: state.user, is_active: true })
  }
}))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => state.request(...args),
  bgUpload: vi.fn()
}))
vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn() })
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasFlashcards: true },
    loading: false
  })
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { getURL: (path: string) => `chrome-extension://owned${path}` },
    tabs: {
      query: (...args: unknown[]) => state.tabsQuery(...args),
      create: vi.fn()
    },
    scripting: {
      executeScript: (...args: unknown[]) => state.executeScript(...args)
    }
  }
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: string | Record<string, unknown>) =>
      typeof options === "string"
        ? options
        : String(options?.defaultValue || key).replace(
            /\{\{(\w+)\}\}/g,
            (_match, name: string) => String(options?.[name] ?? "")
          )
  })
}))

const tokenFor = (user: number, rotation: string) =>
  `header.${btoa(JSON.stringify({ sub: String(user), nonce: rotation }))}.signature`
const selectUser = (user: number, rotation = "original") => {
  state.user = user
  window.localStorage.setItem(
    "tldwConfig",
    JSON.stringify({
      serverUrl: "https://server.test",
      authMode: "multi-user",
      accessToken: tokenFor(user, rotation),
      refreshToken: `refresh-${user}-${rotation}`
    })
  )
  window.dispatchEvent(
    new CustomEvent("tldw:config-updated", {
      detail: { authorityChanged: rotation === "original" }
    })
  )
}
const deckName = () =>
  state.user === 1 ? "Alice private deck" : "Bob private deck"
const clients: QueryClient[] = []
const mount = (strict = false) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  clients.push(client)
  const panel = (
    <QueryClientProvider client={client}>
      <SidepanelFlashcards />
    </QueryClientProvider>
  )
  return render(strict ? <React.StrictMode>{panel}</React.StrictMode> : panel)
}
const press = async (name: string) => {
  const button = await screen.findByRole("button", { name, exact: true })
  await waitFor(() => expect(button).toBeEnabled())
  fireEvent.click(button)
}
const capture = () => press("Capture page selection")
const deferred = () => {
  let resolve!: (value: unknown) => void
  const promise = new Promise<unknown>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
const requests = (path: string, method = "GET") =>
  state.request.mock.calls
    .map(([input]) => input)
    .filter(
      (input) =>
        input.path.split("?")[0] === path && (input.method ?? "GET") === method
    )
const reviewCard = (user: number) => ({
  uuid: `00000000-0000-0000-0000-00000000000${user}`,
  deck_id: 7,
  front: user === 1 ? "Alice private question" : "Bob current question",
  back: user === 1 ? "Alice private answer" : "Bob current answer",
  model_type: "basic",
  version: 1
})
const openReview = async (user: number) => {
  await press("Review due card")
  await screen.findByText(reviewCard(user).front)
}

describe("native Flashcards with the real query cache and account boundaries", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    window.sessionStorage.clear()
    selectUser(1)
    state.tabsQuery.mockResolvedValue([
      { id: 7, title: "Current source", url: "https://source.test" }
    ])
    state.executeScript.mockImplementation(async () => [
      {
        result: state.user === 1 ? "Alice private source" : "Bob current source"
      }
    ])
    state.request.mockImplementation(
      async ({ path, method }: { path: string; method?: string }) => {
        if (path.split("?")[0] === "/api/v1/flashcards/decks")
          return [{ id: 7, name: deckName() }]
        if (path.split("?")[0] === "/api/v1/flashcards" && method === "POST")
          return { uuid: "00000000-0000-0000-0000-000000000001", version: 1 }
        if (path.split("?")[0] === "/api/v1/flashcards/review/next")
          return { card: null }
        if (path.split("?")[0] === "/api/v1/flashcards/templates")
          return { items: [] }
        return {}
      }
    )
  })
  afterEach(() => {
    clients.splice(0).forEach((client) => client.clear())
    vi.unstubAllGlobals()
  })

  it("retires Alice's warmed deck and draft before Bob can save to a colliding deck ID", async () => {
    mount()
    await capture()
    await screen.findByDisplayValue("Alice private source")
    await screen.findByText("Alice private deck")
    await act(async () => selectUser(2))
    await waitFor(() =>
      expect(
        screen.queryByDisplayValue("Alice private source")
      ).not.toBeInTheDocument()
    )
    await capture()
    await screen.findByDisplayValue("Bob current source")
    await screen.findByText("Bob private deck")
    expect(screen.queryByText("Alice private deck")).not.toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Save card", exact: true })
    )
    await waitFor(() =>
      expect(
        state.request.mock.calls.some(
          ([input]) =>
            input.method === "POST" &&
            input.body?.back === "Bob current source" &&
            input.servicePromptConfig?.expectedUserId === 2
        )
      ).toBe(true)
    )
  })

  it("does not read page text after retirement while the tab query is held", async () => {
    let finish!: (value: unknown) => void
    state.tabsQuery.mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve
      })
    )
    mount()
    await capture()
    await waitFor(() => expect(state.tabsQuery).toHaveBeenCalledTimes(1))
    await act(async () => selectUser(2))
    await act(async () =>
      finish([{ id: 7, title: "Replacement page", url: "https://source.test" }])
    )
    expect(state.executeScript).not.toHaveBeenCalled()
  })

  it("preserves the current draft and deck on a same-owner refresh", async () => {
    mount()
    await capture()
    await screen.findByDisplayValue("Alice private source")
    await screen.findByText("Alice private deck")
    await act(async () => selectUser(1, "rotated"))
    expect(screen.getByDisplayValue("Alice private source")).toBeInTheDocument()
    expect(screen.getByText("Alice private deck")).toBeInTheDocument()
  })

  it("does not publish held capture text after an Alice-Bob-Alice replacement", async () => {
    let finish!: (value: unknown) => void
    state.executeScript.mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve
      })
    )
    mount()
    await capture()
    await waitFor(() => expect(state.executeScript).toHaveBeenCalledTimes(1))
    await act(async () => selectUser(2))
    await capture()
    await screen.findByDisplayValue("Bob current source")
    await screen.findByText("Bob private deck")
    await act(async () => selectUser(1))
    await capture()
    await screen.findByDisplayValue("Alice private source")
    await screen.findByText("Alice private deck")
    await act(async () => finish([{ result: "Retired Alice capture" }]))
    expect(
      screen.queryByDisplayValue("Retired Alice capture")
    ).not.toBeInTheDocument()
    expect(screen.getAllByLabelText("Back", { exact: true })).toHaveLength(1)
  })
  it("retires a held generation without publishing cards into Bob's draft queue", async () => {
    const held = deferred()
    const fallback = state.request.getMockImplementation()!
    state.request.mockImplementation((input) =>
      input.path === "/api/v1/flashcards/generate"
        ? held.promise
        : fallback(input)
    )
    mount()
    await press("Generate draft cards")
    await waitFor(() =>
      expect(requests("/api/v1/flashcards/generate", "POST")).toHaveLength(1)
    )
    const request = requests("/api/v1/flashcards/generate", "POST")[0]
    expect(request.servicePromptConfig.expectedUserId).toBe(1)
    await act(async () => selectUser(2))
    await capture()
    await screen.findByDisplayValue("Bob current source")
    expect(request.abortSignal.aborted).toBe(true)
    await act(async () =>
      held.resolve({
        flashcards: [
          { front: "Retired generated front", back: "Retired generated answer" }
        ]
      })
    )
    expect(
      screen.queryByDisplayValue("Retired generated answer")
    ).not.toBeInTheDocument()
    expect(screen.getByDisplayValue("Bob current source")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Generate draft cards", exact: true })
    ).toBeEnabled()
  })

  it.each(["Save card", "Save all cards"])(
    "retires held %s and stops the old queue across an account boundary",
    async (action) => {
      const held = deferred()
      const fallback = state.request.getMockImplementation()!
      state.request.mockImplementation((input) =>
        input.path === "/api/v1/flashcards" && input.method === "POST"
          ? held.promise
          : fallback(input)
      )
      mount()
      await capture()
      await screen.findByDisplayValue("Alice private source")
      if (action === "Save all cards") {
        await capture()
        await waitFor(() =>
          expect(
            screen.getAllByDisplayValue("Alice private source")
          ).toHaveLength(2)
        )
      }
      await press(action)
      await waitFor(() =>
        expect(requests("/api/v1/flashcards", "POST")).toHaveLength(1)
      )
      const request = requests("/api/v1/flashcards", "POST")[0]
      expect(request.headers["X-TLDW-Expected-User-ID"]).toBe("1")
      await act(async () => selectUser(2))
      await capture()
      await screen.findByDisplayValue("Bob current source")
      await screen.findByText("Bob private deck")
      expect(request.abortSignal.aborted).toBe(true)
      await act(async () =>
        held.resolve({
          uuid: "00000000-0000-0000-0000-000000000001",
          version: 1
        })
      )
      expect(requests("/api/v1/flashcards", "POST")).toHaveLength(1)
      expect(screen.getByDisplayValue("Bob current source")).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: "Save card", exact: true })
      ).toBeEnabled()
      expect(
        screen.queryByText("Saved to Alice private deck.")
      ).not.toBeInTheDocument()
    }
  )

  it("does not display a held review read after the account changes", async () => {
    const held = deferred()
    const fallback = state.request.getMockImplementation()!
    state.request.mockImplementation((input) => {
      if (input.path.split("?")[0] === "/api/v1/flashcards/review/next")
        return state.user === 1 ? held.promise : { card: reviewCard(2) }
      return fallback(input)
    })
    mount()
    await press("Review due card")
    await waitFor(() =>
      expect(requests("/api/v1/flashcards/review/next")).toHaveLength(1)
    )
    const request = requests("/api/v1/flashcards/review/next")[0]
    await act(async () => selectUser(2))
    await openReview(2)
    expect(request.abortSignal.aborted).toBe(true)
    await act(async () => held.resolve({ card: reviewCard(1) }))
    expect(screen.queryByText(reviewCard(1).front)).not.toBeInTheDocument()
    expect(screen.getByText(reviewCard(2).front)).toBeInTheDocument()
    expect(
      requests("/api/v1/flashcards/review/next").at(-1).servicePromptConfig
        .expectedUserId
    ).toBe(2)
  })

  it("does not advance Bob's review when Alice's held rating completes", async () => {
    const held = deferred()
    const fallback = state.request.getMockImplementation()!
    state.request.mockImplementation((input) => {
      if (input.path.split("?")[0] === "/api/v1/flashcards/review/next")
        return { card: reviewCard(state.user) }
      if (input.path.includes("/review") && input.method === "POST")
        return held.promise
      return fallback(input)
    })
    mount()
    await openReview(1)
    await press("Reveal answer")
    await press("Good")
    await waitFor(() =>
      expect(
        state.request.mock.calls.filter(
          ([input]) => input.path.includes("/review") && input.method === "POST"
        )
      ).toHaveLength(1)
    )
    const request = state.request.mock.calls.find(
      ([input]) => input.path.includes("/review") && input.method === "POST"
    )![0]
    await act(async () => selectUser(2))
    await openReview(2)
    await press("Reveal answer")
    await act(async () => held.resolve({ ...reviewCard(1), version: 2 }))
    expect(request.abortSignal.aborted).toBe(true)
    expect(screen.getByText(reviewCard(2).back)).toBeInTheDocument()
    expect(
      screen.queryByText("Reviewed 1 card in this sidepanel session.")
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Good", exact: true })
    ).toBeEnabled()
  })

  it("clears the warmed template modal and loads Bob's templates when reopened", async () => {
    const fallback = state.request.getMockImplementation()!
    state.request.mockImplementation((input) => {
      if (input.path === "/api/v1/flashcards/templates")
        return {
          items: [
            {
              id: 7,
              name:
                state.user === 1
                  ? "Alice private template"
                  : "Bob current template",
              model_type: "basic",
              front_template: "Question",
              back_template: "Answer",
              placeholder_definitions: [],
              version: 1
            }
          ]
        }
      return fallback(input)
    })
    mount()
    await capture()
    await screen.findByDisplayValue("Alice private source")
    await press("Apply template to draft 1")
    await screen.findByText("Alice private template")
    await act(async () => selectUser(2))
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    )
    await capture()
    await screen.findByDisplayValue("Bob current source")
    await press("Apply template to draft 1")
    await screen.findByText("Bob current template")
    expect(screen.queryByText("Alice private template")).not.toBeInTheDocument()
    expect(
      requests("/api/v1/flashcards/templates").at(-1).servicePromptConfig
        .expectedUserId
    ).toBe(2)
  })

  it("aborts a held source read on unmount without creating a draft or transfer", async () => {
    const held = deferred()
    state.executeScript.mockReturnValueOnce(held.promise)
    const view = mount()
    await capture()
    await waitFor(() => expect(state.executeScript).toHaveBeenCalledTimes(1))
    view.unmount()
    await act(async () => held.resolve([{ result: "Retired source" }]))
    expect(screen.queryByDisplayValue("Retired source")).not.toBeInTheDocument()
    expect(requests("/api/v1/flashcards", "POST")).toHaveLength(0)
    expect(requests("/api/v1/flashcards/generate", "POST")).toHaveLength(0)
  })
  it("acquires a fresh lease after StrictMode effect replay and retires it on account replacement", async () => {
    mount(true)
    await capture()
    await screen.findByDisplayValue("Alice private source")
    await screen.findByText("Alice private deck")
    await act(async () => selectUser(2))
    await capture()
    await screen.findByDisplayValue("Bob current source")
    await screen.findByText("Bob private deck")
    expect(
      screen.queryByDisplayValue("Alice private source")
    ).not.toBeInTheDocument()
    expect(screen.queryByText("Alice private deck")).not.toBeInTheDocument()
  })

  it("aborts held generation on unmount and does not invalidate the retired cache", async () => {
    const held = deferred()
    const fallback = state.request.getMockImplementation()!
    state.request.mockImplementation((input) =>
      input.path === "/api/v1/flashcards/generate"
        ? held.promise
        : fallback(input)
    )
    const view = mount()
    await press("Generate draft cards")
    await waitFor(() =>
      expect(requests("/api/v1/flashcards/generate", "POST")).toHaveLength(1)
    )
    const request = requests("/api/v1/flashcards/generate", "POST")[0]
    const invalidate = vi.spyOn(clients.at(-1)!, "invalidateQueries")
    view.unmount()
    expect(request.abortSignal.aborted).toBe(true)
    await act(async () =>
      held.resolve({
        flashcards: [{ front: "Old question", back: "Old answer" }]
      })
    )
    expect(invalidate).not.toHaveBeenCalled()
    expect(requests("/api/v1/flashcards", "POST")).toHaveLength(0)
  })
})
