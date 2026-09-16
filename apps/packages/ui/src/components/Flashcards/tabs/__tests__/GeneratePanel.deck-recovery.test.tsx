import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { GeneratePanel } from "../ImportExport/GeneratePanel"
import { createDeck, createFlashcard, generateFlashcards, listDecks, type Deck } from "@/services/flashcards"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

const messages = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn() }))
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => messages }))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasFlashcards: true }, loading: false })
}))
vi.mock("@/services/prompt-studio", () => ({
  getLlmProviders: vi.fn(async () => ({ providers: ["local"] }))
}))
vi.mock("@/services/flashcards", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/flashcards")>(),
  listDecks: vi.fn(),
  generateFlashcards: vi.fn(),
  createDeck: vi.fn(),
  createFlashcard: vi.fn()
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key
  })
}))

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}

const makeScope = (userId = 1) => {
  const controller = new AbortController()
  return {
    controller,
    scope: {
      scopeKey: `owner-${userId}`,
      requestScope: { config: { serverUrl: "https://owner.test", authMode: "multi-user" }, userId },
      scopeSignal: controller.signal,
      scopeInvalidatedSignal: controller.signal,
      capability: "unchecked",
      definitions: {},
      release: () => controller.abort()
    } as ServicePromptSnapshot
  }
}
const deck = (id: number, name = "Alice deck") => ({ id, name, version: 1 } as Deck)
const source = {
  text: "Citrine study volunteers meet every Tuesday at 14:00.",
  sourceType: "note" as const,
  sourceId: "citrine-note",
  sourceTitle: "Citrine study"
}

const mount = (scope: ServicePromptSnapshot | null = makeScope().scope) => {
  const client = new QueryClient({ defaultOptions: {
    queries: { retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false },
    mutations: { retry: false }
  } })
  const view = (owner: ServicePromptSnapshot | null) => (
    <QueryClientProvider client={client}>
      <MemoryRouter><GeneratePanel generationScope={owner} initialIntent={source} /></MemoryRouter>
    </QueryClientProvider>
  )
  const result = render(view(scope))
  const deckQuery = () => client.getQueryCache().getAll().find(query => query.queryKey[0] === "flashcards:decks:scoped" && query.queryKey[1] === scope?.scopeKey)
  return { ...result, client, deckQuery, changeOwner: (owner: ServicePromptSnapshot | null) => result.rerender(view(owner)) }
}
const generate = async () => {
  fireEvent.click(screen.getByTestId("flashcards-generate-button"))
  await screen.findByDisplayValue("When do volunteers meet?")
}
const save = () => fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))
const failedSave = async () => {
  save()
  await waitFor(() => expect(screen.getByTestId("flashcards-generate-save-retry")).toBeEnabled())
  expect(createDeck).not.toHaveBeenCalled()
  expect(createFlashcard).not.toHaveBeenCalled()
}

describe("GeneratePanel recovery through its scoped deck query", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(listDecks).mockReset().mockRejectedValue(new Error("Deck service unavailable"))
    vi.mocked(generateFlashcards).mockReset().mockResolvedValue({
      flashcards: [{ front: "When do volunteers meet?", back: "Tuesday at 14:00", tags: ["Citrine"], model_type: "basic" }],
      count: 1
    })
    vi.mocked(createDeck).mockReset().mockResolvedValue(deck(12, "Generated Flashcards"))
    vi.mocked(createFlashcard).mockReset().mockResolvedValue({ uuid: "created-card" } as Awaited<ReturnType<typeof createFlashcard>>)
  })

  it("recovers via visible Retry and saves the edited draft with its original account and source", async () => {
    const { scope } = makeScope()
    const { deckQuery } = mount(scope)
    await waitFor(() => expect(deckQuery()?.state.status).toBe("error"))
    await generate()
    fireEvent.change(screen.getByDisplayValue("When do volunteers meet?"), { target: { value: "Edited question" } })
    await failedSave()
    expect(screen.getByDisplayValue("Edited question")).toBeVisible()
    expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(source.text)
    const recoveryRead = deferred<Deck[]>()
    vi.mocked(listDecks).mockReturnValueOnce(recoveryRead.promise).mockResolvedValue([deck(7)])
    const beforeRetry = vi.mocked(listDecks).mock.calls.length
    fireEvent.click(screen.getByTestId("flashcards-generate-save-retry"))
    await waitFor(() => expect(listDecks).toHaveBeenCalledTimes(beforeRetry + 1))
    expect(createFlashcard).not.toHaveBeenCalled()
    save()
    expect(listDecks).toHaveBeenCalledTimes(beforeRetry + 1)
    await act(async () => { recoveryRead.resolve([deck(7)]) })
    await waitFor(() => expect(createFlashcard).toHaveBeenCalledTimes(1))
    expect(createFlashcard).toHaveBeenCalledWith(expect.objectContaining({
      deck_id: 7, front: "Edited question", back: "Tuesday at 14:00", tags: ["Citrine"],
      source_ref_type: "note", source_ref_id: "citrine-note"
    }), { requestScope: scope.requestScope, signal: scope.scopeSignal })
    expect(createDeck).not.toHaveBeenCalled()
    expect(generateFlashcards).toHaveBeenCalledTimes(1)
    for (const [, options] of vi.mocked(listDecks).mock.calls) {
      expect(options?.requestScope).toBe(scope.requestScope)
    }
    await waitFor(() => expect(screen.queryByDisplayValue("Edited question")).not.toBeInTheDocument())
  })

  it("retains drafts and clears busy state when Retry still cannot read the deck list", async () => {
    const { deckQuery } = mount()
    await waitFor(() => expect(deckQuery()?.state.status).toBe("error"))
    await generate()
    await failedSave()
    const beforeRetry = vi.mocked(listDecks).mock.calls.length
    fireEvent.click(screen.getByTestId("flashcards-generate-save-retry"))
    await waitFor(() => expect(listDecks).toHaveBeenCalledTimes(beforeRetry + 1))
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-save-retry")).toBeEnabled())
    expect(screen.getByTestId("flashcards-generate-save-status")).toHaveTextContent("Deck service unavailable")
    expect(screen.getByDisplayValue("When do volunteers meet?")).toBeVisible()
    expect(createDeck).not.toHaveBeenCalled()
    expect(createFlashcard).not.toHaveBeenCalled()
  })

  it("does not restart or bypass an initially pending deck list", async () => {
    const pending = deferred<Deck[]>()
    vi.mocked(listDecks).mockReturnValue(pending.promise)
    mount()
    await generate()
    await failedSave()
    expect(listDecks).toHaveBeenCalledTimes(1)
    expect(screen.getByTestId("flashcards-generate-save-status")).toHaveTextContent("Wait for the current account's decks")
  })

  it.each(["aborted", "changed", "unresolved"])("rejects a recovered list after the account becomes %s", async transition => {
    const { scope, controller } = makeScope()
    const { deckQuery, changeOwner } = mount(scope)
    await waitFor(() => expect(deckQuery()?.state.status).toBe("error"))
    await generate()
    await failedSave()
    const recoveryRead = deferred<Deck[]>()
    vi.mocked(listDecks).mockReturnValueOnce(recoveryRead.promise).mockResolvedValue([deck(99, "Other account")])
    const beforeRetry = vi.mocked(listDecks).mock.calls.length
    fireEvent.click(screen.getByTestId("flashcards-generate-save-retry"))
    await waitFor(() => expect(listDecks).toHaveBeenCalledTimes(beforeRetry + 1))
    if (transition === "aborted") controller.abort()
    else changeOwner(transition === "unresolved" ? null : makeScope(2).scope)
    await act(async () => { recoveryRead.resolve([deck(7)]) })
    expect(createDeck).not.toHaveBeenCalled()
    expect(createFlashcard).not.toHaveBeenCalled()
    expect(messages.success).toHaveBeenCalledTimes(1) // Generation only, never save success.
  })

  it.each(["aborted", "unresolved"])("does not retry a read when the account is already %s", async transition => {
    const { scope, controller } = makeScope()
    const { deckQuery, changeOwner } = mount(scope)
    await waitFor(() => expect(deckQuery()?.state.status).toBe("error"))
    await generate()
    await failedSave()
    const beforeRetry = vi.mocked(listDecks).mock.calls.length
    if (transition === "aborted") controller.abort()
    else changeOwner(null)
    await act(async () => { fireEvent.click(screen.getByTestId("flashcards-generate-save-retry")) })
    expect(listDecks).toHaveBeenCalledTimes(beforeRetry)
    expect(createDeck).not.toHaveBeenCalled()
    expect(createFlashcard).not.toHaveBeenCalled()
  })

  it("rejects a selected deck deleted from the successfully recovered catalogue", async () => {
    vi.mocked(listDecks).mockResolvedValue([deck(7)])
    const { client, deckQuery } = mount()
    await waitFor(() => expect(deckQuery()?.state.status).toBe("success"))
    await generate()
    vi.mocked(listDecks).mockRejectedValue(new Error("Deck service unavailable"))
    await act(async () => {
      await client.refetchQueries({ queryKey: ["flashcards:decks:scoped"] })
      // Query notifications are batched into the next task; let the panel see the failure before Save.
      await new Promise(resolve => setTimeout(resolve, 0))
    })
    await failedSave()
    vi.mocked(listDecks).mockResolvedValue([])
    fireEvent.click(screen.getByTestId("flashcards-generate-save-retry"))
    await waitFor(() => expect(screen.getByTestId("flashcards-generate-save-status")).toHaveTextContent("Choose a deck from the current account"))
    expect(createDeck).not.toHaveBeenCalled()
    expect(createFlashcard).not.toHaveBeenCalled()
    expect(screen.getByDisplayValue("When do volunteers meet?")).toBeVisible()
  })

  it("creates a deck once after recovering an empty current-account catalogue", async () => {
    const { scope } = makeScope()
    const { deckQuery } = mount(scope)
    await waitFor(() => expect(deckQuery()?.state.status).toBe("error"))
    await generate()
    await failedSave()
    vi.mocked(listDecks).mockResolvedValue([])
    fireEvent.click(screen.getByTestId("flashcards-generate-save-retry"))
    await waitFor(() => expect(createFlashcard).toHaveBeenCalledTimes(1))
    expect(createDeck).toHaveBeenCalledTimes(1)
    expect(createDeck).toHaveBeenCalledWith(expect.objectContaining({ name: "Generated Flashcards" }), {
      requestScope: scope.requestScope, signal: scope.scopeSignal
    })
    expect(createFlashcard).toHaveBeenCalledWith(expect.objectContaining({ deck_id: 12 }), {
      requestScope: scope.requestScope, signal: scope.scopeSignal
    })
  })

  it("uses a ready list without another read before the card save", async () => {
    vi.mocked(listDecks).mockResolvedValue([deck(7)])
    const { deckQuery } = mount()
    await waitFor(() => expect(deckQuery()?.state.status).toBe("success"))
    await generate()
    const pendingSave = deferred<Awaited<ReturnType<typeof createFlashcard>>>()
    vi.mocked(createFlashcard).mockReturnValue(pendingSave.promise)
    save()
    await waitFor(() => expect(createFlashcard).toHaveBeenCalledTimes(1))
    expect(listDecks).toHaveBeenCalledTimes(1)
    expect(createDeck).not.toHaveBeenCalled()
  })
})
