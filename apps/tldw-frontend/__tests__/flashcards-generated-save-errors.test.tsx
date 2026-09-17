import React from "react"
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { runInNewContext } from "node:vm"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { GeneratePanel } from "@/components/Flashcards/tabs/ImportExport/GeneratePanel"
import { useCreateDeckMutation, useCreateFlashcardMutation, useGenerateFlashcardsMutation, useFlashcardAssistantQuery, useFlashcardAssistantRespondMutation, useReviewFlashcardMutation, useEndFlashcardReviewSessionMutation } from "@/components/Flashcards/hooks/useFlashcardQueries"
import { FlashcardStudyAssistantPanel } from "@/components/Flashcards/components/FlashcardStudyAssistantPanel"
import { useFlashcardReviewRun } from "@/components/Flashcards/hooks/useFlashcardReviewRun"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { StudyAssistantContextResponse, StudyAssistantRespondResponse } from "@/services/flashcards"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

const boundary = vi.hoisted(() => ({ get: vi.fn(), set: vi.fn(), fetch: vi.fn() }))
const messages = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: boundary.get, set: boundary.set, remove: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }), safeStorageSerde: { serialize: JSON.stringify, deserialize: JSON.parse } }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => messages }))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useServerCapabilities", () => ({ useServerCapabilities: () => ({ capabilities: { hasFlashcards: true }, loading: false }) }))
vi.mock("@/services/prompt-studio", () => ({ getLlmProviders: async () => ({ providers: ["local"] }) }))
vi.mock("@/hooks/useSpeechRecognition", () => ({ useSpeechRecognition: () => ({ supported: false, isListening: false, transcript: "", start: vi.fn(), stop: vi.fn(), resetTranscript: vi.fn() }) }))
vi.mock("@/hooks/useTTS", () => ({ useTTS: () => ({ speak: vi.fn(), cancel: vi.fn(), isSpeaking: false }) }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key }) }))

// Run the installed Pages Router handler unchanged, observing its dispatcher.
// Only developer-log forwarding is replaced; it must not open network traffic.
const installNextPagesErrorObserver = () => {
  const require = createRequire(import.meta.url)
  const filename = require.resolve("next/dist/next-devtools/userspace/pages/pages-dev-overlay-setup.js")
  const localRequire = createRequire(filename)
  const errors: unknown[] = []
  const rejections: unknown[] = []
  const unhandled: unknown[] = []
  const listeners: Array<[string, EventListener]> = []
  const captureUnhandled = (error: unknown) => { unhandled.push(error) }
  process.on("unhandledRejection", captureUnhandled)
  const originalError = console.error
  const originalAdd = window.addEventListener.bind(window)
  const add = vi.spyOn(window, "addEventListener").mockImplementation((type, listener, options) => {
    listeners.push([type, listener as EventListener])
    originalAdd(type, listener, options)
  })
  const loadedModule = { exports: {} as { register: () => void } }
  const load = runInNewContext(`(function(require,module,exports){${readFileSync(filename, "utf8")}\n})`, { window, console, process, Error })
  load((id: string) => id === "next/dist/compiled/next-devtools" ? {
    dispatcher: { onUnhandledError: (error: unknown) => errors.push(error), onUnhandledRejection: (error: unknown) => rejections.push(error) }
  } : id === "../app/forward-logs" ? {
    initializeDebugLogForwarding() {}, forwardErrorLog() {}, forwardUnhandledError() {}, logUnhandledRejection() {}
  } : localRequire(id), loadedModule, loadedModule.exports)
  loadedModule.exports.register()
  return {
    errors, rejections, unhandled,
    restore: () => {
      console.error = originalError
      for (const [type, listener] of listeners) window.removeEventListener(type, listener)
      add.mockRestore()
      process.removeListener("unhandledRejection", captureUnhandled)
    }
  }
}

const response = (data: unknown, status = 200) => new Response(JSON.stringify(data), {
  status, headers: { "Content-Type": "application/json" }
})
const source = { text: "Citrine volunteers meet Tuesday at 14:00.", sourceType: "note" as const, sourceId: "citrine-note", sourceTitle: "Citrine" }
const makeScope = () => {
  const controller = new AbortController()
  return {
    scopeKey: "owner-1", requestScope: { config: { serverUrl: "https://source.test", authMode: "multi-user" }, userId: 1 },
    scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal,
    capability: "unchecked", definitions: {}, release: () => controller.abort()
  } as ServicePromptSnapshot
}
const makeClient = () => new QueryClient({ defaultOptions: {
  queries: { retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false }, mutations: { retry: false }
} })
const wrapper = (client: QueryClient) => function Wrapper({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={client}><MemoryRouter>{children}</MemoryRouter></QueryClientProvider>
}

describe("Flashcards mutations through the real request and Next Pages error paths", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? {
      serverUrl: "https://source.test", authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: "1" }))}.signature`
    } : null)
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it("keeps a generation claim-verification 422 inline, retains options, and allows another generation", async () => {
    const sourceText = "The trial response rate was 10%."
    const guidance = "The generated cards could not be verified against your source. Add clearer source details or request fewer cards, then generate again."
    const failure = { detail: { code: "claim_verification_failed", claimVerification: {
      verdict: "failed", report: { numerical_error_count: 1, claims: [{
        claim_id: "flashcard:1:back:c1", claim_text: "The trial response rate was 90%.",
        status: "numerical_error", confidence: 0.85
      }] }, unitResults: [{ unit_id: "flashcard:1:back", verdict: "failed", statuses: ["numerical_error"] }]
    } } }
    let failing = true
    boundary.fetch.mockImplementation(async (url: string, init: RequestInit) => {
      const pathname = new URL(url).pathname
      if (pathname.endsWith("/decks") && init.method === "GET") return response([])
      if (pathname.endsWith("/generate") && init.method === "POST") return failing
        ? response(failure, 422)
        : response({ flashcards: [{ front: "What was the trial response rate?", back: "10%", tags: [], model_type: "basic" }], count: 1 })
      throw new Error(`Unexpected request: ${init.method} ${pathname}`)
    })
    const observer = installNextPagesErrorObserver()
    const warning = vi.spyOn(console, "warn")
    const client = makeClient()
    const scope = makeScope()
    const transfer = vi.fn()
    try {
      render(<GeneratePanel generationScope={scope} initialIntent={{ ...source, text: sourceText }} onTransferAction={transfer} />, { wrapper: wrapper(client) })
      await waitFor(() => expect(client.getQueryCache().getAll().find(query => query.queryKey[0] === "flashcards:decks:scoped")?.state.status).toBe("success"))
      fireEvent.change(screen.getByTestId("flashcards-generate-count"), { target: { value: "2" } })
      fireEvent.mouseDown(screen.getByTestId("flashcards-generate-card-type").querySelector("input")!)
      fireEvent.click(await screen.findByText("Basic (reverse)", { selector: ".ant-select-item-option-content" }))
      fireEvent.mouseDown(screen.getByTestId("flashcards-generate-difficulty").querySelector("input")!)
      fireEvent.click(await screen.findByText("Hard", { selector: ".ant-select-item-option-content" }))
      fireEvent.change(screen.getByTestId("flashcards-generate-focus-topics"), { target: { value: "trial, response rate" } })
      fireEvent.change(screen.getByTestId("flashcards-generate-provider"), { target: { value: "custom-openai-api-99" } })
      fireEvent.change(screen.getByTestId("flashcards-generate-model"), { target: { value: "uat024-claim-negative-control" } })
      fireEvent.click(screen.getByTestId("flashcards-generate-button"))
      expect(await screen.findByText(guidance)).toBeVisible()
      await waitFor(() => expect(screen.getByTestId("flashcards-generate-button")).toBeEnabled())
      expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(sourceText)
      expect(screen.getByTestId("flashcards-generate-count")).toHaveValue(2)
      expect(screen.getByTestId("flashcards-generate-card-type")).toHaveTextContent("Basic (reverse)")
      expect(screen.getByTestId("flashcards-generate-difficulty")).toHaveTextContent("Hard")
      expect(screen.getByTestId("flashcards-generate-focus-topics")).toHaveValue("trial, response rate")
      expect(screen.getByTestId("flashcards-generate-provider")).toHaveValue("custom-openai-api-99")
      expect(screen.getByTestId("flashcards-generate-model")).toHaveValue("uat024-claim-negative-control")
      expect(screen.queryByTestId("flashcards-generate-save-button")).not.toBeInTheDocument()
      expect(screen.queryByDisplayValue("What was the trial response rate?")).not.toBeInTheDocument()
      expect(document.body).not.toHaveTextContent("claim_verification_failed")
      expect(document.body).not.toHaveTextContent("numerical_error")
      expect(transfer).toHaveBeenCalledWith({ area: "generate", status: "error", message: guidance })
      const failedMutation = client.getMutationCache().getAll().find(mutation => mutation.state.status === "error")
      expect(failedMutation?.state.error).toMatchObject({ status: 422 })
      expect(observer.unhandled).toEqual([])
      expect(observer.rejections).toEqual([])
      expect(observer.errors).toEqual([])
      expect(warning).toHaveBeenCalledWith("Failed to generate flashcards:", failedMutation?.state.error)

      failing = false
      fireEvent.click(screen.getByTestId("flashcards-generate-button"))
      expect(await screen.findByDisplayValue("What was the trial response rate?")).toBeVisible()
      expect(screen.getByDisplayValue("10%")).toBeVisible()
      expect(screen.queryByText(guidance)).not.toBeInTheDocument()
      expect(screen.getByTestId("flashcards-generate-save-button")).toBeEnabled()
      const posts = boundary.fetch.mock.calls.filter(([, init]) => init.method === "POST")
      expect(posts).toHaveLength(2)
      for (const [url, init] of posts) {
        expect(url).toBe("https://source.test/api/v1/flashcards/generate")
        expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("1")
        expect(JSON.parse(init.body as string)).toEqual({ text: sourceText, num_cards: 2, card_type: "basic_reverse", difficulty: "hard", focus_topics: ["trial", "response rate"], provider: "custom-openai-api-99", model: "uat024-claim-negative-control" })
      }
      expect(observer.errors).toEqual([])
    } finally { observer.restore(); client.clear() }
  })

  it.each([["deck", 409], ["deck", 500], ["card", 422], ["card", 500]] as const)("keeps a %s HTTP %i save failure inline and retries the edited draft", async (stage, status) => {
    let failing = true
    let deckCreated = false
    const deck = { id: 12, name: "New Citrine deck", version: 1 }
    boundary.fetch.mockImplementation(async (url: string, init: RequestInit) => {
      const pathname = new URL(url).pathname
      if (pathname.endsWith("/decks") && init.method === "GET") return response(deckCreated ? [deck] : [])
      if (pathname.endsWith("/generate")) return response({ flashcards: [{ front: "When?", back: "Tuesday at 14:00", tags: ["Citrine"], model_type: "basic" }], count: 1 })
      if (pathname.endsWith("/decks") && init.method === "POST") {
        if (stage === "deck" && failing) return response({ detail: "Failed to create deck" }, status)
        deckCreated = true
        return response(deck, 201)
      }
      if (pathname === "/api/v1/flashcards" && init.method === "POST") {
        if (stage === "card" && failing) return response({ detail: "Failed to create flashcard" }, status)
        return response({ uuid: "created-card" }, 201)
      }
      throw new Error(`Unexpected request: ${init.method} ${pathname}`)
    })
    const observer = installNextPagesErrorObserver()
    const warning = vi.spyOn(console, "warn")
    const client = makeClient()
    const scope = makeScope()
    const postCalls = (suffix: string) => boundary.fetch.mock.calls.filter(([url, init]) => new URL(url).pathname.endsWith(suffix) && init.method === "POST")
    try {
      render(<GeneratePanel generationScope={scope} initialIntent={source} />, { wrapper: wrapper(client) })
      await waitFor(() => expect(client.getQueryCache().getAll().find(query => query.queryKey[0] === "flashcards:decks:scoped")?.state.status).toBe("success"))
      fireEvent.mouseDown(screen.getByTestId("flashcards-generate-deck").querySelector("input")!)
      fireEvent.click(await screen.findByText("Create new deck", { selector: ".ant-select-item-option-content" }))
      fireEvent.change(screen.getByTestId("flashcards-generate-new-deck-name"), { target: { value: deck.name } })
      fireEvent.click(screen.getByTestId("flashcards-generate-button"))
      fireEvent.change(await screen.findByDisplayValue("When?"), { target: { value: "Edited question" } })
      fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))
      await waitFor(() => expect(screen.getByTestId("flashcards-generate-save-retry")).toBeEnabled())
      expect(screen.getByDisplayValue("Edited question")).toBeVisible()
      expect(screen.getByTestId("flashcards-generate-save-status")).toHaveTextContent(stage === "deck" ? "Failed to create deck" : "Failed to save generated cards")
      const failedMutation = client.getMutationCache().getAll().find(mutation => mutation.state.status === "error")
      expect(failedMutation?.state.error).toMatchObject({ status })
      expect(observer.unhandled).toEqual([])
      expect(observer.rejections).toEqual([])
      expect(observer.errors).toEqual([])
      expect(warning).toHaveBeenCalledWith(expect.stringMatching(/^Failed to create /), failedMutation?.state.error)

      // The picker and draft stay usable; retry does not regenerate or recreate
      // the deck that was already acknowledged before a card failure.
      const picker = screen.getByTestId("flashcards-generate-deck").querySelector("input")!
      fireEvent.keyDown(picker, { key: "ArrowDown", keyCode: 40 })
      expect(picker).toHaveAttribute("aria-expanded", "true")
      fireEvent.keyDown(picker, { key: "Escape", keyCode: 27 })
      await waitFor(() => expect(picker).toHaveAttribute("aria-expanded", "false"))
      fireEvent.change(screen.getByDisplayValue("Edited question"), { target: { value: "Edited after failure" } })
      failing = false
      fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))
      await waitFor(() => expect(screen.queryByDisplayValue("Edited after failure")).not.toBeInTheDocument())
      expect(postCalls("/generate")).toHaveLength(1)
      expect(postCalls("/decks")).toHaveLength(stage === "deck" ? 2 : 1)
      const cards = postCalls("/flashcards")
      expect(cards).toHaveLength(stage === "card" ? 2 : 1)
      expect(JSON.parse(cards.at(-1)![1].body as string)).toMatchObject({ front: "Edited after failure", deck_id: 12, source_ref_type: "note", source_ref_id: source.sourceId })
      for (const [url, init] of boundary.fetch.mock.calls) {
        expect(new URL(url).origin).toBe("https://source.test")
        expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("1")
      }
      expect(observer.errors).toEqual([])
    } finally { observer.restore(); client.clear() }
  })

  it("keeps an assistant HTTP500 inline, retains the question and card, and accepts a successful retry", async () => {
    const cardUuid = "citrine-card"
    const question = "When do Citrine volunteers meet?"
    const answer = "Tuesday at 14:00."
    const context: StudyAssistantContextResponse = {
      thread: { id: 7, context_type: "flashcard", flashcard_uuid: cardUuid, quiz_attempt_id: null, question_id: null,
        last_message_at: null, message_count: 0, deleted: false, client_id: "1", version: 1,
        created_at: "2026-09-17T00:00:00Z", last_modified: "2026-09-17T00:00:00Z" },
      messages: [], context_snapshot: { flashcard: { uuid: cardUuid, front: question, back: answer } },
      available_actions: ["explain", "follow_up"]
    }
    const messageBase = { thread_id: 7, action_type: "follow_up" as const, input_modality: "text" as const,
      structured_payload: {}, context_snapshot: context.context_snapshot, provider: null, model: null,
      created_at: "2026-09-17T00:01:00Z", client_id: "1" }
    const successful: StudyAssistantRespondResponse = {
      thread: { ...context.thread, version: 3, message_count: 2, last_message_at: messageBase.created_at },
      user_message: { ...messageBase, id: 11, role: "user", content: question },
      assistant_message: { ...messageBase, id: 12, role: "assistant", content: answer },
      structured_payload: {}, context_snapshot: context.context_snapshot
    }
    let failing = true
    boundary.fetch.mockImplementation(async (url: string, init: RequestInit) => {
      const pathname = new URL(url).pathname
      if (pathname === `/api/v1/flashcards/${cardUuid}/assistant` && init.method === "GET") return response(context)
      if (pathname === `/api/v1/flashcards/${cardUuid}/assistant/respond` && init.method === "POST") {
        return failing ? response({ detail: "Internal server error" }, 500) : response(successful)
      }
      throw new Error(`Unexpected request: ${init.method} ${pathname}`)
    })
    function Assistant() {
      const query = useFlashcardAssistantQuery(cardUuid)
      const mutation = useFlashcardAssistantRespondMutation()
      return <FlashcardStudyAssistantPanel cardUuid={cardUuid} threadVersion={query.data?.thread.version}
        messages={query.data?.messages ?? []} assistantContext={query.data?.context_snapshot}
        availableActions={query.data?.available_actions} isLoading={query.isLoading} defaultExpanded
        isResponding={mutation.isPending} onRespond={request => mutation.mutateAsync({ cardUuid, request })} />
    }
    const observer = installNextPagesErrorObserver()
    const warning = vi.spyOn(console, "warn")
    const client = makeClient()
    try {
      render(<Assistant />, { wrapper: wrapper(client) })
      await waitFor(() => expect(client.getQueryData(["flashcards:assistant", cardUuid])).toEqual(context))
      const input = screen.getByRole("textbox", { name: "Ask the study assistant" })
      fireEvent.change(input, { target: { value: question } })
      fireEvent.click(screen.getByRole("button", { name: "Ask assistant" }))
      expect(await screen.findByText("The server returned an error. Please try again or check server logs.")).toBeVisible()
      expect(input).toHaveValue(question)
      await waitFor(() => expect(screen.getByRole("button", { name: "Ask assistant" })).toBeEnabled())
      expect(client.getQueryData(["flashcards:assistant", cardUuid])).toEqual(context)
      const failedMutation = client.getMutationCache().getAll().find(mutation => mutation.state.status === "error")
      expect(failedMutation?.state.error).toMatchObject({ status: 500 })
      expect(observer.unhandled).toEqual([])
      expect(observer.rejections).toEqual([])
      expect(observer.errors).toEqual([])
      expect(warning).toHaveBeenCalledWith("Failed to respond with flashcard assistant:", failedMutation?.state.error)

      failing = false
      fireEvent.click(screen.getByRole("button", { name: "Ask assistant" }))
      expect(await screen.findByText(answer)).toBeVisible()
      expect(input).toHaveValue("")
      expect(screen.queryByText("The server returned an error. Please try again or check server logs.")).not.toBeInTheDocument()
      expect(client.getQueryData<StudyAssistantContextResponse>(["flashcards:assistant", cardUuid])?.messages.map(message => message.id)).toEqual([11, 12])
      const posts = boundary.fetch.mock.calls.filter(([, init]) => init.method === "POST")
      expect(posts).toHaveLength(2)
      for (const [url, init] of posts) {
        expect(url).toBe(`https://source.test/api/v1/flashcards/${cardUuid}/assistant/respond`)
        expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer test.${btoa(JSON.stringify({ sub: "1" }))}.signature`)
        expect(JSON.parse(init.body as string)).toEqual({ action: "follow_up", message: question, input_modality: "text", expected_thread_version: 1 })
      }
      expect(observer.errors).toEqual([])
    } finally { observer.restore(); client.clear() }
  })

  it("preserves a scheduled rating after HTTP500 and retries within the real review run without a Next overlay", async () => {
    const config = { serverUrl: "https://source.test", authMode: "multi-user" as const, accessToken: `test.${btoa(JSON.stringify({ sub: "1" }))}.signature` }
    vi.spyOn(tldwClient, "initialize").mockResolvedValue()
    vi.spyOn(tldwClient, "getConfig").mockResolvedValue(config)
    vi.spyOn(tldwClient, "ensureConfigForRequest").mockResolvedValue(config)
    let failing = true
    boundary.fetch.mockImplementation(async (url: string, init: RequestInit) => {
      const pathname = new URL(url).pathname
      if (pathname.endsWith("/auth/me") || pathname.endsWith("/api/auth/session")) return response({ id: 1, is_active: true, authenticated: true, user: { id: 1, is_active: true } })
      if (pathname === "/api/v1/flashcards/review") return failing ? response({ detail: "Internal server error" }, 500) : response({ review_session_id: 77, interval_days: 1 })
      if (pathname === "/api/v1/flashcards/review-sessions/end") return response({ id: 77, status: "completed", cards_reviewed: 1 })
      throw new Error(`Unexpected request: ${init.method} ${pathname}`)
    })
    const client = makeClient()
    const invalidation = vi.spyOn(client, "invalidateQueries")
    const observer = installNextPagesErrorObserver()
    const warning = vi.spyOn(console, "warn")
    const closeError = vi.fn()
    const context = { review_mode: "due" as const, deck_id: 1, tag_filter: null }
    const rating = { cardUuid: "citrine-card", rating: 3, answerTimeMs: 250 }
    const view = renderHook(() => {
      const review = useReviewFlashcardMutation()
      const end = useEndFlashcardReviewSessionMutation()
      return useFlashcardReviewRun({ context, enabled: true, review: review.mutateAsync, end: end.mutateAsync, onCloseError: closeError })
    }, { wrapper: wrapper(client) })
    try {
      let caught: unknown
      await act(async () => { try { await view.result.current.submit(rating) } catch (error) { caught = error } })
      expect(caught).toMatchObject({ status: 500 })
      expect(view.result.current.activeSessionId).toBeNull()
      expect(invalidation).not.toHaveBeenCalled()
      expect(observer.unhandled).toEqual([])
      expect(observer.rejections).toEqual([])
      expect(observer.errors).toEqual([])
      expect(warning).toHaveBeenCalledWith("Failed to submit flashcard review:", caught)
      failing = false
      await act(async () => { expect(await view.result.current.submit(rating)).toMatchObject({ review_session_id: 77 }) })
      expect(view.result.current.activeSessionId).toBe(77)
      expect(invalidation).toHaveBeenCalledTimes(1)
      const posts = boundary.fetch.mock.calls.filter(([url]) => new URL(url).pathname === "/api/v1/flashcards/review")
      expect(posts).toHaveLength(2)
      for (const [, init] of posts) {
        expect(JSON.parse(init.body as string)).toEqual({ card_uuid: rating.cardUuid, rating: 3, answer_time_ms: 250, review_context: context })
        expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("1")
      }
      expect(observer.errors).toEqual([])
    } finally {
      await act(async () => view.unmount())
      expect(closeError).not.toHaveBeenCalled()
      observer.restore(); client.clear()
    }
  })

  it.each(["deck", "card", "generation", "assistant", "rating"] as const)("preserves the %s programming error rejection and Next diagnostic", async stage => {
    const observer = installNextPagesErrorObserver()
    const client = makeClient()
    try {
      const { result } = renderHook(() => ({ deck: useCreateDeckMutation(), card: useCreateFlashcardMutation(), generation: useGenerateFlashcardsMutation(), assistant: useFlashcardAssistantRespondMutation(), rating: useReviewFlashcardMutation() }), { wrapper: wrapper(client) })
      let caught: unknown
      await act(async () => {
        try {
          if (stage === "deck") await result.current.deck.mutateAsync({ name: null as unknown as string })
          else if (stage === "generation") await result.current.generation.mutateAsync({ get text(): string { throw new TypeError("Invalid source getter") } })
          else if (stage === "assistant") await result.current.assistant.mutateAsync({ cardUuid: "citrine-card", get request(): never { throw new TypeError("Invalid assistant request getter") } })
          else if (stage === "rating") await result.current.rating.mutateAsync({ get cardUuid(): never { throw new TypeError("Invalid rating card getter") }, rating: 3 })
          else await result.current.card.mutateAsync({ get front(): string { throw new TypeError("Invalid draft getter") }, back: "A" })
        } catch (error) { caught = error }
      })
      expect(caught).toBeInstanceOf(TypeError)
      expect(observer.errors).toEqual([caught])
      expect(observer.rejections).toEqual([])
      expect(observer.unhandled).toEqual([])
      expect(boundary.fetch).not.toHaveBeenCalled()
    } finally { observer.restore(); client.clear() }
  })
})
