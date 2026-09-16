import React from "react"
import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { runInNewContext } from "node:vm"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { GeneratePanel } from "@/components/Flashcards/tabs/ImportExport/GeneratePanel"
import { useCreateDeckMutation, useCreateFlashcardMutation } from "@/components/Flashcards/hooks/useFlashcardQueries"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

const boundary = vi.hoisted(() => ({ get: vi.fn(), set: vi.fn(), fetch: vi.fn() }))
const messages = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: boundary.get, set: boundary.set, remove: vi.fn() }) }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => messages }))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useServerCapabilities", () => ({ useServerCapabilities: () => ({ capabilities: { hasFlashcards: true }, loading: false }) }))
vi.mock("@/services/prompt-studio", () => ({ getLlmProviders: async () => ({ providers: ["local"] }) }))
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

describe("generated saves through the real request and Next Pages error paths", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? {
      serverUrl: "https://source.test", authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: "1" }))}.signature`
    } : null)
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

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

  it.each(["deck", "card"] as const)("preserves the %s programming error rejection and Next diagnostic", async stage => {
    const observer = installNextPagesErrorObserver()
    const client = makeClient()
    try {
      const { result } = renderHook(() => ({ deck: useCreateDeckMutation(), card: useCreateFlashcardMutation() }), { wrapper: wrapper(client) })
      let caught: unknown
      await act(async () => {
        try {
          if (stage === "deck") await result.current.deck.mutateAsync({ name: null as unknown as string })
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
