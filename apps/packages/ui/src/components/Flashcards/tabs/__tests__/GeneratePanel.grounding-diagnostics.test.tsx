import React from "react"
import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { GeneratePanel } from "../ImportExport/GeneratePanel"
import { mapFlashcardsUiError } from "../../utils/error-taxonomy"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }),
  safeStorageSerde: { serialize: JSON.stringify, deserialize: JSON.parse }
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => ({ success: vi.fn(), error: vi.fn() }) }))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasFlashcards: true }, loading: false })
}))
vi.mock("@/services/prompt-studio", () => ({ getLlmProviders: async () => ({ providers: ["ollama"] }) }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key })
}))

// The HTTP failure shape comes from ArtifactVerificationResult.to_dict and
// VerificationReport.to_dict; only the provider's response is doubled.
const failure = {
  detail: {
    code: "claim_verification_failed",
    claimVerification: {
      verdict: "failed",
      report: {
        report_id: "private-report-id",
        claims: [{
          claim_id: "flashcard:1:back:c1",
          claim_text: "Question: What was the response rate?\nAnswer: 90%.",
          status: "numerical_error",
          confidence: 0.85,
          rationale: "The source reports 10%, not 90%.",
          evidence: [{ doc_id: "private-source-id", snippet: "The trial response rate was 10%.", score: 0.9, authority: "primary" }],
          citations: [{ private_token: "do-not-display" }]
        }, {
          claim_id: "flashcard:2:front:c1", claim_text: "The trial enrolled 20 people.",
          status: "verified", confidence: 0.9, rationale: "The source states the enrollment.", evidence: [], citations: []
        }, {
          claim_id: "flashcard:2:notes:c1", claim_text: "The treatment cures everyone.",
          status: "unverified", confidence: 0, rationale: "The question and answer require semantic evidence support.", evidence: [], citations: []
        }, {
          claim_id: "flashcard:2:extra:c1", claim_text: "All participants improved.",
          status: "contested", confidence: 0.4, rationale: null, evidence: [], citations: []
        }]
      },
      unitResults: [
        { unit_id: "flashcard:1:back", verdict: "failed", claim_ids: ["flashcard:1:back:c1"], statuses: ["numerical_error"], metadata: {} },
        { unit_id: "flashcard:2:front", verdict: "grounded", claim_ids: ["flashcard:2:front:c1"], statuses: ["verified"], metadata: {} },
        { unit_id: "flashcard:2:notes", verdict: "needs_revision", claim_ids: ["flashcard:2:notes:c1"], statuses: ["unverified"], metadata: {} },
        { unit_id: "flashcard:2:extra", verdict: "needs_revision", claim_ids: ["flashcard:2:extra:c1"], statuses: ["contested"], metadata: {} }
      ],
      metadata: {
        generation_provider: "ollama", generation_model: "generation-model",
        verification_provider: "custom-openai-api", verification_model: "verification-model",
        verification_llm_source: "config", api_key: "do-not-display"
      }
    }
  }
}
const response = (data: unknown, status = 200) => new Response(JSON.stringify(data), {
  status, headers: { "Content-Type": "application/json" }
})
const source = { text: "The trial response rate was 10%. The trial enrolled 20 people.", sourceType: "note" as const, sourceId: "trial-note" }
const makeScope = (userId = 1) => {
  const controller = new AbortController()
  const invalidated = new AbortController()
  return {
    controller,
    invalidated,
    scope: {
      scopeKey: `owner-${userId}`,
      requestScope: { config: { serverUrl: "https://source.test", authMode: "multi-user" }, userId },
      scopeSignal: controller.signal, scopeInvalidatedSignal: invalidated.signal,
      capability: "unchecked", definitions: {}, release: () => controller.abort()
    } as ServicePromptSnapshot
  }
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
const mount = () => {
  const owner = makeScope()
  const transfer = vi.fn()
  const client = new QueryClient({ defaultOptions: {
    queries: { retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false }, mutations: { retry: false }
  } })
  const view = (scope: ServicePromptSnapshot | null, intent = source, sourceReviewIntent?: React.ComponentProps<typeof GeneratePanel>["sourceReviewIntent"]) => (
    <QueryClientProvider client={client}><MemoryRouter>
      <GeneratePanel generationScope={scope} initialIntent={intent} sourceReviewIntent={sourceReviewIntent} onTransferAction={transfer} />
    </MemoryRouter></QueryClientProvider>
  )
  const result = render(view(owner.scope))
  return { ...result, ...owner, client, transfer, change: (scope: ServicePromptSnapshot | null, intent = source, sourceReviewIntent?: React.ComponentProps<typeof GeneratePanel>["sourceReviewIntent"]) => result.rerender(view(scope, intent, sourceReviewIntent)) }
}
const generate = () => fireEvent.click(screen.getByTestId("flashcards-generate-button"))
const explanation = () => screen.getByText("Why these cards were not saved").closest("details")!

describe("current-attempt Flashcards grounding diagnostics", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? {
      serverUrl: "https://source.test", authMode: "multi-user",
      accessToken: `test.${btoa(JSON.stringify({ sub: "1" }))}.signature`
    } : null)
    boundary.fetch.mockImplementation(async (url: string, init: RequestInit) => {
      if (new URL(url).pathname.endsWith("/decks") && init.method === "GET") return response([])
      if (new URL(url).pathname.endsWith("/generate") && init.method === "POST") return response(failure, 422)
      throw new Error(`Unexpected request: ${init.method} ${url}`)
    })
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => { vi.unstubAllGlobals() })

  it("shows collapsed human-readable card claims and separate model labels from the retained HTTP failure", async () => {
    const { client } = mount()
    generate()
    await screen.findByText("Why these cards were not saved")
    const details = explanation()
    expect(details).not.toHaveAttribute("open")
    fireEvent.click(within(details).getByText("Why these cards were not saved"))
    // jsdom does not implement summary's native toggle; the content is still
    // inspected through the actual details subtree.
    expect(details).toHaveTextContent("Failed")
    expect(details).toHaveTextContent("Card 1 · Back")
    expect(details).toHaveTextContent("Card 2 · Front")
    expect(details).toHaveTextContent("Card 2 · Notes")
    expect(details).toHaveTextContent("Card 2 · Extra")
    expect(details).toHaveTextContent("Numerical error")
    expect(details).toHaveTextContent("Unverified")
    expect(details).toHaveTextContent("The source reports 10%, not 90%.")
    expect(details).toHaveTextContent("The trial response rate was 10%.")
    expect(details).toHaveTextContent("Generation model: ollama / generation-model")
    expect(details).toHaveTextContent("Verification model: custom-openai-api / verification-model")
    expect(details).not.toHaveTextContent("private-report-id")
    expect(details).not.toHaveTextContent("private-source-id")
    expect(details).not.toHaveTextContent("flashcard:")
    expect(details).not.toHaveTextContent("do-not-display")
    expect(details).not.toHaveTextContent("provider failure")
    expect(screen.queryByTestId("flashcards-generate-save-button")).not.toBeInTheDocument()
    expect(client.getMutationCache().getAll().find(mutation => mutation.state.status === "error")?.state.error).toMatchObject({ status: 422 })
    expect(boundary.fetch.mock.calls.filter(([, init]) => init.method === "POST")).toHaveLength(1)
  })

  it("removes previous diagnostics as soon as the next generation starts", async () => {
    mount()
    generate()
    await screen.findByText("Why these cards were not saved")
    const next = deferred<Response>()
    boundary.fetch.mockImplementation(async (url: string, init: RequestInit) => {
      if (init.method === "GET") return response([])
      if (new URL(url).pathname.endsWith("/generate")) return next.promise
      throw new Error("Unexpected save")
    })
    generate()
    expect(screen.queryByText("Why these cards were not saved")).not.toBeInTheDocument()
    await act(async () => { next.resolve(response({ flashcards: [], count: 0 })) })
  })

  it.each(["abort", "invalidate", "change", "unresolved", "source", "source review"] as const)("removes retained private details when the context becomes %s", async transition => {
    const { scope, controller, invalidated, change } = mount()
    generate()
    await screen.findByText("Why these cards were not saved")
    await act(async () => {
      if (transition === "abort") controller.abort()
      else if (transition === "invalidate") invalidated.abort()
      else if (transition === "change") change(makeScope(2).scope)
      else if (transition === "unresolved") change(null)
      else if (transition === "source") change(scope, { ...source, text: "A different source.", sourceId: "different-note" })
      else change(scope, source, { activity_type: "flashcards", text: "A source review.", source_items: [] })
    })
    expect(screen.queryByText("Why these cards were not saved")).not.toBeInTheDocument()
    expect(document.body).not.toHaveTextContent("The source reports 10%, not 90%.")
  })

  it("does not publish a late failure from the previous source handoff", async () => {
    const pending = deferred<Response>()
    boundary.fetch.mockImplementation(async (_url: string, init: RequestInit) => init.method === "GET" ? response([]) : pending.promise)
    const { scope, change, transfer } = mount()
    generate()
    change(scope, { ...source, text: "A different source.", sourceId: "different-note" })
    await act(async () => { pending.resolve(response(failure, 422)) })
    expect(screen.queryByText("Why these cards were not saved")).not.toBeInTheDocument()
    expect(document.body).not.toHaveTextContent("The source reports 10%, not 90%.")
    expect(transfer).not.toHaveBeenCalled()
  })

  it("labels an unknown status neutrally without interpreting it as a provider failure", () => {
    const report = structuredClone(failure)
    report.detail.claimVerification.report.claims[0].status = "private-provider-failure"
    const mapped = mapFlashcardsUiError(Object.assign(new Error("claim_verification_failed"), { details: report }), {
      operation: "generating cards", fallback: "Generation failed"
    })
    expect(mapped.grounding?.claims[0].status).toBe("Not reported")
    expect(JSON.stringify(mapped.grounding)).not.toContain("private-provider-failure")
  })

  it("projects only known report fields and excludes malformed unit links", () => {
    const report = structuredClone(failure)
    report.detail.claimVerification.unitResults.push({
      unit_id: "private-unit", verdict: "failed", claim_ids: ["flashcard:1:back:c1"], statuses: ["numerical_error"], metadata: {}
    })
    const mapped = mapFlashcardsUiError(Object.assign(new Error("claim_verification_failed"), { status: 422, details: report }), {
      operation: "generating cards", fallback: "Generation failed"
    })
    expect(mapped).toHaveProperty("grounding.verdict", "Failed")
    expect(mapped).toHaveProperty("grounding.claims.length", 4)
    expect(JSON.stringify(mapped.grounding)).not.toMatch(/private-|do-not-display|flashcard:/)
  })

  it.each([null, { verdict: "failed", report: { claims: "malformed" }, unitResults: [] }])("keeps malformed or missing report data on the existing general error path", claimVerification => {
    const mapped = mapFlashcardsUiError(Object.assign(new Error("claim_verification_failed"), {
      details: { detail: { code: "claim_verification_failed", claimVerification } }
    }), { operation: "generating cards", fallback: "Generation failed" })
    expect(mapped.code).toBe("FLASHCARDS_GROUNDING")
    expect(mapped.message).toMatch(/source/i)
    expect(mapped).not.toHaveProperty("grounding")
  })
})
