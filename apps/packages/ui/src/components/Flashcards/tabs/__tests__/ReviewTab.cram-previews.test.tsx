import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { bgRequest } from "@/services/background-proxy"
import { clearSetting } from "@/services/settings/registry"
import { FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING } from "@/services/settings/ui-settings"
import { ReviewTab } from "../ReviewTab"

const state = vi.hoisted(() => ({ review: vi.fn(), end: vi.fn(), previewAlwaysPresent: false }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn(), bgUpload: vi.fn() }))
vi.mock("@/services/service-prompts", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  ...await import("./review-scope-fixture")
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasFlashcards: true }, loading: false })
}))
vi.mock("react-router-dom", async importOriginal => ({
  ...await importOriginal<typeof import("react-router-dom")>(), useNavigate: () => vi.fn()
}))
vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next")
  const { default: ICU } = await import("@/i18n/icu-format")
  const instance = createInstance().use(ICU)
  await instance.init({ lng: "en", fallbackLng: false, resources: {} })
  return { useTranslation: () => ({ t: instance.t.bind(instance) }) }
})
vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() })
}))
vi.mock("@/hooks/useTTS", () => ({ useTTS: () => ({ speak: vi.fn(), cancel: vi.fn(), isSpeaking: false }) }))
vi.mock("@/hooks/useSpeechRecognition", () => ({
  useSpeechRecognition: () => ({ supported: false, isListening: false, transcript: "", resetTranscript: vi.fn() })
}))
vi.mock("@/components/StudySuggestions/StudySuggestionsPanel", () => ({ StudySuggestionsPanel: () => null }))
vi.mock("../../hooks", async importOriginal => {
  const actual = await importOriginal<typeof import("../../hooks")>()
  const emptyQuery = () => ({ data: null, isLoading: false, isError: false, refetch: vi.fn() })
  const emptyMutation = () => ({ mutateAsync: vi.fn(), isPending: false })
  return {
    ...actual,
    // Preserve the real Cram query, pagination, service and URL serializer.
    useDecksQuery: () => ({ data: [{ id: 1, name: "Preview deck" }] }),
    useReviewQuery: emptyQuery,
    useReviewFlashcardMutation: () => ({ mutateAsync: state.review, isPending: false }),
    useEndFlashcardReviewSessionMutation: () => ({ mutateAsync: state.end, isPending: false }),
    useRecentFlashcardReviewSessionsQuery: () => ({ data: [], isLoading: false }),
    useGlobalFlashcardTagSuggestionsQuery: () => ({ data: { items: [] } }),
    useFlashcardAssistantQuery: emptyQuery,
    useFlashcardAssistantRespondMutation: emptyMutation,
    useUpdateFlashcardMutation: emptyMutation,
    useResetFlashcardSchedulingMutation: emptyMutation,
    useDeleteFlashcardMutation: emptyMutation,
    useFlashcardShortcuts: vi.fn(),
    useDebouncedFormField: vi.fn(),
    useDueCountsQuery: () => ({ data: { due: 0, new: 1, learning: 0, total: 1 } }),
    useDeckDueCountsQuery: () => ({ data: {} }),
    useReviewAnalyticsSummaryQuery: emptyQuery,
    useHasCardsQuery: () => ({ data: true }),
    useNextDueQuery: emptyQuery
  }
})

const previews = { again: "1 min", hard: "6 min", good: "10 min", easy: "4 days" }
const card = {
  uuid: "cram-preview-card", deck_id: 1, front: "Preview question", back: "Preview answer",
  ef: 2.5, interval_days: 0, repetitions: 0, lapses: 0, queue_state: "new", step_index: 0,
  tags: ["preview"], is_cloze: false, model_type: "basic", reverse: false,
  version: 1, client_id: "test", deleted: false, source_ref_type: "manual",
  scheduler_type: null, next_intervals: null
}
const clients: QueryClient[] = []
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()) })
beforeEach(async () => {
  vi.clearAllMocks()
  state.previewAlwaysPresent = false
  state.review.mockResolvedValue({
    uuid: card.uuid, interval_days: 0, due_at: "2026-09-17T02:19:54Z",
    last_reviewed_at: "2026-09-17T02:09:54Z", review_session_id: 77
  })
  state.end.mockResolvedValue({ id: 77 })
  await clearSetting(FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING)
  vi.mocked(bgRequest).mockImplementation(async request => {
    const url = new URL(request.path, "https://fixture.test")
    if (url.pathname !== "/api/v1/flashcards") throw Error(`Unexpected request ${url.pathname}`)
    // The companion real-router test proves the opt-in payload and scheduler
    // outcome. This boundary simulates that wire contract, never the hook.
    const enabled = state.previewAlwaysPresent || url.searchParams.get("include_scheduler_preview") === "true"
    return { items: [{ ...card, ...(enabled ? { scheduler_type: "sm2_plus", next_intervals: previews } : {}) }], count: 1, total: 1 }
  })
})

async function mountCram(scheduled = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  clients.push(client)
  render(<QueryClientProvider client={client}><ReviewTab
    onNavigateToCreate={vi.fn()} onNavigateToImport={vi.fn()} reviewDeckId={1}
    onReviewDeckChange={vi.fn()} isActive
  /></QueryClientProvider>)
  fireEvent.click(screen.getByText("Cram", { exact: true }))
  await screen.findByText(card.front)
  if (scheduled) fireEvent.click(screen.getByTestId("flashcards-review-cram-update-schedule"))
  fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
}

describe("Cram authoritative scheduler previews", () => {
  it.each([
    { rating: 0, key: 1, label: "1 min" }, { rating: 2, key: 2, label: "6 min" },
    { rating: 3, key: 3, label: "10 min" }, { rating: 5, key: 4, label: "4 days" }
  ])("requests and displays the real scheduler label for rating $rating", async ({ rating, key, label }) => {
    await mountCram()
    const button = screen.getByTestId(`flashcards-review-rate-${key}`)
    expect(within(button).getByText(label, { exact: true })).toBeVisible()
    expect(vi.mocked(bgRequest).mock.calls).toHaveLength(1)
    const url = new URL(vi.mocked(bgRequest).mock.calls[0][0].path, "https://fixture.test")
    expect(Object.fromEntries(url.searchParams)).toMatchObject({
      deck_id: "1", due_status: "all", include_workspace_items: "false", include_scheduler_preview: "true"
    })
    fireEvent.click(button)
    await waitFor(() => expect(state.review).toHaveBeenCalledTimes(1))
    expect(state.review.mock.calls[0][0]).toMatchObject({ cardUuid: card.uuid, rating })
  })

  it("already prefers a supplied scheduler preview over the legacy SM-2 fallback", async () => {
    state.previewAlwaysPresent = true
    await mountCram()
    expect(within(screen.getByTestId("flashcards-review-rate-3")).getByText("10 min")).toBeVisible()
    expect(state.review).not.toHaveBeenCalled()
    expect(state.end).not.toHaveBeenCalled()
  })

  it("keeps practice-only Good free of rating and session-end mutations", async () => {
    await mountCram(false)
    fireEvent.click(screen.getByTestId("flashcards-review-rate-3"))
    await waitFor(() => expect(screen.getByTestId("flashcards-review-empty-card")).toBeVisible())
    expect(state.review).not.toHaveBeenCalled()
    expect(state.end).not.toHaveBeenCalled()
  })
})
