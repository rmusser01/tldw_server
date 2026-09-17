import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import ICU from "@/i18n/icu-format"
import optionEnglish from "@/assets/locale/en/option.json"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ReviewTab } from "../ReviewTab"
import { clearSetting } from "@/services/settings/registry"
import { listFlashcards, type Flashcard, type FlashcardListResponse } from "@/services/flashcards"
import { useCramQueueQuery as useRealCramQueueQuery } from "../../hooks/useFlashcardQueries"
import {
  FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING,
  FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING
} from "@/services/settings/ui-settings"
import {
  useDecksQuery,
  useCramQueueQuery,
  useReviewQuery,
  useReviewFlashcardMutation,
  useEndFlashcardReviewSessionMutation,
  useGlobalFlashcardTagSuggestionsQuery,
  useFlashcardAssistantQuery,
  useFlashcardAssistantRespondMutation,
  useUpdateFlashcardMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useRecentFlashcardReviewSessionsQuery,
  useFlashcardShortcuts,
  useDebouncedFormField,
  useDueCountsQuery,
  useDeckDueCountsQuery,
  useReviewAnalyticsSummaryQuery,
  useHasCardsQuery,
  useNextDueQuery
} from "../../hooks"

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}

vi.mock("@/services/service-prompts", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  ...await import("./review-scope-fixture")
}))

vi.mock("@/services/flashcards", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/flashcards")>(),
  listFlashcards: vi.fn()
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasFlashcards: true }, loading: false })
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => vi.fn()
  }
})

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
}))

vi.mock("@/components/StudySuggestions/StudySuggestionsPanel", () => ({ StudySuggestionsPanel: () => null }))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({
    speak: vi.fn(),
    cancel: vi.fn(),
    isSpeaking: false
  })
}))

vi.mock("@/hooks/useSpeechRecognition", () => ({
  useSpeechRecognition: () => ({
    supported: false,
    isListening: false,
    transcript: "",
    start: vi.fn(),
    stop: vi.fn(),
    resetTranscript: vi.fn()
  })
}))

vi.mock("../../hooks", () => ({
  useDecksQuery: vi.fn(),
  useCramQueueQuery: vi.fn(),
  useReviewQuery: vi.fn(),
  useReviewFlashcardMutation: vi.fn(),
  useEndFlashcardReviewSessionMutation: vi.fn(() => ({ mutateAsync: vi.fn().mockResolvedValue({ id: 77 }), isPending: false })),
  useRecentFlashcardReviewSessionsQuery: vi.fn(() => ({
    data: [],
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn()
  })),
  useGlobalFlashcardTagSuggestionsQuery: vi.fn(),
  useFlashcardAssistantQuery: vi.fn(),
  useFlashcardAssistantRespondMutation: vi.fn(),
  useUpdateFlashcardMutation: vi.fn(),
  useResetFlashcardSchedulingMutation: vi.fn(),
  useDeleteFlashcardMutation: vi.fn(),
  useFlashcardShortcuts: vi.fn(),
  useDebouncedFormField: vi.fn(() => undefined),
  useDueCountsQuery: vi.fn(),
  useDeckDueCountsQuery: vi.fn(),
  useReviewAnalyticsSummaryQuery: vi.fn(),
  useHasCardsQuery: vi.fn(),
  useNextDueQuery: vi.fn()
}))

if (!globalThis.ResizeObserver) {
  ;globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

describe("ReviewTab cram completion", () => {
  const reviewMutateAsync = vi.fn()
  const endMutateAsync = vi.fn()

  beforeEach(async () => {
    vi.clearAllMocks()
    vi.mocked(listFlashcards).mockReset()
    reviewMutateAsync.mockResolvedValue({ review_session_id: 77, interval_days: 1 })
    endMutateAsync.mockResolvedValue({ id: 77 })
    vi.mocked(useEndFlashcardReviewSessionMutation).mockReturnValue({ mutateAsync: endMutateAsync } as ReturnType<typeof useEndFlashcardReviewSessionMutation>)
    await clearSetting(FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING)
    await clearSetting(FLASHCARDS_REVIEW_ONBOARDING_DISMISSED_SETTING)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [{ id: 1, name: "Biology" }],
      isLoading: false
    } as ReturnType<typeof useEndFlashcardReviewSessionMutation>)
    vi.mocked(useReviewQuery).mockReturnValue({
      data: {
        uuid: "due-card-1",
        deck_id: 1,
        front: "Due front",
        back: "Due back",
        notes: null,
        extra: null,
        is_cloze: false,
        tags: [],
        ef: 2.5,
        interval_days: 2,
        repetitions: 1,
        lapses: 0,
        due_at: null,
        last_reviewed_at: null,
        last_modified: null,
        deleted: false,
        client_id: "test",
        version: 2,
        model_type: "basic",
        reverse: false
      }
    } as ReturnType<typeof useReviewQuery>)
    vi.mocked(useCramQueueQuery).mockReturnValue({
      data: [
        {
          uuid: "cram-card-1",
          deck_id: 1,
          front: "Cram front",
          back: "Cram back",
          notes: null,
          extra: null,
          is_cloze: false,
          tags: ["biology"],
          ef: 2.5,
          interval_days: 2,
          repetitions: 1,
          lapses: 0,
          due_at: null,
          last_reviewed_at: null,
          last_modified: null,
          deleted: false,
          client_id: "test",
          version: 2,
          model_type: "basic",
          reverse: false
        }
      ],
      isSuccess: true,
      isError: false,
      isLoading: false,
      isFetching: false
    } as ReturnType<typeof useCramQueueQuery>)
    vi.mocked(useReviewFlashcardMutation).mockReturnValue({
      mutateAsync: reviewMutateAsync
    } as ReturnType<typeof useReviewFlashcardMutation>)
    vi.mocked(useRecentFlashcardReviewSessionsQuery).mockReturnValue({
      data: [],
      isLoading: false
    } as ReturnType<typeof useRecentFlashcardReviewSessionsQuery>)
    vi.mocked(useFlashcardAssistantQuery).mockReturnValue({
      data: null,
      isLoading: false,
      isError: false
    } as ReturnType<typeof useFlashcardAssistantQuery>)
    vi.mocked(useFlashcardAssistantRespondMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as ReturnType<typeof useFlashcardAssistantRespondMutation>)
    vi.mocked(useUpdateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as ReturnType<typeof useUpdateFlashcardMutation>)
    vi.mocked(useResetFlashcardSchedulingMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as ReturnType<typeof useResetFlashcardSchedulingMutation>)
    vi.mocked(useDeleteFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as ReturnType<typeof useDeleteFlashcardMutation>)
    vi.mocked(useGlobalFlashcardTagSuggestionsQuery).mockReturnValue({
      data: { items: [] },
      isLoading: false,
      isFetching: false,
      isError: false
    } as ReturnType<typeof useGlobalFlashcardTagSuggestionsQuery>)
    vi.mocked(useFlashcardShortcuts).mockImplementation(() => undefined)
    vi.mocked(useDebouncedFormField).mockReturnValue(undefined as ReturnType<typeof useDebouncedFormField>)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: { due: 1, new: 0, learning: 0, total: 1 }
    } as ReturnType<typeof useDueCountsQuery>)
    vi.mocked(useDeckDueCountsQuery).mockReturnValue({ data: {} } as ReturnType<typeof useDeckDueCountsQuery>)
    vi.mocked(useReviewAnalyticsSummaryQuery).mockReturnValue({
      data: null,
      isLoading: false
    } as ReturnType<typeof useReviewAnalyticsSummaryQuery>)
    vi.mocked(useHasCardsQuery).mockReturnValue({ data: true } as ReturnType<typeof useHasCardsQuery>)
    vi.mocked(useNextDueQuery).mockReturnValue({ data: null } as ReturnType<typeof useNextDueQuery>)
  })

  const makeI18n = async (resources = true) => {
    const i18n = createInstance().use(ICU)
    await i18n.init({
      lng: "en", fallbackLng: false,
      resources: resources ? { en: { option: optionEnglish } } : {}
    })
    return i18n
  }

  const response = (items: Flashcard[]): FlashcardListResponse => ({
    items, count: items.length, total: items.length
  })

  const mountQueue = async ({ count = 1, resources = true, tag = "biology" } = {}) => {
    const template = vi.mocked(useCramQueueQuery)().data![0]
    const cards = Array.from({ length: count }, (_, index) => ({
      ...template, uuid: `completion-${index}`, front: `Card ${index + 1}`
    }))
    vi.mocked(listFlashcards).mockResolvedValue(response(cards))
    vi.mocked(useCramQueueQuery).mockImplementation(useRealCramQueueQuery)
    const client = new QueryClient({ defaultOptions: { queries: {
      retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false
    } } })
    const i18n = await makeI18n(resources)
    const props = {
      onNavigateToCreate: vi.fn(), onNavigateToImport: vi.fn(),
      reviewDeckId: 1, onReviewDeckChange: vi.fn(), isActive: true
    }
    const tree = () => <I18nextProvider i18n={i18n}><QueryClientProvider client={client}>
      <ReviewTab {...props} />
    </QueryClientProvider></I18nextProvider>
    const view = render(tree())
    fireEvent.click(screen.getByText("Cram"))
    if (tag) fireEvent.change(screen.getByTestId("flashcards-review-cram-tag"), { target: { value: tag } })
    await waitFor(() => expect(listFlashcards).toHaveBeenCalledWith(expect.objectContaining({
      deck_id: 1, ...(tag ? { tag } : {}), due_status: "all"
    })))
    return { client, cards, props, rerender: () => view.rerender(tree()) }
  }

  const rate = async (front: string) => {
    await screen.findByText(front)
    fireEvent.click(screen.getByTestId("flashcards-review-show-answer"))
    await act(async () => { fireEvent.click(screen.getByTestId("flashcards-review-rate-3")) })
  }

  it.each([0, 1, 2])("formats the English Cram completion resource at count %i", async count => {
    const i18n = await makeI18n()
    expect(i18n.t("option:flashcards.reviewedThisCramSession", { count })).toBe(
      `${count} ${count === 1 ? "card" : "cards"} practiced in this cram session`
    )
  })

  it.each([true, false])("renders singular completion with real ICU (English resource=%s)", async resources => {
    await mountQueue({ resources, tag: "" })
    await rate("Card 1")
    expect(screen.getByText("1 card practiced in this cram session")).toBeInTheDocument()
    expect(reviewMutateAsync).not.toHaveBeenCalled()
    expect(endMutateAsync).not.toHaveBeenCalled()
  })

  it.each([true, false])("completes a nonempty filtered queue after its final card (schedule=%s)", async scheduled => {
    const { cards, client } = await mountQueue()
    if (scheduled) fireEvent.click(screen.getByTestId("flashcards-review-cram-update-schedule"))
    await rate("Card 1")
    expect(screen.getByText("Cram session complete!")).toBeInTheDocument()
    expect(screen.queryByText("No cards match this cram tag filter.")).not.toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-review-progress")).not.toBeInTheDocument()
    expect(screen.getByTestId("flashcards-review-cram-tag")).toHaveValue("biology")
    // A successful refresh still contains the saved matching card, as in native UAT190.
    vi.mocked(listFlashcards).mockResolvedValue(response(cards))
    await act(async () => { await client.refetchQueries({ queryKey: ["flashcards:review:cram-queue"] }) })
    expect(screen.getByText("Cram session complete!")).toBeInTheDocument()
    if (scheduled) {
      expect(reviewMutateAsync).toHaveBeenCalledTimes(1)
      expect(reviewMutateAsync).toHaveBeenCalledWith(expect.objectContaining({ cardUuid: cards[0].uuid, rating: 3 }))
      await waitFor(() => expect(endMutateAsync).toHaveBeenCalledTimes(1))
    } else {
      expect(reviewMutateAsync).not.toHaveBeenCalled()
      expect(endMutateAsync).not.toHaveBeenCalled()
    }
  })

  it("keeps initially empty successful tag guidance and does not invent practiced statistics", async () => {
    await mountQueue({ count: 0 })
    expect(await screen.findByText("No cards match this cram tag filter.")).toBeInTheDocument()
    expect(screen.queryByText("Cram session complete!")).not.toBeInTheDocument()
    expect(screen.queryByText(/cards? practiced in this cram session/)).not.toBeInTheDocument()
    expect(reviewMutateAsync).not.toHaveBeenCalled()
    expect(endMutateAsync).not.toHaveBeenCalled()
  })

  it("shows loading then failure for an exhausted cached tag queue, and completion only after successful Retry", async () => {
    const { client, cards } = await mountQueue()
    await rate("Card 1")
    let reject!: (reason: Error) => void
    const pending = new Promise<FlashcardListResponse>((_resolve, fail) => { reject = fail })
    vi.mocked(listFlashcards).mockReturnValue(pending)
    let refreshing!: Promise<void>
    await act(async () => { refreshing = client.refetchQueries({ queryKey: ["flashcards:review:cram-queue"] }) })
    expect(await screen.findByText("Loading cram cards...")).toBeInTheDocument()
    expect(screen.queryByText("Cram session complete!")).not.toBeInTheDocument()
    expect(screen.queryByText("No cards match this cram tag filter.")).not.toBeInTheDocument()
    await act(async () => { reject(new Error("HTTP500 Failed to list flashcards")); await refreshing })
    expect(await screen.findByText("Unable to load cram cards")).toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-review-empty-card")).not.toBeInTheDocument()
    vi.mocked(listFlashcards).mockResolvedValue(response(cards))
    fireEvent.click(screen.getByTestId("flashcards-review-cram-retry"))
    expect(await screen.findByText("Cram session complete!")).toBeInTheDocument()
    expect(reviewMutateAsync).not.toHaveBeenCalled()
    expect(endMutateAsync).not.toHaveBeenCalled()
  })

  it("preserves pending identities through reordered tag refetch and completes with the plural count", async () => {
    const { client, cards } = await mountQueue({ count: 2 })
    fireEvent.click(screen.getByTestId("flashcards-review-cram-update-schedule"))
    await rate("Card 1")
    vi.mocked(listFlashcards).mockResolvedValue(response([cards[1], cards[0]]))
    await act(async () => { await client.refetchQueries({ queryKey: ["flashcards:review:cram-queue"] }) })
    expect(screen.getByText("Card 2")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("1 card remaining, 1 reviewed")
    await rate("Card 2")
    expect(screen.getByText("Cram session complete!")).toBeInTheDocument()
    expect(screen.getByText("2 cards practiced in this cram session")).toBeInTheDocument()
    expect(reviewMutateAsync.mock.calls.map(([input]) => input.cardUuid)).toEqual(cards.map(card => card.uuid))
    await waitFor(() => expect(endMutateAsync).toHaveBeenCalledTimes(1))
  })

  it.each(["tag", "deck", "account"])("does not carry completion or practiced identities into a new %s scope", async scope => {
    const { props, rerender } = await mountQueue()
    await rate("Card 1")
    if (scope === "tag") fireEvent.change(screen.getByTestId("flashcards-review-cram-tag"), { target: { value: "other" } })
    if (scope === "deck") { props.reviewDeckId = 2; rerender() }
    if (scope === "account") await act(async () => { window.dispatchEvent(new Event("tldw:auth-principal-changed")) })
    await waitFor(() => expect(screen.getByText("Card 1")).toBeInTheDocument())
    expect(screen.getByTestId("flashcards-review-progress")).toHaveTextContent("1 card remaining, 0 reviewed")
    expect(screen.queryByTestId("flashcards-review-empty-card")).not.toBeInTheDocument()
    expect(reviewMutateAsync).not.toHaveBeenCalled()
    expect(endMutateAsync).not.toHaveBeenCalled()
  })
})
