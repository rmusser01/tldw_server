import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { FlashcardEditDrawer } from "../FlashcardEditDrawer"
import type { Flashcard } from "@/services/flashcards"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"

vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next")
  const { default: ICU } = await import("@/i18n/icu-format")
  const instance = createInstance().use(ICU)
  await instance.init({ lng: "en", fallbackLng: false, resources: {} })
  const t = instance.t.bind(instance)
  return { useTranslation: () => ({ t }) }
})

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    loading: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  })
}))

vi.mock("../FlashcardTagPicker", () => ({
  FlashcardTagPicker: ({ dataTestId }: { dataTestId?: string }) => (
    <div data-testid={dataTestId ?? "flashcard-tag-picker"} />
  )
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
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

const sampleCard: Flashcard = {
  uuid: "card-reset-1",
  deck_id: 1,
  front: "Front text",
  back: "Back text",
  notes: null,
  extra: null,
  is_cloze: false,
  tags: [],
  ef: 1.8,
  interval_days: 17,
  repetitions: 6,
  lapses: 4,
  queue_state: "review",
  due_at: "2026-02-20T10:30:00Z",
  last_reviewed_at: "2026-02-18T08:15:00Z",
  last_modified: null,
  deleted: false,
  client_id: "1",
  version: 5,
  model_type: "basic",
  reverse: false,
  source_ref_type: "message",
  source_ref_id: "m-12",
  conversation_id: "c-3"
}

describe("FlashcardEditDrawer reset scheduling action", () => {
  it.each([
    { queue: "learning", days: 0, confirmation: false },
    { queue: "learning", days: 0, confirmation: true },
    { queue: "relearning", days: 7, confirmation: false },
    { queue: "relearning", days: 7, confirmation: true },
    { queue: "review", days: 17, confirmation: false },
    { queue: "review", days: 17, confirmation: true }
  ])("Drawer $queue confirmation=$confirmation reports the saved ten-minute gap", async ({ queue, days, confirmation }) => {
    const scheduledCard = { ...sampleCard, interval_days: days, queue_state: queue,
      last_reviewed_at: "2026-09-17T06:26:35.074Z", due_at: "2026-09-17T06:36:35.074Z" } as Flashcard
    const onReset = vi.fn()
    render(<FlashcardEditDrawer open onClose={vi.fn()} card={scheduledCard} onSave={vi.fn()}
      onDelete={vi.fn()} onResetScheduling={onReset} decks={[]} />)
    if (confirmation) {
      fireEvent.click(screen.getByRole("button", { name: "Reset scheduling" }))
      expect(screen.getByText("Reset scheduling for this card?")).toBeInTheDocument()
    }
    const expected = queue === "review" ? (confirmation ? "Next review gap: 17 days" : "17d") : (confirmation ? "Next review gap: 10 minutes" : "10 min")
    expect(screen.getByText(expected)).toBeInTheDocument()
    expect(onReset).not.toHaveBeenCalled()
  })

  it("updates the saved gap in the same mounted editor without saving", () => {
    const props = { open: true, onClose: vi.fn(), onSave: vi.fn(), onDelete: vi.fn(), decks: [] }
    const view = render(<FlashcardEditDrawer {...props} card={{ ...sampleCard, interval_days: 1 }} />)
    expect(screen.getByText("1d")).toBeInTheDocument()
    view.rerender(<FlashcardEditDrawer {...props} card={{ ...sampleCard, interval_days: 7,
      queue_state: "relearning", last_reviewed_at: "2026-09-17T06:26:35.074Z",
      due_at: "2026-09-17T06:36:35.074Z" }} />)
    expect(screen.getByText("10 min")).toBeInTheDocument()
    expect(props.onSave).not.toHaveBeenCalled()
  })

  it("confirms and invokes reset scheduling callback", async () => {
    const onResetScheduling = vi.fn().mockResolvedValue(undefined)

    render(
      <FlashcardEditDrawer
        open
        onClose={vi.fn()}
        card={sampleCard}
        onSave={vi.fn()}
        onDelete={vi.fn()}
        onResetScheduling={onResetScheduling}
        decks={[
          {
            id: 1,
            name: "Deck 1",
            description: null,
            deleted: false,
            client_id: "1",
            version: 1,
            review_prompt_side: "front",
            scheduler_type: "sm2_plus",
            scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
          }
        ]}
      />
    )

    expect(screen.getByText("Conversation for message #m-12")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Reset scheduling" }))
    expect(
      screen.getByText("Reset scheduling for this card?")
    ).toBeInTheDocument()
    expect(screen.getByText("Current scheduling state:")).toBeInTheDocument()
    expect(screen.getByText("Memory strength: 1.80")).toBeInTheDocument()
    expect(screen.getByText("Next review gap: 17 days")).toBeInTheDocument()
    expect(screen.getByText("Recall runs: 6")).toBeInTheDocument()
    expect(screen.getByText("Relearns: 4")).toBeInTheDocument()

    const resetButtons = screen.getAllByRole("button", { name: "Reset scheduling" })
    fireEvent.click(resetButtons[resetButtons.length - 1])

    await waitFor(() => {
      expect(onResetScheduling).toHaveBeenCalledTimes(1)
    })
  }, 15000)
})
