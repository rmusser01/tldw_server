import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { FlashcardEditDrawer } from "../FlashcardEditDrawer"
import { FLASHCARDS_DRAWER_WIDTH_PX } from "../../constants"
import type { Flashcard } from "@/services/flashcards"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"
import { formatFlashcardAbsoluteDateTime } from "../../utils/date-display"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

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

describe("FlashcardEditDrawer scheduling metadata panel", () => {
  it("renders read-only scheduling details including last reviewed and due timestamps", () => {
    const dueAt = "2026-02-20T10:30:00Z"
    const lastReviewedAt = "2026-02-18T08:15:00Z"

    const card: Flashcard = {
      uuid: "card-2",
      deck_id: 1,
      front: "Question",
      back: "Answer",
      notes: null,
      extra: null,
      is_cloze: false,
      tags: [],
      ef: 2.5,
      interval_days: 7,
      repetitions: 4,
      lapses: 2,
      queue_state: "review",
      due_at: dueAt,
      last_reviewed_at: lastReviewedAt,
      last_modified: null,
      deleted: false,
      client_id: "1",
      version: 5,
      model_type: "basic",
      reverse: false
    }

    render(
      <FlashcardEditDrawer
        open
        onClose={vi.fn()}
        card={card}
        onSave={vi.fn()}
        onDelete={vi.fn()}
        decks={[
          {
            id: 1,
            name: "Deck 1",
            description: null,
            deleted: false,
            client_id: "1",
            version: 1,
            scheduler_type: "sm2_plus",
            review_prompt_side: "front",
            scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
          }
        ]}
      />
    )

    expect(screen.getByText("Scheduling")).toBeInTheDocument()
    expect(screen.getByText("Memory strength")).toBeInTheDocument()
    expect(screen.getByText("2.50")).toBeInTheDocument()
    expect(screen.getByText("Next review gap")).toBeInTheDocument()
    expect(screen.getByText("7d")).toBeInTheDocument()
    expect(screen.getByText("Recall runs")).toBeInTheDocument()
    expect(screen.getByText("4")).toBeInTheDocument()
    expect(screen.getByText("Relearns")).toBeInTheDocument()
    expect(screen.getByText("2")).toBeInTheDocument()

    const expectedDueAbsolute = formatFlashcardAbsoluteDateTime(dueAt) ?? ""
    const expectedLastReviewedAbsolute =
      formatFlashcardAbsoluteDateTime(lastReviewedAt) ?? ""
    expect(
      screen.getByText((content) => content.includes(expectedDueAbsolute))
    ).toBeInTheDocument()
    expect(
      screen.getByText((content) => content.includes(expectedLastReviewedAbsolute))
    ).toBeInTheDocument()

    const wrapper = document.querySelector(".ant-drawer-content-wrapper") as HTMLElement | null
    expect(wrapper?.style.width).toBe(`${FLASHCARDS_DRAWER_WIDTH_PX}px`)
  })
})
