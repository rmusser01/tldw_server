import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardEditDrawer } from "../FlashcardEditDrawer"
import type { Flashcard } from "@/services/flashcards"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"

const mockFlashcardTagPicker = vi.hoisted(() => vi.fn())

vi.mock("../FlashcardTagPicker", () => ({
  FlashcardTagPicker: (props: Record<string, unknown>) => mockFlashcardTagPicker(props)
}))

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
  uuid: "card-1",
  deck_id: 1,
  front: "Front text",
  back: "Back text",
  notes: null,
  extra: null,
  is_cloze: false,
  tags: ["biology", "chapter-1"],
  ef: 2.5,
  interval_days: 0,
  repetitions: 0,
  lapses: 0,
  queue_state: "new",
  due_at: null,
  last_reviewed_at: null,
  last_modified: null,
  deleted: false,
  client_id: "1",
  version: 3,
  model_type: "basic",
  reverse: false
}

const decks = [
  {
    id: 1,
    name: "Deck 1",
    description: null,
    deleted: false,
    client_id: "1",
    version: 1,
    scheduler_type: "sm2_plus" as const,
    scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
  }
]

const renderMockTagPicker = ({
  value = [],
  onChange,
  active,
  placeholder,
  dataTestId
}: {
  value?: string[]
  onChange?: (next: string[]) => void
  active?: boolean
  placeholder?: string
  dataTestId?: string
}) => (
  <div
    data-testid={dataTestId}
    data-active={String(active)}
    data-placeholder={placeholder}
  >
    <div data-testid={`${dataTestId}-value`}>{JSON.stringify(value)}</div>
    <button
      type="button"
      onClick={() => onChange?.([...(value ?? []), "astronomy"])}
    >
      add suggested tag
    </button>
    <button
      type="button"
      onClick={() => onChange?.([...(value ?? []), "   "])}
    >
      add whitespace tag
    </button>
  </div>
)

describe("FlashcardEditDrawer tags", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockFlashcardTagPicker.mockImplementation(renderMockTagPicker)
  })

  it("shows existing tags on open", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined)

    render(
      <FlashcardEditDrawer
        open
        onClose={vi.fn()}
        card={sampleCard}
        onSave={onSave}
        onDelete={vi.fn()}
        decks={decks}
      />
    )

    await waitFor(() => {
      expect(screen.getByTestId("flashcards-edit-tag-picker")).toHaveAttribute(
        "data-active",
        "true"
      )
    })
    expect(screen.getByTestId("flashcards-edit-tag-picker-value")).toHaveTextContent(
      JSON.stringify(sampleCard.tags)
    )
    expect(screen.getByTestId("flashcards-edit-tag-picker")).toHaveAttribute(
      "data-placeholder",
      "Add tags..."
    )
  }, 15000)

  it("selecting a suggested tag appends it", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined)

    render(
      <FlashcardEditDrawer
        open
        onClose={vi.fn()}
        card={sampleCard}
        onSave={onSave}
        onDelete={vi.fn()}
        decks={decks}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "add suggested tag" }))
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          tags: ["biology", "chapter-1", "astronomy"]
        })
      )
    })
  }, 15000)

  it("drops whitespace-only edits before onSave", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined)

    render(
      <FlashcardEditDrawer
        open
        onClose={vi.fn()}
        card={sampleCard}
        onSave={onSave}
        onDelete={vi.fn()}
        decks={decks}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "add whitespace tag" }))
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          tags: ["biology", "chapter-1"]
        })
      )
    })
  }, 15000)
})
