import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardEditDrawer } from "../FlashcardEditDrawer"
import type { Flashcard } from "@/services/flashcards"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"

const uploadFlashcardAsset = vi.hoisted(() => vi.fn())

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

vi.mock("../MarkdownWithBoundary", () => ({
  MarkdownWithBoundary: ({ content }: { content: string }) => <div>{content}</div>
}))

vi.mock("../FlashcardTagPicker", () => ({
  FlashcardTagPicker: ({ dataTestId }: { dataTestId?: string }) => (
    <div data-testid={dataTestId ?? "flashcard-tag-picker"} />
  )
}))

vi.mock("@/services/flashcard-assets", () => ({
  uploadFlashcardAsset
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
  front: "Alpha Omega",
  back: "Back text",
  notes: null,
  extra: null,
  is_cloze: false,
  tags: [],
  ef: 2.5,
  interval_days: 0,
  repetitions: 0,
  lapses: 0,
  queue_state: "new",
  due_at: null,
  created_at: null,
  last_reviewed_at: null,
  last_modified: null,
  deleted: false,
  client_id: "1",
  version: 3,
  model_type: "basic",
  reverse: false
}

describe("FlashcardEditDrawer image insertion", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("inserts an uploaded image snippet into the front field at the cursor", async () => {
    uploadFlashcardAsset.mockResolvedValue({
      asset_uuid: "asset-1",
      reference: "flashcard-asset://asset-1",
      markdown_snippet: "![Slide](flashcard-asset://asset-1)"
    })

    render(
      <FlashcardEditDrawer
        open
        onClose={vi.fn()}
        card={sampleCard}
        onSave={vi.fn().mockResolvedValue(undefined)}
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
            scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
          }
        ]}
      />
    )

    const frontTextarea = await screen.findByDisplayValue("Alpha Omega")
    fireEvent.change(frontTextarea, { target: { value: "Alpha Omega" } })
    ;(frontTextarea as HTMLTextAreaElement).focus()
    ;(frontTextarea as HTMLTextAreaElement).setSelectionRange(6, 6)
    fireEvent.select(frontTextarea)

    const uploadInput = screen.getByLabelText("Upload image for Front")
    fireEvent.change(uploadInput, {
      target: {
        files: [new File(["binary"], "slide.png", { type: "image/png" })]
      }
    })

    await waitFor(() => {
      expect((frontTextarea as HTMLTextAreaElement).value).toBe(
        "Alpha ![Slide](flashcard-asset://asset-1)Omega"
      )
    })
  })
})
