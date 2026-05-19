import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardCreateDrawer } from "../FlashcardCreateDrawer"
import { useCreateDeckMutation, useCreateFlashcardMutation, useDecksQuery } from "../../hooks"
import type { DeckSchedulerSettings, DeckSchedulerSettingsEnvelope } from "@/services/flashcards"

const uploadFlashcardAsset = vi.hoisted(() => vi.fn())

const defaultFsrsSettings = {
  target_retention: 0.9,
  maximum_interval_days: 36500,
  enable_fuzz: false
}

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

vi.mock("../../hooks", () => ({
  useDecksQuery: vi.fn(),
  useCreateFlashcardMutation: vi.fn(),
  useCreateDeckMutation: vi.fn(),
  useDebouncedFormField: vi.fn(() => undefined),
  useFlashcardDeckRecentCardsQuery: vi.fn(() => ({
    data: [],
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn()
  })),
  useFlashcardDeckSearchQuery: vi.fn(() => ({
    data: [],
    isLoading: false,
    isError: false,
    error: null,
    refetch: vi.fn()
  }))
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

describe("FlashcardCreateDrawer image insertion", () => {
  const createDeckMutateAsync = vi.fn()
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.clearAllMocks()
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1,
          scheduler_settings_json: null,
          scheduler_settings: {
            new_steps_minutes: [1, 10],
            relearn_steps_minutes: [10],
            graduating_interval_days: 1,
            easy_interval_days: 4,
            easy_bonus: 1.3,
            interval_modifier: 1,
            max_interval_days: 36500,
            leech_threshold: 8,
            enable_fuzz: false
          }
        }
      ],
      isLoading: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useCreateDeckMutation).mockReturnValue({
      mutateAsync: createDeckMutateAsync,
      isPending: false
    } as any)
    createDeckMutateAsync.mockReset()
  })

  afterEach(() => {
    expect(consoleErrorSpy).not.toHaveBeenCalled()
    consoleErrorSpy.mockRestore()
  })

  it("inserts an uploaded image snippet into the front field at the cursor", async () => {
    uploadFlashcardAsset.mockResolvedValue({
      asset_uuid: "asset-1",
      reference: "flashcard-asset://asset-1",
      markdown_snippet: "![Slide](flashcard-asset://asset-1)"
    })

    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    const frontTextarea = screen.getByPlaceholderText("Question or prompt...") as HTMLTextAreaElement
    fireEvent.change(frontTextarea, { target: { value: "Alpha Omega" } })
    frontTextarea.focus()
    frontTextarea.setSelectionRange(6, 6)
    fireEvent.select(frontTextarea)

    const uploadInput = screen.getByLabelText("Upload image for Front")
    fireEvent.change(uploadInput, {
      target: {
        files: [new File(["binary"], "slide.png", { type: "image/png" })]
      }
    })

    await waitFor(() => {
      expect(frontTextarea.value).toBe(
        "Alpha ![Slide](flashcard-asset://asset-1)Omega"
      )
    })
  })

  it("creates an inline deck with scheduler settings and selects it in the form", async () => {
    const createdDeckSettings: DeckSchedulerSettings = {
      new_steps_minutes: [1, 5, 15],
      relearn_steps_minutes: [10],
      graduating_interval_days: 1,
      easy_interval_days: 3,
      easy_bonus: 1.15,
      interval_modifier: 0.9,
      max_interval_days: 3650,
      leech_threshold: 10,
      enable_fuzz: false
    }
    const createdDeckEnvelope: DeckSchedulerSettingsEnvelope = {
      sm2_plus: createdDeckSettings,
      fsrs: defaultFsrsSettings
    }
    createDeckMutateAsync.mockResolvedValue({
      id: 7,
      name: "New deck",
      description: null,
      deleted: false,
      client_id: "test",
      version: 1,
      scheduler_type: "sm2_plus",
      scheduler_settings_json: JSON.stringify(createdDeckEnvelope),
      scheduler_settings: createdDeckEnvelope
    })

    render(<FlashcardCreateDrawer open onClose={vi.fn()} onSuccess={vi.fn()} />)

    fireEvent.mouseDown(screen.getByLabelText("Deck"))
    fireEvent.click(await screen.findByText("Create new deck"))

    fireEvent.change(screen.getByPlaceholderText("New deck name"), {
      target: { value: "New deck" }
    })
    fireEvent.click(screen.getByTestId("deck-scheduler-editor-preset-fast_acquisition"))
    fireEvent.click(screen.getByTestId("flashcards-inline-create-deck-submit"))

    await waitFor(() =>
      expect(createDeckMutateAsync).toHaveBeenCalledWith({
        name: "New deck",
        scheduler_type: "sm2_plus",
        scheduler_settings: createdDeckEnvelope
      })
    )

    await waitFor(() => {
      expect(screen.queryByPlaceholderText("New deck name")).not.toBeInTheDocument()
    })
    expect(screen.getByTitle("7")).toBeInTheDocument()
  })
})
