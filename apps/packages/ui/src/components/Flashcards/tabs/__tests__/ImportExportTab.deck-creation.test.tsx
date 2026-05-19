import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ImportExportTab } from "../ImportExportTab"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useCreateFlashcardsBulkMutation,
  useDecksQuery,
  useGenerateFlashcardsMutation,
  useImportFlashcardsApkgMutation,
  useImportFlashcardsJsonMutation,
  useImportFlashcardsMutation,
  useImportLimitsQuery,
  usePreviewStructuredQaImportMutation
} from "../../hooks"
import type { DeckSchedulerSettings, DeckSchedulerSettingsEnvelope } from "@/services/flashcards"

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}

const createDeckMutateAsync = vi.hoisted(() => vi.fn())
const createCardMutateAsync = vi.hoisted(() => vi.fn())
const createBulkMutateAsync = vi.hoisted(() => vi.fn())
const generateMutateAsync = vi.hoisted(() => vi.fn())
const previewStructuredMutateAsync = vi.hoisted(() => vi.fn())
const invalidateQueriesMock = vi.hoisted(() => vi.fn())
const useQueryMock = vi.hoisted(() => vi.fn())
const defaultFsrsSettings = {
  target_retention: 0.9,
  maximum_interval_days: 36500,
  enable_fuzz: false
}

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>("@tanstack/react-query")
  return {
    ...actual,
    useQuery: useQueryMock,
    useQueryClient: () => ({
      invalidateQueries: invalidateQueriesMock
    })
  }
})

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("react-router-dom", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-router-dom")>()
  return {
    ...actual,
    useNavigate: () => vi.fn()
  }
})

vi.mock("../components/StudyPackCreateDrawer", () => ({
  StudyPackCreateDrawer: () => null
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

vi.mock("../../hooks", () => ({
  useCreateDeckMutation: vi.fn(),
  useCreateFlashcardMutation: vi.fn(),
  useCreateFlashcardsBulkMutation: vi.fn(),
  useDecksQuery: vi.fn(),
  useGenerateFlashcardsMutation: vi.fn(),
  useImportFlashcardsMutation: vi.fn(),
  useImportFlashcardsApkgMutation: vi.fn(),
  useImportFlashcardsJsonMutation: vi.fn(),
  useImportLimitsQuery: vi.fn(),
  usePreviewStructuredQaImportMutation: vi.fn(),
  useStudyPackCreateMutation: vi.fn(() => ({
    mutateAsync: vi.fn(),
    isPending: false
  })),
  useStudyPackJobQuery: vi.fn(() => ({
    data: null,
    isLoading: false,
    isFetching: false
  }))
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (!(Element.prototype as any).scrollIntoView) {
  ;(Element.prototype as any).scrollIntoView = vi.fn()
}

const fastAcquisitionSettings: DeckSchedulerSettings = {
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

const fastAcquisitionEnvelope: DeckSchedulerSettingsEnvelope = {
  sm2_plus: fastAcquisitionSettings,
  fsrs: defaultFsrsSettings
}

describe("ImportExportTab deck creation flows", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    createDeckMutateAsync.mockReset()
    createCardMutateAsync.mockReset()
    createBulkMutateAsync.mockReset()
    generateMutateAsync.mockReset()
    previewStructuredMutateAsync.mockReset()
    useQueryMock.mockReturnValue({
      data: 0,
      isLoading: false
    } as any)

    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1,
          scheduler_type: "sm2_plus",
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
    vi.mocked(useCreateDeckMutation).mockReturnValue({
      mutateAsync: createDeckMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: createCardMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardsBulkMutation).mockReturnValue({
      mutateAsync: createBulkMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
      mutateAsync: generateMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useImportFlashcardsJsonMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useImportFlashcardsApkgMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useImportLimitsQuery).mockReturnValue({
      data: null
    } as any)
    vi.mocked(usePreviewStructuredQaImportMutation).mockReturnValue({
      mutateAsync: previewStructuredMutateAsync,
      isPending: false
    } as any)
  })

  it("creates a new structured-import deck with scheduler settings from the selector flow", async () => {
    createDeckMutateAsync.mockResolvedValue({
      id: 11,
      name: "Structured deck",
      description: null,
      deleted: false,
      client_id: "test",
      version: 1,
      scheduler_type: "sm2_plus",
      scheduler_settings_json: JSON.stringify(fastAcquisitionEnvelope),
      scheduler_settings: fastAcquisitionEnvelope
    })
    previewStructuredMutateAsync.mockResolvedValue({
      drafts: [
        {
          front: "Question",
          back: "Answer",
          line_start: 1,
          line_end: 2,
          notes: null,
          extra: null,
          tags: []
        }
      ],
      errors: [],
      detected_format: "qa_labels",
      skipped_blocks: 0
    })
    createBulkMutateAsync.mockResolvedValue({
      items: [{ uuid: "card-1", deck_id: 11 }],
      count: 1,
      total: 1,
      errors: []
    })

    render(<ImportExportTab />)

    fireEvent.mouseDown(screen.getByTestId("flashcards-import-format"))
    fireEvent.click(await screen.findByText("Structured Q&A"))

    fireEvent.mouseDown(screen.getByTestId("flashcards-structured-target-deck"))
    fireEvent.click(await screen.findByText("Create new deck"))
    fireEvent.change(screen.getByTestId("flashcards-structured-new-deck-name"), {
      target: { value: "Structured deck" }
    })
    fireEvent.mouseDown(screen.getByTestId("deck-study-defaults-review-prompt-side"))
    fireEvent.click(await screen.findByText("Back first"))
    fireEvent.click(screen.getByTestId("deck-scheduler-editor-preset-fast_acquisition"))

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: { value: "Q: Question\nA: Answer" }
    })
    fireEvent.click(screen.getByTestId("flashcards-structured-preview-button"))

    await waitFor(() => {
      expect(previewStructuredMutateAsync).toHaveBeenCalledTimes(1)
    })

    fireEvent.click(await screen.findByTestId("flashcards-structured-save-button"))

    await waitFor(() =>
      expect(createDeckMutateAsync).toHaveBeenCalledWith({
        name: "Structured deck",
        review_prompt_side: "back",
        scheduler_type: "sm2_plus",
        scheduler_settings: fastAcquisitionEnvelope
      })
    )
  })

  it("creates a new generated-cards deck with scheduler settings from the selector flow", async () => {
    createDeckMutateAsync.mockResolvedValue({
      id: 12,
      name: "Generated deck",
      description: null,
      deleted: false,
      client_id: "test",
      version: 1,
      scheduler_type: "sm2_plus",
      scheduler_settings_json: JSON.stringify(fastAcquisitionEnvelope),
      scheduler_settings: fastAcquisitionEnvelope
    })
    generateMutateAsync.mockResolvedValue({
      flashcards: [
        {
          front: "Generated front",
          back: "Generated back",
          tags: ["tag-1"],
          model_type: "basic"
        }
      ],
      count: 1
    })
    createCardMutateAsync.mockResolvedValue({})

    render(<ImportExportTab />)

    fireEvent.mouseDown(screen.getByTestId("flashcards-generate-deck"))
    fireEvent.click(await screen.findByText("Create new deck"))
    fireEvent.change(screen.getByTestId("flashcards-generate-new-deck-name"), {
      target: { value: "Generated deck" }
    })
    fireEvent.mouseDown(screen.getByTestId("deck-study-defaults-review-prompt-side"))
    fireEvent.click(await screen.findByText("Back first"))
    fireEvent.click(screen.getByTestId("deck-scheduler-editor-preset-fast_acquisition"))

    fireEvent.change(screen.getByTestId("flashcards-generate-text"), {
      target: { value: "Photosynthesis notes..." }
    })
    fireEvent.click(screen.getByTestId("flashcards-generate-button"))

    await waitFor(() => {
      expect(generateMutateAsync).toHaveBeenCalledTimes(1)
    })

    fireEvent.click(await screen.findByTestId("flashcards-generate-save-button"))

    await waitFor(() =>
      expect(createDeckMutateAsync).toHaveBeenCalledWith({
        name: "Generated deck",
        review_prompt_side: "back",
        scheduler_type: "sm2_plus",
        scheduler_settings: fastAcquisitionEnvelope
      })
    )
  })
})
