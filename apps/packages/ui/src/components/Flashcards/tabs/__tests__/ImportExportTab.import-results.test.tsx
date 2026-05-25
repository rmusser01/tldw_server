import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ImportExportTab } from "../ImportExportTab"
import { FLASHCARDS_HELP_LINKS, FLASHCARDS_LAYOUT_GUARDRAILS } from "../../constants"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useCreateFlashcardsBulkMutation,
  useDecksQuery,
  useGenerateFlashcardsMutation,
  useGlobalFlashcardTagSuggestionsQuery,
  useImportFlashcardsMutation,
  useImportFlashcardsApkgMutation,
  useImportFlashcardsJsonMutation,
  useImportLimitsQuery,
  usePreviewStructuredQaImportMutation,
  useStudyPackCreateMutation,
  useStudyPackJobQuery,
  useStudyPackQuery,
  useStudyPackRegenerateMutation
} from "../../hooks"
import {
  deleteFlashcard,
  exportFlashcards,
  exportFlashcardsFile,
  getFlashcard,
  listFlashcards
} from "@/services/flashcards"

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}
const showUndoNotificationMock = vi.fn()
const invalidateQueriesMock = vi.fn()
const { useQueryMock } = vi.hoisted(() => ({
  useQueryMock: vi.fn()
}))

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
    showUndoNotification: showUndoNotificationMock
  })
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
  useGlobalFlashcardTagSuggestionsQuery: vi.fn(),
  useImportFlashcardsMutation: vi.fn(),
  useImportFlashcardsApkgMutation: vi.fn(),
  useImportFlashcardsJsonMutation: vi.fn(),
  useImportLimitsQuery: vi.fn(),
  usePreviewStructuredQaImportMutation: vi.fn(),
  useStudyPackCreateMutation: vi.fn(),
  useStudyPackJobQuery: vi.fn(),
  useStudyPackQuery: vi.fn(),
  useStudyPackRegenerateMutation: vi.fn()
}))

vi.mock("@/services/flashcards", async () => {
  const actual = await vi.importActual<typeof import("@/services/flashcards")>("@/services/flashcards")
  return {
    ...actual,
    getFlashcard: vi.fn(),
    deleteFlashcard: vi.fn(),
    listFlashcards: vi.fn(),
    exportFlashcards: vi.fn(),
    exportFlashcardsFile: vi.fn()
  }
})

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    useNavigate: () => vi.fn(),
    useInRouterContext: () => false
  }
})

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

const createApkgFile = (
  sizeBytes: number,
  name = "deck.apkg"
): File & { arrayBuffer: () => Promise<ArrayBuffer> } => {
  const bytes = new Uint8Array(sizeBytes)
  const file = new File([bytes], name, {
    type: "application/apkg"
  }) as File & { arrayBuffer: () => Promise<ArrayBuffer> }
  file.arrayBuffer = async () => Uint8Array.from(bytes).buffer
  return file
}

describe("ImportExportTab import result details", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    invalidateQueriesMock.mockReset()
    useQueryMock.mockReturnValue({
      data: 42,
      isLoading: false
    } as any)
    vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardsBulkMutation).mockReturnValue({
      mutateAsync: vi.fn().mockResolvedValue({
        items: [],
        count: 0,
        total: 0
      }),
      isPending: false
    } as any)
    vi.mocked(useCreateDeckMutation).mockReturnValue({
      mutateAsync: vi.fn().mockResolvedValue({
        id: 999,
        name: "Generated Flashcards",
        description: null,
        deleted: false,
        client_id: "test",
        version: 1
      }),
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
    vi.mocked(usePreviewStructuredQaImportMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useGlobalFlashcardTagSuggestionsQuery).mockReturnValue({
      data: { items: [] },
      isFetching: false,
      isError: false
    } as any)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [],
      isLoading: false
    } as any)
    vi.mocked(useImportLimitsQuery).mockReturnValue({
      data: null
    } as any)
    vi.mocked(useStudyPackCreateMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useStudyPackJobQuery).mockReturnValue({
      data: null,
      isLoading: false
    } as any)
    vi.mocked(useStudyPackQuery).mockReturnValue({
      data: null,
      isLoading: false
    } as any)
    vi.mocked(useStudyPackRegenerateMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [],
      count: 0,
      total: 0
    } as any)
    vi.mocked(exportFlashcards).mockResolvedValue("Deck\tFront\tBack\n" as any)
    vi.mocked(exportFlashcardsFile).mockResolvedValue(
      new Blob([new Uint8Array([1, 2, 3])], { type: "application/apkg" }) as any
    )

    ;(URL as any).createObjectURL = vi.fn(() => "blob:mock")
    ;(URL as any).revokeObjectURL = vi.fn()
  })

  it("keeps transfer summary free of top-level primary CTAs", () => {
    render(<ImportExportTab />)

    const topLevelPrimaryButtons = screen
      .getByTestId("flashcards-transfer-summary")
      .querySelectorAll(".ant-btn-primary")
    expect(topLevelPrimaryButtons).toHaveLength(
      FLASHCARDS_LAYOUT_GUARDRAILS.transfer.maxTopbarPrimaryCtas.active
    )
  })

  it("shows expandable import help references for columns/delimiter/json mapping", async () => {
    render(<ImportExportTab />)

    expect(screen.getByTestId("flashcards-import-help-accordion")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-import-help-columns")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-import-doc-link")).toHaveAttribute(
      "href",
      FLASHCARDS_HELP_LINKS.importFormats
    )
    expect(screen.getByTestId("flashcards-import-cloze-doc-link")).toHaveAttribute(
      "href",
      FLASHCARDS_HELP_LINKS.cloze
    )

    fireEvent.click(screen.getByText("Delimiter troubleshooting"))
    expect(await screen.findByTestId("flashcards-import-help-delimiter")).toBeInTheDocument()

    const formatSelect = screen.getByTestId("flashcards-import-format")
    fireEvent.mouseDown(
      formatSelect.querySelector(".ant-select-selector") ?? formatSelect
    )
    fireEvent.click(screen.getByText("JSON / JSONL"))

    fireEvent.click(screen.getByText("JSON field mapping"))
    expect(await screen.findByTestId("flashcards-import-help-json")).toBeInTheDocument()
  })

  it("shows imported/skipped counts and line-level errors on partial imports", async () => {
    const mutateAsync = vi.fn().mockResolvedValue({
      imported: 2,
      items: [
        { uuid: "u1", deck_id: 1 },
        { uuid: "u2", deck_id: 1 }
      ],
      errors: [{ line: 15, error: "Missing required field: Front" }]
    })
    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: {
        value: "Deck\tFront\tBack\nBiology\tQuestion\tAnswer"
      }
    })
    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    expect(await screen.findByText("Last import: 2 imported, 1 skipped")).toBeInTheDocument()
    expect(screen.getByText("Line 15")).toBeInTheDocument()
    expect(screen.getByText("Missing required field: Front")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Add a non-empty Front value on that row, or map your header to the Front column."
      )
    ).toBeInTheDocument()
    const helpLink = screen.getByTestId("flashcards-import-error-help-0")
    fireEvent.click(helpLink)
    expect(screen.getByTestId("flashcards-import-help-columns")).toBeInTheDocument()
    expect(messageSpies.warning).toHaveBeenCalledWith(
      "Imported 2 cards, skipped 1 rows (1 errors)."
    )
  }, 15000)

  it("matches baseline snapshot for import result state", async () => {
    const mutateAsync = vi.fn().mockResolvedValue({
      imported: 2,
      items: [
        { uuid: "snap-u1", deck_id: 1 },
        { uuid: "snap-u2", deck_id: 1 }
      ],
      errors: [{ line: 9, error: "Missing required field: Front" }]
    })
    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: {
        value: "Deck\tFront\tBack\nBiology\tQuestion\tAnswer"
      }
    })
    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    expect(await screen.findByTestId("flashcards-import-last-result")).toMatchSnapshot()
    expect(screen.getByTestId("flashcards-transfer-summary-formats")).toMatchSnapshot()
    expect(screen.getByTestId("flashcards-transfer-summary-limits")).toMatchSnapshot()
  }, 15000)

  it("shows preflight warning when selected delimiter does not match sample content", async () => {
    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: {
        value: "Deck,Front,Back\nBiology,Question,Answer"
      }
    })

    expect(screen.getByTestId("flashcards-import-preflight-warning")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Selected delimiter (Tab) may be incorrect. This sample looks Comma-delimited."
      )
    ).toBeInTheDocument()
  })

  it("requires confirmation before importing very large batches", async () => {
    const mutateAsync = vi.fn().mockResolvedValue({
      imported: 0,
      items: [],
      errors: []
    })
    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    const rows = Array.from({ length: 301 }, (_, idx) => `Deck\tFront ${idx}\tBack ${idx}`).join("\n")

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: {
        value: `Deck\tFront\tBack\n${rows}`
      }
    })
    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    expect(screen.getByText("Confirm large import")).toBeInTheDocument()
    expect(
      screen.getByText(
        "Large imports may take a moment to process. You'll have 30 seconds to undo after import completes."
      )
    ).toBeInTheDocument()
    expect(mutateAsync).not.toHaveBeenCalled()

    fireEvent.click(screen.getByTestId("flashcards-import-confirm-large"))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1)
    })
  })

  it("routes JSON/JSONL imports through the JSON upload mutation", async () => {
    const delimitedMutateAsync = vi.fn()
    const jsonMutateAsync = vi.fn().mockResolvedValue({
      imported: 1,
      items: [{ uuid: "json-u1", deck_id: 1 }],
      errors: []
    })
    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync: delimitedMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useImportFlashcardsJsonMutation).mockReturnValue({
      mutateAsync: jsonMutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    const formatSelect = screen.getByTestId("flashcards-import-format")
    fireEvent.mouseDown(
      formatSelect.querySelector(".ant-select-selector") ?? formatSelect
    )
    fireEvent.click(screen.getByText("JSON / JSONL"))

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: {
        value:
          '[{"deck":"Biology","front":"Question","back":"Answer","tags":["tag-1"]}]'
      }
    })
    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    await waitFor(() => {
      expect(jsonMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(jsonMutateAsync).toHaveBeenCalledWith({
      content:
        '[{"deck":"Biology","front":"Question","back":"Answer","tags":["tag-1"]}]',
      filename: "flashcards.json"
    })
    expect(delimitedMutateAsync).not.toHaveBeenCalled()
  })

  it("routes APKG imports through the APKG upload mutation", async () => {
    const delimitedMutateAsync = vi.fn()
    const jsonMutateAsync = vi.fn()
    const apkgMutateAsync = vi.fn().mockResolvedValue({
      imported: 2,
      items: [
        { uuid: "apkg-u1", deck_id: 1 },
        { uuid: "apkg-u2", deck_id: 1 }
      ],
      errors: []
    })

    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync: delimitedMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useImportFlashcardsJsonMutation).mockReturnValue({
      mutateAsync: jsonMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useImportFlashcardsApkgMutation).mockReturnValue({
      mutateAsync: apkgMutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    const formatSelect = screen.getByTestId("flashcards-import-format")
    fireEvent.mouseDown(
      formatSelect.querySelector(".ant-select-selector") ?? formatSelect
    )
    fireEvent.click(screen.getByText("APKG (Anki)"))

    const file = createApkgFile(4, "deck.apkg")
    fireEvent.change(screen.getByTestId("flashcards-import-apkg-input"), {
      target: { files: [file] }
    })

    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    await waitFor(() => {
      expect(apkgMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(apkgMutateAsync.mock.calls[0][0]).toMatchObject({
      filename: "deck.apkg"
    })
    expect(apkgMutateAsync.mock.calls[0][0].bytes).toBeInstanceOf(Uint8Array)
    expect(delimitedMutateAsync).not.toHaveBeenCalled()
    expect(jsonMutateAsync).not.toHaveBeenCalled()
  })

  it("previews structured q and a drafts and saves only selected cards", async () => {
    const previewMutateAsync = vi.fn().mockResolvedValue({
      detected_format: "qa_labels",
      skipped_blocks: 0,
      errors: [],
      drafts: [
        {
          front: "What is ATP?",
          back: "Primary energy currency.",
          line_start: 1,
          line_end: 2,
          tags: []
        },
        {
          front: "What is glycolysis?",
          back: "Cytosolic glucose breakdown.",
          line_start: 4,
          line_end: 5,
          tags: []
        }
      ]
    })
    const createBulkMutateAsync = vi.fn().mockResolvedValue({
      items: [{ uuid: "card-1", version: 1 }],
      count: 1,
      total: 1
    })
    vi.mocked(usePreviewStructuredQaImportMutation).mockReturnValue({
      mutateAsync: previewMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardsBulkMutation).mockReturnValue({
      mutateAsync: createBulkMutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    const formatSelect = screen.getByTestId("flashcards-import-format")
    fireEvent.mouseDown(
      formatSelect.querySelector(".ant-select-selector") ?? formatSelect
    )
    fireEvent.click(screen.getByText("Structured Q&A"))

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: { value: "Q: What is ATP?\nA: Primary energy currency." }
    })
    fireEvent.click(screen.getByTestId("flashcards-structured-preview-button"))

    await waitFor(() => {
      expect(screen.getByDisplayValue("What is ATP?")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("flashcards-structured-draft-selected-1"))
    fireEvent.click(screen.getByTestId("flashcards-structured-save-button"))

    await waitFor(() => {
      expect(createBulkMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(createBulkMutateAsync).toHaveBeenCalledWith([
      expect.objectContaining({
        deck_id: 999,
        front: "What is ATP?",
        back: "Primary energy currency.",
        model_type: "basic",
        is_cloze: false,
        reverse: false
      })
    ])
  })

  it("keeps selected invalid structured drafts in the editor when saving", async () => {
    const previewMutateAsync = vi.fn().mockResolvedValue({
      detected_format: "qa_labels",
      skipped_blocks: 0,
      errors: [],
      drafts: [
        {
          front: "What is ATP?",
          back: "Primary energy currency.",
          line_start: 1,
          line_end: 2,
          tags: []
        },
        {
          front: "What is glycolysis?",
          back: "Cytosolic glucose breakdown.",
          line_start: 4,
          line_end: 5,
          tags: []
        }
      ]
    })
    const createBulkMutateAsync = vi.fn().mockResolvedValue({
      items: [{ uuid: "card-1", version: 1 }],
      count: 1,
      total: 1
    })
    vi.mocked(usePreviewStructuredQaImportMutation).mockReturnValue({
      mutateAsync: previewMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardsBulkMutation).mockReturnValue({
      mutateAsync: createBulkMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useImportLimitsQuery).mockReturnValue({
      data: {
        max_field_length: 32
      }
    } as any)

    render(<ImportExportTab />)

    const formatSelect = screen.getByTestId("flashcards-import-format")
    fireEvent.mouseDown(
      formatSelect.querySelector(".ant-select-selector") ?? formatSelect
    )
    fireEvent.click(screen.getByText("Structured Q&A"))

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: { value: "Q: What is ATP?\nA: Primary energy currency." }
    })
    fireEvent.click(screen.getByTestId("flashcards-structured-preview-button"))

    await waitFor(() => {
      expect(screen.getByDisplayValue("What is ATP?")).toBeInTheDocument()
    })

    fireEvent.change(screen.getByDisplayValue("What is glycolysis?"), {
      target: { value: "What is glycolysis after repeated repetition?" }
    })
    fireEvent.click(screen.getByTestId("flashcards-structured-save-button"))

    await waitFor(() => {
      expect(createBulkMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(createBulkMutateAsync).toHaveBeenCalledWith([
      expect.objectContaining({
        front: "What is ATP?",
        back: "Primary energy currency."
      })
    ])
    expect(
      screen.getByDisplayValue("What is glycolysis after repeated repetition?")
    ).toBeInTheDocument()
    expect(messageSpies.warning).toHaveBeenCalled()
    expect(screen.getByTestId("flashcards-import-last-result")).toHaveTextContent(
      "Last import: 1 imported, 1 skipped"
    )
  })

  it("allows clearing the structured target deck without auto-restoring the first deck", async () => {
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 7,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ],
      isLoading: false
    } as any)

    render(<ImportExportTab />)

    const formatSelect = screen.getByTestId("flashcards-import-format")
    fireEvent.mouseDown(
      formatSelect.querySelector(".ant-select-selector") ?? formatSelect
    )
    fireEvent.click(screen.getByText("Structured Q&A"))

    const deckSelect = screen.getByTestId("flashcards-structured-target-deck")
    await waitFor(() => {
      expect(deckSelect).toHaveTextContent("Biology")
    })

    const clearControl =
      deckSelect.querySelector(".ant-select-clear") ??
      within(deckSelect).queryByLabelText(/clear/i)
    expect(clearControl).toBeTruthy()

    fireEvent.mouseDown(clearControl as Element)
    fireEvent.click(clearControl as Element)

    await waitFor(() => {
      expect(deckSelect).not.toHaveTextContent("Biology")
    })
  })

  it("requires confirmation before importing large APKG files", async () => {
    const apkgMutateAsync = vi.fn().mockResolvedValue({
      imported: 2,
      items: [
        { uuid: "apkg-large-u1", deck_id: 1 },
        { uuid: "apkg-large-u2", deck_id: 1 }
      ],
      errors: []
    })
    vi.mocked(useImportFlashcardsApkgMutation).mockReturnValue({
      mutateAsync: apkgMutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    const formatSelect = screen.getByTestId("flashcards-import-format")
    fireEvent.mouseDown(
      formatSelect.querySelector(".ant-select-selector") ?? formatSelect
    )
    fireEvent.click(screen.getByText("APKG (Anki)"))

    const largeFile = createApkgFile(6 * 1024 * 1024, "large.apkg")
    fireEvent.change(screen.getByTestId("flashcards-import-apkg-input"), {
      target: { files: [largeFile] }
    })

    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    expect(screen.getByText("Confirm large import")).toBeInTheDocument()
    expect(screen.getByText("Summary: file large.apkg, size 6291456 bytes, estimated 1536 cards.")).toBeInTheDocument()
    expect(apkgMutateAsync).not.toHaveBeenCalled()

    fireEvent.click(screen.getByTestId("flashcards-import-confirm-large"))

    await waitFor(() => {
      expect(apkgMutateAsync).toHaveBeenCalledTimes(1)
    })
  })

  it("imports small APKG files without confirmation", async () => {
    const apkgMutateAsync = vi.fn().mockResolvedValue({
      imported: 1,
      items: [{ uuid: "apkg-small-u1", deck_id: 1 }],
      errors: []
    })
    vi.mocked(useImportFlashcardsApkgMutation).mockReturnValue({
      mutateAsync: apkgMutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    const formatSelect = screen.getByTestId("flashcards-import-format")
    fireEvent.mouseDown(
      formatSelect.querySelector(".ant-select-selector") ?? formatSelect
    )
    fireEvent.click(screen.getByText("APKG (Anki)"))

    const smallFile = createApkgFile(4, "small.apkg")
    fireEvent.change(screen.getByTestId("flashcards-import-apkg-input"), {
      target: { files: [smallFile] }
    })

    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    await waitFor(() => {
      expect(apkgMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(screen.queryByText("Confirm large import")).not.toBeInTheDocument()
  })

  it("maps export options and normalized tag filters to preview and export params", async () => {
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 7,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ],
      isLoading: false
    } as any)
    vi.mocked(listFlashcards).mockResolvedValue({
      items: [],
      count: 0,
      total: 42
    } as any)
    vi.mocked(exportFlashcards).mockResolvedValue("Deck,Front,Back\nBiology,Q,A")

    render(<ImportExportTab />)

    const deckSelect = screen.getByTestId("flashcards-export-deck")
    fireEvent.mouseDown(deckSelect.querySelector(".ant-select-selector") ?? deckSelect)
    const biologyOptions = screen.getAllByText("Biology")
    fireEvent.click(biologyOptions[biologyOptions.length - 1])

    fireEvent.change(screen.getByTestId("flashcards-export-tag"), {
      target: { value: "Chapter-1" }
    })
    fireEvent.change(screen.getByTestId("flashcards-export-query"), {
      target: { value: "mitosis" }
    })
    fireEvent.click(screen.getByTestId("flashcards-export-include-reverse"))

    const delimiterSelect = screen.getByTestId("flashcards-export-delimiter")
    fireEvent.mouseDown(
      delimiterSelect.querySelector(".ant-select-selector") ?? delimiterSelect
    )
    fireEvent.click(screen.getByText(", (Comma)"))

    fireEvent.click(screen.getByTestId("flashcards-export-include-header"))
    fireEvent.click(screen.getByTestId("flashcards-export-extended-header"))

    await waitFor(() => {
      expect(screen.getByTestId("flashcards-export-preview")).toHaveTextContent(
        "42 cards from Biology"
      )
    })
    expect(screen.getByTestId("flashcards-export-preview")).toHaveAttribute(
      "data-ds-component",
      "Alert"
    )
    const exportPreviewQuery = useQueryMock.mock.calls
      .map(([options]) => options as any)
      .findLast((options) => options.queryKey?.[0] === "flashcards:export-preview-count")
    expect(exportPreviewQuery.queryKey[2]).toBe("chapter-1")
    await exportPreviewQuery.queryFn()
    expect(listFlashcards).toHaveBeenCalledWith(
      expect.objectContaining({
        tag: "chapter-1"
      })
    )

    fireEvent.click(screen.getByTestId("flashcards-export-button"))

    await waitFor(() => {
      expect(exportFlashcards).toHaveBeenCalledTimes(1)
    })
    expect(exportFlashcards).toHaveBeenCalledWith({
      deck_id: 7,
      tag: "chapter-1",
      q: "mitosis",
      format: "csv",
      include_reverse: true,
      delimiter: ",",
      include_header: true,
      extended_header: true
    })
  })

  it("serializes object payloads before creating JSON export blobs", async () => {
    const jsonPayload = {
      cards: [
        {
          front: "Question",
          back: "Answer"
        }
      ]
    }
    vi.mocked(exportFlashcards).mockResolvedValue(jsonPayload as any)
    const createObjectURLMock = vi.fn((_blob: Blob | MediaSource) => "blob:mock")
    ;(URL as any).createObjectURL = createObjectURLMock

    render(<ImportExportTab />)

    const formatSelect = screen.getByTestId("flashcards-export-format")
    fireEvent.mouseDown(formatSelect.querySelector(".ant-select-selector") ?? formatSelect)
    fireEvent.click(screen.getByText("JSON"))
    fireEvent.click(screen.getByTestId("flashcards-export-button"))

    await waitFor(() => {
      expect(createObjectURLMock).toHaveBeenCalledTimes(1)
    })
    const blob = createObjectURLMock.mock.calls[0][0] as Blob
    await expect(blob.text()).resolves.toBe(JSON.stringify(jsonPayload, null, 2))
  })

  it("renders transfer summary cards for formats, limits, and last action", () => {
    vi.mocked(useImportLimitsQuery).mockReturnValue({
      data: {
        max_lines: 500,
        max_line_length: 32768,
        max_field_length: 1048576
      }
    } as any)

    render(<ImportExportTab />)

    expect(screen.getByTestId("flashcards-transfer-summary")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-transfer-summary-formats")).toHaveTextContent(
      "Import: CSV, TSV, JSON, JSONL, Structured Q&A, APKG · Author: Generate, Image Occlusion · Export: TSV, CSV, JSON, APKG"
    )
    const limits = screen.getByTestId("flashcards-transfer-summary-limits")
    expect(limits).toHaveTextContent(`${(500).toLocaleString()} lines`)
    expect(limits).toHaveTextContent(`${(32768).toLocaleString()} bytes per line`)
    expect(limits).toHaveTextContent(`${(1048576).toLocaleString()} bytes per field`)
    expect(screen.getByTestId("flashcards-transfer-summary-last-action")).toHaveTextContent(
      "No transfer actions yet in this session."
    )
  })

  it("updates transfer summary after import actions", async () => {
    const mutateAsync = vi.fn().mockResolvedValue({
      imported: 3,
      items: [
        { uuid: "u1", deck_id: 1 },
        { uuid: "u2", deck_id: 1 },
        { uuid: "u3", deck_id: 1 }
      ],
      errors: []
    })
    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: {
        value: "Deck\tFront\tBack\nBiology\tQuestion\tAnswer"
      }
    })
    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    await waitFor(() => {
      expect(screen.getByTestId("flashcards-transfer-summary-last-action")).toHaveTextContent(
        "Import Flashcards · Imported 3 cards."
      )
    })
  })

  it("supports generate preview, edit, and save flow", async () => {
    const generateMutateAsync = vi.fn().mockResolvedValue({
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
    const createCardMutateAsync = vi.fn().mockResolvedValue({})
    vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
      mutateAsync: generateMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: createCardMutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-generate-text"), {
      target: { value: "Photosynthesis notes..." }
    })
    fireEvent.click(screen.getByTestId("flashcards-generate-button"))

    await waitFor(() => {
      expect(generateMutateAsync).toHaveBeenCalledTimes(1)
    })

    const frontEditor = screen.getByDisplayValue("Generated front")
    fireEvent.change(frontEditor, {
      target: { value: "Edited front" }
    })

    fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))

    await waitFor(() => {
      expect(createCardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(createCardMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        front: "Edited front",
        back: "Generated back",
        tags: ["tag-1"],
        model_type: "basic",
        is_cloze: false,
        reverse: false
      })
    )
    await waitFor(() => {
      expect(screen.queryByDisplayValue("Edited front")).not.toBeInTheDocument()
    })
  })

  it("retains only failed generated drafts after partial save", async () => {
    const generateMutateAsync = vi.fn().mockResolvedValue({
      flashcards: [
        { front: "Card A", back: "Back A", tags: ["a"], model_type: "basic" },
        { front: "Card B", back: "Back B", tags: ["b"], model_type: "basic" },
        { front: "Card C", back: "Back C", tags: ["c"], model_type: "basic" }
      ],
      count: 3
    })
    const createCardMutateAsync = vi
      .fn()
      .mockResolvedValueOnce({})
      .mockRejectedValueOnce(new Error("save failed"))
      .mockResolvedValueOnce({})

    vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
      mutateAsync: generateMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: createCardMutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-generate-text"), {
      target: { value: "Interleaved save case" }
    })
    fireEvent.click(screen.getByTestId("flashcards-generate-button"))

    await waitFor(() => {
      expect(generateMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(screen.getByDisplayValue("Card A")).toBeInTheDocument()
    expect(screen.getByDisplayValue("Card B")).toBeInTheDocument()
    expect(screen.getByDisplayValue("Card C")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))

    await waitFor(() => {
      expect(createCardMutateAsync).toHaveBeenCalledTimes(3)
    })
    expect(messageSpies.warning).toHaveBeenCalledWith("Saved 2 cards; 1 failed.")

    await waitFor(() => {
      expect(screen.queryByDisplayValue("Card A")).not.toBeInTheDocument()
      expect(screen.getByDisplayValue("Card B")).toBeInTheDocument()
      expect(screen.queryByDisplayValue("Card C")).not.toBeInTheDocument()
    })
  })

  it("keeps generated drafts when all generated-card saves fail", async () => {
    const generateMutateAsync = vi.fn().mockResolvedValue({
      flashcards: [
        { front: "Fail A", back: "Back A", tags: ["a"], model_type: "basic" },
        { front: "Fail B", back: "Back B", tags: ["b"], model_type: "basic" }
      ],
      count: 2
    })
    const createCardMutateAsync = vi
      .fn()
      .mockRejectedValueOnce(new Error("save failed"))
      .mockRejectedValueOnce(new Error("save failed"))

    vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
      mutateAsync: generateMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: createCardMutateAsync,
      isPending: false
    } as any)

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-generate-text"), {
      target: { value: "All fail case" }
    })
    fireEvent.click(screen.getByTestId("flashcards-generate-button"))

    await waitFor(() => {
      expect(generateMutateAsync).toHaveBeenCalledTimes(1)
    })

    fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))

    await waitFor(() => {
      expect(createCardMutateAsync).toHaveBeenCalledTimes(2)
    })
    expect(messageSpies.error).toHaveBeenCalledWith("Failed to save generated cards.")

    expect(screen.getByDisplayValue("Fail A")).toBeInTheDocument()
    expect(screen.getByDisplayValue("Fail B")).toBeInTheDocument()
  })

  it("attaches source attribution when launched from deep-link intent", async () => {
    const generateMutateAsync = vi.fn().mockResolvedValue({
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
    const createCardMutateAsync = vi.fn().mockResolvedValue({})
    vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
      mutateAsync: generateMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useCreateFlashcardMutation).mockReturnValue({
      mutateAsync: createCardMutateAsync,
      isPending: false
    } as any)

    render(
      <ImportExportTab
        generateIntent={{
          text: "Cell respiration notes",
          sourceType: "media",
          sourceId: "55",
          sourceTitle: "Lecture 2"
        }}
      />
    )

    expect(screen.getByTestId("flashcards-generate-source-context")).toHaveTextContent(
      "Source context attached"
    )
    expect(screen.getByTestId("flashcards-generate-text")).toHaveValue(
      "Cell respiration notes"
    )

    fireEvent.click(screen.getByTestId("flashcards-generate-button"))
    await waitFor(() => {
      expect(generateMutateAsync).toHaveBeenCalledTimes(1)
    })
    fireEvent.click(screen.getByTestId("flashcards-generate-save-button"))

    await waitFor(() => {
      expect(createCardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(createCardMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        source_ref_type: "media",
        source_ref_id: "55"
      })
    )
  })

  it("offers undo rollback for the latest import batch", async () => {
    const mutateAsync = vi.fn().mockResolvedValue({
      imported: 2,
      items: [
        { uuid: "undo-u1", deck_id: 1 },
        { uuid: "undo-u2", deck_id: 1 }
      ],
      errors: []
    })
    vi.mocked(useImportFlashcardsMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)
    vi.mocked(getFlashcard)
      .mockResolvedValueOnce({ uuid: "undo-u1", version: 7 } as any)
      .mockResolvedValueOnce({ uuid: "undo-u2", version: 8 } as any)
    vi.mocked(deleteFlashcard).mockResolvedValue(undefined as any)

    render(<ImportExportTab />)

    fireEvent.change(screen.getByTestId("flashcards-import-textarea"), {
      target: {
        value: "Deck\tFront\tBack\nBiology\tQuestion\tAnswer"
      }
    })
    fireEvent.click(screen.getByTestId("flashcards-import-button"))

    await waitFor(() => {
      expect(showUndoNotificationMock).toHaveBeenCalledTimes(1)
    })
    const undoConfig = showUndoNotificationMock.mock.calls[0][0]
    expect(undoConfig.duration).toBe(30)
    expect(String(undoConfig.description)).toContain("Undo within 30s")

    await undoConfig.onUndo()

    expect(getFlashcard).toHaveBeenCalledTimes(2)
    expect(deleteFlashcard).toHaveBeenCalledWith("undo-u1", 7)
    expect(deleteFlashcard).toHaveBeenCalledWith("undo-u2", 8)
    expect(invalidateQueriesMock).toHaveBeenCalled()
  })
})
