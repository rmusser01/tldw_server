import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ImportExportTab } from "../ImportExportTab"
import {
  useCreateDeckMutation,
  useCreateFlashcardMutation,
  useCreateFlashcardsBulkMutation,
  useDecksQuery,
  useGenerateFlashcardsMutation,
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

const messageSpies = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}

const { useQueryMock } = vi.hoisted(() => ({
  useQueryMock: vi.fn()
}))

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>(
    "@tanstack/react-query"
  )
  return {
    ...actual,
    useQuery: useQueryMock,
    useQueryClient: () => ({
      invalidateQueries: vi.fn()
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

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | { defaultValue?: string }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String(
              (defaultValueOrOptions as Record<string, unknown>)[token] ??
                `{{${token}}}`
            )
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
  useStudyPackCreateMutation: vi.fn(),
  useStudyPackJobQuery: vi.fn(),
  useStudyPackQuery: vi.fn(),
  useStudyPackRegenerateMutation: vi.fn()
}))

vi.mock("@/services/flashcards", async () => {
  const actual = await vi.importActual<typeof import("@/services/flashcards")>(
    "@/services/flashcards"
  )
  return {
    ...actual,
    getFlashcard: vi.fn(),
    deleteFlashcard: vi.fn(),
    listFlashcards: vi.fn(),
    exportFlashcards: vi.fn(),
    exportFlashcardsFile: vi.fn()
  }
})

vi.mock("@/services/prompt-studio", () => ({
  getLlmProviders: vi.fn()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
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

function setupMutationMocks() {
  vi.mocked(useGenerateFlashcardsMutation).mockReturnValue({
    mutateAsync: vi.fn(),
    isPending: false
  } as any)
  vi.mocked(useCreateFlashcardMutation).mockReturnValue({
    mutateAsync: vi.fn(),
    isPending: false
  } as any)
  vi.mocked(useCreateFlashcardsBulkMutation).mockReturnValue({
    mutateAsync: vi.fn().mockResolvedValue({ items: [], count: 0, total: 0 }),
    isPending: false
  } as any)
  vi.mocked(useCreateDeckMutation).mockReturnValue({
    mutateAsync: vi.fn().mockResolvedValue({
      id: 999,
      name: "Test",
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
  vi.mocked(useDecksQuery).mockReturnValue({
    data: [{ id: 1, name: "Default", description: null, deleted: false }],
    isLoading: false
  } as any)
  vi.mocked(useImportLimitsQuery).mockReturnValue({
    data: null,
    isLoading: false
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
}

describe("ImportExportTab LLM provider gating", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setupMutationMocks()
  })

  it("shows no-LLM banner when providers query returns empty list", () => {
    useQueryMock.mockImplementation((opts: { queryKey: string[] }) => {
      if (opts?.queryKey?.[1] === "llm-providers") {
        return {
          data: { ok: true, status: 200, data: { providers: [], total_configured: 0 } },
          isLoading: false,
          isError: false
        }
      }
      return { data: 42, isLoading: false, isError: false }
    })

    render(<ImportExportTab />)

    expect(
      screen.getByTestId("flashcards-generate-no-llm-banner")
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Flashcard generation requires an LLM provider. Configure one in Settings \u2192 LLM Providers."
      )
    ).toBeInTheDocument()
  })

  it("disables Generate button when no LLM providers available", () => {
    useQueryMock.mockImplementation((opts: { queryKey: string[] }) => {
      if (opts?.queryKey?.[1] === "llm-providers") {
        return {
          data: { ok: true, status: 200, data: { providers: [], total_configured: 0 } },
          isLoading: false,
          isError: false
        }
      }
      return { data: 42, isLoading: false, isError: false }
    })

    render(<ImportExportTab />)

    // Seed sourceText so the only disable reason is missing LLM providers
    fireEvent.change(screen.getByTestId("flashcards-generate-text"), {
      target: { value: "Some study material for testing" }
    })

    const generateButton = screen.getByTestId("flashcards-generate-button")
    expect(generateButton).toBeDisabled()
  })

  it("does not show no-LLM banner when providers are available", () => {
    useQueryMock.mockImplementation((opts: { queryKey: string[] }) => {
      if (opts?.queryKey?.[1] === "llm-providers") {
        return {
          data: { ok: true, status: 200, data: { providers: [{ id: "openai" }], total_configured: 1 } },
          isLoading: false,
          isError: false
        }
      }
      return { data: 42, isLoading: false, isError: false }
    })

    render(<ImportExportTab />)

    // Seed sourceText so button enable state reflects LLM availability
    fireEvent.change(screen.getByTestId("flashcards-generate-text"), {
      target: { value: "Some study material for testing" }
    })

    expect(
      screen.queryByTestId("flashcards-generate-no-llm-banner")
    ).not.toBeInTheDocument()

    expect(screen.getByTestId("flashcards-generate-button")).not.toBeDisabled()
  })

  it("treats query errors optimistically (no banner shown)", () => {
    useQueryMock.mockImplementation((opts: { queryKey: string[] }) => {
      if (opts?.queryKey?.[1] === "llm-providers") {
        return {
          data: null,
          isLoading: false,
          isError: true
        }
      }
      return { data: 42, isLoading: false, isError: false }
    })

    render(<ImportExportTab />)

    expect(
      screen.queryByTestId("flashcards-generate-no-llm-banner")
    ).not.toBeInTheDocument()
  })
})
