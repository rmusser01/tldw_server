import React from "react"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { GenerateTab } from "../GenerateTab"
import { useGenerateQuizMutation } from "../../hooks"
import { tldwClient } from "@/services/tldw"
import {
  createDeck,
  createFlashcard,
  generateFlashcards,
  listDecks,
  listFlashcards
} from "@/services/flashcards"

const navigationMocks = {
  navigate: vi.fn()
}

const interpolate = (template: string, values: Record<string, unknown> | undefined) =>
  template.replace(/\{\{\s*([^\s}]+)\s*\}\}/g, (_, key: string) => {
    const value = values?.[key]
    return value == null ? "" : String(value)
  })

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      const defaultValue = defaultValueOrOptions?.defaultValue
      if (typeof defaultValue === "string") {
        return interpolate(defaultValue, defaultValueOrOptions)
      }
      return key
    }
  })
}))

vi.mock("../../hooks", () => ({
  useGenerateQuizMutation: vi.fn()
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigationMocks.navigate
}))

vi.mock("@/services/tldw", () => ({
  tldwClient: {
    listMedia: vi.fn(),
    searchMedia: vi.fn(),
    getMediaDetails: vi.fn(),
    listNotes: vi.fn(),
    searchNotes: vi.fn()
  }
}))

vi.mock("@/services/flashcards", () => ({
  generateFlashcards: vi.fn(),
  createDeck: vi.fn(),
  createFlashcard: vi.fn(),
  listDecks: vi.fn(),
  listFlashcards: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const renderWithQueryClient = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <GenerateTab onNavigateToTake={() => {}} />
    </QueryClientProvider>
  )
}

const getRow = (type: string) => screen.getByTestId(`generate-question-plan-row-${type}`)

const getInput = (testId: string) => {
  const input = screen.getByTestId(testId).querySelector("input")
  if (!input) throw new Error(`Missing input for ${testId}`)
  return input
}

const setNumber = (testId: string, value: number) => {
  fireEvent.change(getInput(testId), { target: { value: String(value) } })
}

const selectDefaultMedia = async () => {
  await waitFor(() => {
    expect(screen.getByText("1 media items available")).toBeInTheDocument()
  })
  fireEvent.mouseDown(screen.getAllByRole("combobox")[0])
  fireEvent.click(await screen.findByText("Biology Notes (pdf)"))
}

describe("GenerateTab question plan controls", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    navigationMocks.navigate.mockReset()

    vi.mocked(tldwClient.listMedia).mockResolvedValue({
      items: [{ id: 10, title: "Biology Notes", type: "pdf" }],
      pagination: { total_items: 1 }
    } as any)
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({} as any)
    vi.mocked(tldwClient.listNotes).mockResolvedValue({ items: [] } as any)
    vi.mocked(tldwClient.searchNotes).mockResolvedValue({ items: [] } as any)
    vi.mocked(generateFlashcards).mockResolvedValue({ flashcards: [], count: 0 } as any)
    vi.mocked(listDecks).mockResolvedValue([] as any)
    vi.mocked(listFlashcards).mockResolvedValue({ items: [], count: 0 } as any)
    vi.mocked(createDeck).mockResolvedValue({ id: 100, name: "Generated Deck" } as any)
    vi.mocked(createFlashcard).mockResolvedValue({ uuid: "card-1" } as any)

    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        quiz: { id: 1, name: "Generated Quiz" },
        questions: Array.from({ length: 10 }, (_, index) => ({ id: index + 1 }))
      })),
      isPending: false
    } as any)
  })

  it("renders the fixed five-row plan without a table layout", async () => {
    renderWithQueryClient()

    expect(screen.getByTestId("generate-question-plan")).toBeInTheDocument()
    expect(screen.getByTestId("generate-question-plan")).not.toHaveAttribute("role", "table")
    expect(getRow("multiple_choice")).toBeInTheDocument()
    expect(getRow("true_false")).toBeInTheDocument()
    expect(getRow("fill_blank")).toBeInTheDocument()
    expect(getRow("multi_select")).toBeInTheDocument()
    expect(getRow("matching")).toBeInTheDocument()
  })

  it("shows default row state and total", () => {
    renderWithQueryClient()

    expect(within(getRow("multiple_choice")).getByRole("checkbox")).toBeChecked()
    expect(getInput("generate-question-plan-count-multiple_choice")).toHaveValue("5")
    expect(getInput("generate-question-plan-option-count-multiple_choice")).toHaveValue("4")

    expect(within(getRow("true_false")).getByRole("checkbox")).toBeChecked()
    expect(getInput("generate-question-plan-count-true_false")).toHaveValue("3")

    expect(within(getRow("fill_blank")).getByRole("checkbox")).toBeChecked()
    expect(getInput("generate-question-plan-count-fill_blank")).toHaveValue("2")

    expect(within(getRow("multi_select")).getByRole("checkbox")).not.toBeChecked()
    expect(getInput("generate-question-plan-count-multi_select")).toHaveValue("1")
    expect(getInput("generate-question-plan-option-count-multi_select")).toHaveValue("4")

    expect(within(getRow("matching")).getByRole("checkbox")).not.toBeChecked()
    expect(getInput("generate-question-plan-count-matching")).toHaveValue("1")
    expect(getInput("generate-question-plan-pair-count-matching")).toHaveValue("4")

    expect(screen.getByTestId("generate-question-plan-total")).toHaveTextContent("Total: 10")
  })

  it("updates total when counts and rows change", () => {
    renderWithQueryClient()

    setNumber("generate-question-plan-count-multiple_choice", 6)
    fireEvent.click(within(getRow("multi_select")).getByRole("checkbox"))

    expect(screen.getByTestId("generate-question-plan-total")).toHaveTextContent("Total: 12")
  })

  it("disables generate when total is 0 or greater than 100", async () => {
    renderWithQueryClient()
    await selectDefaultMedia()

    fireEvent.click(within(getRow("multiple_choice")).getByRole("checkbox"))
    fireEvent.click(within(getRow("true_false")).getByRole("checkbox"))
    fireEvent.click(within(getRow("fill_blank")).getByRole("checkbox"))
    expect(screen.getByTestId("generate-question-plan-total")).toHaveTextContent("Total: 0")
    expect(screen.getByRole("button", { name: /Generate Quiz/i })).toBeDisabled()

    fireEvent.click(within(getRow("multiple_choice")).getByRole("checkbox"))
    setNumber("generate-question-plan-count-multiple_choice", 100)
    fireEvent.click(within(getRow("true_false")).getByRole("checkbox"))
    expect(screen.getByTestId("generate-question-plan-total")).toHaveTextContent("Total: 103")
    expect(screen.getByRole("button", { name: /Generate Quiz/i })).toBeDisabled()
  }, 20000)

  it("disables row controls while generation is in flight", async () => {
    let resolveGeneration: ((value: unknown) => void) | undefined
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(
        () =>
          new Promise((resolve) => {
            resolveGeneration = resolve
          })
      ),
      isPending: false
    } as any)

    renderWithQueryClient()
    await selectDefaultMedia()

    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }))

    await waitFor(() => {
      expect(screen.getByTestId("generate-cancel-button")).toBeInTheDocument()
    })
    expect(within(getRow("multiple_choice")).getByRole("checkbox")).toBeDisabled()
    expect(getInput("generate-question-plan-count-multiple_choice")).toBeDisabled()
    expect(getInput("generate-question-plan-option-count-multiple_choice")).toBeDisabled()

    await act(async () => {
      resolveGeneration?.({
        quiz: { id: 1, name: "Generated Quiz" },
        questions: []
      })
    })
  }, 20000)

  it("submits a 5-option MCQ row with the question plan payload", async () => {
    const mutateAsync = vi.fn(async () => ({
      quiz: { id: 1, name: "Generated Quiz" },
      questions: Array.from({ length: 10 }, (_, index) => ({ id: index + 1 }))
    }))
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync,
      isPending: false
    } as any)

    renderWithQueryClient()
    await selectDefaultMedia()

    setNumber("generate-question-plan-option-count-multiple_choice", 5)
    fireEvent.click(screen.getByRole("button", { name: /Generate Quiz/i }))

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1)
    })

    expect(mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        request: expect.objectContaining({
          sources: [{ source_type: "media", source_id: "10" }],
          num_questions: 10,
          question_plan: [
            { question_type: "multiple_choice", count: 5, option_count: 5 },
            { question_type: "true_false", count: 3 },
            { question_type: "fill_blank", count: 2 }
          ]
        }),
        signal: expect.any(AbortSignal)
      })
    )
    expect(mutateAsync.mock.calls[0]?.[0]?.request).not.toHaveProperty("question_types")
  }, 20000)
})
