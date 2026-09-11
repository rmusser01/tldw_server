import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { CreateTab } from "../CreateTab"
import { useCreateQuestionMutation, useCreateQuizMutation } from "../../hooks"

const interpolate = (template: string, values: Record<string, unknown> | undefined) => {
  return template.replace(/\{\{\s*([^\s}]+)\s*\}\}/g, (_, key: string) => {
    const value = values?.[key]
    return value == null ? "" : String(value)
  })
}

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
  useCreateQuizMutation: vi.fn(),
  useCreateQuestionMutation: vi.fn()
}))

describe("CreateTab flexible composition", () => {
  const createQuestionMutateAsync = vi.fn(async () => undefined)

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()

    if (!(globalThis.crypto as Crypto | undefined)?.randomUUID) {
      Object.defineProperty(globalThis, "crypto", {
        value: {
          randomUUID: () => `question-${Math.random()}`
        },
        configurable: true
      })
    }

    vi.mocked(useCreateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({ id: 42 })),
      isPending: false
    } as any)

    vi.mocked(useCreateQuestionMutation).mockReturnValue({
      mutateAsync: createQuestionMutateAsync,
      isPending: false
    } as any)
  })

  it("supports reordering questions and persists order_index accordingly", async () => {
    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.change(screen.getByPlaceholderText("e.g., Biology Chapter 5"), {
      target: { value: "My Quiz" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))
    fireEvent.click(screen.getByRole("button", { name: /Add Question/i }))

    const questionInputs = screen.getAllByPlaceholderText("Enter your question...")
    fireEvent.change(questionInputs[0], { target: { value: "First question" } })
    fireEvent.change(questionInputs[1], { target: { value: "Second question" } })

    const option1Inputs = screen.getAllByPlaceholderText("Option 1")
    const option2Inputs = screen.getAllByPlaceholderText("Option 2")
    fireEvent.change(option1Inputs[0], { target: { value: "A1" } })
    fireEvent.change(option2Inputs[0], { target: { value: "A2" } })
    fireEvent.change(option1Inputs[1], { target: { value: "B1" } })
    fireEvent.change(option2Inputs[1], { target: { value: "B2" } })

    fireEvent.click(screen.getByRole("button", { name: "Move question 1 down" }))
    fireEvent.click(screen.getByRole("button", { name: /Save Quiz/i }))

    await waitFor(() => {
      expect(createQuestionMutateAsync).toHaveBeenCalledTimes(2)
    })

    const createQuestionCalls = createQuestionMutateAsync.mock
      .calls as unknown as Array<[{
        question: {
          question_text: string
          order_index: number
        }
      }]>
    const firstPayload = createQuestionCalls[0]?.[0]
    const secondPayload = createQuestionCalls[1]?.[0]

    expect(firstPayload.question.question_text).toBe("Second question")
    expect(firstPayload.question.order_index).toBe(0)
    expect(secondPayload.question.question_text).toBe("First question")
    expect(secondPayload.question.order_index).toBe(1)
  }, 30000)

  it("enforces multiple-choice option bounds between 2 and 6", () => {
    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))

    const addOptionButton = screen.getByRole("button", { name: /Add Option/i })
    fireEvent.click(addOptionButton)
    fireEvent.click(addOptionButton)

    expect(addOptionButton).toBeDisabled()
    expect(screen.getAllByPlaceholderText(/Option /)).toHaveLength(6)

    for (let i = 0; i < 4; i++) {
      const removableButtons = screen
        .getAllByRole("button", { name: /Remove option/i })
        .filter((button) => !button.hasAttribute("disabled"))
      if (removableButtons.length === 0) {
        break
      }
      fireEvent.click(removableButtons[0])
    }

    expect(screen.getAllByPlaceholderText(/Option /)).toHaveLength(2)
    const removeButtons = screen.getAllByRole("button", { name: /Remove option/i })
    removeButtons.forEach((button) => {
      expect(button).toBeDisabled()
    })
  }, 20000)

  it("remaps selected correct answer when an earlier option is removed", () => {
    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))

    const radios = screen.getAllByRole("radio", { name: /Mark option .* as correct/i })
    fireEvent.click(radios[2])
    expect(radios[2]).toBeChecked()

    fireEvent.click(screen.getByRole("button", { name: /Remove option 2 for question 1/i }))

    const radiosAfterRemoval = screen.getAllByRole("radio", { name: /Mark option .* as correct/i })
    expect(radiosAfterRemoval).toHaveLength(3)
    expect(radiosAfterRemoval[1]).toBeChecked()
  }, 15000)

  it("uses mobile-friendly stacked option rows and touch-target sizing", () => {
    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))

    const optionRow = screen.getByTestId("create-option-row-0-0")
    expect(optionRow.className).toContain("flex-col")

    const detailsGrid = screen.getByTestId("create-details-grid")
    expect(detailsGrid.className).toContain("grid-cols-1")
    expect(detailsGrid.className).toContain("sm:grid-cols-2")

    expect(screen.getByRole("button", { name: /Add Option/i }).className).toContain("min-h-11")
    expect(screen.getByRole("button", { name: "Move question 1 up" }).className).toContain("min-h-11")
    expect(screen.getByRole("button", { name: /Remove option 1 for question 1/i }).className).toContain("min-h-11")
  }, 15000)

  it("serializes matching questions with prompt options and pair map", async () => {
    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.change(screen.getByPlaceholderText("e.g., Biology Chapter 5"), {
      target: { value: "Systems Quiz" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))

    const typeSelector = screen.getAllByRole("combobox")[0]
    fireEvent.mouseDown(typeSelector)
    fireEvent.click(await screen.findByText("Matching"))

    fireEvent.change(screen.getByPlaceholderText("Enter your question..."), {
      target: { value: "Match each term to its description." }
    })

    const leftInputs = screen.getAllByPlaceholderText("Left item")
    const rightInputs = screen.getAllByPlaceholderText("Matching item")

    fireEvent.change(leftInputs[0], { target: { value: "CPU" } })
    fireEvent.change(rightInputs[0], { target: { value: "Processor" } })

    fireEvent.change(leftInputs[1], { target: { value: "RAM" } })
    fireEvent.change(rightInputs[1], { target: { value: "Memory" } })

    fireEvent.click(screen.getByRole("button", { name: /Save Quiz/i }))

    await waitFor(() => {
      expect(createQuestionMutateAsync).toHaveBeenCalledTimes(1)
    })

    expect(createQuestionMutateAsync).toHaveBeenCalledWith({
      quizId: 42,
      question: {
        question_type: "matching",
        question_text: "Match each term to its description.",
        options: ["CPU", "RAM"],
        correct_answer: {
          CPU: "Processor",
          RAM: "Memory"
        },
        explanation: undefined,
        order_index: 0
      }
    })
  }, 20000)
})
