import { fireEvent, render, screen, within } from "@testing-library/react"
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
  useCreateQuestionMutation: vi.fn(),
  useCreateOsceStationMutation: vi.fn(() => ({ mutateAsync: vi.fn(), isPending: false }))
}))

describe("CreateTab preview", () => {
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
      mutateAsync: vi.fn(async () => undefined),
      isPending: false
    } as any)
  })

  it("shows explanation visibility helper text in question editor", () => {
    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))

    expect(
      screen.getByText("Shown to the learner after they submit the quiz.")
    ).toBeInTheDocument()
  })

  it("renders a read-only preview with mixed question types and explanations", async () => {
    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.change(screen.getByPlaceholderText("e.g., Biology Chapter 5"), {
      target: { value: "Plant Biology" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))
    fireEvent.click(screen.getByRole("button", { name: /Add Question/i }))
    fireEvent.click(screen.getByRole("button", { name: /Add Question/i }))

    const questionInputs = screen.getAllByPlaceholderText("Enter your question...")
    fireEvent.change(questionInputs[0], { target: { value: "What is chlorophyll?" } })
    fireEvent.change(questionInputs[1], { target: { value: "Plants breathe oxygen." } })
    fireEvent.change(questionInputs[2], { target: { value: "The process is called _____." } })

    const option1Inputs = screen.getAllByPlaceholderText("Option 1")
    const option2Inputs = screen.getAllByPlaceholderText("Option 2")
    fireEvent.change(option1Inputs[0], { target: { value: "A pigment" } })
    fireEvent.change(option2Inputs[0], { target: { value: "A hormone" } })

    const explanations = screen.getAllByPlaceholderText("Explanation (shown after answering)...")
    fireEvent.change(explanations[0], { target: { value: "Base explanation." } })

    const typeSelectors = screen.getAllByRole("combobox")
    fireEvent.mouseDown(typeSelectors[1])
    const trueFalseOptions = await screen.findAllByText("True/False")
    fireEvent.click(trueFalseOptions[trueFalseOptions.length - 1])

    fireEvent.mouseDown(typeSelectors[2])
    const fillBlankOptions = await screen.findAllByText("Fill in the Blank")
    fireEvent.click(fillBlankOptions[fillBlankOptions.length - 1])
    fireEvent.keyDown(document, { key: "Escape" })

    fireEvent.change(screen.getByPlaceholderText("Enter the correct answer..."), {
      target: { value: "photosynthesis" }
    })

    fireEvent.click(screen.getByRole("button", { name: /^Preview$/i }))

    await screen.findByText("Quiz Preview")
    const modalContent = document.querySelector(".ant-modal")
    expect(modalContent).not.toBeNull()
    if (!modalContent) {
      return
    }

    const modalQueries = within(modalContent as HTMLElement)
    expect(modalQueries.getByText("Plant Biology")).toBeInTheDocument()
    expect(modalQueries.getByText("What is chlorophyll?")).toBeInTheDocument()
    expect(modalQueries.getByText("A. A pigment")).toBeInTheDocument()
    expect(modalQueries.getByText("Plants breathe oxygen.")).toBeInTheDocument()
    expect(modalQueries.getByText("True/False")).toBeInTheDocument()
    expect(modalQueries.getByText("The process is called _____.")).toBeInTheDocument()
    expect(modalQueries.getByText("Fill in the Blank")).toBeInTheDocument()
    expect(modalQueries.getByText("Explanation: Base explanation.")).toBeInTheDocument()
  }, 20000)
})
