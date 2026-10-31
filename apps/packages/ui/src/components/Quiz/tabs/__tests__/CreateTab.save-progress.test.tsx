import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
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

describe("CreateTab save pipeline", () => {
  const createQuizMutateAsync = vi.fn(async () => ({ id: 55 }))
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
      mutateAsync: createQuizMutateAsync,
      isPending: false
    } as any)

    vi.mocked(useCreateQuestionMutation).mockReturnValue({
      mutateAsync: createQuestionMutateAsync,
      isPending: false
    } as any)
  })

  const addFilledQuestion = (index: number, questionText: string) => {
    if (index === 0) {
      fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))
    } else {
      fireEvent.click(screen.getByRole("button", { name: /Add Question/i }))
    }

    const questionInputs = screen.getAllByPlaceholderText("Enter your question...")
    const option1Inputs = screen.getAllByPlaceholderText("Option 1")
    const option2Inputs = screen.getAllByPlaceholderText("Option 2")

    fireEvent.change(questionInputs[index], { target: { value: questionText } })
    fireEvent.change(option1Inputs[index], { target: { value: `Option A${index}` } })
    fireEvent.change(option2Inputs[index], { target: { value: `Option B${index}` } })
  }

  it("shows incremental save progress while creating questions", async () => {
    let resolveQuestion: (() => void) | null = null
    const pendingQuestion = new Promise<void>((resolve) => {
      resolveQuestion = resolve
    })

    createQuestionMutateAsync.mockReturnValueOnce(pendingQuestion)

    const onNavigateToTake = vi.fn()
    render(<CreateTab onNavigateToTake={onNavigateToTake} />)

    fireEvent.change(screen.getByPlaceholderText("e.g., Biology Chapter 5"), {
      target: { value: "Progress Quiz" }
    })
    addFilledQuestion(0, "Question one")

    fireEvent.click(screen.getByRole("button", { name: /Save Quiz/i }))

    expect(await screen.findByText("Saving question 1 of 1...")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^Preview$/i })).toBeDisabled()

    await act(async () => {
      resolveQuestion?.()
      await pendingQuestion
    })

    await waitFor(() => {
      expect(screen.queryByText("Saving question 1 of 1...")).not.toBeInTheDocument()
    })

    expect(onNavigateToTake).toHaveBeenCalledWith({
      highlightQuizId: 55,
      sourceTab: "create"
    })
  }, 20000)

  it("reports exact failed question index when save partially fails", async () => {
    createQuestionMutateAsync
      .mockResolvedValueOnce(undefined)
      .mockRejectedValueOnce(new Error("network"))

    const onNavigateToTake = vi.fn()
    render(<CreateTab onNavigateToTake={onNavigateToTake} />)

    fireEvent.change(screen.getByPlaceholderText("e.g., Biology Chapter 5"), {
      target: { value: "Partial Failure Quiz" }
    })

    addFilledQuestion(0, "Question one")
    addFilledQuestion(1, "Question two")

    fireEvent.click(screen.getByRole("button", { name: /Save Quiz/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Quiz was created, but saving failed at question 2 of 2.")
      ).toBeInTheDocument()
    })

    expect(createQuestionMutateAsync).toHaveBeenCalledTimes(2)
    expect(onNavigateToTake).not.toHaveBeenCalled()
  }, 20000)
})
