import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { CreateTab } from "../CreateTab"
import { useCreateQuestionMutation, useCreateQuizMutation } from "../../hooks"

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
      return typeof defaultValue === "string" ? defaultValue : key
    }
  })
}))

vi.mock("../../hooks", () => ({
  useCreateQuizMutation: vi.fn(),
  useCreateQuestionMutation: vi.fn(),
  useCreateOsceStationMutation: vi.fn(() => ({ mutateAsync: vi.fn(), isPending: false }))
}))

describe("CreateTab validation accessibility", () => {
  const createQuizMutateAsync = vi.fn(async () => ({ id: 101 }))
  const createQuestionMutateAsync = vi.fn(async () => undefined)

  beforeEach(() => {
    vi.clearAllMocks()

    if (!(globalThis.crypto as Crypto | undefined)?.randomUUID) {
      Object.defineProperty(globalThis, "crypto", {
        value: {
          randomUUID: () => "test-question-id"
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

  it("shows inline form validation without generic failure toast when quiz name is missing", async () => {
    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.click(
      screen.getByRole("button", { name: /add your first question/i })
    )

    fireEvent.click(screen.getByRole("button", { name: /save quiz/i }))

    await waitFor(() => {
      expect(screen.getByText("Please enter a quiz name")).toBeInTheDocument()
    })

    expect(createQuizMutateAsync).not.toHaveBeenCalled()
    expect(screen.queryByText("Failed to create quiz")).not.toBeInTheDocument()
  }, 15000)
})
