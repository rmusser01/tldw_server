import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
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

describe("CreateTab draft safety", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()

    if (!(globalThis.crypto as Crypto | undefined)?.randomUUID) {
      Object.defineProperty(globalThis, "crypto", {
        value: {
          randomUUID: () => "draft-question-id"
        },
        configurable: true
      })
    }

    vi.mocked(useCreateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({ id: 900 })),
      isPending: false
    } as any)

    vi.mocked(useCreateQuestionMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => undefined),
      isPending: false
    } as any)
  })

  it("offers draft recovery and restores draft state", async () => {
    window.localStorage.setItem(
      "quiz-create-draft-v1",
      JSON.stringify({
        name: "Recovered Quiz",
        description: "Recovered description",
        timeLimit: 15,
        passingScore: 80,
        questions: [
          {
            key: "restored-question",
            question_type: "multiple_choice",
            question_text: "Recovered question?",
            options: ["A", "B", "C", "D"],
            correct_answer: 1,
            explanation: "Recovered explanation"
          }
        ],
        updatedAt: Date.now()
      })
    )

    render(<CreateTab onNavigateToTake={() => {}} />)

    expect(screen.getByText("Saved draft found")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Restore" }))

    await waitFor(() => {
      expect(screen.getByDisplayValue("Recovered Quiz")).toBeInTheDocument()
    })
    expect(screen.getByText(/Questions/)).toHaveTextContent("(1)")
    expect(screen.getByDisplayValue("Recovered question?")).toBeInTheDocument()
  }, 15000)

  it("discards persisted and pending question drafts when switching to OSCE", async () => {
    window.localStorage.setItem(
      "quiz-create-draft-v1",
      JSON.stringify({
        name: "Recovered Quiz",
        description: "Recovered description",
        timeLimit: 15,
        passingScore: 80,
        questions: [{
          key: "discarded-question",
          question_type: "multiple_choice",
          question_text: "Do not restore me",
          options: ["A", "B"],
          correct_answer: 0,
          explanation: ""
        }],
        updatedAt: Date.now()
      })
    )
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)
    const firstRender = render(<CreateTab onNavigateToTake={() => {}} />)
    expect(screen.getByText("Saved draft found")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("radio", { name: "OSCE" }))

    expect(confirmSpy).toHaveBeenCalledWith(
      "Switch activity type and discard the current activity content?"
    )
    expect(window.localStorage.getItem("quiz-create-draft-v1")).toBeNull()
    expect(screen.queryByText("Saved draft found")).not.toBeInTheDocument()

    firstRender.unmount()
    render(<CreateTab onNavigateToTake={() => {}} />)
    expect(screen.queryByText("Saved draft found")).not.toBeInTheDocument()
    expect(screen.queryByDisplayValue("Do not restore me")).not.toBeInTheDocument()
  })

  it("signals dirty state and blocks browser unload while dirty", () => {
    const onDirtyStateChange = vi.fn()

    render(
      <CreateTab
        onNavigateToTake={() => {}}
        onDirtyStateChange={onDirtyStateChange}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))

    expect(onDirtyStateChange).toHaveBeenCalledWith(true)

    const event = new Event("beforeunload", { cancelable: true })
    window.dispatchEvent(event)

    expect(event.defaultPrevented).toBe(true)
  })

  it("shows storage warning when draft autosave cannot write", async () => {
    const storagePrototype = Object.getPrototypeOf(window.localStorage)
    const setItemSpy = vi
      .spyOn(storagePrototype, "setItem")
      .mockImplementation(() => {
        throw new Error("storage unavailable")
      })

    render(<CreateTab onNavigateToTake={() => {}} />)

    fireEvent.click(screen.getByRole("button", { name: /Add Your First Question/i }))

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 350))
    })

    expect(
      screen.getByText(
        "Draft autosave unavailable: your progress will not be preserved if you leave."
      )
    ).toBeInTheDocument()

    setItemSpy.mockRestore()
  })
})
