import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SchedulerTab } from "../SchedulerTab"
import {
  useDecksQuery,
  useDueCountsQuery,
  useUpdateDeckMutation
} from "../../hooks/useFlashcardQueries"

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
      const interpolate = (
        template: string,
        values?: {
          [key: string]: unknown
        }
      ) =>
        template.replace(/\{\{\s*([^\s}]+)\s*\}\}/g, (_match, token: string) => {
          const value = values?.[token]
          return value == null ? "" : String(value)
        })

      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return interpolate(defaultValueOrOptions.defaultValue, defaultValueOrOptions)
      }
      return key
    }
  })
}))

vi.mock("../../hooks/useFlashcardQueries", () => ({
  useDecksQuery: vi.fn(),
  useDueCountsQuery: vi.fn(),
  useUpdateDeckMutation: vi.fn()
}))

const updateDeckMock = vi.fn()
const refetchDecksMock = vi.fn()

const biologySettings = {
  sm2_plus: {
    new_steps_minutes: [1, 10],
    relearn_steps_minutes: [10],
    graduating_interval_days: 1,
    easy_interval_days: 4,
    easy_bonus: 1.3,
    interval_modifier: 1,
    max_interval_days: 36500,
    leech_threshold: 8,
    enable_fuzz: false
  },
  fsrs: {
    target_retention: 0.9,
    maximum_interval_days: 36500,
    enable_fuzz: false
  }
}

const chemistrySettings = {
  sm2_plus: {
    new_steps_minutes: [2, 20],
    relearn_steps_minutes: [15],
    graduating_interval_days: 2,
    easy_interval_days: 5,
    easy_bonus: 1.5,
    interval_modifier: 0.9,
    max_interval_days: 180,
    leech_threshold: 6,
    enable_fuzz: true
  },
  fsrs: {
    target_retention: 0.88,
    maximum_interval_days: 1825,
    enable_fuzz: true
  }
}

const biologyDeck = {
  id: 1,
  name: "Biology",
  description: "Cells",
  review_prompt_side: "front",
  deleted: false,
  client_id: "test",
  version: 2,
  scheduler_type: "sm2_plus",
  scheduler_settings_json: null,
  scheduler_settings: biologySettings
}

const chemistryDeck = {
  id: 2,
  name: "Chemistry",
  description: "Atoms",
  review_prompt_side: "back",
  deleted: false,
  client_id: "test",
  version: 5,
  scheduler_type: "sm2_plus",
  scheduler_settings_json: null,
  scheduler_settings: chemistrySettings
}

let decksData = [biologyDeck, chemistryDeck]

describe("SchedulerTab editor", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    updateDeckMock.mockReset()
    refetchDecksMock.mockReset()
    decksData = [biologyDeck, chemistryDeck]

    refetchDecksMock.mockImplementation(async () => ({ data: decksData }))
    vi.mocked(useDecksQuery).mockImplementation(
      () =>
        ({
          data: decksData,
          isLoading: false,
          refetch: refetchDecksMock
        }) as any
    )
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: {
        due: 3,
        new: 2,
        learning: 1,
        total: 6
      },
      isLoading: false
    } as any)

    vi.mocked(useUpdateDeckMutation).mockReturnValue({
      mutateAsync: updateDeckMock,
      isPending: false
    } as any)
  })

  it("applies presets, copies another deck, and resets to defaults", async () => {
    updateDeckMock
      .mockResolvedValueOnce({
        ...biologyDeck,
        version: 3,
        review_prompt_side: "back",
        scheduler_settings: chemistrySettings
      })

    render(<SchedulerTab isActive />)

    expect(screen.getByTestId("deck-scheduler-editor-field-new-steps")).toHaveValue("1, 10")
    fireEvent.mouseDown(screen.getByLabelText("Review prompt side"))
    fireEvent.click(screen.getByText("Back first"))

    fireEvent.click(screen.getByRole("button", { name: /fast acquisition/i }))
    expect(screen.getByTestId("deck-scheduler-editor-field-new-steps")).toHaveValue("1, 5, 15")

    fireEvent.change(screen.getByTestId("flashcards-scheduler-copy-select"), {
      target: { value: "2" }
    })
    fireEvent.click(screen.getByRole("button", { name: /copy settings/i }))
    expect(screen.getByTestId("deck-scheduler-editor-field-new-steps")).toHaveValue("2, 20")

    fireEvent.click(screen.getByRole("button", { name: /save changes/i }))
    await waitFor(() =>
      expect(updateDeckMock).toHaveBeenCalledWith({
        deckId: 1,
        update: {
          review_prompt_side: "back",
          scheduler_type: "sm2_plus",
          scheduler_settings: chemistrySettings,
          expected_version: 2
        }
      })
    )

    fireEvent.click(screen.getByTestId("deck-scheduler-editor-reset"))
    expect(screen.getByTestId("deck-scheduler-editor-field-new-steps")).toHaveValue("1, 10")
    expect(screen.getByTestId("deck-scheduler-editor-field-leech-threshold")).toHaveValue("8")
    expect(screen.getByTestId("deck-study-defaults-review-prompt-side")).toHaveTextContent(
      "Back first"
    )
  })

  it("blocks save when client validation fails", async () => {
    render(<SchedulerTab isActive />)

    fireEvent.change(screen.getByTestId("deck-scheduler-editor-field-leech-threshold"), {
      target: { value: "0" }
    })
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }))

    expect(updateDeckMock).not.toHaveBeenCalled()
    expect(await screen.findByText(/leech threshold must be >= 1/i)).toBeInTheDocument()
  })

  it("saves scheduler edits with optimistic locking", async () => {
    updateDeckMock.mockResolvedValue({
      ...biologyDeck,
      version: 3,
      review_prompt_side: "front",
      scheduler_settings: {
        ...biologySettings,
        sm2_plus: {
          ...biologySettings.sm2_plus,
          leech_threshold: 9
        }
      }
    })

    render(<SchedulerTab isActive />)

    fireEvent.change(screen.getByTestId("deck-scheduler-editor-field-leech-threshold"), {
      target: { value: "9" }
    })
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }))

    await waitFor(() =>
      expect(updateDeckMock).toHaveBeenCalledWith({
        deckId: 1,
        update: {
          review_prompt_side: "front",
          scheduler_type: "sm2_plus",
          scheduler_settings: {
            sm2_plus: {
              new_steps_minutes: [1, 10],
              relearn_steps_minutes: [10],
              graduating_interval_days: 1,
              easy_interval_days: 4,
              easy_bonus: 1.3,
              interval_modifier: 1,
              max_interval_days: 36500,
              leech_threshold: 9,
              enable_fuzz: false
            },
            fsrs: biologySettings.fsrs
          },
          expected_version: 2
        }
      })
    )

    expect(screen.getByText(/all changes saved/i)).toBeInTheDocument()
  })

  it("saves review prompt side changes with optimistic locking", async () => {
    updateDeckMock.mockResolvedValue({
      ...biologyDeck,
      version: 3,
      review_prompt_side: "back"
    })

    render(<SchedulerTab isActive />)

    fireEvent.mouseDown(screen.getByLabelText("Review prompt side"))
    fireEvent.click(screen.getByText("Back first"))

    expect(screen.getByText(/unsaved changes/i)).toBeInTheDocument()
    expect(screen.queryByText(/all changes saved/i)).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /save changes/i }))

    await waitFor(() =>
      expect(updateDeckMock).toHaveBeenCalledWith({
        deckId: 1,
        update: {
          review_prompt_side: "back",
          scheduler_type: "sm2_plus",
          scheduler_settings: biologySettings,
          expected_version: 2
        }
      })
    )
  })

  it("offers reload and reapply actions after a version conflict", async () => {
    updateDeckMock.mockRejectedValue(
      Object.assign(new Error("Version mismatch"), {
        response: { status: 409 }
      })
    )

    render(<SchedulerTab isActive />)

    fireEvent.change(screen.getByTestId("deck-scheduler-editor-field-leech-threshold"), {
      target: { value: "9" }
    })
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }))

    await screen.findByText(/deck settings changed elsewhere/i)
    expect(screen.getByTestId("deck-scheduler-editor-field-leech-threshold")).toHaveValue("8")

    decksData = [
      {
        ...biologyDeck,
        version: 3,
        scheduler_settings: {
          ...biologySettings,
          sm2_plus: {
            ...biologySettings.sm2_plus,
            leech_threshold: 12
          }
        }
      },
      chemistryDeck
    ]
    fireEvent.click(screen.getByRole("button", { name: /reload latest/i }))

    await waitFor(() => {
      expect(refetchDecksMock).toHaveBeenCalled()
      expect(screen.getByTestId("deck-scheduler-editor-field-leech-threshold")).toHaveValue("12")
    })

    fireEvent.change(screen.getByTestId("deck-scheduler-editor-field-leech-threshold"), {
      target: { value: "9" }
    })
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }))

    await screen.findByText(/deck settings changed elsewhere/i)
    fireEvent.click(screen.getByRole("button", { name: /reapply my draft/i }))
    expect(screen.getByTestId("deck-scheduler-editor-field-leech-threshold")).toHaveValue("9")
  })

  it("keeps the active draft when deck search hides the selected deck", () => {
    render(<SchedulerTab isActive />)

    fireEvent.change(screen.getByTestId("deck-scheduler-editor-field-leech-threshold"), {
      target: { value: "11" }
    })
    fireEvent.change(screen.getByPlaceholderText(/search decks/i), {
      target: { value: "Chemistry" }
    })

    expect(screen.getByText("Biology Scheduler")).toBeInTheDocument()
    expect(screen.getByTestId("deck-scheduler-editor-field-leech-threshold")).toHaveValue("11")
  })

  it("loads active-deck counts only for the selected deck and guards dirty deck switches", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")
    confirmSpy.mockReturnValue(false)

    render(<SchedulerTab isActive />)

    expect(useDueCountsQuery).toHaveBeenCalledWith(1, expect.objectContaining({ enabled: true }))
    expect(screen.getByText(/due review/i)).toBeInTheDocument()
    expect(screen.getByText("3")).toBeInTheDocument()

    fireEvent.change(screen.getByTestId("deck-scheduler-editor-field-leech-threshold"), {
      target: { value: "11" }
    })
    fireEvent.click(screen.getByRole("button", { name: /chemistry/i }))

    expect(confirmSpy).toHaveBeenCalled()
    expect(screen.getByText("Biology Scheduler")).toBeInTheDocument()

    confirmSpy.mockReturnValue(true)
    fireEvent.click(screen.getByRole("button", { name: /chemistry/i }))

    await screen.findByText("Chemistry Scheduler")
    expect(useDueCountsQuery).toHaveBeenLastCalledWith(2, expect.objectContaining({ enabled: true }))

    confirmSpy.mockRestore()
  })
})
