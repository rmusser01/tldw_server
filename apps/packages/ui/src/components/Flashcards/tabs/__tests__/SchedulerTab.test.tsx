import { fireEvent, render, screen } from "@testing-library/react"
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

describe("SchedulerTab", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Biology",
          description: "Cells",
          review_prompt_side: "front",
          deleted: false,
          client_id: "test",
          version: 2,
          scheduler_settings_json: null,
          scheduler_settings: {
            new_steps_minutes: [1, 10],
            relearn_steps_minutes: [10],
            graduating_interval_days: 1,
            easy_interval_days: 4,
            easy_bonus: 1.3,
            interval_modifier: 1,
            max_interval_days: 365,
            leech_threshold: 8,
            enable_fuzz: true
          }
        },
        {
          id: 2,
          name: "Chemistry",
          description: "Atoms",
          review_prompt_side: "back",
          deleted: false,
          client_id: "test",
          version: 4,
          scheduler_settings_json: null,
          scheduler_settings: {
            new_steps_minutes: [2, 20],
            relearn_steps_minutes: [15],
            graduating_interval_days: 2,
            easy_interval_days: 5,
            easy_bonus: 1.5,
            interval_modifier: 0.9,
            max_interval_days: 180,
            leech_threshold: 6,
            enable_fuzz: false
          }
        }
      ],
      isLoading: false
    } as any)
    vi.mocked(useDueCountsQuery).mockReturnValue({
      data: {
        due: 0,
        new: 0,
        learning: 0,
        total: 0
      },
      isLoading: false
    } as any)
    vi.mocked(useUpdateDeckMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
  })

  it("renders a deck navigator with derived scheduler summaries", () => {
    render(<SchedulerTab isActive />)

    expect(screen.getByText("Biology")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /chemistry atoms 2m,20m -> 2d \/ easy 5d \/ leech 6 \/ fuzz off/i })
    ).toBeInTheDocument()
    expect(screen.getAllByText("1m,10m -> 1d / easy 4d / leech 8 / fuzz on").length).toBeGreaterThan(0)
    expect(screen.getAllByText("2m,20m -> 2d / easy 5d / leech 6 / fuzz off").length).toBeGreaterThan(0)
  })

  it("switches the active deck workspace when a deck is selected", () => {
    render(<SchedulerTab isActive />)

    expect(screen.getByText("Biology Scheduler")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /chemistry/i }))
    expect(screen.getByText("Chemistry Scheduler")).toBeInTheDocument()
  })
})
