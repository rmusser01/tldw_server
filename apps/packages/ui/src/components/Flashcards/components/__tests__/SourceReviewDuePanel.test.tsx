import type {
  SourceReviewActivity,
  SourceReviewLaunchState,
  SourceReviewOccurrenceActionResponse
} from "@/services/flashcards"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { ComponentProps } from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { SourceReviewDuePanel } from "../SourceReviewDuePanel"

const mocks = vi.hoisted(() => ({
  dueQuery: vi.fn(),
  start: vi.fn(),
  complete: vi.fn(),
  skip: vi.fn(),
  messageError: vi.fn(),
  telemetry: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      options?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof options === "string") return options
      if (key === "option:flashcards.sourceReviewHeading") {
        return "Translated source reviews"
      }
      if (options?.defaultValue) {
        return options.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) => String(options[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

vi.mock("../../hooks/useSourceReviewQueries", () => ({
  useDueSourceReviewOccurrencesQuery: (...args: unknown[]) =>
    mocks.dueQuery(...args),
  useStartSourceReviewOccurrenceMutation: () => ({
    mutateAsync: mocks.start,
    isPending: false
  }),
  useCompleteSourceReviewOccurrenceMutation: () => ({
    mutateAsync: mocks.complete,
    isPending: false
  }),
  useSkipSourceReviewOccurrenceMutation: () => ({
    mutateAsync: mocks.skip,
    isPending: false
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ error: mocks.messageError, success: vi.fn() })
}))

vi.mock("@/utils/flashcards-error-recovery-telemetry", () => ({
  trackFlashcardsErrorRecoveryTelemetry: mocks.telemetry
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const launchState = (
  activity: SourceReviewActivity
): SourceReviewLaunchState => ({
  activity_type: activity,
  plan_id: 7,
  occurrence_id: 31,
  target_route: activity === "quiz" ? "/quiz" : "/flashcards",
  target_surface: "source_review_due_panel",
  action: "generate",
  source_payload_field: "source_bundle",
  completion_required: true,
  created_at: "2026-07-09T12:00:00Z",
  source_bundle: {
    items: [
      {
        source_type: "note",
        source_id: "note-42",
        label: "Cardiac physiology notes",
        excerpt_text: "The Frank-Starling mechanism increases stroke volume."
      }
    ]
  }
})

const occurrence = (
  activity: SourceReviewActivity,
  status: "pending" | "in_progress" = "pending"
): SourceReviewOccurrenceActionResponse => ({
  id: 31,
  plan_id: 7,
  offset_value: 1,
  offset_unit: "day",
  activity_type: activity,
  due_at: "2026-07-10T00:00:00Z",
  status,
  started_at: status === "in_progress" ? "2026-07-10T01:00:00Z" : null,
  completed_at: null,
  completion_source: null,
  created_at: "2026-07-09T12:00:00Z",
  last_modified: "2026-07-09T12:00:00Z",
  client_id: "source-review-tests",
  version: status === "in_progress" ? 2 : 1,
  plan_title: "Cardiac physiology",
  source_summary: [
    {
      source_type: "note",
      source_id: "note-42",
      label: "Cardiac physiology notes",
      excerpt_preview: "Frank-Starling preview"
    }
  ],
  launch_state: status === "in_progress" ? launchState(activity) : null
})

type SourceReviewDuePanelProps = ComponentProps<typeof SourceReviewDuePanel>
const noopGenerate: SourceReviewDuePanelProps["onSourceReviewGenerate"] =
  () => {}
const noopQuiz: SourceReviewDuePanelProps["onSourceReviewQuiz"] = () => {}

const renderPanel = (
  item?: SourceReviewOccurrenceActionResponse,
  callbacks?: Partial<
    Pick<
      SourceReviewDuePanelProps,
      "onSourceReviewGenerate" | "onSourceReviewQuiz"
    >
  >
) => {
  mocks.dueQuery.mockReturnValue({
    data: {
      items: item ? [item] : [],
      total: item ? 1 : 0,
      now: "2026-07-10T12:00:00Z"
    },
    isLoading: false,
    isError: false,
    refetch: vi.fn()
  })
  return render(
    <SourceReviewDuePanel
      isActive
      onSourceReviewGenerate={callbacks?.onSourceReviewGenerate ?? noopGenerate}
      onSourceReviewQuiz={callbacks?.onSourceReviewQuiz ?? noopQuiz}
    />
  )
}

describe("SourceReviewDuePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.start.mockReset()
    mocks.complete.mockResolvedValue({
      ...occurrence("reread", "in_progress"),
      status: "completed"
    })
    mocks.skip.mockResolvedValue({
      ...occurrence("reread"),
      status: "skipped"
    })
  })

  it("shows a compact empty state when no source reviews are due", () => {
    renderPanel()

    expect(screen.getByText("Translated source reviews")).toBeInTheDocument()
    expect(screen.getByText("No source reviews due")).toBeInTheDocument()
  })

  it("shows a retryable error instead of an empty state when loading fails", () => {
    const refetch = vi.fn()
    mocks.dueQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      refetch
    })
    render(
      <SourceReviewDuePanel
        isActive
        onSourceReviewGenerate={vi.fn()}
        onSourceReviewQuiz={vi.fn()}
      />
    )

    expect(screen.getByText("Source reviews unavailable")).toBeInTheDocument()
    expect(screen.queryByText("No source reviews due")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(refetch).toHaveBeenCalled()
  })

  it("keeps long due queues collapsed above normal review", () => {
    const items = Array.from({ length: 5 }, (_, index) => ({
      ...occurrence("reread"),
      id: index + 1,
      plan_title: `Review plan ${index + 1}`
    }))
    mocks.dueQuery.mockReturnValue({
      data: { items, total: 5, now: "2026-07-10T12:00:00Z" },
      isLoading: false,
      isError: false,
      refetch: vi.fn()
    })
    render(
      <SourceReviewDuePanel
        isActive
        onSourceReviewGenerate={vi.fn()}
        onSourceReviewQuiz={vi.fn()}
      />
    )

    expect(screen.getAllByRole("article")).toHaveLength(3)
    fireEvent.click(screen.getByRole("button", { name: "Show all 5" }))
    expect(screen.getAllByRole("article")).toHaveLength(5)
  })

  it("preserves an expanded due queue across background refetches", async () => {
    const items = Array.from({ length: 5 }, (_, index) => ({
      ...occurrence("reread"),
      id: index + 1,
      plan_title: `Review plan ${index + 1}`
    }))
    const dueResult = (now: string) => ({
      data: { items, total: 5, now },
      isLoading: false,
      isError: false,
      refetch: vi.fn()
    })
    mocks.dueQuery.mockReturnValue(dueResult("2026-07-10T12:00:00Z"))
    const onGenerate = vi.fn<
      SourceReviewDuePanelProps["onSourceReviewGenerate"]
    >()
    const onQuiz = vi.fn<SourceReviewDuePanelProps["onSourceReviewQuiz"]>()
    const view = render(
      <SourceReviewDuePanel
        isActive
        onSourceReviewGenerate={onGenerate}
        onSourceReviewQuiz={onQuiz}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Show all 5" }))
    expect(screen.getAllByRole("article")).toHaveLength(5)

    mocks.dueQuery.mockReturnValue(dueResult("2026-07-10T12:01:00Z"))
    view.rerender(
      <SourceReviewDuePanel
        isActive
        onSourceReviewGenerate={onGenerate}
        onSourceReviewQuiz={onQuiz}
      />
    )

    await waitFor(() =>
      expect(screen.getAllByRole("article")).toHaveLength(5)
    )
  })

  it("tolerates a missing reread source bundle", () => {
    const resumed = occurrence("reread", "in_progress")
    const launchState = resumed.launch_state as any
    launchState.source_bundle = undefined
    mocks.start.mockResolvedValue(resumed)
    renderPanel(resumed)

    expect(() =>
      fireEvent.click(screen.getByRole("button", { name: "Resume" }))
    ).not.toThrow()
  })

  it.each(["quiz", "flashcards", "cloze"] as const)(
    "does not launch a %s handoff without a source bundle",
    async (activity) => {
      const resumed = occurrence(activity, "in_progress")
      resumed.launch_state!.source_bundle = undefined as never
      mocks.start.mockResolvedValue(resumed)
      const onGenerate =
        vi.fn<SourceReviewDuePanelProps["onSourceReviewGenerate"]>()
      const onQuiz = vi.fn<SourceReviewDuePanelProps["onSourceReviewQuiz"]>()
      renderPanel(occurrence(activity, "in_progress"), {
        onSourceReviewGenerate: onGenerate,
        onSourceReviewQuiz: onQuiz
      })

      fireEvent.click(screen.getByRole("button", { name: "Resume" }))

      await waitFor(() => expect(mocks.start).toHaveBeenCalledWith(31))
      expect(onGenerate).not.toHaveBeenCalled()
      expect(onQuiz).not.toHaveBeenCalled()
      expect(mocks.messageError).toHaveBeenCalledWith(
        "This review could not load its saved source snapshot."
      )
    }
  )

  it("starts a pending occurrence", async () => {
    const started = occurrence("reread", "in_progress")
    mocks.start.mockResolvedValue(started)
    renderPanel(occurrence("reread"))

    fireEvent.click(screen.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(mocks.start).toHaveBeenCalledWith(31))
  })

  it("shows a bounded source summary before an occurrence starts", () => {
    renderPanel(occurrence("quiz"))

    expect(
      screen.getByText("1 source · Cardiac physiology notes")
    ).toBeInTheDocument()
    expect(screen.getByText("Frank-Starling preview")).toBeInTheDocument()
    expect(
      screen.queryByText(
        "The Frank-Starling mechanism increases stroke volume."
      )
    ).not.toBeInTheDocument()
  })

  it("resumes an in-progress occurrence through the start endpoint", async () => {
    const onQuiz = vi.fn<SourceReviewDuePanelProps["onSourceReviewQuiz"]>()
    mocks.start.mockResolvedValue(occurrence("quiz", "in_progress"))
    renderPanel(occurrence("quiz", "in_progress"), {
      onSourceReviewQuiz: onQuiz
    })

    fireEvent.click(screen.getByRole("button", { name: "Resume" }))

    await waitFor(() => {
      expect(mocks.start).toHaveBeenCalledWith(31)
      expect(onQuiz).toHaveBeenCalledWith({
        occurrence_id: 31,
        plan_id: 7,
        plan_title: "Cardiac physiology",
        activity_type: "quiz",
        source_bundle: launchState("quiz").source_bundle
      })
    })
  })

  it("keeps a completed occurrence hidden while a stale refetch settles", async () => {
    const item = occurrence("reread", "in_progress")
    const dueResult = (now: string) => ({
      data: { items: [item], total: 1, now },
      isLoading: false,
      isError: false,
      refetch: vi.fn()
    })
    mocks.dueQuery.mockReturnValue(dueResult("2026-07-10T12:00:00Z"))
    const view = render(
      <SourceReviewDuePanel
        isActive
        onSourceReviewGenerate={vi.fn()}
        onSourceReviewQuiz={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Complete" }))
    await waitFor(() => expect(screen.queryByRole("article")).toBeNull())

    mocks.dueQuery.mockReturnValue(dueResult("2026-07-10T12:01:00Z"))
    view.rerender(
      <SourceReviewDuePanel
        isActive
        onSourceReviewGenerate={vi.fn()}
        onSourceReviewQuiz={vi.fn()}
      />
    )

    await waitFor(() => expect(screen.queryByRole("article")).toBeNull())
  })

  it("keeps a started occurrence local while a stale refetch settles", async () => {
    const pending = occurrence("reread")
    mocks.start.mockResolvedValue(occurrence("reread", "in_progress"))
    const dueResult = (now: string) => ({
      data: { items: [pending], total: 1, now },
      isLoading: false,
      isError: false,
      refetch: vi.fn()
    })
    mocks.dueQuery.mockReturnValue(dueResult("2026-07-10T12:00:00Z"))
    const view = render(
      <SourceReviewDuePanel
        isActive
        onSourceReviewGenerate={vi.fn()}
        onSourceReviewQuiz={vi.fn()}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Start" }))
    expect(await screen.findByRole("button", { name: "Resume" })).toBeVisible()

    mocks.dueQuery.mockReturnValue(dueResult("2026-07-10T12:01:00Z"))
    view.rerender(
      <SourceReviewDuePanel
        isActive
        onSourceReviewGenerate={vi.fn()}
        onSourceReviewQuiz={vi.fn()}
      />
    )

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Resume" })).toBeVisible()
    )
  })

  it.each([
    ["start", "Start", "source-review-start"],
    ["complete", "Complete", "source-review-complete"],
    ["skip", "Skip", "source-review-skip"]
  ] as const)(
    "reports %s failures through flashcards recovery telemetry",
    async (mutation, buttonName, operation) => {
      mocks[mutation].mockRejectedValueOnce(new Error(`${mutation} failed`))
      renderPanel(
        occurrence("reread", mutation === "start" ? "pending" : "in_progress")
      )

      fireEvent.click(screen.getByRole("button", { name: buttonName }))

      await waitFor(() =>
        expect(mocks.telemetry).toHaveBeenCalledWith(
          expect.objectContaining({
            type: "flashcards_mutation_failed",
            surface: "review",
            operation
          })
        )
      )
    }
  )

  it("shows the grounded source snapshot inline for reread", async () => {
    mocks.start.mockResolvedValue(occurrence("reread", "in_progress"))
    renderPanel(occurrence("reread"))

    fireEvent.click(screen.getByRole("button", { name: "Start" }))

    expect(
      await screen.findByText(
        "The Frank-Starling mechanism increases stroke volume."
      )
    ).toBeInTheDocument()
  })

  it("shows source type, ID, and locator when reread has no excerpt", async () => {
    const started = occurrence("reread", "in_progress")
    started.launch_state!.source_bundle.items = [
      {
        source_type: "note",
        source_id: "note-42",
        label: "Cardiac physiology notes",
        locator: { section: "Hemodynamics" }
      }
    ]
    mocks.start.mockResolvedValue(started)
    renderPanel(occurrence("reread"))

    fireEvent.click(screen.getByRole("button", { name: "Start" }))

    expect(await screen.findByText("Note · note-42")).toBeInTheDocument()
    expect(screen.getByText(/Hemodynamics/)).toBeInTheDocument()
  })

  it("completes an in-progress occurrence", async () => {
    renderPanel(occurrence("reread", "in_progress"))

    fireEvent.click(screen.getByRole("button", { name: "Complete" }))

    await waitFor(() => expect(mocks.complete).toHaveBeenCalledWith(31))
  })

  it("skips a due occurrence", async () => {
    renderPanel(occurrence("reread"))

    fireEvent.click(screen.getByRole("button", { name: "Skip" }))

    await waitFor(() => expect(mocks.skip).toHaveBeenCalledWith(31))
  })

  it.each(["flashcards", "cloze"] as const)(
    "hands %s source text to generation without generating artifacts",
    async (activity) => {
      const onGenerate =
        vi.fn<SourceReviewDuePanelProps["onSourceReviewGenerate"]>()
      mocks.start.mockResolvedValue(occurrence(activity, "in_progress"))
      renderPanel(occurrence(activity), {
        onSourceReviewGenerate: onGenerate
      })

      fireEvent.click(screen.getByRole("button", { name: "Start" }))

      await waitFor(() => {
        expect(onGenerate).toHaveBeenCalledWith({
          activity_type: activity,
          text: "Cardiac physiology notes\nThe Frank-Starling mechanism increases stroke volume.",
          source_items: launchState(activity).source_bundle.items
        })
      })
      expect(mocks.complete).not.toHaveBeenCalled()
    }
  )
})
