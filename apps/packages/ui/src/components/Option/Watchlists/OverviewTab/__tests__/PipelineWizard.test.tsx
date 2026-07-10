import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { PipelineWizard } from "../PipelineWizard"

const i18nMocks = vi.hoisted(() => ({
  language: "en-US",
  translations: {} as Record<string, string>
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      (i18nMocks.translations[_key] || fallback || _key).replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options?.[token]
        return value == null ? "" : String(value)
      }),
    i18n: {
      get language() { return i18nMocks.language },
      get resolvedLanguage() { return i18nMocks.language }
    }
  })
}))

const sources = Array.from({ length: 8 }, (_value, index) => ({
  id: index + 1,
  name: `Lakers source ${index + 1}`,
  url: `https://example.com/lakers-${index + 1}.xml`,
  source_type: "rss" as const,
  active: true,
  tags: [],
  created_at: "2026-07-01T00:00:00Z"
}))

const sportscastDraft = {
  sourceIds: sources.map((source) => source.id),
  monitorName: "Purple and Gold Weekly",
  scheduleMode: "weekly" as const,
  scheduleWeekday: "SUN" as const,
  scheduleHour: 18,
  scheduleMinute: 0,
  timezone: "America/Los_Angeles",
  nextRunAt: "2026-07-12T18:00:00-07:00",
  createScheduledOutput: true,
  templateName: "briefing_markdown",
  programFormat: "sportscast" as const,
  outcomeNoun: "episode" as const,
  showName: "Purple and Gold Weekly",
  showNotes: true,
  audioEnabled: true,
  targetAudioMinutes: 20,
  audioSpeakers: [
    { id: "host", label: "Host", role: "host", voice: "alloy" },
    { id: "analyst", label: "Analyst", role: "analyst", voice: "nova" }
  ]
}

const renderWizard = (
  overrides: Partial<React.ComponentProps<typeof PipelineWizard>> = {}
) => {
  const callbacks = {
    onCancel: vi.fn(),
    onSaveDraft: vi.fn(),
    onTest: vi.fn().mockResolvedValue({ jobId: 77, status: "ready" }),
    onActivate: vi.fn().mockResolvedValue({ jobId: 77, status: "active" }),
    onTestSource: vi.fn().mockResolvedValue({ status: "ready" })
  }
  const props = { ...callbacks, ...overrides }
  render(
    <PipelineWizard
      open
      sources={sources}
      sourcesLoading={false}
      submitting={false}
      submitError={null}
      onCancel={props.onCancel}
      onSaveDraft={props.onSaveDraft}
      onTest={props.onTest}
      onActivate={props.onActivate}
      onTestSource={props.onTestSource}
      {...overrides}
    />
  )
  return props
}

const dialog = () => screen.getByRole("dialog", { name: "Set up briefing" })
const inDialog = () => within(dialog())
const readyActionButton = async (name: string) => {
  const button = inDialog().getByRole("button", { name: new RegExp(`${name}$`) })
  await waitFor(() => expect(button).not.toHaveClass("ant-btn-loading"))
  return button
}

const readyProjection = (stages: Record<string, { status: string }> = {}) => ({
  occurrence_id: 1,
  run_id: 2,
  job_id: 77,
  artifact_status: "ready" as const,
  delivery_status: "not_configured" as const,
  stages,
  output: { id: 3 },
  audio: null,
  editorial: {},
  selection: {},
  next_run_at: null,
  recovery: {}
})

describe("PipelineWizard", () => {
  beforeEach(() => {
    i18nMocks.language = "en-US"
    i18nMocks.translations = {}
  })

  it("uses the single outcome-first step sequence", () => {
    renderWizard()

    expect(inDialog().getAllByRole("listitem").map((item) => item.textContent)).toEqual([
      expect.stringContaining("Sources"),
      expect.stringContaining("Cadence"),
      expect.stringContaining("Briefing"),
      expect.stringContaining("Delivery"),
      expect.stringContaining("Test")
    ])
    expect(inDialog().getAllByRole("button").filter((button) => button.classList.contains("ant-btn-primary"))).toHaveLength(1)
  })

  it("keeps the step sequence scrollable and uses logical alignment for narrow RTL layouts", () => {
    renderWizard()
    dialog().setAttribute("dir", "rtl")

    const progress = inDialog().getByRole("navigation", { name: "Briefing setup steps" })
    expect(within(progress).getByRole("list")).toHaveClass("overflow-x-auto")
    expect(within(progress).getByRole("button", { name: "Sources" })).toHaveClass("text-start")
  })

  it("shows a complete activation receipt", () => {
    renderWizard({ initialStep: "test", initialDraft: sportscastDraft })

    const receipt = screen.getByTestId("watchlists-pipeline-receipt")
    expect(receipt).toHaveTextContent("Sunday, July 12 at 6:00 PM PDT")
    expect(receipt).toHaveTextContent("America/Los_Angeles")
    expect(receipt).toHaveTextContent("8 sources")
    expect(receipt).toHaveTextContent("two-host sportscast")
    expect(receipt).toHaveTextContent("targeting 20 minutes")
    expect(receipt).toHaveTextContent("Reports")
  })

  it("keeps a schedule classified as scheduled when projection is unavailable", () => {
    renderWizard({
      initialStep: "test",
      initialDraft: { ...sportscastDraft, nextRunAt: "projection-unavailable" }
    })

    const receipt = screen.getByTestId("watchlists-pipeline-receipt")
    expect(receipt).toHaveTextContent("Save a valid schedule to calculate the exact time.")
    expect(receipt).not.toHaveTextContent("Manual only")
  })

  it("cancels stale advanced previews and uses the latest exact backend receipt", async () => {
    let resolveFirst: ((value: { nextRunAt?: string; followingRunAt?: string }) => void) | undefined
    const onPreviewSchedule = vi.fn()
      .mockImplementationOnce((_draft, _options) => new Promise((resolve) => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce({
        nextRunAt: "2026-08-03T08:00:00Z",
        followingRunAt: "2026-09-01T08:00:00Z"
      })
    renderWizard({
      initialStep: "cadence",
      initialDraft: {
        ...sportscastDraft,
        scheduleMode: "advanced",
        scheduleAdvancedCron: "0 8 1 * MON",
        timezone: "UTC",
        nextRunAt: undefined,
        followingRunAt: undefined
      },
      onPreviewSchedule
    } as Partial<React.ComponentProps<typeof PipelineWizard>>)

    await waitFor(() => expect(onPreviewSchedule).toHaveBeenCalledTimes(1), { timeout: 2_000 })
    expect(inDialog().getByText("Calculating the exact schedule…")).toBeInTheDocument()
    const firstSignal = onPreviewSchedule.mock.calls[0]?.[1]?.signal as AbortSignal

    fireEvent.change(inDialog().getByLabelText("Cron expression"), {
      target: { value: "0 8 1 * TUE" }
    })

    await waitFor(() => expect(onPreviewSchedule).toHaveBeenCalledTimes(2), { timeout: 2_000 })
    expect(firstSignal.aborted).toBe(true)
    resolveFirst?.({ nextRunAt: "2026-07-01T08:00:00Z" })
    expect(await inDialog().findByText(/Monday, August 3 at 8:00 AM UTC/)).toBeInTheDocument()
    expect(inDialog().queryByText(/Wednesday, July 1/)).not.toBeInTheDocument()

    fireEvent.click(inDialog().getByRole("button", { name: "Next: Briefing" }))
    fireEvent.click(inDialog().getByRole("button", { name: "Next: Delivery" }))
    fireEvent.click(inDialog().getByRole("button", { name: "Next: Test" }))
    expect(screen.getByTestId("watchlists-pipeline-receipt")).toHaveTextContent(
      "Monday, August 3 at 8:00 AM UTC"
    )
    expect(screen.getByTestId("watchlists-pipeline-receipt")).not.toHaveTextContent("Manual only")
  })

  it("isolates identical advanced previews across setup sessions", async () => {
    let resolveSessionA: ((value: { nextRunAt?: string }) => void) | undefined
    const onPreviewSchedule = vi.fn()
      .mockImplementationOnce((_draft, _options) => new Promise((resolve) => {
        resolveSessionA = resolve
      }))
      .mockResolvedValueOnce({
        nextRunAt: "2026-08-03T08:00:00Z",
        followingRunAt: "2026-09-01T08:00:00Z"
      })
    const initialDraft = {
      ...sportscastDraft,
      scheduleMode: "advanced" as const,
      scheduleAdvancedCron: "0 8 1 * MON",
      timezone: "UTC",
      nextRunAt: undefined,
      followingRunAt: undefined
    }
    const commonProps = {
      open: true,
      initialStep: "cadence" as const,
      initialDraft,
      sources,
      onCancel: vi.fn(),
      onPreviewSchedule
    }
    const view = render(<PipelineWizard {...commonProps} sessionKey="session-a" />)

    await waitFor(() => expect(onPreviewSchedule).toHaveBeenCalledTimes(1), { timeout: 2_000 })
    const sessionASignal = onPreviewSchedule.mock.calls[0]?.[1]?.signal as AbortSignal

    view.rerender(<PipelineWizard {...commonProps} sessionKey="session-b" />)

    await waitFor(() => expect(onPreviewSchedule).toHaveBeenCalledTimes(2), { timeout: 2_000 })
    expect(sessionASignal.aborted).toBe(true)
    resolveSessionA?.({ nextRunAt: "2026-07-01T08:00:00Z" })

    expect(await inDialog().findByText(/Monday, August 3 at 8:00 AM UTC/)).toBeInTheDocument()
    expect(inDialog().queryByText(/Wednesday, July 1/)).not.toBeInTheDocument()
  })

  it("does not preview manual schedules and surfaces advanced preview failures", async () => {
    const manualPreview = vi.fn()
    const manualView = render(
      <PipelineWizard
        open
        initialStep="cadence"
        initialDraft={{ ...sportscastDraft, scheduleMode: "manual", nextRunAt: undefined }}
        sources={sources}
        onCancel={vi.fn()}
        onPreviewSchedule={manualPreview as never}
      />
    )
    await new Promise((resolve) => setTimeout(resolve, 350))
    expect(manualPreview).not.toHaveBeenCalled()
    manualView.unmount()

    renderWizard({
      initialStep: "cadence",
      initialDraft: {
        ...sportscastDraft,
        scheduleMode: "advanced",
        scheduleAdvancedCron: "0 8 1 * MON",
        nextRunAt: undefined
      },
      onPreviewSchedule: vi.fn().mockRejectedValue(new Error("invalid_schedule_expr"))
    } as Partial<React.ComponentProps<typeof PipelineWizard>>)

    expect(await inDialog().findByText("Could not calculate the exact schedule. Review the cron expression and timezone.", {}, { timeout: 2_000 })).toBeInTheDocument()
  })

  it("does not send external delivery during the default test", async () => {
    const props = renderWizard({
      initialStep: "test",
      initialDraft: {
        ...sportscastDraft,
        emailDeliveryEnabled: true,
        emailRecipients: ["coach@example.com"]
      }
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))

    await waitFor(() => {
      expect(props.onTest).toHaveBeenCalledWith(
        expect.objectContaining({ emailRecipients: ["coach@example.com"] }),
        expect.objectContaining({
          externalDelivery: false,
          audioSampleSeconds: 60
        }),
        expect.any(Function)
      )
    })
  })

  it("renders only durable stages observed from the exact-run projection", async () => {
    const projection = readyProjection({
      collect: { status: "ready" },
      render_text: { status: "ready" }
    })
    const onTest = vi.fn().mockImplementation(async (
      _draft: unknown,
      _options: unknown,
      onProgress: (value: typeof projection) => void
    ) => {
      onProgress(projection)
      return { jobId: 77, runId: 2, status: "ready", briefing: projection }
    })
    renderWizard({ initialStep: "test", initialDraft: sportscastDraft, onTest })

    expect(inDialog().queryByText("Select updates")).not.toBeInTheDocument()
    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))

    expect(await inDialog().findByText("Collect sources")).toBeInTheDocument()
    expect(inDialog().getByText("Create report")).toBeInTheDocument()
    expect(inDialog().getAllByText("Ready")).toHaveLength(2)
    expect(inDialog().queryByText("Create audio")).not.toBeInTheDocument()
    expect(inDialog().queryByText("Check test delivery")).not.toBeInTheDocument()

    fireEvent.click(inDialog().getByRole("button", { name: "Sources" }))
    fireEvent.click(inDialog().getByLabelText("Lakers source 1"))
    expect(inDialog().queryByText(/Test started/)).not.toBeInTheDocument()
    expect(inDialog().queryByText("Collect sources")).not.toBeInTheDocument()
  })

  it("uses text-only test actions, disclosure, receipt, and metadata", async () => {
    const props = renderWizard({
      initialStep: "test",
      initialDraft: {
        ...sportscastDraft,
        programFormat: "concise_briefing",
        outcomeNoun: "briefing",
        showNotes: false,
        audioEnabled: false,
        audioSpeakers: []
      }
    })

    expect(inDialog().getByText(/Text-only tests do not use text-to-speech/)).toBeInTheDocument()
    expect(inDialog().getByRole("button", { name: "Generate test report" })).toBeInTheDocument()
    expect(inDialog().queryByRole("button", { name: "Generate 60-second sample" })).not.toBeInTheDocument()
    expect(inDialog().queryByRole("button", { name: "Generate full test episode" })).not.toBeInTheDocument()
    expect(screen.getByTestId("watchlists-pipeline-receipt")).toHaveTextContent("text report")
    expect(screen.getByTestId("watchlists-pipeline-receipt")).not.toHaveTextContent("audio")

    fireEvent.click(inDialog().getByRole("button", { name: "Generate test report" }))
    await waitFor(() => expect(props.onTest).toHaveBeenCalledWith(
      expect.any(Object),
      expect.objectContaining({ audioSampleSeconds: null }),
      expect.any(Function)
    ))
  })

  it("offers sample, full, send, and activation actions while reusing the inactive job id", async () => {
    const props = renderWizard({
      initialStep: "test",
      initialDraft: {
        ...sportscastDraft,
        emailDeliveryEnabled: true,
        emailRecipients: ["coach@example.com"]
      }
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))
    await waitFor(() => expect(props.onTest).toHaveBeenCalledTimes(1))

    fireEvent.click(await readyActionButton("Generate full test episode"))
    await waitFor(() => {
      expect(props.onTest).toHaveBeenLastCalledWith(
        expect.any(Object),
        expect.objectContaining({
          externalDelivery: false,
          audioSampleSeconds: null,
          jobId: 77
        }),
        expect.any(Function)
      )
    })

    fireEvent.click(await readyActionButton("Send test"))
    await waitFor(() => {
      expect(props.onTest).toHaveBeenLastCalledWith(
        expect.any(Object),
        expect.objectContaining({
          externalDelivery: true,
          audioSampleSeconds: 60,
          jobId: 77
        }),
        expect.any(Function)
      )
    })

    fireEvent.click(await readyActionButton("Activate schedule"))
    await waitFor(() => {
      expect(props.onActivate).toHaveBeenCalledWith(
        expect.any(Object),
        { jobId: 77 }
      )
    })
  }, 20_000)

  it("starts Briefing with all six formats and reveals custom controls progressively", () => {
    renderWizard({ initialStep: "briefing", initialDraft: sportscastDraft })

    expect(inDialog().getByRole("heading", { name: "What are you making?" })).toBeInTheDocument()
    for (const label of [
      "Concise briefing",
      "Solo update",
      "Host discussion",
      "Sportscast",
      "Culture roundtable",
      "Custom"
    ]) {
      expect(inDialog().getByLabelText(label)).toBeInTheDocument()
    }
    expect(inDialog().getByLabelText("Target duration in minutes")).toHaveAttribute("min", "1")
    expect(inDialog().getByLabelText("Target duration in minutes")).toHaveAttribute("max", "60")

    fireEvent.click(inDialog().getByLabelText("Custom"))
    fireEvent.click(inDialog().getByText("Advanced briefing settings"))
    expect(inDialog().getByLabelText("Custom editorial instructions")).toBeInTheDocument()
    expect(inDialog().getByLabelText("Speaker 1 persona")).toBeInTheDocument()
  })

  it("tests new sources without leaving Sources", async () => {
    const props = renderWizard({
      initialDraft: {
        sourceMode: "new",
        sourceName: "Policy feed",
        sourceUrl: "https://example.com/policy.xml"
      },
      onTestSource: vi.fn().mockResolvedValue({
        status: "ready",
        sourceTest: {
          total: 3,
          ingestable: 2,
          filtered: 1,
          items: [
            { source_id: 1, source_type: "rss", title: "Ready item", decision: "ingest" },
            { source_id: 1, source_type: "rss", title: "Filtered item", decision: "filtered" }
          ]
        }
      })
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Test source" }))

    await waitFor(() => {
      expect(props.onTestSource).toHaveBeenCalledWith(
        expect.objectContaining({
          sourceMode: "new",
          sourceUrl: "https://example.com/policy.xml"
        })
      )
    })
    expect(inDialog().getByText("2 ready, 1 filtered from 3 items.")).toBeInTheDocument()
    expect(inDialog().getByText("Ready item")).toBeInTheDocument()
    expect(inDialog().getByText("Filtered item")).toBeInTheDocument()
  })

  it("keeps the draft after server failure and distinguishes explicit cancellation", async () => {
    const onTest = vi
      .fn()
      .mockRejectedValueOnce(new Error("Provider unavailable"))
      .mockResolvedValueOnce({ jobId: 77, status: "cancelled" })
    renderWizard({ initialStep: "test", initialDraft: sportscastDraft, onTest })

    fireEvent.click(await readyActionButton("Generate 60-second sample"))
    expect(await inDialog().findByText(/Provider unavailable/)).toHaveTextContent(
      "Test failed. Your draft is saved. Provider unavailable"
    )

    fireEvent.click(await readyActionButton("Generate 60-second sample"))
    expect(await inDialog().findByText("Test cancelled. Your draft is saved.")).toBeInTheDocument()

    fireEvent.click(inDialog().getByRole("button", { name: "Back" }))
    fireEvent.click(inDialog().getByRole("button", { name: "Back" }))
    expect(inDialog().getByLabelText("Show name")).toHaveValue("Purple and Gold Weekly")
  }, 10_000)

  it("does not show success when the durable projection failed", async () => {
    const projection = {
      ...readyProjection(),
      artifact_status: "failed" as const,
      stages: {
        persist_text: {
          status: "failed" as const,
          code: "report_persist_failed",
          retryable: true
        }
      }
    }
    renderWizard({
      initialStep: "test",
      initialDraft: sportscastDraft,
      onTest: vi.fn().mockResolvedValue({
        jobId: 77,
        runId: 2,
        status: "ready",
        briefing: projection
      })
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))

    expect(await inDialog().findByTestId("watchlists-pipeline-action-error")).toHaveTextContent(
      "Save report in Reports failed (report_persist_failed). Open run 2 and retry this stage."
    )
    expect(inDialog().queryByText(/Test started/)).not.toBeInTheDocument()
  })

  it("reports a bounded foreground poll as still running with run context", async () => {
    const projection = {
      ...readyProjection({ generate_audio: { status: "running" } }),
      run_id: 88,
      artifact_status: "running" as const,
      delivery_status: "waiting_for_artifacts" as const
    }
    renderWizard({
      initialStep: "test",
      initialDraft: sportscastDraft,
      onTest: vi.fn().mockResolvedValue({
        jobId: 77,
        runId: 88,
        status: "running",
        briefing: projection
      })
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))

    expect(await inDialog().findByText("Run 88 is still running: Create audio. You can close setup and monitor it from Overview.")).toBeInTheDocument()
    expect(inDialog().queryByText(/Test started/)).not.toBeInTheDocument()
  })

  it("localizes ready, failed, and running terminal outcomes", async () => {
    i18nMocks.language = "es"
    i18nMocks.translations = {
      "watchlists:overview.pipelineSetup.test.ready": "Prueba completada. El informe está listo.",
      "watchlists:overview.pipelineSetup.test.failedWithCode": "{{stage}} falló ({{code}}). Abre la ejecución {{runId}} y reintenta esta etapa.",
      "watchlists:overview.pipelineSetup.test.running": "La ejecución {{runId}} sigue activa: {{stage}}. Supervísala en Resumen.",
      "watchlists:overview.pipelineSetup.test.stages.persistence": "Guardar informe en Informes",
      "watchlists:overview.pipelineSetup.test.stages.audio": "Crear audio"
    }
    const outcomes = [
      {
        projection: readyProjection(),
        expected: "Prueba completada. El informe está listo."
      },
      {
        projection: {
          ...readyProjection(),
          artifact_status: "failed" as const,
          stages: { persist_text: { status: "failed" as const, code: "persist_failed" } }
        },
        expected: "Guardar informe en Informes falló (persist_failed). Abre la ejecución 2 y reintenta esta etapa."
      },
      {
        projection: {
          ...readyProjection(),
          artifact_status: "running" as const,
          stages: { generate_audio: { status: "running" as const } }
        },
        expected: "La ejecución 2 sigue activa: Crear audio. Supervísala en Resumen."
      }
    ]

    for (const { projection, expected } of outcomes) {
      renderWizard({
        initialStep: "test",
        initialDraft: sportscastDraft,
        onTest: vi.fn().mockResolvedValue({ status: "ready", briefing: projection })
      })
      fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))
      expect(await inDialog().findByText(expected)).toBeInTheDocument()
      cleanup()
    }
  })

  it("surfaces a current AbortError instead of assuming it is stale", async () => {
    const abortError = new Error("superseded")
    abortError.name = "AbortError"
    renderWizard({
      initialStep: "test",
      initialDraft: sportscastDraft,
      onTest: vi.fn().mockRejectedValue(abortError)
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))

    expect(await inDialog().findByTestId("watchlists-pipeline-action-error")).toHaveTextContent(
      "superseded"
    )
  })

  it("rehydrates a new open session and ignores completion from the superseded session", async () => {
    let resolveFirst: ((value: unknown) => void) | undefined
    const onTest = vi.fn().mockImplementation(() => new Promise((resolve) => {
      resolveFirst = resolve
    }))
    const firstProps = {
      open: true,
      sessionKey: "first",
      sources,
      initialStep: "test" as const,
      initialDraft: sportscastDraft,
      onCancel: vi.fn(),
      onTest,
      onActivate: vi.fn()
    }
    const view = render(<PipelineWizard {...(firstProps as React.ComponentProps<typeof PipelineWizard>)} />)
    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))
    await waitFor(() => expect(onTest).toHaveBeenCalledTimes(1))
    const firstSignal = onTest.mock.calls[0]?.[1]?.signal as AbortSignal

    view.rerender(
      <PipelineWizard
        {...(firstProps as React.ComponentProps<typeof PipelineWizard>)}
        {...({ sessionKey: "second" } as Record<string, unknown>)}
        initialStep="sources"
        initialDraft={{ sourceMode: "new", sourceName: "Fresh feed", sourceUrl: "https://example.com/fresh.xml" }}
      />
    )
    expect(inDialog().getByLabelText("Source name")).toHaveValue("Fresh feed")
    expect(firstSignal.aborted).toBe(true)
    resolveFirst?.({ jobId: 77, status: "ready" })
    await waitFor(() => {
      expect(inDialog().queryByText(/Test started/)).not.toBeInTheDocument()
      expect(inDialog().getByLabelText("Source name")).toHaveValue("Fresh feed")
    })
  })

  it("connects localized validation messages and focuses the first invalid control", async () => {
    renderWizard({ initialDraft: { sourceMode: "new", sourceName: "", sourceUrl: "" } })

    fireEvent.click(inDialog().getByRole("button", { name: "Next: Cadence" }))

    const sourceName = inDialog().getByLabelText("Source name")
    await waitFor(() => expect(sourceName).toHaveFocus())
    expect(sourceName).toHaveAttribute("id", "pipeline-source-name")
    expect(sourceName).toHaveAttribute("aria-invalid", "true")
    expect(sourceName).toHaveAttribute("aria-describedby", "pipeline-source-name-error")
    expect(inDialog().getByText("Enter a source name.")).toHaveAttribute(
      "id",
      "pipeline-source-name-error"
    )
  })

  it("opens advanced details, annotates, and focuses the first invalid voice", async () => {
    renderWizard({
      initialStep: "test",
      initialDraft: {
        ...sportscastDraft,
        audioSpeakers: [
          { id: "host", label: "Host", role: "host", voice: "alloy" },
          { id: "host", label: "Analyst", role: "analyst", voice: "" }
        ]
      }
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))

    const voice = await inDialog().findByRole("combobox", { name: "Speaker 2 voice" })
    await waitFor(() => expect(voice).toHaveFocus())
    expect(voice).toHaveAttribute("id", "pipeline-speaker-2-voice")
    expect(voice).toHaveAttribute("aria-invalid", "true")
    expect(voice).toHaveAttribute("aria-describedby", "pipeline-speaker-2-voice-error")
    expect(inDialog().getByText("Choose a voice for every speaker.")).toHaveAttribute(
      "id",
      "pipeline-speaker-2-voice-error"
    )
    expect(voice.closest("details")).toHaveAttribute("open")
  })

  it("localizes destination step and default source/speaker grammar", () => {
    i18nMocks.translations = {
      "watchlists:overview.pipelineSetup.steps.cadence": "Ritmo",
      "watchlists:overview.pipelineSetup.actions.next": "Siguiente: {{step}}",
      "watchlists:overview.pipelineSetup.speaker.defaultLabel": "Locutor {{index}}",
      "watchlists:overview.pipelineSetup.speaker.defaultRole": "presentador",
      "watchlists:overview.pipelineSetup.source.fallback": "Fuente {{id}}"
    }
    renderWizard({
      sources: [{ ...sources[0], name: "" }],
      initialDraft: { sourceIds: [1] }
    })

    expect(inDialog().getByRole("button", { name: "Siguiente: Ritmo" })).toBeInTheDocument()
    expect(inDialog().getByLabelText("Fuente 1")).toBeInTheDocument()
    fireEvent.click(inDialog().getByRole("button", { name: "Siguiente: Ritmo" }))
    fireEvent.change(inDialog().getByLabelText("Monitor name"), { target: { value: "Monitor" } })
    fireEvent.click(inDialog().getByRole("button", { name: "Siguiente: Briefing" }))
    expect(inDialog().getByLabelText("Speaker 1 label")).toHaveValue("Locutor 1")
    expect(inDialog().getByLabelText("Speaker 1 role")).toHaveValue("presentador")
  })

  it("constrains the modal and wraps full-width controls for narrow coarse pointers", () => {
    renderWizard({ initialStep: "briefing", initialDraft: sportscastDraft })

    expect(dialog()).toHaveClass("w-[min(760px,calc(100vw-2rem))]")
    expect(inDialog().getByTestId("watchlists-pipeline-scroll-region")).toHaveClass("overflow-y-auto")
    expect(inDialog().getByRole("button", { name: "Cancel" })).toHaveClass("w-full", "whitespace-normal")
    expect(inDialog().getByRole("switch", { name: "Audio" }).className).toContain(
      "[@media(pointer:coarse)]:min-h-11"
    )
    expect(dialog()).toHaveClass("watchlists-pipeline-wizard")
    expect(inDialog().getByLabelText("Target duration in minutes")).toHaveClass("ant-input")
    expect(inDialog().getByRole("combobox", { name: "Cast size" }).closest(".ant-select")).toContainElement(
      inDialog().getByRole("combobox", { name: "Cast size" })
    )
    expect(inDialog().getByLabelText("Sportscast").closest(".ant-radio-wrapper")).toBeInTheDocument()
  })
})
