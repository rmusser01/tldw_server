import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { PipelineWizard } from "../PipelineWizard"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      (fallback || _key).replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options?.[token]
        return value == null ? "" : String(value)
      })
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

describe("PipelineWizard", () => {
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
        })
      )
    })
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

    fireEvent.click(inDialog().getByRole("button", { name: "Generate full test episode" }))
    await waitFor(() => {
      expect(props.onTest).toHaveBeenLastCalledWith(
        expect.any(Object),
        expect.objectContaining({
          externalDelivery: false,
          audioSampleSeconds: null,
          jobId: 77
        })
      )
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Send test" }))
    await waitFor(() => {
      expect(props.onTest).toHaveBeenLastCalledWith(
        expect.any(Object),
        expect.objectContaining({
          externalDelivery: true,
          audioSampleSeconds: 60,
          jobId: 77
        })
      )
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Activate schedule" }))
    await waitFor(() => {
      expect(props.onActivate).toHaveBeenCalledWith(
        expect.any(Object),
        { jobId: 77 }
      )
    })
  })

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
      }
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
    expect(inDialog().getByText("Source is ready.")).toBeInTheDocument()
  })

  it("keeps the draft after server failure and distinguishes explicit cancellation", async () => {
    const onTest = vi
      .fn()
      .mockRejectedValueOnce(new Error("Provider unavailable"))
      .mockResolvedValueOnce({ jobId: 77, status: "cancelled" })
    renderWizard({ initialStep: "test", initialDraft: sportscastDraft, onTest })

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))
    expect(await inDialog().findByText(/Provider unavailable/)).toHaveTextContent(
      "Test failed. Your draft is saved. Provider unavailable"
    )

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))
    expect(await inDialog().findByText("Test cancelled. Your draft is saved.")).toBeInTheDocument()

    fireEvent.click(inDialog().getByRole("button", { name: "Back" }))
    fireEvent.click(inDialog().getByRole("button", { name: "Back" }))
    expect(inDialog().getByLabelText("Show name")).toHaveValue("Purple and Gold Weekly")
  })

  it("silently ignores AbortError from superseded work", async () => {
    const abortError = new Error("superseded")
    abortError.name = "AbortError"
    renderWizard({
      initialStep: "test",
      initialDraft: sportscastDraft,
      onTest: vi.fn().mockRejectedValue(abortError)
    })

    fireEvent.click(inDialog().getByRole("button", { name: "Generate 60-second sample" }))

    await waitFor(() => {
      expect(inDialog().queryByTestId("watchlists-pipeline-action-error")).not.toBeInTheDocument()
    })
  })

  it("moves focus to the validation summary and keeps fallback copy usable", async () => {
    renderWizard()

    fireEvent.click(inDialog().getByRole("button", { name: "Next: Cadence" }))

    const summary = await screen.findByTestId("watchlists-pipeline-validation-summary")
    await waitFor(() => expect(summary).toHaveFocus())
    expect(summary).toHaveTextContent("Choose at least one source before continuing.")
  })
})
