import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { PipelineWizard } from "../PipelineWizard"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
}))

const sources = [
  {
    id: 11,
    name: "AI Feed",
    url: "https://example.com/ai.xml",
    source_type: "rss" as const,
    active: true,
    tags: [],
    created_at: "2026-05-18T00:00:00Z"
  },
  {
    id: 12,
    name: "Security Feed",
    url: "https://example.com/security.xml",
    source_type: "rss" as const,
    active: true,
    tags: [],
    created_at: "2026-05-18T00:00:00Z"
  }
]

const renderWizard = (overrides: Partial<React.ComponentProps<typeof PipelineWizard>> = {}) => {
  const onCancel = vi.fn()
  const onSubmit = vi.fn()
  const onPreview = vi.fn()
  const result = render(
    <PipelineWizard
      open
      sources={sources}
      sourcesLoading={false}
      submitting={false}
      previewLoading={false}
      previewError={null}
      previewRendered={null}
      previewRunId={null}
      previewWarnings={[]}
      onCancel={onCancel}
      onSubmit={onSubmit}
      onPreview={onPreview}
      {...overrides}
    />
  )
  return { ...result, onCancel, onSubmit, onPreview }
}

const getDialog = () => screen.getByRole("dialog", { name: "Briefing pipeline builder" })
const getDialogQueries = () => within(getDialog())

describe("PipelineWizard", () => {
  it("creates a pipeline from an existing source with digest and two-speaker audio", async () => {
    const { onSubmit } = renderWizard()

    await waitFor(() => {
      expect(getDialogQueries().getByLabelText("AI Feed")).toBeInTheDocument()
    })

    fireEvent.click(getDialogQueries().getByLabelText("AI Feed"))
    fireEvent.click(getDialogQueries().getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(getDialogQueries().getByLabelText("Monitor name")).toBeInTheDocument()
    })
    fireEvent.change(getDialogQueries().getByLabelText("Monitor name"), {
      target: { value: "Five Hour Brief" }
    })
    fireEvent.mouseDown(getDialogQueries().getByLabelText("Schedule"))
    fireEvent.click(await screen.findByText("Every N hours/minutes"))
    fireEvent.change(getDialogQueries().getByLabelText("Every"), {
      target: { value: "5" }
    })
    fireEvent.click(getDialogQueries().getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(getDialogQueries().getByLabelText("Template")).toBeInTheDocument()
    })
    fireEvent.change(getDialogQueries().getByLabelText("Template"), {
      target: { value: "newsletter_markdown" }
    })
    fireEvent.click(getDialogQueries().getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(getDialogQueries().getByLabelText("Audio briefing")).toBeInTheDocument()
    })
    fireEvent.mouseDown(getDialogQueries().getByLabelText("Speaker count"))
    fireEvent.click(await screen.findByText("2 speakers"))
    fireEvent.change(getDialogQueries().getByLabelText("Speaker 1 label"), {
      target: { value: "Host" }
    })
    fireEvent.change(getDialogQueries().getByLabelText("Speaker 2 label"), {
      target: { value: "Analyst" }
    })
    fireEvent.click(getDialogQueries().getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-pipeline-review-summary")).toHaveTextContent("AI Feed")
    })
    expect(screen.getByTestId("watchlists-pipeline-review-summary")).toHaveTextContent("Every 5 hours")
    expect(screen.getByTestId("watchlists-pipeline-review-summary")).toHaveTextContent("2 speakers audio briefing")

    fireEvent.click(getDialogQueries().getByRole("button", { name: "Create pipeline" }))

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          sourceMode: "existing",
          sourceIds: [11],
          monitorName: "Five Hour Brief",
          scheduleMode: "interval",
          scheduleIntervalValue: 5,
          templateName: "newsletter_markdown",
          audioEnabled: true,
          audioSpeakers: expect.arrayContaining([
            expect.objectContaining({ label: "Host" }),
            expect.objectContaining({ label: "Analyst" })
          ])
        }),
        { mode: "create" }
      )
    })
  })

  it("supports creating a new source before monitor setup", async () => {
    const { onSubmit } = renderWizard()

    await waitFor(() => {
      expect(getDialogQueries().getByLabelText("Create a new feed")).toBeInTheDocument()
    })
    fireEvent.click(getDialogQueries().getByLabelText("Create a new feed"))
    fireEvent.change(getDialogQueries().getByLabelText("Feed name"), {
      target: { value: "Policy Feed" }
    })
    fireEvent.change(getDialogQueries().getByLabelText("Feed URL"), {
      target: { value: "https://example.com/policy.xml" }
    })

    fireEvent.click(getDialogQueries().getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(getDialogQueries().getByLabelText("Monitor name")).toBeInTheDocument()
    })
    fireEvent.change(getDialogQueries().getByLabelText("Monitor name"), {
      target: { value: "Policy Brief" }
    })
    fireEvent.click(getDialogQueries().getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(getDialogQueries().getByLabelText("Template")).toBeInTheDocument()
    })
    fireEvent.click(getDialogQueries().getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(getDialogQueries().getByLabelText("Audio briefing")).toBeInTheDocument()
    })
    fireEvent.click(getDialogQueries().getByLabelText("Audio briefing"))
    fireEvent.click(getDialogQueries().getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-pipeline-review-summary")).toHaveTextContent("Policy Feed")
    })
    fireEvent.click(getDialogQueries().getByRole("button", { name: "Create pipeline" }))

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          sourceMode: "new",
          sourceName: "Policy Feed",
          sourceUrl: "https://example.com/policy.xml",
          monitorName: "Policy Brief",
          audioEnabled: false,
          audioSpeakers: []
        }),
        { mode: "create" }
      )
    })
  })
})
