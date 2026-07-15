// @vitest-environment jsdom
import React from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"

import { FloatingProgressWidget } from "../FloatingProgressWidget"
import { IngestWizardProvider } from "../IngestWizardContext"
import type { ItemProgressStatus, ProcessingStatus } from "../types"
import {
  createEmptyQuickIngestSession,
  useQuickIngestSessionStore,
} from "@/store/quick-ingest-session"

const translationMock = vi.hoisted(() =>
  vi.fn(
    (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [k: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      const value = defaultValueOrOptions?.defaultValue || key
      return value.replace(/\{\{(\w+)\}\}/g, (_match: string, token: string) =>
        defaultValueOrOptions?.[token] == null
          ? `{{${token}}}`
          : String(defaultValueOrOptions[token])
      )
    }
  )
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translationMock,
  }),
}))

vi.mock("lucide-react", () => {
  const icon = (name: string) => (props: any) => (
    <span data-icon={name} aria-hidden={props?.["aria-hidden"]} />
  )
  return {
    AlertTriangle: icon("AlertTriangle"),
    Check: icon("Check"),
    ExternalLink: icon("ExternalLink"),
    Loader2: icon("Loader2"),
    XCircle: icon("XCircle"),
  }
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  translationMock.mockClear()
  useQuickIngestSessionStore.setState({
    session: null,
    triggerSummary: { count: 0, label: null, hadFailure: false },
  })
})

const renderWidget = ({
  processingStatus,
  itemStatus,
  lifecycle = "completed",
  estimatedRemaining = 0,
}: {
  processingStatus: ProcessingStatus
  itemStatus: ItemProgressStatus
  lifecycle?:
    | "processing"
    | "completed"
    | "partial_failure"
    | "cancelled"
    | "interrupted"
  estimatedRemaining?: number
}) => {
  useQuickIngestSessionStore.setState({
    session: {
      ...createEmptyQuickIngestSession(),
      visibility: "hidden",
      lifecycle,
    },
  })

  render(
    <IngestWizardProvider
      initialState={{
        isMinimized: true,
        processingState: {
          status: processingStatus,
          perItemProgress: [
            {
              id: "item-1",
              status: itemStatus,
              progressPercent: 100,
              currentStage: itemStatus,
              estimatedRemaining,
            },
          ],
          elapsed: 10,
          estimatedRemaining,
        },
      }}
    >
      <FloatingProgressWidget />
    </IngestWizardProvider>
  )
}

describe("FloatingProgressWidget", () => {
  it("shows Done for complete minimized sessions", () => {
    renderWidget({ processingStatus: "complete", itemStatus: "complete" })

    expect(screen.getByRole("status")).toHaveTextContent("Done")
  })

  it("shows Failed for error minimized sessions", () => {
    renderWidget({
      processingStatus: "error",
      itemStatus: "failed",
      lifecycle: "partial_failure",
    })

    expect(screen.getByRole("status")).toHaveTextContent("Failed")
  })

  it("shows Cancelled for cancelled minimized sessions", () => {
    renderWidget({
      processingStatus: "cancelled",
      itemStatus: "cancelled",
      lifecycle: "cancelled",
    })

    expect(screen.getByRole("status")).toHaveTextContent("Cancelled")
  })

  it("shows Interrupted for interrupted hidden sessions", () => {
    renderWidget({
      processingStatus: "error",
      itemStatus: "failed",
      lifecycle: "interrupted",
    })

    expect(screen.getByRole("status")).toHaveTextContent("Interrupted")
  })

  it("summarizes completed conference runs without overstating search readiness", () => {
    useQuickIngestSessionStore.setState({
      session: {
        ...createEmptyQuickIngestSession(),
        visibility: "hidden",
        lifecycle: "partial_failure",
      },
      triggerSummary: { count: 4, label: "1 failed", hadFailure: true },
    })

    render(
      <IngestWizardProvider
        initialState={{
          isMinimized: true,
          conferenceBatchMetadata: {
            collectionName: "Conference 2010",
            conferenceName: "Conference",
            eventYear: "2010",
            sharedTags: ["conference"],
            sourcePlaylistUrl: "https://www.youtube.com/playlist?list=PLtest",
          },
          processingState: {
            status: "error",
            elapsed: 120,
            estimatedRemaining: 0,
            perItemProgress: [
              {
                id: "ok-1",
                status: "complete",
                progressPercent: 100,
                currentStage: "Complete",
                estimatedRemaining: 0,
              },
              {
                id: "ok-2",
                status: "complete",
                progressPercent: 100,
                currentStage: "Complete",
                estimatedRemaining: 0,
              },
              {
                id: "skip-1",
                status: "complete",
                progressPercent: 100,
                currentStage: "Complete",
                estimatedRemaining: 0,
              },
              {
                id: "failed-1",
                status: "failed",
                progressPercent: 100,
                currentStage: "Failed",
                estimatedRemaining: 0,
              },
            ],
          },
          results: [
            {
              id: "ok-1",
              status: "ok",
              outcome: "processed",
              type: "video",
            },
            {
              id: "ok-2",
              status: "ok",
              outcome: "processed",
              type: "video",
            },
            {
              id: "skip-1",
              status: "ok",
              outcome: "skipped",
              type: "video",
            },
            {
              id: "failed-1",
              status: "error",
              outcome: "failed",
              type: "video",
              error: "Download failed",
            },
          ],
        }}
      >
        <FloatingProgressWidget />
      </IngestWizardProvider>
    )

    const widget = screen.getByRole("status", { name: "Ingest progress" })
    expect(widget).toHaveTextContent("Conference 2010")
    expect(widget).toHaveTextContent("2 succeeded, 1 skipped, 1 failed")
    expect(widget).toHaveTextContent("Open the wizard for collection readiness")
    expect(widget).not.toHaveTextContent(/searchable/i)
    expect(translationMock).toHaveBeenCalledWith(
      "quickIngest.widget.summary.succeeded",
      expect.objectContaining({ count: 2 })
    )
    expect(translationMock).toHaveBeenCalledWith(
      "quickIngest.widget.summary.skipped",
      expect.objectContaining({ count: 1 })
    )
    expect(translationMock).toHaveBeenCalledWith(
      "quickIngest.widget.summary.failed",
      expect.objectContaining({ count: 1 })
    )
    expect(translationMock).toHaveBeenCalledWith(
      "quickIngest.widget.collectionReadiness",
      expect.anything()
    )
  })

  it("translates the finished-count fallback instead of composing English directly", () => {
    renderWidget({ processingStatus: "complete", itemStatus: "complete" })

    expect(translationMock).toHaveBeenCalledWith(
      "quickIngest.widget.summary.finished",
      expect.objectContaining({ completed: 1, total: 1 })
    )
  })

  it("summarizes active, attention, and terminal lifecycle evidence", () => {
    useQuickIngestSessionStore.setState({
      session: {
        ...createEmptyQuickIngestSession(),
        visibility: "hidden",
        lifecycle: "processing",
      },
    })

    render(
      <IngestWizardProvider
        initialState={{
          isMinimized: true,
          processingState: {
            status: "running",
            elapsed: 12,
            estimatedRemaining: 0,
            perItemProgress: [
              {
                id: "active",
                status: "processing",
                lifecycleState: "running",
                terminalOutcome: null,
                progressPercent: 35,
                currentStage: "Downloading source",
                estimatedRemaining: 0,
              } as any,
              {
                id: "attention",
                status: "processing",
                lifecycleState: "status_unavailable",
                terminalOutcome: null,
                progressPercent: 0,
                currentStage: "Status unavailable",
                estimatedRemaining: 0,
              } as any,
              {
                id: "terminal",
                status: "complete",
                lifecycleState: "terminal",
                terminalOutcome: "completed",
                progressPercent: 100,
                currentStage: "Completed",
                estimatedRemaining: 0,
              } as any,
            ],
          },
        }}
      >
        <FloatingProgressWidget />
      </IngestWizardProvider>
    )

    const widget = screen.getByRole("status", { name: "Ingest progress" })
    expect(widget).toHaveTextContent("1 active, 1 needs attention, 1 terminal")
  })

  it("shows worker progress evidence without inventing analyze or store stages", () => {
    useQuickIngestSessionStore.setState({
      session: {
        ...createEmptyQuickIngestSession(),
        visibility: "hidden",
        lifecycle: "processing",
      },
    })

    render(
      <IngestWizardProvider
        initialState={{
          isMinimized: true,
          processingState: {
            status: "running",
            elapsed: 12,
            estimatedRemaining: 0,
            perItemProgress: [
              {
                id: "active",
                status: "processing",
                lifecycleState: "running",
                terminalOutcome: null,
                progressPercent: 35,
                currentStage: "Downloading source",
                estimatedRemaining: 0,
              } as any,
            ],
          },
        }}
      >
        <FloatingProgressWidget />
      </IngestWizardProvider>
    )

    const widget = screen.getByRole("status", { name: "Ingest progress" })
    expect(widget).toHaveTextContent("Downloading source")
    expect(widget).not.toHaveTextContent(/processing and indexing/i)
    expect(widget).not.toHaveTextContent(/analyz|storing/i)
  })

  it.each([
    [42, "quickIngest.widget.etaSeconds", { count: 42 }],
    [120, "quickIngest.widget.etaMinutes", { count: 2 }],
  ] as const)(
    "localizes active ETA evidence for %s seconds",
    (estimatedRemaining, key, options) => {
      renderWidget({
        processingStatus: "running",
        itemStatus: "processing",
        lifecycle: "processing",
        estimatedRemaining,
      })

      expect(translationMock).toHaveBeenCalledWith(
        key,
        expect.objectContaining(options)
      )
    }
  )
})
