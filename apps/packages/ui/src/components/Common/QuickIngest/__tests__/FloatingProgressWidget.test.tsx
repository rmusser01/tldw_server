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

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
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
    },
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
  useQuickIngestSessionStore.setState({
    session: null,
    triggerSummary: { count: 0, label: null, hadFailure: false },
  })
})

const renderWidget = ({
  processingStatus,
  itemStatus,
  lifecycle = "completed",
}: {
  processingStatus: ProcessingStatus
  itemStatus: ItemProgressStatus
  lifecycle?: "completed" | "partial_failure" | "cancelled" | "interrupted"
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
              estimatedRemaining: 0,
            },
          ],
          elapsed: 10,
          estimatedRemaining: 0,
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
  })
})
