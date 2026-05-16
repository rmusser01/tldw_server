// @vitest-environment jsdom
import React from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"
import {
  IngestWizardProvider,
} from "../IngestWizardContext"
import { FloatingProgressWidget } from "../FloatingProgressWidget"
import type { ItemProgressStatus, ProcessingStatus } from "../types"
import {
  createEmptyQuickIngestSession,
  useQuickIngestSessionStore,
} from "@/store/quick-ingest-session"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultOrOpts?: any) => {
      if (typeof defaultOrOpts === "string") return defaultOrOpts
      if (defaultOrOpts?.defaultValue) {
        return defaultOrOpts.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_: string, token: string) => String(defaultOrOpts[token] ?? "")
        )
      }
      return key
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
  useQuickIngestSessionStore.setState({
    session: null,
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
  const session = {
    ...createEmptyQuickIngestSession(),
    visibility: "hidden" as const,
    lifecycle,
  }
  useQuickIngestSessionStore.setState({ session })

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

describe("FloatingProgressWidget terminal states", () => {
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
})
