import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { ArtifactReviewStatus, GeneratedArtifact } from "@/types/workspace"
import {
  TraceableArtifactDetail,
  TraceableArtifactSummary
} from "../TraceableArtifactDetail"

const baseArtifact = (
  overrides: Partial<GeneratedArtifact> = {}
): GeneratedArtifact => ({
  id: "art-traceable",
  type: "report",
  title: "Reviewed ACP Brief",
  status: "completed",
  reviewStatus: "accepted",
  content: "# Brief\n\nAccepted body",
  contentType: "text/markdown",
  previewText: "Accepted body",
  summary: "A reviewed ACP brief",
  totalTokens: 560,
  totalCostUsd: 0.12,
  version: 3,
  rootArtifactId: "root-art-1",
  artifactVersionId: "version-art-3",
  previousVersionId: "version-art-2",
  schemaVersion: 1,
  producerMetadata: {
    producerType: "acp",
    producerId: "task-42",
    runId: "run-7",
    sessionId: "session-abc"
  },
  sourceLineage: [
    {
      sourceId: "src-1",
      sourceType: "media",
      title: "Transcript",
      mediaId: 42,
      citationCount: 1
    }
  ],
  reviewMetadata: {
    reviewerId: "reviewer-1",
    decision: "accepted"
  },
  versionMetadata: {
    revisionReason: "Reviewer accepted the brief"
  },
  exportTargets: ["markdown"],
  exportRefs: [{ format: "markdown", fileId: 101, status: "ready" }],
  redaction: {
    supportSafe: true,
    redacted: false,
    retentionClass: "standard"
  },
  createdAt: new Date("2026-05-06T12:05:00Z"),
  completedAt: new Date("2026-05-06T12:06:00Z"),
  ...overrides
})

describe("TraceableArtifactDetail", () => {
  it("renders review state, provenance, lineage, versioning, redaction, and export details", () => {
    render(<TraceableArtifactDetail artifact={baseArtifact()} />)

    expect(screen.getByTestId("traceable-artifact-review-state")).toHaveTextContent(
      "Accepted"
    )
    expect(screen.getByText("ACP provenance")).toBeInTheDocument()
    expect(screen.getByText("task-42")).toBeInTheDocument()
    expect(screen.getByText("run-7")).toBeInTheDocument()
    expect(screen.getByText("session-abc")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open session" })).toHaveAttribute(
      "href",
      "/acp-playground?session=session-abc"
    )
    expect(screen.getByRole("link", { name: "Diagnostics" })).toHaveAttribute(
      "href",
      "/acp-playground?session=session-abc&view=diagnostics"
    )

    expect(screen.getAllByText("Version").length).toBeGreaterThan(0)
    expect(screen.getAllByText("v3").length).toBeGreaterThan(0)
    expect(screen.getByText("version-art-2")).toBeInTheDocument()
    expect(screen.getByText("Reviewer accepted the brief")).toBeInTheDocument()

    expect(screen.getByText("Transcript")).toBeInTheDocument()
    expect(screen.getByText("src-1")).toBeInTheDocument()
    expect(screen.getByText("1 citation")).toBeInTheDocument()

    expect(screen.getAllByText("Support safe").length).toBeGreaterThan(0)
    expect(screen.getByText("Not redacted")).toBeInTheDocument()
    expect(screen.getByText("standard")).toBeInTheDocument()

    expect(screen.getByText("Markdown")).toBeInTheDocument()
    expect(screen.getByText("file #101")).toBeInTheDocument()
  })

  it.each<ArtifactReviewStatus>([
    "accepted",
    "needs_revision",
    "rejected",
    "assigned",
    "archived"
  ])("differentiates %s review state in the summary", (reviewStatus) => {
    render(<TraceableArtifactSummary artifact={baseArtifact({ reviewStatus })} />)

    const summary = screen.getByTestId("traceable-artifact-summary")
    expect(
      within(summary).getByText(
        reviewStatus
          .split("_")
          .map((part) => part[0].toUpperCase() + part.slice(1))
          .join(" ")
      )
    ).toBeInTheDocument()
  })

  it("renders unavailable metadata states without leaking raw support payloads", () => {
    render(
      <TraceableArtifactDetail
        artifact={baseArtifact({
          producerMetadata: undefined,
          sourceLineage: undefined,
          exportRefs: undefined,
          exportTargets: undefined,
          redaction: { supportSafe: false, redacted: true }
        })}
      />
    )

    expect(screen.getByText("No ACP provenance recorded")).toBeInTheDocument()
    expect(screen.getByText("No source lineage recorded")).toBeInTheDocument()
    expect(screen.getByText("No exports recorded")).toBeInTheDocument()
    expect(screen.getAllByText("Restricted").length).toBeGreaterThan(0)
    expect(screen.getByText("Redacted")).toBeInTheDocument()
    expect(screen.queryByText("{")).not.toBeInTheDocument()
  })

  it("exposes review-state controls when a transition handler is provided", () => {
    const onReviewStateChange = vi.fn()
    render(
      <TraceableArtifactDetail
        artifact={baseArtifact({ reviewStatus: "reviewing" })}
        onReviewStateChange={onReviewStateChange}
      />
    )

    const controls = screen.getByRole("group", {
      name: "Review state controls"
    })
    fireEvent.click(within(controls).getByRole("button", { name: "Needs Revision" }))

    expect(onReviewStateChange).toHaveBeenCalledWith("needs_revision")
  })
})
