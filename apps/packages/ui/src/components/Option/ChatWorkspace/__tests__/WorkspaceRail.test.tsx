import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { WorkspaceRail } from "../WorkspaceRail"
import type { WorkspaceSource } from "@/types/workspace"

const sources: WorkspaceSource[] = [
  {
    id: "source-1",
    mediaId: 101,
    title: "Operator Notes",
    type: "document",
    status: "ready",
    addedAt: new Date("2026-05-03T00:00:00Z")
  },
  {
    id: "source-2",
    mediaId: 202,
    title: "Research Clip",
    type: "video",
    status: "ready",
    addedAt: new Date("2026-05-03T00:00:00Z")
  }
]

describe("WorkspaceRail", () => {
  it("selecting a source for browsing does not stage it", () => {
    const onBrowseSource = vi.fn()
    const onStageSources = vi.fn()

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={onBrowseSource}
        onStageSources={onStageSources}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Browse Operator Notes" })
    )

    expect(onBrowseSource).toHaveBeenCalledWith("source-1")
    expect(onStageSources).not.toHaveBeenCalled()
    expect(screen.queryByText("Context staged")).not.toBeInTheDocument()
  })

  it("stages only through the explicit Stage for Chat action", () => {
    const onStageSources = vi.fn()

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={onStageSources}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Stage Operator Notes for chat" })
    )

    expect(onStageSources).toHaveBeenCalledWith(["source-1"])
  })

  it("filters sources without changing browse or staged state", () => {
    const onBrowseSource = vi.fn()
    const onStageSources = vi.fn()

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={onBrowseSource}
        onStageSources={onStageSources}
      />
    )

    fireEvent.change(screen.getByRole("searchbox", { name: "Filter sources" }), {
      target: { value: "clip" }
    })

    expect(screen.queryByText("Operator Notes")).not.toBeInTheDocument()
    expect(screen.getByText("Research Clip")).toBeInTheDocument()
    expect(onBrowseSource).not.toHaveBeenCalled()
    expect(onStageSources).not.toHaveBeenCalled()
  })

  it("shows staged state, workspace name, and honest study v1 labels", () => {
    render(
      <WorkspaceRail
        workspaceName="Project Alpha"
        sources={sources}
        browsedSourceId="source-1"
        stagedSourceIds={["source-1"]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
      />
    )

    expect(
      screen.getByRole("heading", { name: "Project Alpha" })
    ).toBeInTheDocument()
    expect(screen.getByText("Context staged")).toBeInTheDocument()
    expect(screen.getByText("No generated study set")).toBeInTheDocument()
  })

  it("shows non-ready source status and prevents staging it", () => {
    const onStageSources = vi.fn()

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={[
          ...sources,
          {
            id: "source-3",
            mediaId: 303,
            title: "Processing Brief",
            type: "document",
            status: "processing",
            addedAt: new Date("2026-05-03T00:00:00Z")
          }
        ]}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={onStageSources}
      />
    )

    const stageButton = screen.getByRole("button", {
      name: "Stage Processing Brief for chat"
    })

    expect(screen.getByText("processing")).toBeInTheDocument()
    expect(stageButton).toBeDisabled()

    fireEvent.click(stageButton)

    expect(onStageSources).not.toHaveBeenCalled()
  })

  it("disambiguates duplicate source title action names", () => {
    const duplicateSources: WorkspaceSource[] = [
      {
        ...sources[0],
        id: "source-a",
        mediaId: 401,
        title: "Meeting Notes"
      },
      {
        ...sources[1],
        id: "source-b",
        mediaId: 402,
        title: "Meeting Notes"
      }
    ]

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={duplicateSources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
      />
    )

    expect(
      screen.getByRole("button", {
        name: "Browse Meeting Notes source-a"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: "Stage Meeting Notes source-b for chat"
      })
    ).toBeInTheDocument()
  })

  it("wraps long source titles inside action buttons", () => {
    const longTitle = "x".repeat(160)

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={[{ ...sources[0], title: longTitle }]}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
      />
    )

    const browseButton = screen.getByRole("button", {
      name: `Browse ${longTitle}`
    })
    const stageButton = screen.getByRole("button", {
      name: `Stage ${longTitle} for chat`
    })

    expect(browseButton).toHaveClass("min-w-0")
    expect(browseButton).toHaveClass("break-words")
    expect(stageButton).toHaveClass("min-w-0")
    expect(stageButton).toHaveClass("break-words")
  })
})
