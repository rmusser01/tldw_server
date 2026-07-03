import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"
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

  it("routes source-management actions to canonical surfaces", () => {
    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
        onUnstageSource={vi.fn()}
      />
    )

    expect(screen.getByRole("link", { name: "Add source" })).toHaveAttribute(
      "href",
      "/research-workspace?tab=sources"
    )
    expect(screen.getByRole("link", { name: "Open library" })).toHaveAttribute(
      "href",
      "/media"
    )
  })

  it("uses SPA links for source-management actions inside a router", () => {
    render(
      <MemoryRouter>
        <WorkspaceRail
          workspaceName="Default workspace"
          sources={sources}
          browsedSourceId={null}
          stagedSourceIds={[]}
          onBrowseSource={vi.fn()}
          onStageSources={vi.fn()}
          onUnstageSource={vi.fn()}
        />
      </MemoryRouter>
    )

    expect(screen.getByRole("link", { name: "Add source" })).toHaveAttribute(
      "href",
      "/research-workspace?tab=sources"
    )
    expect(screen.getByRole("link", { name: "Open library" })).toHaveAttribute(
      "href",
      "/media"
    )
  })

  it("unstages one source without staging another", () => {
    const onStageSources = vi.fn()
    const onUnstageSource = vi.fn()

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={["source-1"]}
        onBrowseSource={vi.fn()}
        onStageSources={onStageSources}
        onUnstageSource={onUnstageSource}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Unstage Operator Notes from chat" })
    )

    expect(onUnstageSource).toHaveBeenCalledWith("source-1")
    expect(onStageSources).not.toHaveBeenCalled()
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

  it("keeps ready sources without structured media ids stageable for fallback context", () => {
    const onStageSources = vi.fn()

    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={[
          {
            ...sources[0],
            mediaId: 0,
            title: "Fallback Brief",
            statusMessage: "No structured media id available."
          }
        ]}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={onStageSources}
        onUnstageSource={vi.fn()}
      />
    )

    const stageButton = screen.getByRole("button", {
      name: "Stage Fallback Brief for chat"
    })

    expect(stageButton).toBeEnabled()

    fireEvent.click(stageButton)

    expect(onStageSources).toHaveBeenCalledWith(["source-1"])
  })

  it("renders processing, error, and unavailable states with disabled stage actions", () => {
    render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={[
          {
            ...sources[0],
            id: "processing-source",
            title: "Processing Brief",
            status: "processing",
            statusMessage: "Still indexing."
          },
          {
            ...sources[0],
            id: "error-source",
            title: "Failed Brief",
            status: "error",
            statusMessage: "Indexing failed."
          },
          {
            ...sources[0],
            id: "unavailable-source",
            title: "Unavailable Brief",
            status: "blocked" as WorkspaceSource["status"]
          }
        ]}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
        onUnstageSource={vi.fn()}
      />
    )

    expect(screen.getByText("processing")).toBeInTheDocument()
    expect(screen.getByText("error")).toBeInTheDocument()
    expect(screen.getByText("unavailable")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Stage Processing Brief for chat" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Stage Failed Brief for chat" })
    ).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Stage Unavailable Brief for chat" })
    ).toBeDisabled()
  })

  it("distinguishes loading, source error, no-source, and filtered-empty states", () => {
    const { rerender } = render(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={[]}
        sourcesLoading
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
        onUnstageSource={vi.fn()}
      />
    )

    expect(screen.getByText("Loading workspace sources")).toBeInTheDocument()

    rerender(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={[]}
        sourcesError="Could not load workspace sources."
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
        onUnstageSource={vi.fn()}
      />
    )

    expect(screen.getByText("Could not load workspace sources.")).toBeInTheDocument()

    rerender(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={[]}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
        onUnstageSource={vi.fn()}
      />
    )

    expect(screen.getByText("No workspace sources yet")).toBeInTheDocument()

    rerender(
      <WorkspaceRail
        workspaceName="Default workspace"
        sources={sources}
        browsedSourceId={null}
        stagedSourceIds={[]}
        onBrowseSource={vi.fn()}
        onStageSources={vi.fn()}
        onUnstageSource={vi.fn()}
      />
    )
    fireEvent.change(screen.getByRole("searchbox", { name: "Filter sources" }), {
      target: { value: "missing" }
    })

    expect(screen.getByText("No sources match the filter")).toBeInTheDocument()
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
