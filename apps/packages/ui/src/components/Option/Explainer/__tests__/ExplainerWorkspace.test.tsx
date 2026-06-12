import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { ExplainerSession } from "../types"
import { ExplainerWorkspace } from "../ExplainerWorkspace"

const api = vi.hoisted(() => ({
  listSessions: vi.fn(),
  getSession: vi.fn(),
  createSession: vi.fn(),
  updateSession: vi.fn(),
  deleteSession: vi.fn(),
  createNode: vi.fn(),
  updateNode: vi.fn(),
  deleteNode: vi.fn(),
  expandNode: vi.fn(),
  answerQuestion: vi.fn(),
  getJob: vi.fn(),
  exportChatbook: vi.fn(),
  searchSources: vi.fn()
}))

vi.mock("../explainerApi", () => ({
  explainerApi: api
}))

const sampleSession: ExplainerSession = {
  id: "session-1",
  ownerUserId: "7",
  title: "Learn attention",
  mode: "goal",
  status: "active",
  outputIntent: "both",
  grounding: "open",
  depthPreset: "deep",
  selectedSources: [
    {
      sourceId: "media-42",
      sourceType: "media",
      title: "Attention notes",
      addedAt: "2026-06-09T00:00:00Z",
      metadata: { snapshotHash: "sha256:test" }
    }
  ],
  rootNodeIds: ["root"],
  nodes: {
    root: {
      id: "root",
      sessionId: "session-1",
      parentId: null,
      ordinal: 0,
      title: "Explain transformer attention",
      body: "Attention lets tokens route information to each other.",
      kind: "summary",
      intent: "both",
      status: "complete",
      evidenceState: "supported",
      outsideKnowledgeUsed: false,
      citations: [
        {
          id: "cite-1",
          sourceId: "media-42",
          sourceType: "media",
          title: "Attention notes",
          excerpt: "Attention weights are computed from query-key similarity.",
          locationLabel: "chunk 3",
          snapshotHash: "sha256:citation"
        }
      ],
      childNodeIds: ["child"],
      createdAt: "2026-06-09T00:00:00Z",
      updatedAt: "2026-06-09T00:00:01Z"
    },
    child: {
      id: "child",
      sessionId: "session-1",
      parentId: "root",
      ordinal: 1,
      title: "Scaled dot-product attention",
      body: "Compares query and key vectors.",
      kind: "explanation",
      intent: "explain",
      status: "complete",
      evidenceState: "partially_supported",
      outsideKnowledgeUsed: true,
      citations: [],
      childNodeIds: [],
      createdAt: "2026-06-09T00:00:02Z",
      updatedAt: "2026-06-09T00:00:03Z"
    }
  },
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:01Z",
  archivedAt: null
}

const summaryOf = (session: ExplainerSession) => ({
  ...session,
  nodeCount: Object.keys(session.nodes).length,
  selectedSourceCount: session.selectedSources.length
})

const renderWorkspace = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <ExplainerWorkspace />
    </QueryClientProvider>
  )
}

describe("ExplainerWorkspace", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    api.listSessions.mockResolvedValue({
      items: [summaryOf(sampleSession)],
      total: 1,
      limit: 50,
      offset: 0
    })
    api.getSession.mockResolvedValue(sampleSession)
    api.createSession.mockResolvedValue(sampleSession)
    api.updateSession.mockResolvedValue(sampleSession)
    api.deleteSession.mockResolvedValue(sampleSession)
    api.deleteNode.mockResolvedValue({ id: "child", status: "deleted" })
    api.exportChatbook.mockResolvedValue({
      success: true,
      message: "Export job started: job-1",
      job_id: "job-1"
    })
    api.searchSources.mockResolvedValue([
      {
        sourceId: "media-99",
        sourceType: "media",
        title: "New attention source",
        description: "Imported PDF",
        metadata: { mediaId: 99 }
      }
    ])
  })

  it("renders the heading and collapses the composer once a session is active", async () => {
    renderWorkspace()

    expect(await screen.findByRole("heading", { name: "Explainer" })).toBeInTheDocument()
    expect(await screen.findByLabelText("Saved Explainer sessions")).toHaveValue("session-1")
    await waitFor(() => {
      expect(screen.queryByLabelText("Learning goal")).not.toBeInTheDocument()
    })
    expect(screen.getByRole("button", { name: "New explainer" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "New explainer" }))
    expect(screen.getByRole("tab", { name: "Goal" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Sources" })).toBeInTheDocument()
    expect(screen.getByLabelText("Learning goal")).toBeInTheDocument()
  })

  it("creates a goal session from the reopened composer and collapses afterwards", async () => {
    renderWorkspace()

    fireEvent.click(await screen.findByRole("button", { name: "New explainer" }))
    fireEvent.change(screen.getByLabelText("Learning goal"), {
      target: { value: "Explain transformer attention" }
    })
    fireEvent.change(screen.getByLabelText("Output intent"), { target: { value: "plan" } })
    fireEvent.change(screen.getByLabelText("Depth preset"), { target: { value: "deep" } })
    fireEvent.click(screen.getByRole("button", { name: "Create Explainer" }))

    await waitFor(() => {
      expect(api.createSession).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "goal",
          title: "Explain transformer attention",
          outputIntent: "plan",
          grounding: "open",
          depthPreset: "deep"
        })
      )
    })
    await waitFor(() => {
      expect(screen.queryByLabelText("Learning goal")).not.toBeInTheDocument()
    })
  })

  it("shows the loaded session's settings in the rail, not composer drafts", async () => {
    renderWorkspace()

    const rail = await screen.findByLabelText("Explainer session settings")
    // sampleSession is both/open/deep — composer defaults are explain/source_led/standard
    await waitFor(() => {
      expect(within(rail).getByText("Open")).toBeInTheDocument()
    })
    expect(within(rail).getByText("Deep")).toBeInTheDocument()
    expect(within(rail).getByText("Explain & plan")).toBeInTheDocument()
  })

  it("sanitizes error messages and lets users dismiss them", async () => {
    api.expandNode.mockRejectedValue(
      new Error(
        "Explainer generation is not configured (POST /api/v1/explainer/sessions/s/nodes/n/expand)"
      )
    )
    renderWorkspace()

    const detail = await screen.findByRole("region", { name: "Explainer detail" })
    await within(detail).findByText("Explain transformer attention")
    fireEvent.click(within(detail).getByRole("button", { name: "Break down" }))

    const banner = await screen.findByText("Explainer generation is not configured")
    expect(banner.textContent).not.toContain("/api/v1")

    fireEvent.click(screen.getByRole("button", { name: "Dismiss error" }))
    expect(screen.queryByText("Explainer generation is not configured")).not.toBeInTheDocument()
  })

  it("renames the session inline from the header", async () => {
    renderWorkspace()

    fireEvent.click(await screen.findByRole("button", { name: "Rename session" }))
    const input = screen.getByLabelText("Session title")
    fireEvent.change(input, { target: { value: "Attention, properly" } })
    fireEvent.click(screen.getByRole("button", { name: "Save title" }))

    await waitFor(() => {
      expect(api.updateSession).toHaveBeenCalledWith("session-1", { title: "Attention, properly" })
    })
  })

  it("archives the session behind an explicit confirm", async () => {
    renderWorkspace()

    fireEvent.click(await screen.findByRole("button", { name: "Archive session" }))
    expect(api.deleteSession).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Confirm archive" }))

    await waitFor(() => {
      expect(api.deleteSession).toHaveBeenCalledWith("session-1")
    })
  })

  it("deletes a non-root node behind an explicit confirm", async () => {
    renderWorkspace()

    const tree = await screen.findByRole("tree", { name: "Explainer outline" })
    fireEvent.click(await within(tree).findByText("Scaled dot-product attention"))

    const detail = screen.getByRole("region", { name: "Explainer detail" })
    fireEvent.click(within(detail).getByRole("button", { name: "Delete node" }))
    expect(api.deleteNode).not.toHaveBeenCalled()
    fireEvent.click(within(detail).getByRole("button", { name: "Confirm delete" }))

    await waitFor(() => {
      expect(api.deleteNode).toHaveBeenCalledWith("session-1", "child")
    })
  })

  it("marks the node being generated and disables its break-down control", async () => {
    api.expandNode.mockResolvedValue({
      jobId: "job-9",
      sessionId: "session-1",
      nodeId: "root",
      status: "queued"
    })
    api.getJob.mockResolvedValue({
      jobId: "job-9",
      sessionId: "session-1",
      nodeId: "root",
      status: "running",
      progressMessage: "Generating expansion"
    })
    renderWorkspace()

    const detail = await screen.findByRole("region", { name: "Explainer detail" })
    await within(detail).findByText("Explain transformer attention")
    fireEvent.click(within(detail).getByRole("button", { name: "Break down" }))

    const tree = screen.getByRole("tree", { name: "Explainer outline" })
    await waitFor(() => {
      expect(within(tree).getByText("Generating")).toBeInTheDocument()
    })
    expect(
      within(tree).getByRole("button", { name: "Break down Explain transformer attention" })
    ).toBeDisabled()
    expect(within(detail).getByRole("button", { name: "Break down" })).toBeDisabled()
  })

  it("links to the chatbook download when the export returns one", async () => {
    api.exportChatbook.mockResolvedValue({
      success: true,
      message: "Export complete",
      job_id: "job-1",
      download_url: "/api/v1/chatbooks/download/job-1"
    })
    renderWorkspace()

    await screen.findByLabelText("Saved Explainer sessions")
    const exportButton = screen.getByRole("button", { name: "Export to Chatbook" })
    await waitFor(() => expect(exportButton).toBeEnabled())
    fireEvent.click(exportButton)

    const link = await screen.findByRole("link", { name: "Download chatbook" })
    expect(link).toHaveAttribute("href", "/api/v1/chatbooks/download/job-1")
  })

  it("scrolls the detail pane into view on narrow viewports when selecting a node", async () => {
    const scrollSpy = vi.fn()
    window.HTMLElement.prototype.scrollIntoView = scrollSpy
    const originalWidth = window.innerWidth
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 390 })

    renderWorkspace()
    const tree = await screen.findByRole("tree", { name: "Explainer outline" })
    fireEvent.click(await within(tree).findByText("Scaled dot-product attention"))

    await waitFor(() => {
      expect(scrollSpy).toHaveBeenCalled()
    })

    Object.defineProperty(window, "innerWidth", { configurable: true, value: originalWidth })
  })

  it("searches sources when pressing Enter in the query field", async () => {
    renderWorkspace()

    fireEvent.click(await screen.findByRole("button", { name: "New explainer" }))
    fireEvent.click(screen.getByRole("tab", { name: "Sources" }))
    const input = screen.getByLabelText("Source search")
    fireEvent.change(input, { target: { value: "attention" } })
    fireEvent.keyDown(input, { key: "Enter" })

    await waitFor(() => {
      expect(api.searchSources).toHaveBeenCalledWith("attention")
    })
  })

  it("clears active job progress when switching sessions", async () => {
    api.listSessions.mockResolvedValue({
      items: [
        summaryOf(sampleSession),
        summaryOf({ ...sampleSession, id: "session-2", title: "Second session" })
      ],
      total: 2,
      limit: 50,
      offset: 0
    })
    api.getSession.mockImplementation(async (sessionId: string) =>
      sessionId === "session-2"
        ? { ...sampleSession, id: "session-2", title: "Second session" }
        : sampleSession
    )
    api.expandNode.mockResolvedValue({
      jobId: "job-9",
      sessionId: "session-1",
      nodeId: "root",
      status: "queued"
    })
    api.getJob.mockResolvedValue({
      jobId: "job-9",
      sessionId: "session-1",
      nodeId: "root",
      status: "running",
      progressMessage: "Generating expansion"
    })
    renderWorkspace()

    const detail = await screen.findByRole("region", { name: "Explainer detail" })
    await within(detail).findByText("Explain transformer attention")
    fireEvent.click(within(detail).getByRole("button", { name: "Break down" }))
    expect(await screen.findByText("Generating expansion")).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText("Saved Explainer sessions"), {
      target: { value: "session-2" }
    })

    await waitFor(() => {
      expect(screen.queryByText("Generating expansion")).not.toBeInTheDocument()
    })
  })
})

describe("ExplainerWorkspace upgrades", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    api.getSession.mockResolvedValue(sampleSession)
    api.createSession.mockResolvedValue(sampleSession)
    api.answerQuestion.mockResolvedValue(sampleSession.nodes.child)
    api.searchSources.mockResolvedValue([])
    api.exportChatbook.mockResolvedValue({ success: true, message: "ok", job_id: "j" })
  })

  it("persists a clarifying answer through the API", async () => {
    const questionSession: ExplainerSession = {
      ...sampleSession,
      nodes: {
        ...sampleSession.nodes,
        child: {
          ...sampleSession.nodes.child,
          kind: "question",
          questionOptions: [{ id: "math", label: "Focus on math" }],
          selectedOptionId: null,
          selectedCustomAnswer: null
        }
      }
    }
    api.listSessions.mockResolvedValue({
      items: [summaryOf(questionSession)],
      total: 1,
      limit: 50,
      offset: 0
    })
    api.getSession.mockResolvedValue(questionSession)
    renderWorkspace()

    const tree = await screen.findByRole("tree", { name: "Explainer outline" })
    fireEvent.click(await within(tree).findByText("Scaled dot-product attention"))
    const detail = screen.getByRole("region", { name: "Explainer detail" })
    fireEvent.click(within(detail).getByRole("button", { name: "Focus on math" }))

    await waitFor(() => {
      expect(api.answerQuestion).toHaveBeenCalledWith("session-1", "child", {
        selectedOptionId: "math"
      })
    })
  })

  it("prefills the goal composer from a template card", async () => {
    api.listSessions.mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 })
    renderWorkspace()

    const gallery = await screen.findByRole("region", { name: "Explainer templates" })
    fireEvent.click(within(gallery).getByRole("button", { name: /Prepare a study plan/ }))

    expect((screen.getByLabelText("Learning goal") as HTMLTextAreaElement).value).toContain(
      "study plan"
    )
    expect(screen.getByLabelText("Output intent")).toHaveValue("plan")
    expect(screen.getByLabelText("Depth preset")).toHaveValue("deep")
  })
})
