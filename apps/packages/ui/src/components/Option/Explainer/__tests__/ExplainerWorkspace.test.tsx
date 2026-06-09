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
  grounding: "source_led",
  depthPreset: "standard",
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
      childNodeIds: [],
      createdAt: "2026-06-09T00:00:00Z",
      updatedAt: "2026-06-09T00:00:01Z"
    }
  },
  createdAt: "2026-06-09T00:00:00Z",
  updatedAt: "2026-06-09T00:00:01Z",
  archivedAt: null
}

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
      items: [{ ...sampleSession, nodeCount: 1, selectedSourceCount: 1 }],
      total: 1,
      limit: 50,
      offset: 0
    })
    api.getSession.mockResolvedValue(sampleSession)
    api.createSession.mockResolvedValue(sampleSession)
    api.updateSession.mockResolvedValue(sampleSession)
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

  it("renders the Explainer heading and explicit Goal/Sources tabs", async () => {
    renderWorkspace()

    expect(await screen.findByRole("heading", { name: "Explainer" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Goal" })).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Sources" })).toBeInTheDocument()
    expect(await screen.findByLabelText("Saved Explainer sessions")).toHaveValue("session-1")
  })

  it("creates a persisted goal session with configurable intent and depth", async () => {
    renderWorkspace()

    fireEvent.change(await screen.findByLabelText("Learning goal"), {
      target: { value: "Explain transformer attention" }
    })
    fireEvent.change(screen.getByLabelText("Output intent"), {
      target: { value: "plan" }
    })
    fireEvent.change(screen.getByLabelText("Depth preset"), {
      target: { value: "deep" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Create Explainer" }))

    await waitFor(() => {
      expect(api.createSession).toHaveBeenCalledWith(
        expect.objectContaining({
          mode: "goal",
          title: "Explain transformer attention",
          outputIntent: "plan",
          grounding: "open",
          depthPreset: "deep",
          rootPrompt: "Explain transformer attention",
          selectedSources: []
        })
      )
    })
  })

  it("lets users search and select sources in the page", async () => {
    renderWorkspace()

    fireEvent.click(await screen.findByRole("tab", { name: "Sources" }))
    fireEvent.change(screen.getByLabelText("Source search"), {
      target: { value: "attention" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Search sources" }))

    expect(await screen.findByText("New attention source")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Add New attention source" }))

    expect(api.searchSources).toHaveBeenCalledWith("attention")
    expect(screen.getByText("Selected sources")).toBeInTheDocument()
    expect(screen.getAllByText("New attention source")[0]).toBeInTheDocument()
  })

  it("shows configurable grounding, persisted tree details, citations, and export action", async () => {
    renderWorkspace()

    fireEvent.click(await screen.findByRole("tab", { name: "Sources" }))
    expect(screen.getByLabelText("Grounding mode")).toHaveValue("source_led")

    const tree = await screen.findByRole("tree", { name: "Explainer outline" })
    expect(
      await within(tree).findByText("Explain transformer attention")
    ).toBeInTheDocument()
    expect(screen.getByText("Attention lets tokens route information to each other.")).toBeInTheDocument()
    expect(screen.getByText("Attention weights are computed from query-key similarity.")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Export to Chatbook" }))

    await waitFor(() => {
      expect(api.exportChatbook).toHaveBeenCalledWith("session-1", {
        name: "Learn attention Explainer Session",
        asyncMode: true
      })
    })
    expect(await screen.findByText("Export job started: job-1")).toBeInTheDocument()
  })
})
