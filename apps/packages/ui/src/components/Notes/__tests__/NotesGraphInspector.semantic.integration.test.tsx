// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphInspector from "../NotesGraphInspector"
import { useNotesSemanticIndex } from "../hooks/useNotesSemanticIndex"
import capabilityDriftApi from "./fixtures/semantic-capability-drift-api.json"

const mocks = vi.hoisted(() => ({
  capabilities: vi.fn(),
  status: vi.fn(),
  run: vi.fn(),
  createRun: vi.fn(),
  confirmDanger: vi.fn()
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mocks.confirmDanger
}))
vi.mock("@/services/note-semantic-index", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/note-semantic-index")>()
  return {
    ...actual,
    getNotesSemanticCapabilities: mocks.capabilities,
    getNotesSemanticStatus: mocks.status,
    getNotesSemanticRun: mocks.run,
    createNotesSemanticRun: mocks.createRun,
    createNotesSemanticCommand: () => ({
      idempotencyKey: "integration-command-key"
    })
  }
})
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) => {
      const labels: Record<string, string> = {
        "notesSearch.graphDetails": "Details",
        "notesSearch.graphSimilarContent": "Similar content",
        "notesSearch.semanticIndex": "Semantic index",
        "notesSearch.semanticState.ready": "Ready",
        "notesSearch.semanticState.updating": "Updating",
        "notesSearch.semanticState.needs_attention": "Needs attention",
        "notesSearch.semanticProvider": "Provider",
        "notesSearch.semanticModel": "Model",
        "notesSearch.semanticExecutionBoundary": "Embedding execution",
        "notesSearch.semanticStorageBoundary": "Vector storage",
        "notesSearch.semanticBoundary.external": "External",
        "notesSearch.semanticBoundary.local": "Local",
        "notesSearch.semanticOutboundData": "Data sent for indexing",
        "notesSearch.semanticOutbound.note_content_chunks": "Note body chunks",
        "notesSearch.semanticOutbound.note_title": "Note titles",
        "notesSearch.semanticActiveNotes": "{{count}} active Notes",
        "notesSearch.semanticEstimate":
          "About {{chunks}} chunks across {{runs}} runs",
        "notesSearch.semanticProgress":
          "{{indexed}} of {{total}} Notes indexed",
        "notesSearch.semanticRebuild": "Rebuild index",
        "notesSearch.semanticRenewConsent": "Review consent and rebuild",
        "notesSearch.semanticDelete": "Delete index",
        "notesSearch.semanticShowEdges": "Show similar content",
        "notesSearch.semanticCancel": "Cancel indexing",
        "notesSearch.semanticDetail.building": "Indexing is in progress.",
        "notesSearch.semanticDetail.stale_configuration":
          "Configuration changed; review consent.",
        "notesSearch.semanticStatusAnnouncement":
          "Semantic index {{state}}. {{detail}}",
        "notesSearch.semanticTechnicalDetails": "Technical details",
        "notesSearch.semanticVectorBackend": "Vector backend",
        "notesSearch.semanticDimensions": "Dimensions",
        "notesSearch.semanticMetric": "Similarity metric",
        "notesSearch.semanticMetricValue.cosine": "Cosine",
        "notesSearch.semanticConfigurationRevision": "Configuration revision",
        "notesSearch.semanticIndexRevision": "Semantic index revision",
        "notesSearch.semanticPublishedChunks": "Published chunks",
        "notesSearch.semanticRebuildStarted": "Semantic index rebuild started."
      }
      return Object.entries(options ?? {}).reduce(
        (value, [name, replacement]) =>
          value.replace(`{{${name}}}`, String(replacement)),
        labels[key] ?? key
      )
    }
  })
}))

const graph = {
  nodes: [
    {
      id: "note:source",
      type: "note" as const,
      label: "Source",
      created_at: null,
      deleted: false,
      degree: 0,
      tag_count: 0,
      primary_source_id: null
    }
  ],
  edges: [],
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 20, max_edges: 20, max_degree: 20 },
  radius_cap_applied: false,
  active_note_count: 1,
  all_notes_note_cap: 20,
  all_notes_eligible: true,
  suggestions_authorized: false
}

const capability = {
  active_note_count: 12,
  estimated_chunk_count: 36,
  estimated_run_count: 2,
  provider_label: "OpenAI",
  model: "text-embedding-3-small",
  endpoint_display: "https://api.openai.com",
  execution_boundary: "external",
  storage_boundary: "local",
  storage_label: "ChromaDB",
  outbound_data_categories: ["note_content_chunks", "note_title"],
  capability_revision: `sha256:${"a".repeat(64)}`,
  indexing_available: true,
  unavailable_reason: null,
  metric: "cosine",
  resolved_dimensions: 1536,
  dimension_probe_required: false,
  renewal_requires_delete: false,
  manage_authorized: true
}

const run = (indexed: number) => ({
  run_id: "run-a",
  mode: "rebuild",
  status: indexed === 2 ? "queued" : "processing",
  revision: indexed === 2 ? 1 : 2,
  indexed_notes: indexed,
  excluded_notes: 0,
  failed_notes: 0,
  pending_notes: 12 - indexed,
  published_chunks: indexed * 3,
  cleanup_complete: false,
  error_code: null,
  link: "/api/v1/notes/graph/semantic-index/runs/run-a"
})

const status = (
  state: "ready" | "updating",
  activeRun: ReturnType<typeof run> | null = null
) => ({
  state,
  detail_reason: state === "ready" ? null : "building",
  desired_state: "enabled",
  configuration_revision: 7,
  semantic_index_revision: 3,
  active_generation_id: "generation-a",
  active_generation_usable: true,
  indexed_notes: state === "ready" ? 12 : 2,
  excluded_notes: 0,
  failed_notes: 0,
  pending_notes: state === "ready" ? 0 : 10,
  published_chunks: state === "ready" ? 36 : 6,
  cleanup_pending: false,
  active_run: activeRun
})

const suggestions = {
  capabilities: null,
  activeRun: null,
  lastTerminalRun: null,
  suggestions: [],
  provisionalBySuggestionId: {},
  mutations: {},
  capabilitiesQuery: { isLoading: false, error: null },
  suggestionsQuery: { isLoading: false, error: null }
}

function Harness() {
  const semantic = useNotesSemanticIndex({
    authorityScope: "authority-a",
    enabled: true,
    isOnline: true,
    datasetId: "dataset-a",
    pollIntervalMs: 20
  })
  return (
    <NotesGraphInspector
      graph={graph}
      selectedNodeId="note:source"
      suggestionsAuthorized={false}
      isOnline
      controller={suggestions as never}
      semanticController={semantic}
      onSelectNode={vi.fn()}
      onAnnounce={vi.fn()}
      onDecideSuggestion={vi.fn().mockResolvedValue(true)}
    />
  )
}

describe("Notes semantic hook-to-inspector transition", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    mocks.confirmDanger.mockResolvedValue(true)
  })

  afterEach(() => {
    vi.clearAllTimers()
  })

  it("moves from Ready to Updating and renders current polled progress", async () => {
    const admitted = run(2)
    mocks.capabilities.mockResolvedValue(capability)
    mocks.status
      .mockResolvedValueOnce(status("ready"))
      .mockResolvedValue(status("updating", admitted))
    mocks.createRun.mockResolvedValue(admitted)
    mocks.run.mockResolvedValueOnce(admitted).mockResolvedValue(run(5))
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    render(
      <QueryClientProvider client={client}>
        <Harness />
      </QueryClientProvider>
    )

    await act(async () => {
      await Promise.resolve()
    })
    fireEvent.click(await screen.findByRole("tab", { name: "Similar content" }))
    expect(await screen.findByText("Ready")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Rebuild index" }))
    expect(await screen.findByText("Updating")).toBeInTheDocument()
    expect(await screen.findByText("2 of 12 Notes indexed")).toBeInTheDocument()

    await waitFor(() =>
      expect(screen.getByText("5 of 12 Notes indexed")).toBeInTheDocument()
    )
  })

  it("consumes API-produced capability drift and suppresses semantic edges", async () => {
    mocks.capabilities.mockResolvedValue(capabilityDriftApi.capabilities)
    mocks.status.mockResolvedValue(capabilityDriftApi.status)
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    render(
      <QueryClientProvider client={client}>
        <Harness />
      </QueryClientProvider>
    )

    fireEvent.click(await screen.findByRole("tab", { name: "Similar content" }))

    expect(await screen.findByText("Needs attention")).toBeInTheDocument()
    expect(
      screen.queryByRole("checkbox", { name: "Show similar content" })
    ).toBeNull()
    expect(
      screen.getByRole("button", { name: "Review consent and rebuild" })
    ).toBeEnabled()
    expect(screen.getByRole("button", { name: "Delete index" })).toBeEnabled()
  })
})
