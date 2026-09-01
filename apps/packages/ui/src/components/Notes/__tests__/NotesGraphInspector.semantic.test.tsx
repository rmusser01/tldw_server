// @vitest-environment jsdom
import { NotesSemanticClientError } from "@/services/note-semantic-index"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import NotesGraphInspector from "../NotesGraphInspector"

const uiMocks = vi.hoisted(() => ({ confirmDanger: vi.fn() }))
vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => uiMocks.confirmDanger
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) => {
      const labels: Record<string, string> = {
        "notesSearch.graphDetails": "Details",
        "notesSearch.graphSimilarContent": "Similar content",
        "notesSearch.semanticIndex": "Semantic index",
        "notesSearch.semanticState.off": "Off",
        "notesSearch.semanticState.preparing": "Preparing",
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
        "notesSearch.semanticLoading": "Loading semantic index details.",
        "notesSearch.semanticEnable": "Enable semantic index",
        "notesSearch.semanticShowEdges": "Show similar content",
        "notesSearch.semanticRebuild": "Rebuild index",
        "notesSearch.semanticRetry": "Retry failed Notes",
        "notesSearch.semanticCancel": "Cancel indexing",
        "notesSearch.semanticDelete": "Delete index",
        "notesSearch.semanticDetail.building": "Indexing is in progress.",
        "notesSearch.semanticDetail.degraded":
          "Some Notes could not be indexed.",
        "notesSearch.semanticDetail.stale_configuration":
          "Configuration changed; review consent.",
        "notesSearch.semanticDetail.consent_required":
          "Consent must be renewed.",
        "notesSearch.semanticDetail.cleanup_pending":
          "Vector cleanup is pending.",
        "notesSearch.semanticDetail.cleanup_stalled":
          "Vector cleanup needs attention.",
        "notesSearch.semanticUnavailable": "Semantic indexing is unavailable.",
        "notesSearch.semanticPermissionReadOnly":
          "You can view similar content but cannot manage this index.",
        "notesSearch.semanticEnableStarted": "Semantic indexing started.",
        "notesSearch.semanticRebuildStarted": "Semantic index rebuild started.",
        "notesSearch.semanticRetryStarted": "Failed Notes retry started.",
        "notesSearch.semanticCancelRequested":
          "Semantic index cancellation requested.",
        "notesSearch.semanticDeleteStarted": "Semantic index deletion started.",
        "notesSearch.semanticDimensionProbeDisclosure":
          "A fixed non-user probe will resolve dimensions after you consent and before any Note text is read or transferred.",
        "notesSearch.semanticTechnicalDetails": "Technical details",
        "notesSearch.semanticVectorBackend": "Vector backend",
        "notesSearch.semanticDimensions": "Dimensions",
        "notesSearch.semanticDimensionsPending": "Pending probe",
        "notesSearch.semanticMetric": "Similarity metric",
        "notesSearch.semanticMetricValue.cosine": "Cosine",
        "notesSearch.semanticConfigurationRevision": "Configuration revision",
        "notesSearch.semanticIndexRevision": "Semantic index revision",
        "notesSearch.semanticPublishedChunks": "Published chunks",
        "notesSearch.semanticError.refresh":
          "Semantic index details changed. Refresh and try again."
      }
      const label = labels[key] ?? key
      return Object.entries(options ?? {}).reduce(
        (value, [name, replacement]) =>
          value.replace(`{{${name}}}`, String(replacement)),
        label
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

const capabilities = (overrides: Record<string, unknown> = {}) => ({
  active_note_count: 12,
  estimated_chunk_count: 36,
  estimated_run_count: 2,
  provider_label: "OpenAI",
  model: "text-embedding-3-small",
  execution_boundary: "external",
  storage_boundary: "local",
  storage_label: "ChromaDB",
  outbound_data_categories: ["note_content_chunks", "note_title"],
  capability_revision: `sha256:${"a".repeat(64)}`,
  indexing_available: true,
  unavailable_reason: null,
  metric: "cosine",
  resolved_dimensions: 1536,
  manage_authorized: true,
  ...overrides
})

const status = (overrides: Record<string, unknown> = {}) => ({
  state: "off",
  detail_reason: null,
  desired_state: "disabled",
  configuration_revision: 0,
  semantic_index_revision: 0,
  active_generation_id: null,
  indexed_notes: 0,
  excluded_notes: 0,
  failed_notes: 0,
  pending_notes: 0,
  published_chunks: 0,
  cleanup_pending: false,
  active_run: null,
  ...overrides
})

const semanticController = (overrides: Record<string, unknown> = {}) => ({
  capabilities: capabilities(),
  status: status(),
  activeRun: null,
  lastTerminalRun: null,
  isOffline: false,
  capabilitiesQuery: { isLoading: false, error: null },
  statusQuery: { isLoading: false, error: null },
  runQuery: { isLoading: false, error: null },
  mutations: {
    enable: { isPending: false },
    rebuild: { isPending: false },
    retry: { isPending: false },
    cancel: { isPending: false },
    deleteIndex: { isPending: false }
  },
  enable: vi.fn().mockResolvedValue(undefined),
  rebuild: vi.fn().mockResolvedValue(undefined),
  retryFailed: vi.fn().mockResolvedValue(undefined),
  cancel: vi.fn().mockResolvedValue(undefined),
  deleteIndex: vi.fn().mockResolvedValue(undefined),
  ...overrides
})

const suggestionController = {
  capabilities: null,
  activeRun: null,
  lastTerminalRun: null,
  suggestions: [],
  provisionalBySuggestionId: {},
  mutations: {},
  capabilitiesQuery: { isLoading: false, error: null },
  suggestionsQuery: { isLoading: false, error: null }
}

const renderInspector = (
  semantic: ReturnType<typeof semanticController>,
  options: {
    semanticEnabled?: boolean
    onAnnounce?: ReturnType<typeof vi.fn>
  } = {}
) => {
  const onAnnounce = options.onAnnounce ?? vi.fn()
  const view = render(
    <NotesGraphInspector
      graph={graph}
      selectedNodeId="note:source"
      suggestionsAuthorized={false}
      isOnline
      controller={suggestionController as never}
      semanticController={semantic as never}
      semanticEnabled={options.semanticEnabled ?? false}
      onSemanticEnabledChange={vi.fn()}
      onSelectNode={vi.fn()}
      onAnnounce={onAnnounce}
      onDecideSuggestion={vi.fn().mockResolvedValue(true)}
    />
  )
  fireEvent.click(screen.getByRole("tab", { name: "Similar content" }))
  return { onAnnounce, ...view }
}

describe("NotesGraphInspector semantic setup", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    uiMocks.confirmDanger.mockResolvedValue(true)
  })

  it("shows source-grounded setup disclosure and no arbitrary configuration controls", () => {
    renderInspector(semanticController())

    expect(
      screen.getByRole("heading", { name: "Semantic index" })
    ).toBeInTheDocument()
    expect(screen.getByText("Off")).toBeInTheDocument()
    expect(screen.getByText("12 active Notes")).toBeInTheDocument()
    expect(
      screen.getByText("About 36 chunks across 2 runs")
    ).toBeInTheDocument()
    expect(screen.getByText("OpenAI")).toBeInTheDocument()
    expect(screen.getByText("text-embedding-3-small")).toBeInTheDocument()
    const storageDisclosure =
      screen.getByText("Vector storage").nextElementSibling
    expect(storageDisclosure).toHaveTextContent("ChromaDB")
    expect(storageDisclosure).toHaveTextContent("Local")
    expect(screen.getByText("Note body chunks")).toBeInTheDocument()
    expect(screen.getByText("Note titles")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Enable semantic index" })
    ).toBeEnabled()
    expect(screen.queryByLabelText(/provider/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/model/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/endpoint/i)).not.toBeInTheDocument()
  })

  it("does not mislabel loading capabilities as unavailable or read-only", () => {
    renderInspector(
      semanticController({
        capabilities: null,
        status: null,
        capabilitiesQuery: { isLoading: true, error: null },
        statusQuery: { isLoading: true, error: null }
      })
    )

    expect(
      screen.getByText("Loading semantic index details.")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Semantic indexing is unavailable.")
    ).not.toBeInTheDocument()
    expect(screen.queryByText(/cannot manage this index/i)).toBeNull()
  })

  it("discloses unresolved dimensions before consent and keeps technical metadata restrained", () => {
    renderInspector(
      semanticController({
        capabilities: capabilities({ resolved_dimensions: null })
      })
    )

    expect(
      screen.getByText(
        "A fixed non-user probe will resolve dimensions after you consent and before any Note text is read or transferred."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByText("Technical details", { selector: "summary" })
    ).toBeInTheDocument()
    expect(screen.getByText("Pending probe")).toBeInTheDocument()
    expect(screen.getByText("Cosine")).toBeInTheDocument()
    expect(screen.queryByLabelText(/dimensions|metric/i)).toBeNull()
  })

  it("shows the edge filter only for a usable generation", () => {
    renderInspector(
      semanticController({
        status: status({ state: "preparing", detail_reason: "building" })
      })
    )

    expect(
      screen.queryByRole("checkbox", { name: "Show similar content" })
    ).toBeNull()
  })

  it.each([
    ["preparing", "building", "Preparing", "Indexing is in progress."],
    ["ready", null, "Ready", null],
    ["updating", "building", "Updating", "Indexing is in progress."],
    [
      "needs_attention",
      "degraded",
      "Needs attention",
      "Some Notes could not be indexed."
    ],
    [
      "needs_attention",
      "stale_configuration",
      "Needs attention",
      "Configuration changed; review consent."
    ],
    [
      "needs_attention",
      "consent_required",
      "Needs attention",
      "Consent must be renewed."
    ],
    ["off", "cleanup_pending", "Off", "Vector cleanup is pending."]
  ])(
    "renders %s / %s as a named lifecycle state",
    (stateName, reason, label, detail) => {
      const active = ["preparing", "updating"].includes(String(stateName))
        ? {
            run_id: "run-a",
            mode: "rebuild",
            status: "processing",
            revision: 2,
            indexed_notes: 4,
            excluded_notes: 1,
            failed_notes: 0,
            pending_notes: 7,
            published_chunks: 12,
            cleanup_complete: false,
            error_code: null,
            link: "/api/v1/notes/graph/semantic-index/runs/run-a"
          }
        : null
      renderInspector(
        semanticController({
          status: status({
            state: stateName,
            detail_reason: reason,
            desired_state: stateName === "off" ? "disabled" : "enabled",
            indexed_notes: 4,
            excluded_notes: 1,
            failed_notes: reason === "degraded" ? 2 : 0,
            pending_notes: active ? 7 : 0,
            cleanup_pending: reason === "cleanup_pending",
            active_run: active
          }),
          activeRun: active
        })
      )

      expect(screen.getByText(label)).toBeInTheDocument()
      expect(screen.getByText("4 of 12 Notes indexed")).toBeInTheDocument()
      if (detail) expect(screen.getByText(detail)).toBeInTheDocument()
    }
  )

  it("confirms every lifecycle command and announces completion", async () => {
    const semantic = semanticController({
      status: status({
        state: "needs_attention",
        detail_reason: "degraded",
        desired_state: "enabled",
        configuration_revision: 7,
        failed_notes: 2
      })
    })
    const { onAnnounce } = renderInspector(semantic)

    fireEvent.click(screen.getByRole("button", { name: "Retry failed Notes" }))
    await waitFor(() => expect(semantic.retryFailed).toHaveBeenCalledTimes(1))
    expect(uiMocks.confirmDanger).toHaveBeenCalledTimes(1)
    expect(onAnnounce).toHaveBeenCalledWith("Failed Notes retry started.")

    fireEvent.click(screen.getByRole("button", { name: "Rebuild index" }))
    await waitFor(() => expect(semantic.rebuild).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole("button", { name: "Delete index" }))
    await waitFor(() => expect(semantic.deleteIndex).toHaveBeenCalledTimes(1))
    expect(uiMocks.confirmDanger).toHaveBeenCalledTimes(3)
    expect(onAnnounce).toHaveBeenCalledWith("Semantic index rebuild started.")
    expect(onAnnounce).toHaveBeenCalledWith("Semantic index deletion started.")
  })

  it("confirms enablement and disables it while cleanup is pending", async () => {
    const semantic = semanticController()
    const { onAnnounce, unmount } = renderInspector(semantic)

    fireEvent.click(
      screen.getByRole("button", { name: "Enable semantic index" })
    )
    await waitFor(() => expect(semantic.enable).toHaveBeenCalledTimes(1))
    expect(uiMocks.confirmDanger).toHaveBeenCalledTimes(1)
    expect(onAnnounce).toHaveBeenCalledWith("Semantic indexing started.")
    unmount()

    renderInspector(
      semanticController({
        status: status({ cleanup_pending: true })
      })
    )
    expect(
      screen.getByRole("button", { name: "Enable semantic index" })
    ).toBeDisabled()
  })

  it("renders and announces an actionable revision-conflict recovery", async () => {
    const semantic = semanticController({
      status: status({
        state: "ready",
        desired_state: "enabled",
        active_generation_id: "generation-a"
      }),
      rebuild: vi
        .fn()
        .mockRejectedValueOnce(
          new NotesSemanticClientError(
            409,
            "notes_semantic_configuration_revision_conflict"
          )
        )
        .mockResolvedValueOnce(undefined)
    })
    const { onAnnounce } = renderInspector(semantic)

    fireEvent.click(screen.getByRole("button", { name: "Rebuild index" }))
    const alert = await screen.findByText(
      "Semantic index details changed. Refresh and try again."
    )
    expect(alert).toHaveAttribute("role", "alert")
    expect(onAnnounce).toHaveBeenCalledWith(
      "Semantic index details changed. Refresh and try again."
    )

    fireEvent.click(screen.getByRole("button", { name: "Rebuild index" }))
    await waitFor(() => expect(semantic.rebuild).toHaveBeenCalledTimes(2))
    await waitFor(() =>
      expect(
        screen.queryByText(
          "Semantic index details changed. Refresh and try again."
        )
      ).toBeNull()
    )
    expect(onAnnounce).toHaveBeenCalledWith("Semantic index rebuild started.")
  })

  it("offers cancel during active work and confirms before requesting it", async () => {
    const active = {
      run_id: "run-a",
      mode: "rebuild",
      status: "processing",
      revision: 2,
      indexed_notes: 4,
      excluded_notes: 0,
      failed_notes: 0,
      pending_notes: 8,
      published_chunks: 12,
      cleanup_complete: false,
      error_code: null,
      link: "/api/v1/notes/graph/semantic-index/runs/run-a"
    }
    const semantic = semanticController({
      status: status({ state: "preparing", active_run: active }),
      activeRun: active
    })
    const { onAnnounce } = renderInspector(semantic)

    fireEvent.click(screen.getByRole("button", { name: "Cancel indexing" }))
    await waitFor(() => expect(semantic.cancel).toHaveBeenCalledTimes(1))
    expect(uiMocks.confirmDanger).toHaveBeenCalledTimes(1)
    expect(onAnnounce).toHaveBeenCalledWith(
      "Semantic index cancellation requested."
    )
  })

  it("uses capability authority to hide management and blocks unavailable enablement", () => {
    const { unmount } = render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized={false}
        isOnline
        controller={suggestionController as never}
        semanticController={
          semanticController({
            capabilities: capabilities({ manage_authorized: false }),
            status: status({ state: "ready", desired_state: "enabled" })
          }) as never
        }
        semanticEnabled
        onSemanticEnabledChange={vi.fn()}
        onSelectNode={vi.fn()}
        onAnnounce={vi.fn()}
        onDecideSuggestion={vi.fn().mockResolvedValue(true)}
      />
    )
    fireEvent.click(screen.getByRole("tab", { name: "Similar content" }))
    expect(screen.getByText(/cannot manage this index/i)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /rebuild|delete|cancel|retry/i })
    ).toBeNull()
    expect(
      screen.getByRole("checkbox", { name: "Show similar content" })
    ).toBeEnabled()
    unmount()

    renderInspector(
      semanticController({
        capabilities: capabilities({
          indexing_available: false,
          unavailable_reason: "notes_semantic_provider_unavailable"
        })
      })
    )
    expect(
      screen.getByText("Semantic indexing is unavailable.")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Enable semantic index" })
    ).toBeDisabled()
  })
})
