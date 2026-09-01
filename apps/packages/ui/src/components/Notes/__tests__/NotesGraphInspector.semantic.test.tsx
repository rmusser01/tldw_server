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
        "notesSearch.graphSuggestions": "Suggestions",
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
        "notesSearch.semanticRenewConsent": "Review consent and rebuild",
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
        "notesSearch.semanticDetail.generic":
          "Semantic index details changed. Review the current status.",
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
          "Semantic index details changed. Refresh and try again.",
        "notesSearch.semanticStatusAnnouncement":
          "Semantic index {{state}}. {{detail}}",
        "notesSearch.semanticConfirm.renew.title":
          "Review consent and rebuild?",
        "notesSearch.semanticConfirm.renew.body":
          "Review the current provider, model, data boundaries, and outbound data above before renewing consent.",
        "notesSearch.semanticConfirm.renew.confirm":
          "Renew consent and rebuild",
        "notesSearch.semanticConfirm.deleteIndex.body":
          "Published vectors and semantic relationships will be removed. Offline backups may retain derived vectors until normal backup retention expires."
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
  const element = (semanticValue: ReturnType<typeof semanticController>) => (
    <NotesGraphInspector
      graph={graph}
      selectedNodeId="note:source"
      suggestionsAuthorized={false}
      isOnline
      controller={suggestionController as never}
      semanticController={semanticValue as never}
      semanticEnabled={options.semanticEnabled ?? false}
      onSemanticEnabledChange={vi.fn()}
      onSelectNode={vi.fn()}
      onAnnounce={onAnnounce}
      onDecideSuggestion={vi.fn().mockResolvedValue(true)}
    />
  )
  const view = render(element(semantic))
  fireEvent.click(screen.getByRole("tab", { name: "Similar content" }))
  return {
    onAnnounce,
    rerenderSemantic: (next: ReturnType<typeof semanticController>) =>
      view.rerender(element(next)),
    ...view
  }
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
            active_generation_id:
              stateName === "updating" ? "generation-a" : null,
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

  it("confirms enablement and hides it while cleanup is pending", async () => {
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
      screen.queryByRole("button", { name: "Enable semantic index" })
    ).toBeNull()
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
      status: status({
        state: "preparing",
        desired_state: "enabled",
        active_run: active
      }),
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
      screen.queryByRole("button", { name: "Enable semantic index" })
    ).toBeNull()
  })

  it("uses the active indexing run for Updating state and current progress", () => {
    const active = {
      run_id: "run-progress",
      mode: "rebuild",
      status: "processing",
      revision: 8,
      indexed_notes: 5,
      excluded_notes: 1,
      failed_notes: 0,
      pending_notes: 6,
      published_chunks: 18,
      cleanup_complete: false,
      error_code: null,
      link: "/api/v1/notes/graph/semantic-index/runs/run-progress"
    }
    renderInspector(
      semanticController({
        status: status({
          state: "ready",
          desired_state: "enabled",
          active_generation_id: "generation-a",
          indexed_notes: 12,
          pending_notes: 0,
          active_run: active
        }),
        activeRun: active
      })
    )

    expect(screen.getByText("Updating")).toBeInTheDocument()
    expect(screen.getByText("5 of 12 Notes indexed")).toBeInTheDocument()
  })

  it.each([
    ["healthy Off", status(), capabilities(), null, ["Enable semantic index"]],
    [
      "cleanup-pending Off",
      status({ detail_reason: "cleanup_pending", cleanup_pending: true }),
      capabilities(),
      null,
      []
    ],
    [
      "stalled cleanup",
      status({
        state: "needs_attention",
        detail_reason: "cleanup_stalled",
        cleanup_pending: true
      }),
      capabilities(),
      null,
      []
    ],
    [
      "Ready",
      status({
        state: "ready",
        desired_state: "enabled",
        active_generation_id: "generation-a"
      }),
      capabilities(),
      null,
      ["Rebuild index", "Delete index"]
    ],
    [
      "degraded",
      status({
        state: "needs_attention",
        detail_reason: "degraded",
        desired_state: "enabled",
        active_generation_id: "generation-a",
        failed_notes: 2
      }),
      capabilities(),
      null,
      ["Retry failed Notes", "Rebuild index", "Delete index"]
    ],
    [
      "unavailable enabled index",
      status({
        state: "needs_attention",
        detail_reason: "unavailable",
        desired_state: "enabled",
        active_generation_id: "generation-a"
      }),
      capabilities({
        indexing_available: false,
        unavailable_reason: "notes_semantic_provider_unavailable"
      }),
      null,
      ["Delete index"]
    ],
    [
      "stale configuration",
      status({
        state: "needs_attention",
        detail_reason: "stale_configuration",
        desired_state: "enabled",
        active_generation_id: "generation-a"
      }),
      capabilities(),
      null,
      ["Review consent and rebuild", "Delete index"]
    ],
    [
      "active rebuild",
      status({
        state: "updating",
        detail_reason: "building",
        desired_state: "enabled",
        active_generation_id: "generation-a"
      }),
      capabilities(),
      {
        run_id: "run-rebuild",
        mode: "rebuild",
        status: "processing",
        revision: 7,
        indexed_notes: 3,
        excluded_notes: 0,
        failed_notes: 0,
        pending_notes: 9,
        published_chunks: 8,
        cleanup_complete: false,
        error_code: null,
        link: "/api/v1/notes/graph/semantic-index/runs/run-rebuild"
      },
      ["Cancel indexing"]
    ],
    [
      "delete cleanup",
      status({
        state: "off",
        detail_reason: "cleanup_pending",
        desired_state: "disabled",
        cleanup_pending: true
      }),
      capabilities(),
      {
        run_id: "run-delete",
        mode: "delete",
        status: "processing",
        revision: 8,
        indexed_notes: 0,
        excluded_notes: 0,
        failed_notes: 0,
        pending_notes: 0,
        published_chunks: 0,
        cleanup_complete: false,
        error_code: null,
        link: "/api/v1/notes/graph/semantic-index/runs/run-delete"
      },
      []
    ]
  ])(
    "shows only backend-valid management actions for %s",
    (
      _label,
      semanticStatus,
      semanticCapability,
      activeRun,
      expectedActions
    ) => {
      renderInspector(
        semanticController({
          status: semanticStatus,
          capabilities: semanticCapability,
          activeRun
        })
      )
      const actionNames = [
        "Enable semantic index",
        "Review consent and rebuild",
        "Retry failed Notes",
        "Rebuild index",
        "Cancel indexing",
        "Delete index"
      ]

      expect(
        actionNames.filter((name) => screen.queryByRole("button", { name }))
      ).toEqual(expectedActions)
    }
  )

  it("renews stale consent through the enable command after fresh disclosure confirmation", async () => {
    const semantic = semanticController({
      status: status({
        state: "needs_attention",
        detail_reason: "stale_configuration",
        desired_state: "enabled",
        active_generation_id: "generation-a",
        configuration_revision: 7
      })
    })
    renderInspector(semantic)

    fireEvent.click(
      screen.getByRole("button", { name: "Review consent and rebuild" })
    )
    await waitFor(() => expect(semantic.enable).toHaveBeenCalledTimes(1))
    expect(uiMocks.confirmDanger).toHaveBeenCalledWith(
      expect.objectContaining({
        content:
          "Review the current provider, model, data boundaries, and outbound data above before renewing consent."
      })
    )
  })

  it("hides enablement when a direct capability value lacks the complete consent disclosure", () => {
    renderInspector(
      semanticController({
        capabilities: capabilities({
          outbound_data_categories: ["note_title"]
        })
      })
    )

    expect(
      screen.queryByRole("button", { name: "Enable semantic index" })
    ).toBeNull()
  })

  it("uses one localized generic detail for unknown server reasons", () => {
    const { onAnnounce } = renderInspector(
      semanticController({
        status: status({
          state: "needs_attention",
          detail_reason: "new_server_reason",
          desired_state: "enabled"
        })
      })
    )

    expect(
      screen.getByText(
        "Semantic index details changed. Review the current status."
      )
    ).toBeInTheDocument()
    expect(screen.queryByText(/semanticDetail\.new_server_reason/)).toBeNull()
    expect(onAnnounce).toHaveBeenCalledWith(
      "Semantic index Needs attention. Semantic index details changed. Review the current status."
    )
  })

  it("discloses normal backup retention in the actual delete confirmation", async () => {
    const semantic = semanticController({
      status: status({
        state: "ready",
        desired_state: "enabled",
        active_generation_id: "generation-a"
      })
    })
    renderInspector(semantic)

    fireEvent.click(screen.getByRole("button", { name: "Delete index" }))
    await waitFor(() => expect(semantic.deleteIndex).toHaveBeenCalledTimes(1))
    expect(uiMocks.confirmDanger).toHaveBeenCalledWith(
      expect.objectContaining({
        content:
          "Published vectors and semantic relationships will be removed. Offline backups may retain derived vectors until normal backup retention expires."
      })
    )
  })

  it("moves focus to the stable semantic heading after enable and cancel commands", async () => {
    const enableSemantic = semanticController()
    const first = renderInspector(enableSemantic)
    const enableButton = screen.getByRole("button", {
      name: "Enable semantic index"
    })
    enableButton.focus()
    fireEvent.click(enableButton)
    await waitFor(() => expect(enableSemantic.enable).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Semantic index" })
      ).toHaveFocus()
    )
    first.unmount()

    const active = {
      run_id: "run-focus",
      mode: "rebuild",
      status: "processing",
      revision: 7,
      indexed_notes: 3,
      excluded_notes: 0,
      failed_notes: 0,
      pending_notes: 9,
      published_chunks: 8,
      cleanup_complete: false,
      error_code: null,
      link: "/api/v1/notes/graph/semantic-index/runs/run-focus"
    }
    const cancelSemantic = semanticController({
      status: status({
        state: "updating",
        desired_state: "enabled",
        active_generation_id: "generation-a",
        active_run: active
      }),
      activeRun: active
    })
    renderInspector(cancelSemantic)
    const cancelButton = screen.getByRole("button", {
      name: "Cancel indexing"
    })
    cancelButton.focus()
    fireEvent.click(cancelButton)
    await waitFor(() => expect(cancelSemantic.cancel).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Semantic index" })
      ).toHaveFocus()
    )
  })

  it("preserves focus when an active run becomes terminal and replaces its action", async () => {
    const active = {
      run_id: "run-terminal-focus",
      mode: "rebuild",
      status: "processing",
      revision: 7,
      indexed_notes: 10,
      excluded_notes: 0,
      failed_notes: 0,
      pending_notes: 2,
      published_chunks: 30,
      cleanup_complete: false,
      error_code: null,
      link: "/api/v1/notes/graph/semantic-index/runs/run-terminal-focus"
    }
    const view = renderInspector(
      semanticController({
        status: status({
          state: "updating",
          desired_state: "enabled",
          active_generation_id: "generation-a",
          active_run: active
        }),
        activeRun: active
      })
    )
    screen.getByRole("button", { name: "Cancel indexing" }).focus()

    view.rerenderSemantic(
      semanticController({
        status: status({
          state: "ready",
          desired_state: "enabled",
          active_generation_id: "generation-a"
        })
      })
    )

    await waitFor(() =>
      expect(
        screen.getByRole("heading", { name: "Semantic index" })
      ).toHaveFocus()
    )
  })

  it("supports roving keyboard focus across all three inspector tabs", () => {
    render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized
        isOnline
        controller={suggestionController as never}
        semanticController={semanticController() as never}
        semanticEnabled={false}
        onSemanticEnabledChange={vi.fn()}
        onSelectNode={vi.fn()}
        onAnnounce={vi.fn()}
        onDecideSuggestion={vi.fn().mockResolvedValue(true)}
      />
    )
    const details = screen.getByRole("tab", { name: "Details" })
    const suggestions = screen.getByRole("tab", { name: "Suggestions" })
    const semantic = screen.getByRole("tab", { name: "Similar content" })
    details.focus()

    fireEvent.keyDown(details, { key: "ArrowRight" })
    expect(suggestions).toHaveFocus()
    fireEvent.keyDown(suggestions, { key: "ArrowRight" })
    expect(semantic).toHaveFocus()
    fireEvent.keyDown(semantic, { key: "ArrowRight" })
    expect(details).toHaveFocus()
  })
})
