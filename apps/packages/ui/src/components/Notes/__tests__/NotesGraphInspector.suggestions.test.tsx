// @vitest-environment jsdom
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
        "notesSearch.graphGroundedSuggestions": "Grounded suggestions",
        "notesSearch.graphProvider": "Provider",
        "notesSearch.graphModel": "Model",
        "notesSearch.graphDataBoundary": "Data boundary",
        "notesSearch.graphBoundary.external": "External",
        "notesSearch.graphOutboundData": "Data sent for generation",
        "notesSearch.graphOutbound.selected_note_excerpts":
          "Selected note excerpts",
        "notesSearch.graphOutbound.candidate_note_titles":
          "Candidate note titles",
        "notesSearch.graphStrongMatch": "Strong match",
        "notesSearch.graphPossibleMatch": "Possible match",
        "notesSearch.graphSourceEvidence": "Source evidence",
        "notesSearch.graphTargetEvidence": "Target evidence",
        "notesSearch.graphGenerate": "Generate",
        "notesSearch.graphRegenerate": "Regenerate",
        "notesSearch.graphSuggestionMenu": "Suggestion actions",
        "notesSearch.graphResetDismissed": "Reset dismissed suggestions",
        "notesSearch.graphExistingTag": "Use existing tag",
        "notesSearch.graphNewTag": "Create new tag"
      }
      if (key.startsWith("notesSearch.graphRunState."))
        return key.split(".").at(-1)
      const label = labels[key] ?? key
      return Object.entries(options ?? {}).reduce(
        (value, [name, replacement]) =>
          value.replace(`{{${name}}}`, String(replacement)),
        label
      )
    }
  })
}))

const fingerprint = `sha256:${"a".repeat(64)}`
const graph = {
  nodes: [
    {
      id: "note:source",
      type: "note" as const,
      label: "Source <script>alert(1)</script>",
      created_at: null,
      deleted: false,
      degree: 1,
      tag_count: 0,
      primary_source_id: null
    },
    {
      id: "note:target",
      type: "note" as const,
      label: "Target",
      created_at: null,
      deleted: false,
      degree: 1,
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
  active_note_count: 2,
  all_notes_note_cap: 20,
  all_notes_eligible: true,
  suggestions_authorized: true
}

const suggestion = {
  id: "suggestion-one",
  run_id: "run-one",
  kind: "related_note" as const,
  state: "pending" as const,
  revision: 1,
  source_note_id: "source",
  source_fingerprint: fingerprint,
  target_note_id: "target",
  target_fingerprint: fingerprint,
  target_title: "Target <img src=x onerror=alert(1)>",
  normalized_tag: null,
  display_tag: null,
  existing_tag: false,
  match_strength: "strong" as const,
  rationale: "Grounded <b>reason</b>",
  evidence: [
    {
      side: "source" as const,
      note_id: "source",
      field: "content" as const,
      start_offset: 0,
      end_offset: 4,
      text: "Source evidence"
    },
    {
      side: "target" as const,
      note_id: "target",
      field: "content" as const,
      start_offset: 0,
      end_offset: 4,
      text: "Target evidence"
    }
  ],
  updated_at: "2026-08-27T12:00:00Z"
}

const controller = (overrides: Record<string, unknown> = {}) => ({
  capabilities: {
    provider: "provider <script>",
    model: "model <img>",
    endpoint_origin_revision: fingerprint,
    data_boundary: "unknown",
    disclosure_external: true,
    outbound_data_categories: [
      "selected_note_excerpts",
      "candidate_note_titles"
    ],
    generation_available: true,
    unavailable_reason: null,
    limits: {},
    allowed_actions: [
      "generate",
      "cancel",
      "accept",
      "reject",
      "reset_rejections"
    ],
    revision: fingerprint,
    etag: `"${fingerprint}"`
  },
  activeRun: null,
  lastTerminalRun: null,
  suggestions: [suggestion],
  isOffline: false,
  generate: vi.fn(),
  cancel: vi.fn(),
  accept: vi.fn(),
  reject: vi.fn(),
  resetRejections: vi.fn(),
  mutations: {
    generation: { isPending: false },
    cancellation: { isPending: false },
    acceptance: { isPending: false },
    rejection: { isPending: false },
    reset: { isPending: false }
  },
  capabilitiesQuery: { isLoading: false, error: null },
  suggestionsQuery: { isLoading: false, error: null },
  ...overrides
})
const sharedCallbacks = () => ({
  onAnnounce: vi.fn(),
  onDecideSuggestion: vi.fn().mockResolvedValue(true)
})

describe("NotesGraphInspector", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    uiMocks.confirmDanger.mockResolvedValue(true)
  })
  it("omits all suggestion affordances when the graph is read-only", () => {
    render(
      <NotesGraphInspector
        graph={{ ...graph, suggestions_authorized: false }}
        selectedNodeId="note:source"
        suggestionsAuthorized={false}
        isOnline
        controller={controller() as never}
        {...sharedCallbacks()}
        onSelectNode={vi.fn()}
      />
    )

    expect(screen.getByRole("tab", { name: "Details" })).toBeInTheDocument()
    expect(
      screen.queryByRole("tab", { name: "Suggestions" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /generate/i })
    ).not.toBeInTheDocument()
  })

  it("discloses grounded generation and renders hostile server text as ordinary text", () => {
    render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized
        isOnline
        controller={controller() as never}
        {...sharedCallbacks()}
        onSelectNode={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("tab", { name: "Suggestions" }))

    expect(screen.getByText("provider <script>")).toBeInTheDocument()
    expect(screen.getByText("model <img>")).toBeInTheDocument()
    expect(screen.getByText(/external/i)).toBeInTheDocument()
    expect(screen.getByText(/selected note excerpts/i)).toBeInTheDocument()
    expect(screen.getByText(/candidate note titles/i)).toBeInTheDocument()
    expect(
      screen.getByText("Target <img src=x onerror=alert(1)>")
    ).toBeInTheDocument()
    expect(screen.getByText("Grounded <b>reason</b>")).toBeInTheDocument()
    expect(document.querySelector("script")).toBeNull()
    expect(document.querySelector("img")).toBeNull()
    expect(screen.getByText("Strong match")).toBeInTheDocument()
    expect(screen.getAllByText("Source evidence")).toHaveLength(2)
    expect(screen.getAllByText("Target evidence")).toHaveLength(2)
  })

  it("moves and activates inspector tabs with roving keyboard focus", () => {
    render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized
        isOnline
        controller={controller() as never}
        {...sharedCallbacks()}
        onSelectNode={vi.fn()}
      />
    )
    const details = screen.getByRole("tab", { name: "Details" })
    const suggestions = screen.getByRole("tab", { name: "Suggestions" })

    expect(details).toHaveAttribute("tabindex", "0")
    expect(suggestions).toHaveAttribute("tabindex", "-1")
    details.focus()
    fireEvent.keyDown(details, { key: "ArrowRight" })
    expect(suggestions).toHaveFocus()
    expect(suggestions).toHaveAttribute("aria-selected", "true")
    expect(details).toHaveAttribute("tabindex", "-1")

    fireEvent.keyDown(suggestions, { key: "Home" })
    expect(details).toHaveFocus()
    expect(details).toHaveAttribute("aria-selected", "true")
    fireEvent.keyDown(details, { key: "End" })
    expect(suggestions).toHaveFocus()
    fireEvent.keyDown(suggestions, { key: "ArrowLeft" })
    expect(details).toHaveFocus()
  })

  it.each([
    "admitting",
    "queued",
    "running",
    "cancelling",
    "publishing",
    "succeeded",
    "failed",
    "cancelled",
    "stale"
  ])("shows a non-color-only %s run status", (state) => {
    const terminal = ["succeeded", "failed", "cancelled", "stale"].includes(
      state
    )
    const run = {
      id: "run-one",
      state,
      provider: "provider",
      model: "model",
      revision: 1
    }
    render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized
        isOnline
        controller={
          controller({
            activeRun: terminal ? null : run,
            lastTerminalRun: terminal ? run : null
          }) as never
        }
        {...sharedCallbacks()}
        onSelectNode={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("tab", { name: "Suggestions" }))
    expect(
      screen.getByTestId("notes-graph-suggestion-run-status")
    ).toHaveTextContent(new RegExp(state, "i"))
  })

  it("gates offline commands while leaving live announcements to the workspace", () => {
    render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized
        isOnline={false}
        controller={controller({ isOffline: true }) as never}
        {...sharedCallbacks()}
        onSelectNode={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("tab", { name: "Suggestions" }))
    expect(screen.getByRole("button", { name: /generate/i })).toBeDisabled()
    expect(document.querySelectorAll('[aria-live="polite"]')).toHaveLength(0)
  })

  it("distinguishes existing and new tag suggestions", () => {
    const existingTag = {
      ...suggestion,
      id: "existing-tag",
      kind: "tag" as const,
      target_note_id: null,
      target_fingerprint: null,
      target_title: null,
      normalized_tag: "cardiology",
      display_tag: "Cardiology",
      existing_tag: true,
      match_strength: "possible" as const,
      evidence: suggestion.evidence.filter((entry) => entry.side === "source")
    }
    const newTag = {
      ...existingTag,
      id: "new-tag",
      normalized_tag: "hemodynamics",
      display_tag: "Hemodynamics",
      existing_tag: false
    }
    render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized
        isOnline
        controller={controller({ suggestions: [existingTag, newTag] }) as never}
        {...sharedCallbacks()}
        onSelectNode={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("tab", { name: "Suggestions" }))

    expect(screen.getByText("Use existing tag")).toBeInTheDocument()
    expect(screen.getByText("Create new tag")).toBeInTheDocument()
    expect(screen.getByText("Cardiology")).toBeInTheDocument()
    expect(screen.getByText("Hemodynamics")).toBeInTheDocument()
    expect(screen.getAllByText("Possible match")).toHaveLength(2)
  })

  it("moves focus forward after a successful middle-row decision", async () => {
    const suggestions = [
      { ...suggestion, id: "suggestion-one", target_title: "First target" },
      { ...suggestion, id: "suggestion-two", target_title: "Second target" },
      { ...suggestion, id: "suggestion-three", target_title: "Third target" }
    ]
    render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized
        isOnline
        controller={controller({ suggestions }) as never}
        onAnnounce={vi.fn()}
        onDecideSuggestion={vi.fn().mockResolvedValue(true)}
        onSelectNode={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("tab", { name: "Suggestions" }))
    fireEvent.click(
      screen.getAllByRole("button", {
        name: "notesSearch.graphAcceptSuggestion"
      })[1]
    )

    await waitFor(() =>
      expect(
        screen.getAllByRole("button", {
          name: "notesSearch.graphAcceptSuggestion"
        })[2]
      ).toHaveFocus()
    )
  })

  it("confirms reset, cancels an active run, and restores focus after a failed decision", async () => {
    const resetRejections = vi.fn().mockResolvedValue({})
    const cancel = vi.fn().mockResolvedValue({})
    const onDecideSuggestion = vi.fn().mockResolvedValue(false)
    render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:source"
        suggestionsAuthorized
        isOnline
        controller={
          controller({
            activeRun: {
              id: "run-one",
              state: "running",
              provider: "provider",
              model: "model",
              revision: 1,
              cancellation_available: true
            },
            cancel,
            resetRejections
          }) as never
        }
        onAnnounce={vi.fn()}
        onDecideSuggestion={onDecideSuggestion}
        onSelectNode={vi.fn()}
      />
    )
    fireEvent.click(screen.getByRole("tab", { name: "Suggestions" }))
    const menuTrigger = screen.getByRole("button", {
      name: "Suggestion actions"
    })
    fireEvent.click(menuTrigger)
    const resetMenuItem = screen.getByRole("menuitem", {
      name: "Reset dismissed suggestions"
    })
    await waitFor(() => expect(resetMenuItem).toHaveFocus())
    fireEvent.keyDown(resetMenuItem, { key: "Escape" })
    expect(
      screen.queryByRole("menuitem", {
        name: "Reset dismissed suggestions"
      })
    ).not.toBeInTheDocument()
    await waitFor(() => expect(menuTrigger).toHaveFocus())

    fireEvent.click(menuTrigger)
    fireEvent.click(
      screen.getByRole("menuitem", { name: "Reset dismissed suggestions" })
    )
    await waitFor(() => expect(resetRejections).toHaveBeenCalledTimes(1))
    expect(uiMocks.confirmDanger).toHaveBeenCalledTimes(1)

    fireEvent.click(
      screen.getByRole("button", { name: "notesSearch.graphCancelRun" })
    )
    await waitFor(() => expect(cancel).toHaveBeenCalledTimes(1))

    const rejectButton = screen.getByRole("button", {
      name: "notesSearch.graphRejectSuggestion"
    })
    rejectButton.focus()
    fireEvent.click(rejectButton)
    await waitFor(() =>
      expect(onDecideSuggestion).toHaveBeenCalledWith(
        "reject",
        "suggestion-one"
      )
    )
    await waitFor(() => expect(rejectButton).toHaveFocus())
    expect(document.querySelectorAll('[aria-live="polite"]')).toHaveLength(0)
  })
})
