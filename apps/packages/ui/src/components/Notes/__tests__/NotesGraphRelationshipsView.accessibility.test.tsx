// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import NotesGraphRelationshipsView, {
  NotesSemanticRelationshipDetails,
  buildNotesGraphRelationshipGroups
} from "../NotesGraphRelationshipsView"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, unknown>) => {
      const labels: Record<string, string> = {
        "notesSearch.graphRelationships": "Relationships",
        "notesSearch.graphRelationshipGroup.outgoing": "Outgoing",
        "notesSearch.graphRelationshipGroup.incoming": "Incoming",
        "notesSearch.graphRelationshipGroup.connected": "Connected",
        "notesSearch.graphRelationshipGroup.suggested": "Suggested",
        "notesSearch.graphPossibleMatch": "Possible match",
        "notesSearch.graphStrongMatch": "Strong match",
        "notesSearch.graphSourceEvidence": "Source evidence",
        "notesSearch.graphTargetEvidence": "Target evidence",
        "notesSearch.graphSimilarContent": "Similar content",
        "notesSearch.graphPassageSimilarity": "Passage similarity: {{value}}",
        "notesSearch.graphSimilarityBand.high": "High similarity",
        "notesSearch.graphSemanticProviderModel": "{{provider}} / {{model}}",
        "notesSearch.graphSemanticVersions":
          "Source v{{source}}, target v{{target}}",
        "notesSearch.graphSemanticGeneration": "Generation {{generation}}",
        "notesSearch.graphSemanticModelRevision": "Model revision",
        "notesSearch.graphSemanticNormalizationVersion":
          "Normalization version",
        "notesSearch.graphSemanticChunkerVersion": "Chunker version",
        "notesSearch.graphEvidenceOmitted":
          "Evidence omitted by response limit",
        "notesSearch.graphCreateManualLink": "Create manual link",
        "notesSearch.graphEdgeType.manual": "Manual link",
        "notesSearch.graphEdgeType.wikilink": "Note link",
        "notesSearch.graphEdgeType.semantic": "Similar content",
        "notesSearch.graphNextPage": "Next page",
        "notesSearch.graphPreviousPage": "Previous page",
        "notesSearch.graphAcceptSuggestion": "Accept {{title}}",
        "notesSearch.graphRejectSuggestion": "Reject {{title}}"
      }
      return Object.entries(options ?? {}).reduce(
        (value, [name, replacement]) =>
          value.replace(`{{${name}}}`, String(replacement)),
        labels[key] ?? key
      )
    }
  })
}))

const node = (id: string, label: string) => ({
  id,
  type: "note" as const,
  label,
  created_at: null,
  deleted: false,
  degree: 1,
  tag_count: 0,
  primary_source_id: null
})
const graph = {
  nodes: [
    node("note:a", "Alpha"),
    node("note:b", "Beta"),
    node("note:c", "Charlie")
  ],
  edges: [
    {
      id: "z",
      source: "note:a",
      target: "note:c",
      type: "wikilink" as const,
      directed: true,
      weight: null,
      label: null
    },
    {
      id: "a",
      source: "note:b",
      target: "note:a",
      type: "manual" as const,
      directed: true,
      weight: null,
      label: null
    },
    {
      id: "u",
      source: "note:a",
      target: "note:b",
      type: "backlink" as const,
      directed: false,
      weight: null,
      label: null
    }
  ],
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 200, max_edges: 300, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 3,
  all_notes_note_cap: 200,
  all_notes_eligible: true,
  suggestions_authorized: true
}

describe("NotesGraphRelationshipsView", () => {
  it("projects fixed directional groups from the loaded graph with no fetch", () => {
    const groups = buildNotesGraphRelationshipGroups({
      graph,
      selectedNodeId: "note:a",
      provisionalOverlays: [],
      suggestions: []
    })
    expect(
      groups.map((group) => [
        group.id,
        group.rows.map((row) => row.counterpart.label)
      ])
    ).toEqual([
      ["outgoing", ["Charlie"]],
      ["incoming", ["Beta"]]
    ])
    expect(groups[1]?.rows[0]?.edgeIds).toEqual(["a", "u"])
    expect(groups[1]?.rows[0]?.edgeTypes).toEqual(["manual", "backlink"])
  })

  it("uses group-relative set positions across pagination boundaries", async () => {
    const onSelectNode = vi.fn()
    const manyGraph = {
      ...graph,
      nodes: [
        node("note:a", "Alpha"),
        ...Array.from({ length: 101 }, (_, index) =>
          node(`note:${index}`, `Node ${String(index).padStart(3, "0")}`)
        ),
        node("note:incoming-1", "Incoming 1"),
        node("note:incoming-2", "Incoming 2")
      ],
      edges: [
        ...Array.from({ length: 101 }, (_, index) => ({
          id: `edge:${index}`,
          source: "note:a",
          target: `note:${index}`,
          type: "manual" as const,
          directed: true,
          weight: null,
          label: null
        })),
        ...[1, 2].map((index) => ({
          id: `incoming:${index}`,
          source: `note:incoming-${index}`,
          target: "note:a",
          type: "manual" as const,
          directed: true,
          weight: null,
          label: null
        }))
      ]
    }
    const { rerender } = render(
      <NotesGraphRelationshipsView
        graph={manyGraph}
        selectedNodeId="note:a"
        queryIdentity="query-a"
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        isOnline
        onSelectNode={onSelectNode}
      />
    )
    const firstPageButtons = screen.getAllByTestId(
      "notes-graph-relationship-row"
    )
    const firstPageItems = screen.getAllByRole("listitem")
    expect(firstPageButtons).toHaveLength(100)
    expect(firstPageButtons[99]).not.toHaveAttribute("aria-posinset")
    expect(firstPageButtons[99]).not.toHaveAttribute("aria-setsize")
    expect(firstPageItems[99]).toHaveAttribute("aria-posinset", "100")
    expect(firstPageItems[99]).toHaveAttribute("aria-setsize", "101")
    fireEvent.click(screen.getByRole("button", { name: "Next page" }))
    const secondPageItems = screen.getAllByRole("listitem")
    expect(secondPageItems).toHaveLength(3)
    expect(secondPageItems[0]).toHaveAttribute("aria-posinset", "101")
    expect(secondPageItems[0]).toHaveAttribute("aria-setsize", "101")
    expect(secondPageItems[1]).toHaveAttribute("aria-posinset", "1")
    expect(secondPageItems[1]).toHaveAttribute("aria-setsize", "2")
    expect(secondPageItems[2]).toHaveAttribute("aria-posinset", "2")
    expect(secondPageItems[2]).toHaveAttribute("aria-setsize", "2")
    const finalOutgoingRow = screen.getByRole("button", { name: "Node 100" })
    await waitFor(() => expect(finalOutgoingRow).toHaveFocus())
    fireEvent.click(finalOutgoingRow)
    expect(onSelectNode).toHaveBeenCalledWith("note:100")

    fireEvent.click(screen.getByRole("button", { name: "Previous page" }))
    await waitFor(() =>
      expect(
        screen.getAllByTestId("notes-graph-relationship-row")[0]
      ).toHaveFocus()
    )

    fireEvent.click(screen.getByRole("button", { name: "Next page" }))
    rerender(
      <NotesGraphRelationshipsView
        graph={manyGraph}
        selectedNodeId="note:a"
        queryIdentity="query-b"
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        isOnline
        onSelectNode={onSelectNode}
      />
    )
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Node 000" })).toBeVisible()
    )

    fireEvent.click(screen.getByRole("button", { name: "Next page" }))
    rerender(
      <NotesGraphRelationshipsView
        graph={manyGraph}
        selectedNodeId="note:a"
        queryIdentity="query-b"
        visibleEdgeTypes={new Set(["wikilink"])}
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        isOnline
        onSelectNode={onSelectNode}
      />
    )
    expect(screen.queryByRole("button", { name: "Next page" })).toBeNull()
    rerender(
      <NotesGraphRelationshipsView
        graph={manyGraph}
        selectedNodeId="note:a"
        queryIdentity="query-b"
        visibleEdgeTypes={new Set(["manual"])}
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        isOnline
        onSelectNode={onSelectNode}
      />
    )
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Node 000" })).toBeVisible()
    )
  })

  it("uses one shared decision coordinator and restores failure focus", async () => {
    const decide = vi.fn().mockResolvedValue(false)
    const item = {
      id: "s1",
      kind: "related_note" as const,
      target_title: "Suggested target",
      target_note_id: "target",
      match_strength: "possible" as const,
      rationale: "Possible relation",
      evidence: [{ side: "source" as const, text: "Grounded excerpt" }]
    }
    render(
      <NotesGraphRelationshipsView
        graph={graph}
        selectedNodeId="note:a"
        provisionalOverlays={[
          {
            edge: {
              id: "suggestion-edge:s1",
              suggestionId: "s1",
              source: "note:a",
              target: "suggestion-node:s1",
              type: "provisional_suggestion",
              directed: false
            },
            node: {
              id: "suggestion-node:s1",
              suggestionId: "s1",
              type: "provisional_note",
              label: "Suggested target"
            }
          }
        ]}
        suggestions={[item] as never}
        suggestionsAuthorized
        isOnline
        canAccept
        canReject
        onSelectNode={vi.fn()}
        onDecideSuggestion={decide}
      />
    )

    const acceptButton = screen.getByRole("button", {
      name: "Accept Suggested target"
    })
    acceptButton.focus()
    fireEvent.click(acceptButton)

    await waitFor(() => expect(decide).toHaveBeenCalledWith("accept", "s1"))
    await waitFor(() => expect(acceptButton).toHaveFocus())
  })

  it("keeps provisional titles non-navigational while loaded suggestion targets remain selectable", () => {
    const onSelectNode = vi.fn()
    const suggestion = (id: string, title: string, targetNoteId: string) => ({
      id,
      kind: "related_note" as const,
      target_title: title,
      target_note_id: targetNoteId,
      match_strength: "possible" as const,
      rationale: "Possible relation",
      evidence: []
    })
    render(
      <NotesGraphRelationshipsView
        graph={graph}
        selectedNodeId="note:a"
        provisionalOverlays={[
          {
            edge: {
              id: "suggestion-edge:s1",
              suggestionId: "s1",
              source: "note:a",
              target: "suggestion-node:s1",
              type: "provisional_suggestion",
              directed: false
            },
            node: {
              id: "suggestion-node:s1",
              suggestionId: "s1",
              type: "provisional_note",
              label: "Provisional target"
            }
          },
          {
            edge: {
              id: "suggestion-edge:s2",
              suggestionId: "s2",
              source: "note:a",
              target: "note:b",
              type: "provisional_suggestion",
              directed: false
            },
            node: null
          }
        ]}
        suggestions={
          [
            suggestion("s1", "Provisional target", "unloaded"),
            suggestion("s2", "Loaded suggestion target", "b")
          ] as never
        }
        suggestionsAuthorized
        isOnline
        onSelectNode={onSelectNode}
      />
    )

    expect(
      screen.getByRole("heading", { name: "Provisional target", level: 3 })
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Provisional target" })
    ).not.toBeInTheDocument()
    fireEvent.click(screen.getByText("Provisional target"))
    expect(onSelectNode).not.toHaveBeenCalled()

    fireEvent.click(
      screen.getByRole("button", { name: "Loaded suggestion target" })
    )
    expect(onSelectNode).toHaveBeenCalledTimes(1)
    expect(onSelectNode).toHaveBeenCalledWith("note:b")
  })

  it("moves focus to the next review row after a successful decision", async () => {
    const decide = vi.fn().mockResolvedValue(true)
    const item = (id: string, title: string) => ({
      id,
      kind: "related_note" as const,
      target_title: title,
      target_note_id: id,
      match_strength: "possible" as const,
      rationale: "Possible relation",
      evidence: []
    })
    const overlay = (id: string, target: string) => ({
      edge: {
        id: `suggestion-edge:${id}`,
        suggestionId: id,
        source: "note:a",
        target,
        type: "provisional_suggestion" as const,
        directed: false as const
      },
      node: {
        id: target,
        suggestionId: id,
        type: "provisional_note" as const,
        label: target
      }
    })
    render(
      <NotesGraphRelationshipsView
        graph={graph}
        selectedNodeId="note:a"
        provisionalOverlays={[
          overlay("s1", "suggestion-node:s1"),
          overlay("s2", "suggestion-node:s2"),
          overlay("s3", "suggestion-node:s3")
        ]}
        suggestions={
          [
            item("s1", "First target"),
            item("s2", "Second target"),
            item("s3", "Third target")
          ] as never
        }
        suggestionsAuthorized
        isOnline
        canAccept
        canReject
        onSelectNode={vi.fn()}
        onDecideSuggestion={decide}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Accept Second target" })
    )
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Accept Third target" })
      ).toHaveFocus()
    )
  })

  it("shows grounded provisional evidence and decisions only when authorized", () => {
    const decide = vi.fn().mockResolvedValue(true)
    const item = {
      id: "s1",
      kind: "related_note" as const,
      target_title: "Suggested target",
      target_note_id: "target",
      match_strength: "possible" as const,
      rationale: "Possible relation",
      evidence: [{ side: "source" as const, text: "Grounded excerpt" }]
    }
    render(
      <NotesGraphRelationshipsView
        graph={graph}
        selectedNodeId="note:a"
        provisionalOverlays={[
          {
            edge: {
              id: "suggestion-edge:s1",
              suggestionId: "s1",
              source: "note:a",
              target: "suggestion-node:s1",
              type: "provisional_suggestion",
              directed: false
            },
            node: {
              id: "suggestion-node:s1",
              suggestionId: "s1",
              type: "provisional_note",
              label: "Suggested target"
            }
          }
        ]}
        suggestions={[item] as never}
        suggestionsAuthorized
        isOnline
        canAccept
        canReject
        onSelectNode={vi.fn()}
        onDecideSuggestion={decide}
      />
    )
    expect(screen.getByText("Suggested")).toBeInTheDocument()
    expect(screen.getByText("Possible match")).toBeInTheDocument()
    expect(screen.getByText("Grounded excerpt")).toBeInTheDocument()
    fireEvent.click(
      screen.getByRole("button", { name: "Accept Suggested target" })
    )
    fireEvent.click(
      screen.getByRole("button", { name: "Reject Suggested target" })
    )
    expect(decide).toHaveBeenNthCalledWith(1, "accept", "s1")
    expect(decide).toHaveBeenNthCalledWith(2, "reject", "s1")
  })

  it("groups parallel relationship types while preserving semantic evidence and conversion", async () => {
    const convert = vi.fn().mockResolvedValue(true)
    const onSelectEdge = vi.fn()
    const semantic = {
      id: "semantic:a:b",
      source: "note:a",
      target: "note:b",
      type: "semantic" as const,
      directed: false,
      weight: 0.8765,
      label: null,
      evidence: {
        similarity: 0.8765,
        qualitative_band: "high" as const,
        source_note_id: "note:a",
        target_note_id: "note:b",
        source_content_version: 4,
        target_content_version: 7,
        generation_id: "generation-a",
        semantic_index_revision: 9,
        configuration_revision: 3,
        normalization_version: "normalize-v1",
        chunker_version: "chunk-v1",
        provider_label: "Local provider",
        model_label: "Embedding model",
        model_revision: "model-r2",
        excerpt_pairs: [
          {
            source: {
              field: "content" as const,
              start_code_point: 0,
              end_code_point: 12,
              text: "Source match"
            },
            target: {
              field: "content" as const,
              start_code_point: 0,
              end_code_point: 12,
              text: "Target match"
            }
          }
        ]
      }
    }
    const parallel = {
      ...graph,
      edges: [
        {
          id: "manual:a:b",
          source: "note:a",
          target: "note:b",
          type: "manual" as const,
          directed: false,
          weight: 1,
          label: null
        },
        {
          id: "wikilink:a:b",
          source: "note:a",
          target: "note:b",
          type: "wikilink" as const,
          directed: false,
          weight: 1,
          label: null
        },
        semantic
      ],
      manual_link_authorized: true
    }
    const groups = buildNotesGraphRelationshipGroups({
      graph: parallel,
      selectedNodeId: "note:a",
      provisionalOverlays: [],
      suggestions: []
    })
    const beta = groups
      .flatMap((group) => group.rows)
      .find((row) => row.counterpart.id === "note:b")
    expect(beta?.edgeIds).toEqual([
      "manual:a:b",
      "wikilink:a:b",
      "semantic:a:b"
    ])
    expect(beta?.edgeTypes).toEqual(["manual", "wikilink", "semantic"])

    const { unmount } = render(
      <NotesGraphRelationshipsView
        graph={parallel}
        selectedNodeId="note:a"
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        manualLinkAuthorized
        isOnline
        onSelectNode={vi.fn()}
        onSelectEdge={onSelectEdge}
        onCreateManualLink={convert}
      />
    )

    expect(screen.getAllByTestId("notes-graph-relationship-row")).toHaveLength(
      1
    )
    expect(screen.getByText("Source match")).not.toBeVisible()
    const disclosure = screen.getByTestId(
      "notes-graph-semantic-evidence-toggle"
    )
    expect(disclosure).toHaveTextContent("High similarity")
    expect(disclosure).toHaveTextContent("Passage similarity: 0.8765")
    fireEvent.click(disclosure)
    expect(screen.getByText("Local provider / Embedding model")).toBeVisible()
    expect(screen.getByText("Model revision")).toBeVisible()
    expect(screen.getByText("model-r2")).toBeVisible()
    expect(screen.getByText("Normalization version")).toBeVisible()
    expect(screen.getByText("normalize-v1")).toBeVisible()
    expect(screen.getByText("Chunker version")).toBeVisible()
    expect(screen.getByText("chunk-v1")).toBeVisible()
    expect(screen.getByText("Source match")).toBeVisible()
    expect(screen.getByText("Target match")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Similar content" }))
    expect(onSelectEdge).toHaveBeenCalledWith("semantic:a:b")
    expect(
      screen.queryByRole("button", { name: "Create manual link" })
    ).not.toBeInTheDocument()

    unmount()
    const semanticOnly = render(
      <NotesGraphRelationshipsView
        graph={{ ...parallel, edges: [semantic] }}
        selectedNodeId="note:a"
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        manualLinkAuthorized
        isOnline
        onSelectNode={vi.fn()}
        onSelectEdge={vi.fn()}
        onCreateManualLink={convert}
      />
    )
    fireEvent.click(screen.getByTestId("notes-graph-semantic-evidence-toggle"))
    const convertButton = screen.getByRole("button", {
      name: "Create manual link"
    })
    convertButton.focus()
    fireEvent.click(convertButton)
    expect(convert).toHaveBeenCalledWith(semantic, convertButton)

    semanticOnly.unmount()
    const detailsTab = document.createElement("button")
    detailsTab.id = "notes-graph-details-tab"
    document.body.append(detailsTab)
    render(
      <NotesSemanticRelationshipDetails
        edge={semantic}
        manualLinkAuthorized
        isOnline
        hasManualRelationship={false}
        onCreateManualLink={convert}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Create manual link" }))
    await waitFor(() => expect(detailsTab).toHaveFocus())
    detailsTab.remove()
  })

  it("hides capped conversion for readers but permits writers with fresh graph authority", () => {
    const convert = vi.fn().mockResolvedValue(true)
    const omittedGraph = {
      ...graph,
      manual_link_authorized: true,
      edges: [
        {
          id: "semantic:omitted",
          source: "note:a",
          target: "note:b",
          type: "semantic" as const,
          directed: false,
          weight: 0.8,
          label: null,
          evidence_omitted: "response_byte_cap" as const
        }
      ]
    }
    const { unmount } = render(
      <NotesGraphRelationshipsView
        graph={{
          ...omittedGraph,
          manual_link_authorized: false
        }}
        selectedNodeId="note:a"
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        manualLinkAuthorized={false}
        isOnline
        onSelectNode={vi.fn()}
        onSelectEdge={vi.fn()}
      />
    )
    expect(
      screen.getByText("Evidence omitted by response limit")
    ).not.toBeVisible()
    fireEvent.click(screen.getByTestId("notes-graph-semantic-evidence-toggle"))
    expect(screen.getByText("Evidence omitted by response limit")).toBeVisible()
    expect(
      screen.queryByRole("button", { name: "Create manual link" })
    ).not.toBeInTheDocument()

    unmount()
    const writer = render(
      <NotesGraphRelationshipsView
        graph={omittedGraph}
        selectedNodeId="note:a"
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        manualLinkAuthorized
        manualLinkPendingEdgeIds={new Set(["semantic:omitted"])}
        isOnline
        onSelectNode={vi.fn()}
        onSelectEdge={vi.fn()}
        onCreateManualLink={convert}
      />
    )
    fireEvent.click(screen.getByTestId("notes-graph-semantic-evidence-toggle"))
    expect(
      screen.getByRole("button", { name: "Create manual link" })
    ).toBeDisabled()

    writer.rerender(
      <NotesGraphRelationshipsView
        graph={omittedGraph}
        selectedNodeId="note:a"
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        manualLinkAuthorized
        manualLinkPendingEdgeIds={new Set()}
        isOnline
        onSelectNode={vi.fn()}
        onSelectEdge={vi.fn()}
        onCreateManualLink={convert}
      />
    )
    fireEvent.click(screen.getByRole("button", { name: "Create manual link" }))
    expect(convert).toHaveBeenCalledWith(
      omittedGraph.edges[0],
      expect.any(HTMLButtonElement)
    )
  })
})
