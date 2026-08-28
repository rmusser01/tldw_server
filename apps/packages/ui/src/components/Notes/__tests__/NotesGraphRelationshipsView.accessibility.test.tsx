// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import NotesGraphRelationshipsView, {
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
      ["incoming", ["Beta"]],
      ["connected", ["Beta"]]
    ])
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
    render(
      <NotesGraphRelationshipsView
        graph={manyGraph}
        selectedNodeId="note:a"
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
      expect(screen.getByRole("button", { name: "Third target" })).toHaveFocus()
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
})
