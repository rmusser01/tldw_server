// @vitest-environment jsdom
import { render } from "@testing-library/react"
import axe from "axe-core"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import NotesGraphInspector from "../NotesGraphInspector"
import NotesGraphRelationshipsView from "../NotesGraphRelationshipsView"

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn(async () => false)
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key })
}))

describe("Notes graph Task 11 accessibility", () => {
  it("has no serious inspector accessibility violations", async () => {
    const graph = {
      nodes: [
        {
          id: "note:a",
          type: "note" as const,
          label: "Alpha",
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
    const { container } = render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:a"
        suggestionsAuthorized={false}
        isOnline
        controller={{ suggestions: [] } as never}
        onSelectNode={vi.fn()}
        onAnnounce={vi.fn()}
        onDecideSuggestion={vi.fn().mockResolvedValue(true)}
      />
    )
    const result = await axe.run(container, {
      resultTypes: ["violations"],
      rules: { "color-contrast": { enabled: false } }
    })
    expect(
      result.violations.filter((violation) =>
        ["serious", "critical"].includes(violation.impact ?? "")
      )
    ).toEqual([])
  })

  it("has no serious grouped-relationship accessibility violations", async () => {
    const graph = {
      nodes: [
        {
          id: "note:a",
          type: "note" as const,
          label: "Alpha",
          created_at: null,
          deleted: false,
          degree: 1,
          tag_count: 0,
          primary_source_id: null
        },
        {
          id: "note:b",
          type: "note" as const,
          label: "Beta",
          created_at: null,
          deleted: false,
          degree: 1,
          tag_count: 0,
          primary_source_id: null
        }
      ],
      edges: [
        {
          id: "edge:a-b",
          source: "note:a",
          target: "note:b",
          type: "manual" as const,
          directed: true,
          weight: null,
          label: null
        }
      ],
      truncated: false,
      truncated_by: [],
      has_more: false,
      cursor: null,
      limits: { max_nodes: 20, max_edges: 20, max_degree: 20 },
      radius_cap_applied: false,
      active_note_count: 2,
      all_notes_note_cap: 20,
      all_notes_eligible: true,
      suggestions_authorized: false
    }
    const { container } = render(
      <NotesGraphRelationshipsView
        graph={graph}
        selectedNodeId="note:a"
        provisionalOverlays={[]}
        suggestions={[]}
        suggestionsAuthorized={false}
        isOnline
        onSelectNode={vi.fn()}
      />
    )
    const result = await axe.run(container, {
      resultTypes: ["violations"],
      rules: { "color-contrast": { enabled: false } }
    })
    expect(
      result.violations.filter((violation) =>
        ["serious", "critical"].includes(violation.impact ?? "")
      )
    ).toEqual([])
  })
})
