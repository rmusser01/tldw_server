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
          id: "semantic:a:b",
          source: "note:a",
          target: "note:b",
          type: "semantic" as const,
          directed: false,
          weight: 0.81234,
          label: null,
          evidence: {
            similarity: 0.81234,
            qualitative_band: "high" as const,
            source_note_id: "note:a",
            target_note_id: "note:b",
            source_content_version: 2,
            target_content_version: 5,
            generation_id: "generation-with-a-long-stable-identifier",
            semantic_index_revision: 8,
            configuration_revision: 3,
            normalization_version: "normalize-v1",
            chunker_version: "chunk-v1",
            provider_label: "Provider with a long localized display label",
            model_label: "Embedding model with a long revision label",
            model_revision: "model-r1",
            excerpt_pairs: [
              {
                source: {
                  field: "content" as const,
                  start_code_point: 0,
                  end_code_point: 12,
                  text: "A long source excerpt remains readable without overlapping adjacent content."
                },
                target: {
                  field: "content" as const,
                  start_code_point: 0,
                  end_code_point: 12,
                  text: "A long target excerpt stacks in narrow inspectors and reflows when space permits."
                }
              }
            ]
          }
        }
      ],
      truncated: false,
      truncated_by: [],
      has_more: false,
      cursor: null,
      limits: { max_nodes: 20, max_edges: 20, max_degree: 20 },
      radius_cap_applied: false,
      active_note_count: 1,
      all_notes_note_cap: 20,
      all_notes_eligible: true,
      suggestions_authorized: false,
      manual_link_authorized: true
    }
    const { container } = render(
      <NotesGraphInspector
        graph={graph}
        selectedNodeId="note:a"
        selectedEdgeId="semantic:a:b"
        suggestionsAuthorized={false}
        manualLinkAuthorized
        isOnline
        controller={{ suggestions: [] } as never}
        onSelectNode={vi.fn()}
        onSelectEdge={vi.fn()}
        onCreateManualLink={vi.fn().mockResolvedValue(true)}
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
      suggestions_authorized: false,
      manual_link_authorized: false
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
