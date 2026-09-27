import { fireEvent, render, screen } from "@testing-library/react"
import React from "react"
import { describe, expect, it, vi } from "vitest"

import NotesGraphToolbar from "../NotesGraphToolbar"

const translations: Record<string, string> = {
  "option:notesSearch.graphViewMode": "VIEW_MODE_X",
  "option:notesSearch.graphCanvas": "CANVAS_X",
  "option:notesSearch.graphRelationships": "RELATIONSHIPS_X",
  "option:notesSearch.graphSearchLoaded": "SEARCH_X",
  "option:notesSearch.graphRadiusLabel": "RADIUS_X",
  "option:notesSearch.graphRadiusAria": "RADIUS_ARIA_X",
  "option:notesSearch.graphMaxNodesLabel": "MAX_NODES_X",
  "option:notesSearch.graphMaxNodesAria": "MAX_NODES_ARIA_X",
  "option:notesSearch.graphLayoutLabel": "LAYOUT_X",
  "option:notesSearch.graphLayoutAria": "LAYOUT_ARIA_X",
  "option:notesSearch.graphLayoutOption.dagre": "DAGRE_X",
  "option:notesSearch.graphLayoutOption.circle": "CIRCLE_X",
  "option:notesSearch.graphLayoutOption.grid": "GRID_X",
  "option:notesSearch.graphLayoutOption.concentric": "CONCENTRIC_X",
  "option:notesSearch.graphScope": "SCOPE_X",
  "option:notesSearch.graphScopeFocused": "FOCUSED_X",
  "option:notesSearch.graphScopeAll": "ALL_NOTES_X",
  "option:notesSearch.graphFocusCurrent": "FOCUS_CURRENT_X",
  "option:notesSearch.graphExpand": "EXPAND_X",
  "option:notesSearch.graphRefresh": "REFRESH_X",
  "option:notesSearch.graphZoomIn": "ZOOM_IN_X",
  "option:notesSearch.graphZoomOut": "ZOOM_OUT_X",
  "option:notesSearch.graphFit": "FIT_X",
  "option:notesSearch.graphEdgeVisibility": "EDGE_VISIBILITY_X",
  "option:notesSearch.graphEdgeVisibilityFilters": "EDGE_FILTERS_X",
  "option:notesSearch.graphEdgeType.manual": "MANUAL_X",
  "option:notesSearch.graphEdgeType.wikilink": "WIKILINK_X",
  "option:notesSearch.graphEdgeType.backlink": "BACKLINK_X",
  "option:notesSearch.graphEdgeType.tag_membership": "TAG_MEMBERSHIP_X",
  "option:notesSearch.graphEdgeType.source_membership": "SOURCE_MEMBERSHIP_X",
  "option:notesSearch.graphSuggestions": "SUGGESTIONS_X"
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => translations[key] ?? key
  })
}))

describe("NotesGraphToolbar localization", () => {
  it("routes visible and assistive toolbar copy through translation keys", () => {
    render(
      <NotesGraphToolbar
        viewMode="canvas"
        suggestionsAuthorized
        search=""
        searchResults={[]}
        radius={1}
        maxNodes={100}
        maxNodeCap={500}
        layout="dagre"
        scope="focused"
        allNotes={{
          activeNoteCount: 3,
          effectiveNoteCap: 500,
          eligible: true
        }}
        visibleEdgeTypes={
          new Set([
            "manual",
            "wikilink",
            "backlink",
            "tag_membership",
            "source_membership"
          ])
        }
        showProvisional
        canExpand
        isRefreshing={false}
        onSearchChange={vi.fn()}
        onViewModeChange={vi.fn()}
        onSelectSearchResult={vi.fn()}
        onRadiusChange={vi.fn()}
        onMaxNodesChange={vi.fn()}
        onLayoutChange={vi.fn()}
        onShowFocused={vi.fn()}
        onShowAllNotes={vi.fn()}
        onToggleEdgeType={vi.fn()}
        onToggleProvisional={vi.fn()}
        onFocusCurrent={vi.fn()}
        onExpand={vi.fn()}
        onRefresh={vi.fn()}
        onZoomIn={vi.fn()}
        onZoomOut={vi.fn()}
        onFit={vi.fn()}
      />
    )

    expect(screen.getByRole("searchbox", { name: "SEARCH_X" })).toBeVisible()
    expect(
      screen.getByRole("combobox", { name: "RADIUS_ARIA_X" })
    ).toBeVisible()
    expect(
      screen.getByRole("spinbutton", { name: "MAX_NODES_ARIA_X" })
    ).toBeVisible()
    expect(screen.getByRole("combobox", { name: "LAYOUT_ARIA_X" })).toHaveValue(
      "dagre"
    )
    expect(screen.getByRole("option", { name: "DAGRE_X" })).toBeVisible()
    expect(screen.getByRole("group", { name: "SCOPE_X" })).toBeVisible()
    expect(screen.getByRole("button", { name: "FOCUSED_X" })).toBeVisible()
    expect(screen.getByRole("button", { name: "ALL_NOTES_X" })).toBeVisible()
    for (const name of [
      "FOCUS_CURRENT_X",
      "EXPAND_X",
      "REFRESH_X",
      "ZOOM_IN_X",
      "ZOOM_OUT_X",
      "FIT_X"
    ]) {
      expect(screen.getByRole("button", { name })).toBeVisible()
    }

    fireEvent.click(screen.getByRole("button", { name: "EDGE_VISIBILITY_X" }))

    expect(screen.getByRole("group", { name: "EDGE_FILTERS_X" })).toBeVisible()
    for (const name of [
      "MANUAL_X",
      "WIKILINK_X",
      "BACKLINK_X",
      "TAG_MEMBERSHIP_X",
      "SOURCE_MEMBERSHIP_X",
      "SUGGESTIONS_X"
    ]) {
      expect(screen.getByRole("checkbox", { name })).toBeVisible()
    }
  })
})
