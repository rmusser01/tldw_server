// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen } from "@testing-library/react"
import React from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import NotesGraphWorkspace from "../NotesGraphWorkspace"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string } | string) =>
      typeof options === "string" ? options : options?.defaultValue ?? key
  })
}))
vi.mock("../hooks/useNotesGraphWorkspace", () => ({
  useNotesGraphWorkspace: () => ({
    graph: null,
    focusNoteId: null,
    focus: vi.fn(),
    allNotes: { eligible: false, effectiveNoteCap: 0, activeNoteCount: 0 }
  })
}))
vi.mock("../hooks/useNotesGraphSuggestions", () => ({
  useNotesGraphSuggestions: () => ({
    suggestions: [],
    provisionalBySuggestionId: {},
    mutations: {}
  })
}))

describe("NotesGraphWorkspace focus on open (NO-01)", () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it("moves focus to the graph region without scrolling the page", () => {
    const focusSpy = vi.spyOn(HTMLElement.prototype, "focus")
    render(
      <QueryClientProvider client={new QueryClient()}>
        <NotesGraphWorkspace
          authorityScope="scope"
          isOnline
          initialFocusNoteId={null}
          selectedNoteId={null}
          hasActiveNotes={false}
          onSelectNote={vi.fn()}
          onCreateNote={vi.fn()}
        />
      </QueryClientProvider>
    )
    const region = screen.getByTestId("notes-graph-workspace")
    const regionFocusCalls = focusSpy.mock.contexts
      .map((context, index) => ({ context, args: focusSpy.mock.calls[index] }))
      .filter(({ context }) => context === region)
    expect(regionFocusCalls.length).toBeGreaterThan(0)
    // A plain focus() scrolled the window and hid the app header.
    for (const { args } of regionFocusCalls) {
      expect(args[0]).toEqual({ preventScroll: true })
    }
  })
})
