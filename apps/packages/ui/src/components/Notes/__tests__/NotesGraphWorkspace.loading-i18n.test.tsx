import commonEn from "../../../assets/locale/en/common.json"
import optionEn from "../../../assets/locale/en/option.json"
import ICUWithInterpolation from "../../../i18n/icu-format"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import React from "react"
import { I18nextProvider } from "react-i18next"
import { describe, expect, it, vi } from "vitest"

import NotesGraphWorkspace from "../NotesGraphWorkspace"

const { mockUseNotesGraphSuggestions, mockUseNotesGraphWorkspace } = vi.hoisted(
  () => ({
    mockUseNotesGraphSuggestions: vi.fn(),
    mockUseNotesGraphWorkspace: vi.fn()
  })
)

vi.mock("../hooks/useNotesGraphWorkspace", () => ({
  useNotesGraphWorkspace: mockUseNotesGraphWorkspace
}))

vi.mock("../hooks/useNotesGraphSuggestions", () => ({
  useNotesGraphSuggestions: mockUseNotesGraphSuggestions
}))

const loadingWorkspace = () => ({
  graph: null,
  graphQuery: { isFetching: true },
  focusNoteId: "source-note",
  scope: "focused" as const,
  layout: "dagre" as const,
  setLayout: vi.fn(),
  search: "",
  setSearch: vi.fn(),
  searchResults: [],
  visibleEdgeTypes: new Set([
    "manual",
    "wikilink",
    "backlink",
    "tag_membership",
    "source_membership"
  ]),
  toggleEdgeType: vi.fn(),
  allNotes: { activeNoteCount: 1, effectiveNoteCap: 500, eligible: true },
  canExpand: false,
  expand: vi.fn(),
  focus: vi.fn(),
  showAllNotes: vi.fn(),
  refresh: vi.fn(),
  isOffline: false,
  isLoading: true,
  error: null
})

const emptySuggestions = () => ({
  provisionalBySuggestionId: {},
  suggestions: [],
  capabilities: null,
  activeRun: null,
  lastTerminalRun: null,
  mutations: null
})

describe("NotesGraphWorkspace loading translation", () => {
  it("renders the localized loading title through the real ICU path", async () => {
    const i18n = createInstance()
    await i18n.use(ICUWithInterpolation).init({
      lng: "en",
      fallbackLng: false,
      ns: ["option", "common"],
      defaultNS: "option",
      resources: {
        en: {
          common: commonEn,
          option: optionEn
        }
      },
      interpolation: { escapeValue: false }
    })
    mockUseNotesGraphWorkspace.mockReturnValue(loadingWorkspace())
    mockUseNotesGraphSuggestions.mockReturnValue(emptySuggestions())
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })

    render(
      <I18nextProvider i18n={i18n}>
        <QueryClientProvider client={queryClient}>
          <NotesGraphWorkspace
            authorityScope="opaque-authority"
            isOnline
            initialFocusNoteId="source-note"
            selectedNoteId="source-note"
            hasActiveNotes
            onSelectNote={vi.fn()}
            onCreateNote={vi.fn()}
          />
        </QueryClientProvider>
      </I18nextProvider>
    )

    expect(screen.getByRole("status")).toHaveTextContent(commonEn.loading.title)
  })
})
