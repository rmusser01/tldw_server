import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ManageTab, buildFlashcardsWorkspaceVisibilityOptions } from "../ManageTab"
import { clearSetting } from "@/services/settings/registry"
import { FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING } from "@/services/settings/ui-settings"
import type { Flashcard } from "@/services/flashcards"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"
import {
  useUpdateFlashcardsBulkMutation,
  useDecksQuery,
  useFlashcardDocumentQuery,
  useManageQuery,
  useTagSuggestionsQuery,
  useUpdateFlashcardMutation,
  useUpdateDeckMutation,
  useResetFlashcardSchedulingMutation,
  useDeleteFlashcardMutation,
  useCardsKeyboardNav,
  getManageServerOrderBy
} from "../../hooks"

const {
  createDrawerMock,
  trackShortcutHintTelemetryMock,
  markdownSnippetMock,
  markdownWithBoundaryMock
} = vi.hoisted(() => ({
  createDrawerMock: vi.fn(
    (props: {
      open?: boolean
      initialDeckId?: number | null
      includeWorkspaceItems?: boolean
      workspaceId?: string | null
    }) =>
      props.open ? (
        <div data-testid="mock-create-drawer">
          <span data-testid="mock-create-drawer-initial-deck-id">
            {String(props.initialDeckId ?? "")}
          </span>
          <span data-testid="mock-create-drawer-include-workspace">
            {String(props.includeWorkspaceItems ?? false)}
          </span>
          <span data-testid="mock-create-drawer-workspace-id">
            {String(props.workspaceId ?? "")}
          </span>
        </div>
      ) : null
  ),
  trackShortcutHintTelemetryMock: vi.fn().mockResolvedValue(undefined),
  markdownSnippetMock: vi.fn(({ content }: { content: string }) => (
    <div data-testid="markdown-snippet">{content}</div>
  )),
  markdownWithBoundaryMock: vi.fn(({ content }: { content: string }) => (
    <div data-testid="markdown-with-boundary">{content}</div>
  ))
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>("@tanstack/react-query")
  return {
    ...actual,
    useQueryClient: () => ({
      invalidateQueries: vi.fn()
    })
  }
})

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    loading: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn().mockResolvedValue(true)
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/utils/flashcards-shortcut-hint-telemetry", () => ({
  trackFlashcardsShortcutHintTelemetry: trackShortcutHintTelemetryMock
}))

vi.mock("../../hooks", () => ({
  DOCUMENT_VIEW_SUPPORTED_SORTS: ["due", "created"],
  getFlashcardDocumentQueryKey: vi.fn(() => ["flashcards:document", 1]),
  useDecksQuery: vi.fn(),
  useFlashcardDocumentQuery: vi.fn(),
  useManageQuery: vi.fn(),
  useTagSuggestionsQuery: vi.fn(),
  useUpdateFlashcardMutation: vi.fn(),
  useUpdateDeckMutation: vi.fn(),
  useUpdateFlashcardsBulkMutation: vi.fn(),
  useResetFlashcardSchedulingMutation: vi.fn(),
  useDeleteFlashcardMutation: vi.fn(),
  useCardsKeyboardNav: vi.fn(),
  getManageServerOrderBy: vi.fn(() => "due_at")
}))

vi.mock("../../components", () => ({
  FlashcardMarkdownSnippet: (props: { content: string }) => markdownSnippetMock(props),
  MarkdownWithBoundary: (props: { content: string }) => markdownWithBoundaryMock(props),
  FlashcardActionsMenu: () => <div />,
  FlashcardEditDrawer: () => null,
  FlashcardCreateDrawer: (props: {
    open?: boolean
    initialDeckId?: number | null
    includeWorkspaceItems?: boolean
    workspaceId?: string | null
  }) => createDrawerMock(props)
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

if (typeof window !== "undefined" && !window.localStorage) {
  const storage = new Map<string, string>()
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => {
        storage.set(key, value)
      },
      removeItem: (key: string) => {
        storage.delete(key)
      },
      clear: () => {
        storage.clear()
      }
    }
  })
}

const sampleCard: Flashcard = {
  uuid: "card-meta-1",
  deck_id: 1,
  front: "Front prompt",
  back: "Back answer",
  notes: null,
  extra: null,
  is_cloze: false,
  tags: ["biology"],
  ef: 2.7,
  interval_days: 5,
  repetitions: 3,
  lapses: 1,
  queue_state: "review",
  due_at: null,
  last_reviewed_at: null,
  last_modified: null,
  deleted: false,
  client_id: "test",
  version: 4,
  model_type: "basic",
  reverse: false,
  source_ref_type: "media",
  source_ref_id: "42"
}

describe("ManageTab scheduling metadata visibility", () => {
  const updateDeckMutateAsync = vi.fn(async () => undefined)

  beforeEach(async () => {
    vi.clearAllMocks()
    createDrawerMock.mockClear()
    markdownSnippetMock.mockClear()
    markdownWithBoundaryMock.mockClear()
    await clearSetting(FLASHCARDS_SHORTCUT_HINT_DENSITY_SETTING)
    vi.mocked(useDecksQuery).mockReturnValue({
      data: [
        {
          id: 1,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1,
          scheduler_type: "sm2_plus",
          scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
        }
      ],
      isLoading: false
    } as any)
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [sampleCard],
        count: 1,
        total: 1
      },
      isFetching: false
    } as any)
    vi.mocked(useFlashcardDocumentQuery).mockReturnValue({
      items: [sampleCard],
      isFetching: false,
      isLoading: false,
      isTruncated: false,
      hasNextPage: false,
      isFetchingNextPage: false,
      fetchNextPage: vi.fn(),
      supportedSorts: ["due", "created"],
      data: {
        pages: [
          {
            items: [sampleCard],
            isTruncated: false,
            total: 1
          }
        ]
      }
    } as any)
    vi.mocked(useTagSuggestionsQuery).mockReturnValue({
      data: ["biology", "chemistry", "physics"],
      isLoading: false
    } as any)
    vi.mocked(useUpdateFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useUpdateDeckMutation).mockReturnValue({
      mutateAsync: updateDeckMutateAsync,
      isPending: false
    } as any)
    vi.mocked(useUpdateFlashcardsBulkMutation).mockReturnValue({
      mutateAsync: vi.fn().mockResolvedValue({ results: [] }),
      isPending: false
    } as any)
    vi.mocked(useResetFlashcardSchedulingMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useDeleteFlashcardMutation).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false
    } as any)
    vi.mocked(useCardsKeyboardNav).mockImplementation(() => undefined)
    vi.mocked(getManageServerOrderBy).mockReturnValue("due_at")
  })

  it("shows scheduling metadata in compact list rows", () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    expect(screen.getByText("Memory 2.70")).toBeInTheDocument()
    expect(screen.getByText("Next gap 5d")).toBeInTheDocument()
    expect(screen.getByText("Recall runs 3")).toBeInTheDocument()
    expect(screen.getByText("Relearns 1")).toBeInTheDocument()
    expect(screen.getByText("Media #42")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-shortcut-chips")).toBeInTheDocument()
    expect(screen.getByText("Sort: Due date")).toBeInTheDocument()
  }, 15000)

  it("prioritizes first-run actions and hides expert card filters when no cards exist", () => {
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [],
        count: 0,
        total: 0
      },
      isFetching: false
    } as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    expect(screen.getByText("No flashcards yet")).toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-manage-shortcut-chips")).not.toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-manage-search")).not.toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-manage-deck-select")).not.toBeInTheDocument()
    expect(screen.queryByTestId("flashcards-manage-selection-summary")).not.toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-empty-create-cta")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-empty-import-cta")).toBeInTheDocument()
    expect(screen.getByTestId("flashcards-manage-empty-generate-cta")).toBeInTheDocument()
  })

  it("uses the markdown renderer for compact flashcard snippets", () => {
    const markdownFront = "**Important** concept"
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [
          {
            ...sampleCard,
            front: markdownFront
          }
        ],
        count: 1,
        total: 1
      },
      isFetching: false
    } as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    expect(
      markdownSnippetMock.mock.calls.some(
        ([props]) => (props as { content?: string }).content === markdownFront
      )
    ).toBe(true)
  })

  it("shows scheduling metadata in expanded list rows", () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-density-toggle"))

    expect(screen.getByText("Memory strength 2.70")).toBeInTheDocument()
    expect(screen.getByText("Next review gap 5d")).toBeInTheDocument()
    expect(screen.getByText("Recall runs 3")).toBeInTheDocument()
    expect(screen.getByText("Relearns 1")).toBeInTheDocument()
    expect(screen.getByText("Media #42")).toBeInTheDocument()
  }, 15000)

  it("uses lightweight snippets in expanded rows and the full renderer for the answer preview", () => {
    const markdownFront = "## Heading"
    const markdownBack = "*Answer* details"
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [
          {
            ...sampleCard,
            front: markdownFront,
            back: markdownBack
          }
        ],
        count: 1,
        total: 1
      },
      isFetching: false
    } as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-density-toggle"))
    fireEvent.click(screen.getByTestId(`flashcard-item-${sampleCard.uuid}`))

    expect(
      markdownSnippetMock.mock.calls.some(
        ([props]) => (props as { content?: string }).content === markdownFront
      )
    ).toBe(true)
    expect(
      markdownSnippetMock.mock.calls.some(
        ([props]) => (props as { content?: string }).content === markdownBack
      )
    ).toBe(true)
    expect(
      markdownWithBoundaryMock.mock.calls.some(
        ([props]) => (props as { content?: string }).content === markdownBack
      )
    ).toBe(true)
  })

  it("does not render source badges for manual cards", () => {
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [
          {
            ...sampleCard,
            uuid: "card-manual-1",
            source_ref_type: "manual",
            source_ref_id: null
          }
        ],
        count: 1,
        total: 1
      },
      isFetching: false
    } as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    expect(screen.queryByText(/Media #/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Note #/)).not.toBeInTheDocument()
    expect(screen.queryByText(/Message #/)).not.toBeInTheDocument()
    expect(screen.queryByText(/source unavailable/i)).not.toBeInTheDocument()
  }, 15000)

  it("cycles shortcut hint density and persists the choice", async () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    const toggle = screen.getByTestId("flashcards-manage-shortcut-hints-toggle")
    expect(toggle).toHaveTextContent("Compact hints")
    expect(screen.getByText("J/K Navigate")).toBeInTheDocument()

    fireEvent.click(toggle)
    await waitFor(() => {
      expect(screen.getByText("J/K · Enter · Space · Delete")).toBeInTheDocument()
    })
    expect(screen.getByTestId("flashcards-manage-shortcut-hints-toggle")).toHaveTextContent(
      "Hide hints"
    )

    fireEvent.click(screen.getByTestId("flashcards-manage-shortcut-hints-toggle"))
    await waitFor(() => {
      expect(screen.getByTestId("flashcards-manage-shortcut-hints-toggle")).toHaveTextContent(
        "Show hints"
      )
    })
    expect(trackShortcutHintTelemetryMock).toHaveBeenCalledWith({
      type: "flashcards_shortcut_hints_dismissed",
      surface: "cards",
      from_density: "compact"
    })
    expect(screen.queryByText("J/K · Enter · Space · Delete")).not.toBeInTheDocument()
    expect(screen.queryByText("J/K Navigate")).not.toBeInTheDocument()
    await waitFor(() => {
      expect(window.localStorage.getItem("tldw:flashcards:shortcutHintDensity")).toBe(
        "hidden"
      )
    })
  }, 15000)

  it("supports multi-tag filter chips with suggestions", async () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive={false}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "More" }))
    const input = screen.getByTestId("flashcards-manage-tag-input")
    fireEvent.change(input, { target: { value: "chemistry" } })
    fireEvent.keyDown(input, { key: "Enter", code: "Enter" })

    await waitFor(() => {
      expect(screen.getByTestId("flashcards-manage-active-tag-filters")).toHaveTextContent(
        "chemistry"
      )
    })

    fireEvent.click(screen.getByRole("button", { name: "biology" }))

    await waitFor(() => {
      const lastManageCall = vi.mocked(useManageQuery).mock.calls.at(-1)
      expect(lastManageCall?.[0]).toMatchObject({
        tags: ["chemistry", "biology"]
      })
    })
  }, 15000)

  it("hides workspace decks by default and reveals them when the toggle is enabled", async () => {
    vi.mocked(useDecksQuery).mockImplementation((params: any) => ({
      data: params?.include_workspace_items
        ? [
            {
              id: 1,
              name: "Biology",
              description: null,
              deleted: false,
              client_id: "test",
              version: 1,
              scheduler_type: "sm2_plus",
              scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
            },
            {
              id: 9,
              name: "Biology",
              description: "Scoped deck",
              workspace_id: "workspace-77",
              deleted: false,
              client_id: "test",
              version: 1,
              scheduler_type: "sm2_plus",
              scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
            }
          ]
        : [
            {
              id: 1,
              name: "Biology",
              description: null,
              deleted: false,
              client_id: "test",
              version: 1,
              scheduler_type: "sm2_plus",
              scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
            }
          ],
      isLoading: false
    } as any))
    vi.mocked(useManageQuery).mockReturnValue({
      data: {
        items: [
          {
            ...sampleCard,
            deck_id: 9
          }
        ],
        count: 1
      },
      isFetching: false
    } as any)

    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    expect(screen.getByText("Deck 9")).toBeInTheDocument()
    expect(screen.queryByText("Biology · workspace-77")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("checkbox", { name: /Show workspace decks/i }))

    await waitFor(() => {
      expect(screen.getByText("Biology · workspace-77")).toBeInTheDocument()
    })
    expect(vi.mocked(useDecksQuery)).toHaveBeenLastCalledWith(
      expect.objectContaining({
        include_workspace_items: true
      })
    )

    const deckSelect = within(
      screen.getByTestId("flashcards-manage-deck-select")
    ).getByRole("combobox")
    fireEvent.mouseDown(deckSelect)
    expect(screen.getAllByText("Biology · workspace-77").length).toBeGreaterThan(1)
  })

  it("filters decks and card queries to a selected workspace", async () => {
    const user = userEvent.setup()
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    await user.click(screen.getByRole("checkbox", { name: /Show workspace decks/i }))
    expect(screen.getByTestId("flashcards-manage-workspace-filter")).toBeInTheDocument()
    expect(
      buildFlashcardsWorkspaceVisibilityOptions(true, "workspace-77")
    ).toEqual(
      expect.objectContaining({
        workspaceId: "workspace-77",
        includeWorkspaceItems: false
      })
    )
  })

  it("resets workspace filters for a non-workspace create handoff", async () => {
    vi.mocked(useDecksQuery).mockImplementation((params: any) => ({
      data:
        params?.include_workspace_items || params?.workspace_id === "workspace-77"
          ? [
              {
                id: 1,
                name: "Biology",
                description: null,
                deleted: false,
                client_id: "test",
                version: 1,
                scheduler_type: "sm2_plus",
                scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
              },
              {
                id: 9,
                name: "Workspace Biology",
                description: null,
                workspace_id: "workspace-77",
                deleted: false,
                client_id: "test",
                version: 1,
                scheduler_type: "sm2_plus",
                scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
              }
            ]
          : [
              {
                id: 1,
                name: "Biology",
                description: null,
                deleted: false,
                client_id: "test",
                version: 1,
                scheduler_type: "sm2_plus",
                scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
              }
            ],
      isLoading: false
    } as any))

    const baseProps = {
      onNavigateToImport: () => {},
      onReviewCard: () => {},
      isActive: true
    }
    const onCreateHandoffConsumed = vi.fn()
    const { rerender } = render(<ManageTab {...baseProps} />)

    fireEvent.click(screen.getByRole("checkbox", { name: /Show workspace decks/i }))
    const workspaceSelect = within(
      screen.getByTestId("flashcards-manage-workspace-filter")
    ).getByRole("combobox")
    fireEvent.mouseDown(workspaceSelect)
    fireEvent.click(
      await screen.findByText("workspace-77", {
        selector: ".ant-select-item-option-content"
      })
    )

    rerender(
      <ManageTab
        {...baseProps}
        openCreateSignal={1}
        createInitialDeckId={12}
        createInitialShowWorkspaceDecks={false}
        onCreateHandoffConsumed={onCreateHandoffConsumed}
      />
    )

    await waitFor(() => {
      expect(screen.getByTestId("mock-create-drawer")).toBeInTheDocument()
    })
    expect(screen.getByTestId("mock-create-drawer-initial-deck-id")).toHaveTextContent("12")
    expect(screen.getByTestId("mock-create-drawer-include-workspace")).toHaveTextContent("false")
    expect(screen.getByTestId("mock-create-drawer-workspace-id")).toBeEmptyDOMElement()
    expect(onCreateHandoffConsumed).toHaveBeenCalledTimes(1)
  })

  it("moves deck scope by patching workspace_id in the update payload", async () => {
    render(
      <ManageTab
        onNavigateToImport={() => {}}
        onReviewCard={() => {}}
        isActive
      />
    )

    const deckSelect = within(
      screen.getByTestId("flashcards-manage-deck-select")
    ).getByRole("combobox")
    fireEvent.mouseDown(deckSelect)
    fireEvent.keyDown(deckSelect, { key: "ArrowDown", code: "ArrowDown" })
    fireEvent.keyDown(deckSelect, { key: "Enter", code: "Enter" })

    await waitFor(() => {
      expect(screen.getByTestId("flashcards-manage-move-scope")).not.toBeDisabled()
    })
    fireEvent.click(screen.getByTestId("flashcards-manage-move-scope"))

    fireEvent.change(screen.getByPlaceholderText("Leave blank for general scope"), {
      target: { value: "workspace-77" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save/i }))

    await waitFor(() => {
      expect(updateDeckMutateAsync).toHaveBeenCalledWith({
        deckId: 1,
        update: {
          workspace_id: "workspace-77",
          parent_deck_id: null,
          expected_version: 1
        }
      })
    })
  })
})
