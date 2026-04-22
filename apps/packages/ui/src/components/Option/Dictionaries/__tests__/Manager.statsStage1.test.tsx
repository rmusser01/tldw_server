import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { DictionariesManager } from "../Manager"

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches:
      /min-width:\s*576px/.test(query) ||
      /min-width:\s*768px/.test(query) ||
      /min-width:\s*992px/.test(query),
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false
  })
})

if (typeof window.ResizeObserver === "undefined") {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  ;(window as any).ResizeObserver = ResizeObserverMock
  ;(globalThis as any).ResizeObserver = ResizeObserverMock
}

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
  tldwClientMock
} = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  useMutationMock: vi.fn(),
  useQueryClientMock: vi.fn(),
  notificationMock: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    open: vi.fn(),
    destroy: vi.fn()
  },
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    listDictionaries: vi.fn(async () => ({ dictionaries: [] })),
    dictionaryStatistics: vi.fn(async () => ({})),
    dictionaryActivity: vi.fn(async () => ({ events: [], total: 0 }))
  }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        if (fallbackOrOptions.defaultValue) {
          const count = (fallbackOrOptions as { count?: number }).count
          return typeof count === "number"
            ? fallbackOrOptions.defaultValue.replace("{{count}}", String(count))
            : fallbackOrOptions.defaultValue
        }
        return key
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    loading: false,
    capabilities: {
      hasChatDictionaries: true
    }
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({
    showUndoNotification: vi.fn()
  })
}))

vi.mock("@/components/Common/FeatureEmptyState", () => ({
  default: ({ title }: { title: string }) => <div>{title}</div>
}))

vi.mock("@/components/Common/LabelWithHelp", () => ({
  LabelWithHelp: ({ label }: { label: React.ReactNode }) => <span>{label}</span>
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn(async () => true)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

const makeUseQueryResult = (value: Record<string, any>) => ({
  data: undefined,
  status: "success",
  error: null,
  isPending: false,
  isFetching: false,
  isLoading: false,
  refetch: vi.fn(),
  ...value
})

const makeUseMutationResult = () => ({
  mutate: vi.fn(),
  mutateAsync: vi.fn(),
  isPending: false
})

describe("DictionariesManager statistics stage-1 coverage", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn(),
      getQueryData: vi.fn()
    })

    useMutationMock.mockImplementation(() => makeUseMutationResult())

    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: [
            {
              id: 88,
              name: "Medical Terms",
              description: "Clinical substitutions",
              is_active: true,
              entry_count: 4
            }
          ]
        })
      }

      return makeUseQueryResult({})
    })
  })

  it("renders expanded statistics fields", async () => {
    const user = userEvent.setup()

    tldwClientMock.dictionaryStatistics.mockResolvedValueOnce({
      dictionary_id: 88,
      name: "Medical Terms",
      total_entries: 4,
      regex_entries: 1,
      literal_entries: 3,
      enabled_entries: 3,
      disabled_entries: 1,
      probabilistic_entries: 2,
      timed_effect_entries: 1,
      groups: ["Clinical", "Abbrev"],
      average_probability: 0.65,
      created_at: "2026-02-01T12:00:00Z",
      updated_at: "2026-02-17T16:00:00Z",
      last_used: "2026-02-18T09:00:00Z",
      zero_usage_entries: 1,
      entry_usage: [
        {
          entry_id: 9,
          pattern: "BP",
          usage_count: 7,
          last_used_at: "2026-02-18T09:00:00Z"
        },
        {
          entry_id: 10,
          pattern: "HR",
          usage_count: 0,
          last_used_at: null
        }
      ],
      pattern_conflict_count: 1,
      pattern_conflicts: [
        {
          entry_id_a: 9,
          entry_id_b: 10,
          pattern_a: "BP",
          pattern_b: "/B.*/",
          type_a: "literal",
          type_b: "regex",
          conflict_type: "literal-regex",
          severity: "medium",
          reason: "Regex pattern overlaps with a literal pattern and may trigger on the same input."
        }
      ],
      total_usage_count: 12
    })

    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", {
        name: "View statistics for Medical Terms"
      })
    )

    expect(
      await screen.findByText("Dictionary Statistics", {}, { timeout: 15000 })
    ).toBeInTheDocument()

    // Summary sentence at top (use getAllByText since "4 entries" also appears in list)
    const summaryEl = screen.getByText(
      (_content, element) =>
        element?.tagName === "P" && (element.textContent || "").includes("used 12 times")
    )
    expect(summaryEl).toBeInTheDocument()
    expect(summaryEl.textContent).toMatch(/4\s+entries/)
    expect(summaryEl.textContent).toMatch(/1 conflicts/)
    expect(summaryEl.textContent).toMatch(/1 unused/)

    // Overview section (open by default)
    expect(screen.getByText("Overview")).toBeInTheDocument()
    expect(screen.getByText("Total Entries")).toBeInTheDocument()

    // Usage section (open by default)
    expect(screen.getByText("Usage")).toBeInTheDocument()
    expect(screen.getByText("Enabled Entries")).toBeInTheDocument()
    expect(screen.getByText("Disabled Entries")).toBeInTheDocument()
    expect(screen.getByText("12")).toBeInTheDocument()

    // Expand Health section (collapsed by default)
    await user.click(screen.getByText("Health"))
    expect(screen.getByText("Unused Entries")).toBeInTheDocument()
    // Note: "Pattern Conflicts" appears both as a Descriptions label
    // and as a section heading below
    expect(screen.getByText("Clinical, Abbrev")).toBeInTheDocument()

    // Expand Advanced section (collapsed by default)
    await user.click(screen.getByText("Advanced"))
    expect(screen.getByText("Probabilistic Entries")).toBeInTheDocument()
    expect(screen.getByText("Timed Effect Entries")).toBeInTheDocument()
    expect(screen.getByText("0.65")).toBeInTheDocument()

    // Sections below the Collapse remain unchanged
    expect(screen.getByText("Entry usage snapshot")).toBeInTheDocument()
    expect(screen.getByText(/7 uses/)).toBeInTheDocument()
    expect(screen.getByText("Pattern conflicts")).toBeInTheDocument()
    expect(screen.getByText(/Regex pattern overlaps with a literal pattern/i)).toBeInTheDocument()
  }, 60000)

  it("shows null-safe fallbacks for optional statistics fields", async () => {
    const user = userEvent.setup()

    tldwClientMock.dictionaryStatistics.mockResolvedValueOnce({
      dictionary_id: 88,
      name: "Medical Terms",
      total_entries: 0,
      regex_entries: 0,
      literal_entries: 0,
      enabled_entries: null,
      disabled_entries: null,
      probabilistic_entries: null,
      timed_effect_entries: null,
      groups: [],
      average_probability: null,
      created_at: null,
      updated_at: null,
      last_used: null,
      zero_usage_entries: null,
      entry_usage: [],
      pattern_conflict_count: null,
      pattern_conflicts: [],
      total_usage_count: null
    })

    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", {
        name: "View statistics for Medical Terms"
      })
    )

    await screen.findByText("Dictionary Statistics")

    // Summary sentence shows 0 entries with no conflict/unused/usage extras
    expect(screen.getByText(/0 entries/)).toBeInTheDocument()

    // Overview and Usage sections are open by default
    await waitFor(() => {
      expect(screen.getAllByText("0").length).toBeGreaterThanOrEqual(3)
    })

    // Expand Health section for null-safe "—" values
    await user.click(screen.getByText("Health"))
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(1)

    // Expand Advanced section for 0.00 probability fallback
    await user.click(screen.getByText("Advanced"))
    expect(screen.getByText("0.00")).toBeInTheDocument()

    expect(screen.getByText("No potential conflicts detected.")).toBeInTheDocument()
  }, 60000)
})
