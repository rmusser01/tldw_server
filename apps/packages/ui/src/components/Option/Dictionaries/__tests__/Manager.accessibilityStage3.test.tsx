import fs from "node:fs"
import path from "node:path"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import axe from "axe-core"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { DictionariesManager } from "../Manager"

if (typeof window.ResizeObserver === "undefined") {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  ;(window as any).ResizeObserver = ResizeObserverMock
  ;(globalThis as any).ResizeObserver = ResizeObserverMock
}

const dictionaryRecords = [
  {
    id: 401,
    name: "Valid Dictionary",
    description: "Accessibility checks",
    is_active: true,
    entry_count: 1,
    version: 3
  },
  {
    id: 402,
    name: "Warning Dictionary",
    description: "Has warning-level issues",
    is_active: true,
    entry_count: 1,
    version: 2
  },
  {
    id: 403,
    name: "Error Dictionary",
    description: "Has error-level issues",
    is_active: true,
    entry_count: 1,
    version: 4
  }
]

const dictionaryById = new Map(dictionaryRecords.map((record) => [record.id, record]))

const entriesByDictionaryId: Record<number, Array<Record<string, any>>> = {
  401: [
    {
      id: 21,
      dictionary_id: 401,
      pattern: "/BP/i",
      replacement: "blood pressure",
      type: "regex",
      probability: 1,
      enabled: true,
      case_sensitive: false,
      max_replacements: 0
    }
  ],
  402: [
    {
      id: 22,
      dictionary_id: 402,
      pattern: "KCL",
      replacement: "Potassium Chloride",
      type: "literal",
      probability: 1,
      enabled: true,
      case_sensitive: false,
      max_replacements: 0
    }
  ],
  403: [
    {
      id: 23,
      dictionary_id: 403,
      pattern: "/(a+)+$/",
      replacement: "unsafe-regex",
      type: "regex",
      probability: 1,
      enabled: true,
      case_sensitive: false,
      max_replacements: 0
    }
  ]
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
    getDictionary: vi.fn(async (dictionaryId: number) =>
      dictionaryById.get(Number(dictionaryId)) || dictionaryRecords[0]
    ),
    listDictionaryEntries: vi.fn(async (dictionaryId: number) => ({
      entries: entriesByDictionaryId[Number(dictionaryId)] || []
    })),
    updateDictionary: vi.fn(async () => ({})),
    validateDictionary: vi.fn(async (payload: any) => {
      const dictionaryName = String(payload?.data?.name || "")
      if (dictionaryName.includes("Warning")) {
        return {
          errors: [],
          warnings: [
            {
              code: "pattern_overlap",
              field: "entries[0].pattern",
              message: "Pattern overlaps with another literal entry."
            }
          ]
        }
      }
      if (dictionaryName.includes("Error")) {
        return {
          errors: [
            {
              code: "regex_safety",
              field: "entries[0].pattern",
              message: "Regex may cause catastrophic backtracking."
            }
          ],
          warnings: []
        }
      }
      return { errors: [], warnings: [] }
    }),
    processDictionaryText: vi.fn(async () => ({
      original_text: "BP",
      processed_text: "blood pressure",
      replacements: 1,
      iterations: 1,
      entries_used: ["BP"]
    })),
    addDictionaryEntry: vi.fn(async () => ({})),
    updateDictionaryEntry: vi.fn(async () => ({})),
    deleteDictionaryEntry: vi.fn(async () => ({})),
    reorderDictionaryEntries: vi.fn(async () => ({})),
    bulkDictionaryEntries: vi.fn(async () => ({
      success: true,
      affected_count: 1,
      failed_ids: []
    })),
    importDictionaryJSON: vi.fn(async () => ({})),
    importDictionaryMarkdown: vi.fn(async () => ({})),
    exportDictionaryJSON: vi.fn(async () => ({
      name: "Accessible Dictionary",
      entries: []
    })),
    exportDictionaryMarkdown: vi.fn(async () => ({ content: "# dictionary" })),
    dictionaryStatistics: vi.fn(async () => ({
      dictionary_id: 401,
      name: "Valid Dictionary",
      total_entries: 1,
      regex_entries: 1,
      literal_entries: 0
    })),
    dictionaryActivity: vi.fn(async () => ({
      events: [],
      total: 0,
      limit: 10,
      offset: 0
    })),
    listChats: vi.fn(async () => [])
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
        return fallbackOrOptions.defaultValue || key
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

vi.mock("@/store/option", () => ({
  useStoreMessageOption: () => ({
    setHistoryId: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatState: vi.fn(),
    setServerChatTitle: vi.fn()
  })
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

const makeUseMutationResult = (opts: any) => ({
  mutate: async (variables: any) => {
    try {
      const result = await opts?.mutationFn?.(variables)
      opts?.onSuccess?.(result, variables, undefined)
      return result
    } catch (error) {
      opts?.onError?.(error, variables, undefined)
      throw error
    } finally {
      opts?.onSettled?.(undefined, undefined, variables, undefined)
    }
  },
  mutateAsync: async (variables: any) => {
    try {
      const result = await opts?.mutationFn?.(variables)
      opts?.onSuccess?.(result, variables, undefined)
      return result
    } catch (error) {
      opts?.onError?.(error, variables, undefined)
      throw error
    } finally {
      opts?.onSettled?.(undefined, undefined, variables, undefined)
    }
  },
  isPending: false
})

const runA11yRules = async (
  context: Element,
  ruleIds: string[]
) =>
  axe.run(context, {
    runOnly: {
      type: "rule",
      values: ruleIds
    },
    resultTypes: ["violations"]
  })

const parseRgbVar = (block: string, variable: string): [number, number, number] => {
  const match = block.match(new RegExp(`${variable}:\\s*(\\d+)\\s+(\\d+)\\s+(\\d+)\\s*;`))
  if (!match) {
    throw new Error(`Missing ${variable} token`)
  }
  return [Number(match[1]), Number(match[2]), Number(match[3])]
}

const srgbToLinear = (value: number) => {
  const normalized = value / 255
  return normalized <= 0.04045
    ? normalized / 12.92
    : ((normalized + 0.055) / 1.055) ** 2.4
}

const luminance = ([r, g, b]: [number, number, number]) =>
  0.2126 * srgbToLinear(r) +
  0.7152 * srgbToLinear(g) +
  0.0722 * srgbToLinear(b)

const contrastRatio = (a: [number, number, number], b: [number, number, number]) => {
  const l1 = luminance(a)
  const l2 = luminance(b)
  const lighter = Math.max(l1, l2)
  const darker = Math.min(l1, l2)
  return (lighter + 0.05) / (darker + 0.05)
}

describe("DictionariesManager accessibility stage-3", () => {
  beforeEach(() => {
    vi.clearAllMocks()

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

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn(),
      setQueryData: vi.fn(),
      getQueryData: vi.fn()
    })
    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))

    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]
      const dictionaryId = Number(queryKey[1])

      if (key === "tldw:listDictionaries") {
        return makeUseQueryResult({
          status: "success",
          data: dictionaryRecords
        })
      }

      if (key === "tldw:getDictionary") {
        return makeUseQueryResult({
          status: "success",
          data: dictionaryById.get(dictionaryId) || dictionaryRecords[0]
        })
      }

      if (key === "tldw:listDictionaryEntries" || key === "tldw:listDictionaryEntriesAll") {
        return makeUseQueryResult({
          status: "success",
          data: entriesByDictionaryId[dictionaryId] || []
        })
      }

      if (key === "tldw:listChatsForDictionaryAssign") {
        return makeUseQueryResult({
          status: "success",
          data: []
        })
      }

      return makeUseQueryResult({})
    })
  })

  it("has no axe violations for core aria naming and region rules in list view", async () => {
    const { container } = render(<DictionariesManager />)
    await screen.findByText("Valid Dictionary")

    const results = await runA11yRules(container, [
      "aria-required-attr",
      "aria-valid-attr",
      "aria-valid-attr-value",
      "button-name",
      "region"
    ])

    expect(results.violations).toEqual([])
  }, 60000)

  it("has no invalid aria/button-name violations in entries drawer workflow", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    await user.click(
      screen.getByRole("button", {
        name: "Manage entries for Valid Dictionary"
      })
    )
    const drawerTitle = await screen.findByText("Manage Entries: Valid Dictionary")
    const drawerScope =
      drawerTitle.closest(".ant-drawer-content") || document.body

    const results = await runA11yRules(drawerScope, [
      "aria-required-attr",
      "aria-valid-attr",
      "aria-valid-attr-value",
      "button-name"
    ])

    expect(results.violations).toEqual([])
  }, 60000)

  it("uses context-aware validation labels so status is not color-only", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)
    await screen.findByText("Valid Dictionary")

    await user.click(
      screen.getByRole("button", { name: "Validate dictionary Valid Dictionary" })
    )
    await waitFor(() => {
      expect(
        screen.getByRole("button", {
          name: "Dictionary Valid Dictionary is valid. Click to re-validate."
        })
      ).toBeInTheDocument()
    })

    await user.click(
      screen.getByRole("button", { name: "Validate dictionary Warning Dictionary" })
    )
    await waitFor(() => {
      expect(
        screen.getByRole("button", {
          name: "Dictionary Warning Dictionary has warnings. Click to re-validate."
        })
      ).toBeInTheDocument()
    })

    await user.click(
      screen.getByRole("button", { name: "Validate dictionary Error Dictionary" })
    )
    await waitFor(() => {
      expect(
        screen.getByRole("button", {
          name: "Dictionary Error Dictionary has errors. Click to re-validate."
        })
      ).toBeInTheDocument()
    })
  }, 60000)

  it("keeps key dictionary actions and advanced toggle semantics explicitly labeled", async () => {
    const user = userEvent.setup()
    render(<DictionariesManager />)

    expect(
      screen.getByRole("button", { name: "Validate dictionary Valid Dictionary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Edit dictionary Valid Dictionary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Manage entries for Valid Dictionary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Quick assign Valid Dictionary to chats" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Export Valid Dictionary as JSON" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Export Valid Dictionary as Markdown" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "View statistics for Valid Dictionary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Version history for Valid Dictionary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Duplicate dictionary Valid Dictionary" })
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Delete dictionary Valid Dictionary" })
    ).toBeInTheDocument()

    await user.click(
      screen.getByRole("button", {
        name: "Manage entries for Valid Dictionary"
      })
    )
    await screen.findByText("Manage Entries: Valid Dictionary")

    const advancedToggle = screen.getByRole("button", {
      name: "Advanced options"
    })
    expect(advancedToggle).toHaveAttribute("aria-expanded", "false")
    expect(advancedToggle).toHaveAttribute("aria-controls")

    await user.click(advancedToggle)

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Simple mode" })).toHaveAttribute(
        "aria-expanded",
        "true"
      )
    })

    const expandedToggle = screen.getByRole("button", { name: "Simple mode" })
    const advancedPanelId = expandedToggle.getAttribute("aria-controls")
    expect(advancedPanelId).toBeTruthy()
    expect(document.getElementById(advancedPanelId || "")).not.toBeNull()
  }, 60000)

  it("keeps status icon color tokens at WCAG AA non-text contrast on light/dark surfaces", () => {
    const tokenFile = path.resolve(__dirname, "../../../../assets/tailwind-shared.css")
    const css = fs.readFileSync(tokenFile, "utf8")

    const rootBlock = css.match(/:root\s*\{([\s\S]*?)\}/)?.[1]
    const darkBlock = css.match(/\.dark\s*\{([\s\S]*?)\}/)?.[1]

    expect(rootBlock).toBeTruthy()
    expect(darkBlock).toBeTruthy()

    const lightSurface = parseRgbVar(rootBlock as string, "--color-surface")
    const darkSurface = parseRgbVar(darkBlock as string, "--color-surface")

    const lightSuccess = parseRgbVar(rootBlock as string, "--color-success")
    const lightWarn = parseRgbVar(rootBlock as string, "--color-warn")
    const lightDanger = parseRgbVar(rootBlock as string, "--color-danger")

    const darkSuccess = parseRgbVar(darkBlock as string, "--color-success")
    const darkWarn = parseRgbVar(darkBlock as string, "--color-warn")
    const darkDanger = parseRgbVar(darkBlock as string, "--color-danger")

    expect(contrastRatio(lightSuccess, lightSurface)).toBeGreaterThanOrEqual(3)
    expect(contrastRatio(lightWarn, lightSurface)).toBeGreaterThanOrEqual(3)
    expect(contrastRatio(lightDanger, lightSurface)).toBeGreaterThanOrEqual(3)
    expect(contrastRatio(darkSuccess, darkSurface)).toBeGreaterThanOrEqual(3)
    expect(contrastRatio(darkWarn, darkSurface)).toBeGreaterThanOrEqual(3)
    expect(contrastRatio(darkDanger, darkSurface)).toBeGreaterThanOrEqual(3)
  })
})
