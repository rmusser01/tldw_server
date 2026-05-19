import fs from "node:fs"
import path from "node:path"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { WorldBooksManager } from "../Manager"

const {
  useQueryMock,
  useMutationMock,
  useQueryClientMock,
  notificationMock,
  undoNotificationMock,
  confirmDangerMock,
  tldwClientMock,
  mockBreakpoints
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
  undoNotificationMock: {
    showUndoNotification: vi.fn()
  },
  confirmDangerMock: vi.fn(async () => true),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    createWorldBook: vi.fn(async () => ({ id: 1 })),
    updateWorldBook: vi.fn(async () => ({})),
    deleteWorldBook: vi.fn(async () => ({})),
    listCharacters: vi.fn(async () => []),
    listCharacterWorldBooks: vi.fn(async () => []),
    listWorldBookEntries: vi.fn(async () => ({ entries: [] })),
    addWorldBookEntry: vi.fn(async () => ({})),
    updateWorldBookEntry: vi.fn(async () => ({})),
    deleteWorldBookEntry: vi.fn(async () => ({})),
    bulkWorldBookEntries: vi.fn(async () => ({ success: true, affected_count: 0, failed_ids: [] })),
    exportWorldBook: vi.fn(async () => ({})),
    worldBookStatistics: vi.fn(async () => ({})),
    importWorldBook: vi.fn(async () => ({})),
    attachWorldBookToCharacter: vi.fn(async () => ({})),
    detachWorldBookFromCharacter: vi.fn(async () => ({}))
  },
  mockBreakpoints: { md: true }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock,
  useMutation: useMutationMock,
  useQueryClient: useQueryClientMock
}))

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  return {
    ...actual,
    Grid: {
      ...actual.Grid,
      useBreakpoint: () => mockBreakpoints
    }
  }
})

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

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => undoNotificationMock
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => confirmDangerMock
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

const makeUseQueryResult = (value: Record<string, any>) => ({
  data: null,
  status: "success",
  isLoading: false,
  isFetching: false,
  isPending: false,
  error: null,
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
    }
  },
  isPending: false
})

const worldBooks = [
  {
    id: 1,
    name: "Arcana",
    description: "Main lore",
    enabled: true,
    entry_count: 2
  }
]

const entryData = [
  {
    entry_id: 1,
    keywords: ["castle"],
    content: "Castle lore variant one.",
    priority: 60,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  },
  {
    entry_id: 2,
    keywords: ["castle"],
    content: "Castle lore variant two.",
    priority: 55,
    enabled: true,
    case_sensitive: false,
    regex_match: false,
    whole_word_match: true,
    appendable: false
  }
]

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

describe("WorldBooksManager accessibility stage-4 alerts, validation wiring, and contrast", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockBreakpoints.md = true

    useQueryClientMock.mockReturnValue({
      invalidateQueries: vi.fn()
    })

    useMutationMock.mockImplementation((opts: any) => makeUseMutationResult(opts))
    useQueryMock.mockImplementation((opts: any) => {
      const queryKey = Array.isArray(opts?.queryKey) ? opts.queryKey : []
      const key = queryKey[0]

      if (key === "tldw:listWorldBooks") {
        return makeUseQueryResult({ data: worldBooks, status: "success" })
      }
      if (key === "tldw:listCharactersForWB") {
        return makeUseQueryResult({ data: [] })
      }
      if (key === "tldw:worldBookAttachments") {
        return makeUseQueryResult({ data: {}, isLoading: false })
      }
      if (key === "tldw:listWorldBookEntries") {
        return makeUseQueryResult({ data: entryData, status: "success" })
      }
      if (key === "tldw:worldBookRuntimeConfig") {
        return makeUseQueryResult({ data: { max_recursive_depth: 10 }, status: "success" })
      }
      return makeUseQueryResult({})
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it("announces keyword conflict tags with explicit aria-label content", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    // Select the world book to show detail panel with entries tab
    await user.click(screen.getByText("Arcana"))
    await user.click(screen.getByText("Keyword Index (1 conflicts)"))

    expect(
      await screen.findByLabelText("castle: conflict - 2 content variations")
    ).toBeInTheDocument()
    expect(
      await screen.findByText("1 keyword conflicts detected.")
    ).toBeInTheDocument()
  }, 30000)

  it("keeps required field errors linked through aria-describedby", async () => {
    const user = userEvent.setup()
    render(<WorldBooksManager />)

    await user.click(screen.getByRole("button", { name: "New World Book" }))
    await user.click(screen.getByRole("button", { name: "Create" }))

    const nameInput = screen.getByRole("textbox", { name: "Name" })
    await waitFor(() => {
      expect(nameInput).toHaveAttribute("aria-invalid", "true")
      expect(nameInput).toHaveAttribute("aria-describedby")
      expect(screen.getByText("Name is required")).toBeInTheDocument()
    })

    const describedBy = nameInput.getAttribute("aria-describedby") || ""
    const describedByIds = describedBy.split(/\s+/).filter(Boolean)
    expect(describedByIds.length).toBeGreaterThan(0)
    expect(describedByIds.some((id) => !!document.getElementById(id))).toBe(true)
  }, 30000)

  it("keeps muted text tokens at WCAG AA contrast against primary backgrounds", () => {
    const tokenFile = path.resolve(__dirname, "../../../../assets/tailwind-shared.css")
    const css = fs.readFileSync(tokenFile, "utf8")

    const rootBlock = css.match(/:root\s*\{([\s\S]*?)\}/)?.[1]
    const darkBlock = css.match(/\.dark\s*\{([\s\S]*?)\}/)?.[1]

    expect(rootBlock).toBeTruthy()
    expect(darkBlock).toBeTruthy()

    const lightMuted = parseRgbVar(rootBlock as string, "--color-text-muted")
    const lightBg = parseRgbVar(rootBlock as string, "--color-bg")
    const lightSurface = parseRgbVar(rootBlock as string, "--color-surface")
    const darkMuted = parseRgbVar(darkBlock as string, "--color-text-muted")
    const darkBg = parseRgbVar(darkBlock as string, "--color-bg")
    const darkSurface = parseRgbVar(darkBlock as string, "--color-surface")

    expect(contrastRatio(lightMuted, lightBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(lightMuted, lightSurface)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(darkMuted, darkBg)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(darkMuted, darkSurface)).toBeGreaterThanOrEqual(4.5)
  })
})
