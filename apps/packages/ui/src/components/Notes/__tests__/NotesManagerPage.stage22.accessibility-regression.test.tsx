import fs from "node:fs"
import path from "node:path"
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import axe from "axe-core"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageInfo,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting,
  mockGetAllNoteKeywordStats,
  mockSearchNoteKeywords,
  mockCytoscapeFactory
} = vi.hoisted(() => {
  const cyInstance: Record<string, any> = {
    on: vi.fn(),
    fit: vi.fn(),
    destroy: vi.fn(),
    zoom: vi.fn(() => 1)
  }
  const cytoscapeFactory: any = vi.fn(() => cyInstance)
  cytoscapeFactory.use = vi.fn()

  return {
    mockBgRequest: vi.fn(),
    mockMessageSuccess: vi.fn(),
    mockMessageError: vi.fn(),
    mockMessageWarning: vi.fn(),
    mockMessageInfo: vi.fn(),
    mockNavigate: vi.fn(),
    mockConfirmDanger: vi.fn(),
    mockGetSetting: vi.fn(),
    mockSetSetting: vi.fn(),
    mockClearSetting: vi.fn(),
    mockGetAllNoteKeywordStats: vi.fn(),
    mockSearchNoteKeywords: vi.fn(),
    mockCytoscapeFactory: cytoscapeFactory
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasNotes: true },
    loading: false
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mockConfirmDanger
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: mockMessageSuccess,
    error: mockMessageError,
    warning: mockMessageWarning,
    info: mockMessageInfo
  })
}))

vi.mock("@/services/note-keywords", () => ({
  getAllNoteKeywordStats: mockGetAllNoteKeywordStats,
  searchNoteKeywords: mockSearchNoteKeywords
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setHistoryId: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatState: vi.fn(),
      setServerChatTopic: vi.fn(),
      setServerChatClusterId: vi.fn(),
      setServerChatSource: vi.fn(),
      setServerChatExternalRef: vi.fn()
    })
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mockGetSetting,
    setSetting: mockSetSetting,
    clearSetting: mockClearSetting
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null)
  }
}))

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel" />
}))

vi.mock("cytoscape", () => ({
  default: mockCytoscapeFactory
}))

vi.mock("cytoscape-dagre", () => ({
  default: {}
}))

const CORE_RULES = [
  "aria-required-attr",
  "aria-valid-attr",
  "aria-valid-attr-value",
  "button-name",
  "link-name"
]

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

const runA11yRules = async (context: Element, ruleIds: string[]) =>
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

const seedAndSaveNote = async () => {
  fireEvent.change(screen.getByPlaceholderText("Title"), {
    target: { value: "A11y regression note" }
  })
  fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
    target: { value: "Seed content" }
  })
  fireEvent.click(screen.getByTestId("notes-save-button"))

  await waitFor(() => {
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 1")
  })
}

describe("NotesManagerPage stage 22 accessibility regression", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockGetAllNoteKeywordStats.mockResolvedValue([{ keyword: "research", noteCount: 3 }])
    mockSearchNoteKeywords.mockResolvedValue(["research"])

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }
      if (path === "/api/v1/notes/" && method === "POST") {
        return { id: "note-a", version: 1, last_modified: "2026-02-18T10:00:00.000Z" }
      }
      if (path === "/api/v1/notes/note-a" && method === "GET") {
        return {
          id: "note-a",
          title: "A11y regression note",
          content: "Seed content",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T10:00:00.000Z"
        }
      }
      if (path.startsWith("/api/v1/notes/note-a/neighbors")) {
        return {
          nodes: [{ id: "note-a", type: "note", label: "A11y regression note" }],
          edges: []
        }
      }
      if (path.startsWith("/api/v1/notes/graph?")) {
        return {
          elements: {
            nodes: [{ data: { id: "note:note-a", type: "note", label: "A11y regression note" } }],
            edges: []
          },
          truncated: false
        }
      }
      return {}
    })
  })

  it("has no core aria/name violations in baseline notes shell", async () => {
    const { container } = renderPage()
    await screen.findByTestId("notes-editor-region")

    const results = await runA11yRules(container, CORE_RULES)
    expect(results.violations).toEqual([])
  }, 15000)

  it("has no core aria/name violations in the keyword picker modal state", async () => {
    renderPage()

    fireEvent.click(screen.getByRole("button", { name: "Browse tags" }))
    const modalBody = await screen.findByTestId("notes-keyword-picker-modal")
    const modalRoot = modalBody.closest(".ant-modal-root") || document.body

    const modalTitle = modalRoot.querySelector(".ant-modal-title")
    expect(modalTitle).toHaveTextContent("Browse tags")
    const results = await runA11yRules(modalRoot, CORE_RULES)
    expect(results.violations).toEqual([])
  })

  it("has no core aria/name violations in the graph modal state", async () => {
    renderPage()
    await seedAndSaveNote()

    fireEvent.click(screen.getByRole("button", { name: "Split" }))
    fireEvent.click(await screen.findByTestId("notes-open-graph-view"))

    const radiusControl = await screen.findByTestId("notes-graph-radius-control")
    const modalRoot = radiusControl.closest(".ant-modal-root") || document.body

    const modalTitle = modalRoot.querySelector(".ant-modal-title")
    expect(modalTitle).toHaveTextContent("Notes graph view")
    const results = await runA11yRules(modalRoot, CORE_RULES)
    expect(results.violations).toEqual([])
  })

  it("keeps notes theme text contrast at WCAG AA levels for light and dark surfaces", () => {
    const cssPath = path.resolve(process.cwd(), "src/assets/tailwind-shared.css")
    const cssText = fs.readFileSync(cssPath, "utf8")
    const rootMatch = cssText.match(/:root\s*\{([\s\S]*?)\}/)
    const darkMatch = cssText.match(/\.dark\s*\{([\s\S]*?)\}/)

    expect(rootMatch).toBeTruthy()
    expect(darkMatch).toBeTruthy()

    const rootBlock = rootMatch?.[1] ?? ""
    const darkBlock = darkMatch?.[1] ?? ""

    const lightText = parseRgbVar(rootBlock, "--color-text")
    const lightSurface = parseRgbVar(rootBlock, "--color-surface")
    const darkText = parseRgbVar(darkBlock, "--color-text")
    const darkSurface = parseRgbVar(darkBlock, "--color-surface")

    expect(contrastRatio(lightText, lightSurface)).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(darkText, darkSurface)).toBeGreaterThanOrEqual(4.5)
  })
})
