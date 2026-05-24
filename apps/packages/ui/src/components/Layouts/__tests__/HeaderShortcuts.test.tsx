import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"
import { HeaderShortcuts } from "../HeaderShortcuts"
import { getHeaderShortcutItems } from "../header-shortcut-items"

const mockState = vi.hoisted(() => ({
  expanded: false,
  launcherView: "current" as "current" | "legacy",
  setExpanded: vi.fn().mockResolvedValue(undefined),
  setLauncherView: vi.fn().mockResolvedValue(undefined),
  setSelection: vi.fn().mockResolvedValue(undefined)
}))

const mockUseSetting = vi.hoisted(() => vi.fn())
const mockT = vi.hoisted(() => (key: string, fallback?: string) => fallback ?? key)

const ALL_SHORTCUT_IDS = [
  "chat", "chat-workspace", "prompts", "prompt-studio", "characters",
  "chat-dictionaries", "world-books", "deep-research", "research-workspace",
  "knowledge-qa", "media", "document-workspace",
  "repo2txt",
  "multi-item-review", "collections",
  "watchlists", "integrations", "mcp-hub", "scheduled-tasks", "notes", "chatbooks-playground", "flashcards",
  "quizzes", "evaluations", "chunking-playground",
  "stt-playground", "tts-playground", "audiobook-studio",
  "workflows", "writing-playground", "acp-playground",
  "skills", "kanban-playground",
  "model-playground", "data-tables",
  "admin-server", "admin-integrations", "documentation", "moderation-review", "moderation-rules",
  "admin-llamacpp", "admin-mlx", "settings"
]

/* ------------------------------------------------------------------ */
/*  Mocks                                                              */
/* ------------------------------------------------------------------ */

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mockT,
    i18n: { language: "en" }
  })
}))

vi.mock("@/hooks/useKeyboardShortcuts", () => ({
  useShortcut: vi.fn(),
  formatShortcut: vi.fn(() => ""),
  isMac: false
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: mockUseSetting
}))

vi.mock("@/services/settings/ui-settings", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/ui-settings")>()
  return {
    ...actual,
    HEADER_SHORTCUTS_EXPANDED_SETTING: { key: "header_shortcuts_expanded", defaultValue: false },
    HEADER_SHORTCUTS_LAUNCHER_VIEW_SETTING: {
      key: "header_shortcuts_launcher_view",
      defaultValue: "current"
    },
    HEADER_SHORTCUT_SELECTION_SETTING: { key: "header_shortcut_selection", defaultValue: [] }
  }
})

const renderWithRouter = (ui: React.ReactElement, initialRoute = "/") =>
  render(<MemoryRouter initialEntries={[initialRoute]}>{ui}</MemoryRouter>)

/* ------------------------------------------------------------------ */
/*  Tests                                                              */
/* ------------------------------------------------------------------ */

describe("HeaderShortcuts launcher modal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockState.expanded = false
    mockState.launcherView = "current"
    mockState.setExpanded = vi.fn().mockResolvedValue(undefined)
    mockState.setLauncherView = vi.fn().mockResolvedValue(undefined)
    mockState.setSelection = vi.fn().mockResolvedValue(undefined)

    mockUseSetting.mockImplementation((setting: { key: string }) => {
      if (setting.key === "header_shortcuts_expanded") {
        return [mockState.expanded, mockState.setExpanded]
      }
      if (setting.key === "header_shortcuts_launcher_view") {
        return [mockState.launcherView, mockState.setLauncherView]
      }
      return [ALL_SHORTCUT_IDS, mockState.setSelection]
    })
  })

  it("renders nothing when closed (expanded=false)", () => {
    const { container } = renderWithRouter(
      <HeaderShortcuts expanded={false} onExpandedChange={vi.fn()} />
    )
    expect(container.innerHTML).toBe("")
  })

  it("routes the chat shortcut directly to /chat", () => {
    const chatShortcut = getHeaderShortcutItems().find((item) => item.id === "chat")

    expect(chatShortcut?.to).toBe("/chat")
  })

  it("includes launcher shortcuts for integrations and scheduled tasks", () => {
    expect(getHeaderShortcutItems()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          to: "/integrations",
          labelDefault: "Integrations"
        }),
        expect.objectContaining({
          to: "/mcp-hub",
          labelDefault: "MCP Hub"
        }),
        expect.objectContaining({
          to: "/scheduled-tasks",
          labelDefault: "Scheduled Tasks"
        }),
        expect.objectContaining({
          to: "/admin/integrations",
          labelDefault: "Workspace Integrations"
        })
      ])
    )
  })

  it("shows MCP Hub in the automation launcher group and search results", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const nav = screen.getByLabelText("Categories")
    fireEvent.click(within(nav).getByText("Automation & Agents"))

    expect(screen.getByText("MCP Hub")).toBeInTheDocument()

    const input = screen.getByPlaceholderText("Search pages...")
    fireEvent.change(input, { target: { value: "mcp" } })

    const listbox = screen.getByRole("listbox")
    expect(within(listbox).getByText("MCP Hub")).toBeInTheDocument()
  })

  it("renders a dialog when open (expanded=true)", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )
    expect(screen.getByRole("dialog")).toBeInTheDocument()
  })

  it("shows search input placeholder", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )
    expect(screen.getByPlaceholderText("Search pages...")).toBeInTheDocument()
  })

  it("shows 'All' category in sidebar", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )
    expect(screen.getByText("All")).toBeInTheDocument()
  })

  it("shows category names in sidebar", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )
    const nav = screen.getByLabelText("Categories")
    expect(within(nav).getByText("Chat & Persona")).toBeInTheDocument()
    expect(within(nav).getByText("Research")).toBeInTheDocument()
    expect(within(nav).getByText("Library")).toBeInTheDocument()
    expect(within(nav).getByText("Creation")).toBeInTheDocument()
    expect(within(nav).getByText("Planning & Learning")).toBeInTheDocument()
    expect(within(nav).getByText("Automation & Agents")).toBeInTheDocument()
    expect(within(nav).getByText("Tools")).toBeInTheDocument()
    expect(within(nav).getByText("Admin & Help")).toBeInTheDocument()
  })

  it("shows items with group headers in All mode", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )
    // Items should be visible
    expect(screen.getByText("Chat")).toBeInTheDocument()
    expect(screen.getByText("Prompts")).toBeInTheDocument()
    expect(screen.getByText("Deep Research")).toBeInTheDocument()
    expect(screen.getByText("Repo2Txt")).toBeInTheDocument()
    expect(screen.getByText("Settings")).toBeInTheDocument()
  })

  it("does not show Content Review in the launcher modal", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const listbox = screen.getByRole("listbox")
    expect(within(listbox).queryByText("Content Review")).not.toBeInTheDocument()
  })

  it("filters items when category is clicked", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    // Click "Creation" category in the sidebar
    const nav = screen.getByLabelText("Categories")
    fireEvent.click(within(nav).getByText("Creation"))

    // Audio items should be visible
    expect(screen.getByText("STT Playground")).toBeInTheDocument()
    expect(screen.getByText("TTS Playground")).toBeInTheDocument()

    // Non-audio items should NOT be visible (in the listbox)
    const listbox = screen.getByRole("listbox")
    expect(within(listbox).queryByText("Chat")).not.toBeInTheDocument()
  })

  it("filters items by search query", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const input = screen.getByPlaceholderText("Search pages...")
    fireEvent.change(input, { target: { value: "Media" } })

    const listbox = screen.getByRole("listbox")
    expect(within(listbox).getByText("Media")).toBeInTheDocument()
    // "Chat" shouldn't match "Media"
    expect(within(listbox).queryByText("Chat")).not.toBeInTheDocument()
  })

  it("shows empty state when search has no results", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const input = screen.getByPlaceholderText("Search pages...")
    fireEvent.change(input, { target: { value: "xyznonexistent" } })

    expect(screen.getByText("No pages match your search")).toBeInTheDocument()
  })

  it("calls onExpandedChange(false) when Escape is pressed", () => {
    const onExpandedChange = vi.fn()
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={onExpandedChange} />
    )

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" })
    expect(onExpandedChange).toHaveBeenCalledWith(false)
  })

  it("calls onExpandedChange(false) when backdrop is clicked", () => {
    const onExpandedChange = vi.fn()
    const { container } = renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={onExpandedChange} />
    )

    // The backdrop is the first fixed overlay element
    const backdrop = container.parentElement!.querySelector(".fixed.inset-0")
    expect(backdrop).toBeInTheDocument()
    fireEvent.click(backdrop!)
    expect(onExpandedChange).toHaveBeenCalledWith(false)
  })

  it("shows keyboard shortcut badges for items with shortcutIndex", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    // Chat has shortcutIndex: 1, so should display "Ctrl+1" (isMac is false in mock)
    expect(screen.getByText("Ctrl+1")).toBeInTheDocument()
    expect(screen.getByText("Ctrl+2")).toBeInTheDocument()
  })

  it("displays match count badges in sidebar when searching", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const input = screen.getByPlaceholderText("Search pages...")
    fireEvent.change(input, { target: { value: "Playground" } })

    // Multiple items have "Playground" in their name, sidebar should show counts
    const nav = screen.getByLabelText("Categories")
    // All category should show total count
    const allButton = within(nav).getByText("All")
    // The count is a sibling span within the same button
    expect(allButton.closest("button")!.querySelector(".text-xs")).toBeInTheDocument()
  })

  it("renders items as NavLink elements with role=option", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const options = screen.getAllByRole("option")
    expect(options.length).toBeGreaterThan(0)
  })

  it("has aria-modal and role=dialog on the modal container", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const dialog = screen.getByRole("dialog")
    expect(dialog).toHaveAttribute("aria-modal", "true")
  })

  it("uses enlarged default modal dimensions", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const dialog = screen.getByRole("dialog")
    expect(dialog).toHaveStyle({ maxWidth: "var(--content-max-width)", maxHeight: "80vh" })
  })

  it("uses a wider left category column", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    const nav = screen.getByLabelText("Categories")
    expect(nav.className).toContain("w-56")
  })

  it("toggles between current and legacy sheet views", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    expect(screen.getByLabelText("Categories")).toBeInTheDocument()
    expect(
      screen.queryByRole("listbox", { name: "Legacy sheet" })
    ).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Legacy sheet view" }))

    expect(
      screen.queryByLabelText("Categories")
    ).not.toBeInTheDocument()
    expect(screen.getByRole("listbox", { name: "Legacy sheet" })).toBeInTheDocument()
    expect(mockState.setLauncherView).toHaveBeenCalledWith("legacy")

    fireEvent.click(screen.getByRole("button", { name: "Current view" }))
    expect(screen.getByLabelText("Categories")).toBeInTheDocument()
    expect(mockState.setLauncherView).toHaveBeenCalledWith("current")
  })

  it("opens in persisted legacy view mode when preference is set", () => {
    mockState.launcherView = "legacy"

    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    expect(screen.queryByLabelText("Categories")).not.toBeInTheDocument()
    expect(screen.getByRole("listbox", { name: "Legacy sheet" })).toBeInTheDocument()
  })

  it("shows footer with keyboard navigation hints", () => {
    renderWithRouter(
      <HeaderShortcuts expanded={true} onExpandedChange={vi.fn()} />
    )

    expect(screen.getByText("navigate")).toBeInTheDocument()
    expect(screen.getByText("select")).toBeInTheDocument()
    expect(screen.queryByText("toggle")).not.toBeInTheDocument()
  })
})
