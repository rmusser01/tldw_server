import React from "react"
import { ConfigProvider } from "antd"
import userEvent from "@testing-library/user-event"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting,
  mockPromptModal,
  capabilityState
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockSetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  mockPromptModal: vi.fn(),
  capabilityState: { hasNotes: true }
}))

vi.mock("@/components/Notes/notes-manager-utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/Notes/notes-manager-utils")>()
  return { ...actual, promptModal: mockPromptModal }
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
    capabilities: capabilityState,
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
    info: vi.fn()
  })
}))

vi.mock("@/services/note-keywords", () => ({
  getAllNoteKeywordStats: vi.fn(async () => []),
  searchNoteKeywords: vi.fn(async () => [])
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

describe("NotesManagerPage stage 40 advanced editing and navigation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    capabilityState.hasNotes = true
    window.localStorage.clear()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 }
        }
      }

      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }

      return {}
    })
  })

  it.each([6, 7])(
    "initializes a fresh %s-heading outline at the disclosure boundary",
    async (count) => {
      renderPage()
      fireEvent.change(
        screen.getByPlaceholderText(
          "Write your note here... (Markdown supported)"
        ),
        {
          target: {
            value: Array.from(
              { length: count },
              (_, index) => `## Boundary ${index}`
            ).join("\n")
          }
        }
      )
      expect(await screen.findByTestId("notes-toc-panel-toggle")).toHaveAttribute(
        "aria-expanded",
        String(count === 6)
      )
    }
  )

  it("targets the renamed duplicate rich heading after an actual input record", async () => {
    renderPage()
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      {
        target: { value: "## Before\n\n## Duplicate\n\n## End" }
      }
    )
    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const editor = await screen.findByTestId("notes-wysiwyg-editor")
    const headings = editor.querySelectorAll("h2")
    expect(headings).toHaveLength(3)
    headings[0].textContent = "Duplicate"
    fireEvent.input(editor)
    fireEvent.click(await screen.findByTestId("notes-toc-item-duplicate-2"))
    await waitFor(() =>
      expect(window.getSelection()?.anchorNode).toBe(headings[1])
    )
    expect(headings[0].getAttribute("data-md-slug")).toBe("before")
    expect(editor.querySelectorAll("h2")[1]).toBe(headings[1])
  })

  it.each([
    { shiftKey: true },
    { ctrlKey: true },
    { metaKey: true },
    { altKey: true },
    { isComposing: true }
  ])("leaves rich task Enter native for %j", async (options) => {
    renderPage()
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      { target: { value: "- [x] task" } }
    )
    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const editor = await screen.findByTestId("notes-wysiwyg-editor")
    const item = editor.querySelector("li")
    expect(item).not.toBeNull()
    const text = item!.lastChild!
    const range = document.createRange()
    range.setStart(text, text.textContent!.length)
    range.collapse(true)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    const before = editor.innerHTML
    expect(fireEvent.keyDown(editor, { key: "Enter", ...options })).toBe(true)
    expect(editor.innerHTML).toBe(before)
    expect(window.getSelection()?.anchorNode).toBe(text)
  })

  it("leaves rich task Enter untouched when Notes capability disables editing", async () => {
    const view = renderPage()
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      { target: { value: "- [x] task" } }
    )
    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const editor = await screen.findByTestId("notes-wysiwyg-editor")
    const item = editor.querySelector("li")
    expect(item).not.toBeNull()
    const range = document.createRange()
    range.selectNodeContents(item!)
    range.collapse(false)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    capabilityState.hasNotes = false
    view.rerender(
      <QueryClientProvider client={new QueryClient()}>
        <NotesManagerPage />
      </QueryClientProvider>
    )
    expect(editor).toHaveAttribute("contenteditable", "false")
    expect(editor.querySelector("input")).toBeDisabled()
    const before = editor.innerHTML
    expect(fireEvent.keyDown(editor, { key: "Enter" })).toBe(true)
    expect(editor.innerHTML).toBe(before)
  })

  it("restores the rich selection when a heading menu is used with the keyboard", async () => {
    const user = userEvent.setup()
    render(
      <ConfigProvider theme={{ token: { motion: false } }}>
        <QueryClientProvider
          client={
            new QueryClient({
              defaultOptions: {
                queries: { retry: false },
                mutations: { retry: false }
              }
            })
          }>
          <NotesManagerPage />
        </QueryClientProvider>
      </ConfigProvider>
    )
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      { target: { value: "first\n\nsecond" } }
    )
    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const editor = await screen.findByTestId("notes-wysiwyg-editor")
    const paragraph = Array.from(editor.querySelectorAll("p")).find(
      (node) => node.textContent === "second"
    )
    expect(paragraph).toBeDefined()
    const text = paragraph!.firstChild!
    editor.focus()
    const range = document.createRange()
    range.setStart(text, 1)
    range.setEnd(text, 4)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    expect(window.getSelection()?.toString()).toBe("eco")
    const original = document.execCommand
    const observed: string[] = []
    document.execCommand = vi.fn((command: string) => {
      if (command === "formatBlock")
        observed.push(window.getSelection()?.toString() || "")
      return true
    })
    try {
      screen.getByTestId("notes-toolbar-heading").focus()
      await user.keyboard("{Enter}")
      const item = await screen.findByRole("menuitem", { name: "Heading 1" })
      await waitFor(() => expect(item).toBeVisible())
      fireEvent.keyDown(window, { key: "Tab", keyCode: 9 })
      await waitFor(() => expect(item).toHaveFocus())
      fireEvent.keyDown(item, { key: "Enter", keyCode: 13 })
      await waitFor(() => expect(observed).toEqual(["eco"]))
      expect(paragraph!.firstChild).toBe(text)
    } finally {
      document.execCommand = original
    }
  })

  it("offers the WYSIWYG input mode by default now that NE-01 is fixed", async () => {
    renderPage()

    expect(
      await screen.findByPlaceholderText("Write your note here... (Markdown supported)")
    ).toBeInTheDocument()
    expect(screen.getByTestId("notes-input-mode-toggle")).toBeInTheDocument()
    expect(screen.getByTestId("notes-input-mode-markdown")).toBeInTheDocument()
    expect(screen.getByTestId("notes-input-mode-wysiwyg")).toBeInTheDocument()
    // Markdown stays the default input mode.
    expect(screen.queryByTestId("notes-wysiwyg-editor")).not.toBeInTheDocument()
  }, 10000)

  // XP-15 (#3124): editor meta labels rendered at 14px in the body colour,
  // louder than section headings, because antd Typography ignored text-[11px].
  it("renders editor meta labels as small muted text", async () => {
    renderPage()

    await screen.findByPlaceholderText("Write your note here... (Markdown supported)")
    for (const label of ["Formatting", "Assist", "Markdown + LaTeX supported"]) {
      const element = screen.getAllByText(label)[0]
      expect(element, label).toHaveClass("text-[11px]", "text-text-muted")
      expect(element.className, label).not.toMatch(/ant-typography/)
    }
  }, 10000)

  // AX-15 (#3124): the active input mode was shown by button fill only.
  it("exposes the active input mode with aria-pressed", async () => {
    renderPage()

    const markdown = await screen.findByTestId("notes-input-mode-markdown")
    const wysiwyg = screen.getByTestId("notes-input-mode-wysiwyg")
    expect(markdown).toHaveAttribute("aria-pressed", "true")
    expect(wysiwyg).toHaveAttribute("aria-pressed", "false")

    fireEvent.click(wysiwyg)
    expect(await screen.findByTestId("notes-wysiwyg-editor")).toBeInTheDocument()
    expect(screen.getByTestId("notes-input-mode-markdown")).toHaveAttribute("aria-pressed", "false")
    expect(screen.getByTestId("notes-input-mode-wysiwyg")).toHaveAttribute("aria-pressed", "true")
  }, 10000)

  it("preserves markdown content when switching to WYSIWYG and back without edits", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    const sourceMarkdown = "# Title\n\n- item\n\nParagraph with **bold** and [link](https://example.com)"
    fireEvent.change(textarea, { target: { value: sourceMarkdown } })

    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    expect(await screen.findByTestId("notes-wysiwyg-editor")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))

    await waitFor(() => {
      expect(
        screen.getByPlaceholderText("Write your note here... (Markdown supported)")
      ).toHaveValue(sourceMarkdown)
    })
  }, 10000)

  it("generates table of contents entries and jumps to heading offsets in markdown mode", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    fireEvent.change(textarea, {
      target: {
        value: "# Intro\n\n## Section A\nalpha\n\n## Section B\nbeta"
      }
    })

    expect(await screen.findByTestId("notes-toc-panel")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("notes-toc-item-section-b"))

    await waitFor(() => {
      const updatedTextarea = screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)"
      ) as HTMLTextAreaElement
      expect(updatedTextarea.selectionStart).toBeGreaterThan(10)
      expect(updatedTextarea.selectionStart).toBe(updatedTextarea.selectionEnd)
    })
  })

  it("filters title and literal TOC headings without changing later heading offsets", async () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Guide" }
    })
    const source = "# Guide\n# Table of contents\n## Section\n## Section\n## End"
    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: source } })
    expect(screen.queryByTestId("notes-toc-item-guide")).not.toBeInTheDocument()
    expect(
      screen.queryByTestId("notes-toc-item-table-of-contents")
    ).not.toBeInTheDocument()
    const focus = vi.spyOn(textarea, "focus")
    fireEvent.click(await screen.findByTestId("notes-toc-item-section-2"))
    await waitFor(() =>
      expect(textarea.selectionStart).toBe(source.lastIndexOf("## Section"))
    )
    expect(focus).toHaveBeenCalledWith({ preventScroll: true })
    expect(textarea.value).toBe(source)
  })

  it("collapses a long outline and remembers the deliberate disclosure choice", async () => {
    const view = renderPage()
    const source = Array.from(
      { length: 7 },
      (_, index) => `## Part ${index}`
    ).join("\n")
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      { target: { value: source } }
    )
    const toggle = await screen.findByTestId("notes-toc-panel-toggle")
    expect(toggle).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByTestId("notes-toc-panel-badge")).toHaveTextContent("7")
    expect(toggle.getAttribute("aria-controls")).toBeTruthy()
    fireEvent.click(toggle)
    expect(await screen.findByTestId("notes-toc-item-part-6")).toBeVisible()
    expect(localStorage.getItem("notes-section-toc")).toBe("true")
    view.unmount()
    renderPage()
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      { target: { value: source } }
    )
    expect(await screen.findByTestId("notes-toc-panel-toggle")).toHaveAttribute(
      "aria-expanded",
      "true"
    )
  })

  it("does not show a TOC below three meaningful headings", () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Guide" }
    })
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      {
        target: { value: "# Guide\n## Table of contents\n## One\n## Two" }
      }
    )
    expect(screen.queryByTestId("notes-toc-panel")).not.toBeInTheDocument()
  })

  it.each(["markdown", "wysiwyg"])(
    "mounts the %s editor before a Preview TOC jump",
    async (mode) => {
      renderPage()
      const source = "# Intro\n\n## Middle\nbody\n\n## End\nlast"
      fireEvent.change(
        screen.getByPlaceholderText(
          "Write your note here... (Markdown supported)"
        ),
        { target: { value: source } }
      )
      if (mode === "wysiwyg")
        fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
      fireEvent.click(
        screen.getByRole("button", { name: "Preview", exact: true })
      )
      fireEvent.click(screen.getByTestId("notes-toc-item-end"))
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "Split", exact: true })
        ).toHaveAttribute("aria-pressed", "true")
      )
      await waitFor(() => {
        if (mode === "wysiwyg") {
          const editor = screen.getByTestId("notes-wysiwyg-editor")
          expect(editor).toHaveFocus()
          expect(window.getSelection()?.anchorNode).toBe(
            editor.querySelectorAll("h2")[1]
          )
        } else {
          const editor = screen.getByPlaceholderText(
            "Write your note here... (Markdown supported)"
          ) as HTMLTextAreaElement
          expect(editor.selectionStart).toBe(source.indexOf("## End"))
        }
      })
    }
  )

  it("records a real rich task toggle and another body edit before switching back", async () => {
    renderPage()
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      {
        target: { value: "3. ordered\n4. next\n\n- [ ] task" }
      }
    )
    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const editor = await screen.findByTestId("notes-wysiwyg-editor")
    fireEvent.click(editor.querySelector("input")!)
    const ordered = editor.querySelector("li")!
    ordered.append(document.createTextNode(" edited"))
    fireEvent.input(editor)
    fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))
    await waitFor(() =>
      expect(
        screen.getByPlaceholderText(
          "Write your note here... (Markdown supported)"
        )
      ).toHaveValue("3. ordered edited\n4. next\n\n- [x] task")
    )
  })

  it("connects rich task Enter to dirty Markdown without rewriting neighbouring nodes", async () => {
    renderPage()
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      {
        target: { value: "- [x] first\n- neighbour" }
      }
    )
    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const editor = await screen.findByTestId("notes-wysiwyg-editor")
    const item = editor.querySelector("li")!
    const neighbour = editor.querySelectorAll("li")[1]
    const text = item.lastChild!
    const range = document.createRange()
    range.setStart(text, text.textContent!.length)
    range.collapse(true)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    fireEvent.keyDown(editor, { key: "Enter" })
    expect(editor.querySelectorAll("li")).toHaveLength(3)
    expect(editor.querySelectorAll("li")[2]).toBe(neighbour)
    fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))
    await waitFor(() =>
      expect(
        screen.getByPlaceholderText(
          "Write your note here... (Markdown supported)"
        )
      ).toHaveValue("- [x] first\n- [ ]\n- neighbour")
    )
  })

  it("maps Split scroll ranges in both directions without changing content", async () => {
    renderPage()
    const source = "# Intro\nbody\n## Middle\nbody\n## End\nbody"
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      { target: { value: source } }
    )
    fireEvent.click(screen.getByRole("button", { name: "Split", exact: true }))
    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    const preview = await screen.findByTestId("notes-split-preview-viewport")
    Object.defineProperties(textarea, {
      scrollHeight: { configurable: true, value: 1000 },
      clientHeight: { configurable: true, value: 200 }
    })
    Object.defineProperties(preview, {
      scrollHeight: { configurable: true, value: 1800 },
      clientHeight: { configurable: true, value: 200 }
    })
    textarea.scrollTop = 400
    fireEvent.scroll(textarea)
    expect(preview.scrollTop).toBe(800)
    fireEvent.scroll(preview)
    expect(textarea.scrollTop).toBe(400)
    preview.scrollTop = 1200
    fireEvent.scroll(preview)
    expect(textarea.scrollTop).toBe(600)
    Object.defineProperty(textarea, "scrollHeight", {
      configurable: true,
      value: 200
    })
    textarea.scrollTop = 0
    fireEvent.scroll(textarea)
    expect(preview.scrollTop).toBe(1200)
    expect(textarea.value).toBe(source)
  })

  it("converts WYSIWYG edits back to markdown with heading and inline formatting", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: "# Title\n\nParagraph" } })

    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const richEditor = await screen.findByTestId("notes-wysiwyg-editor")

    ;(richEditor as HTMLDivElement).innerHTML =
      '<h1 data-md-slug="title">Title</h1><p>Paragraph</p><h2 data-md-slug="new-section">New Section</h2><p>Added <strong>bold</strong></p>'
    fireEvent.input(richEditor)

    fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))

    await waitFor(() => {
      const markdownTextarea = screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)"
      ) as HTMLTextAreaElement
      expect(markdownTextarea.value).toContain("## New Section")
      expect(markdownTextarea.value).toContain("Added **bold**")
    })
  })

  it("keeps the collapsed text position when a native list command replaces heading nodes", async () => {
    renderPage()
    fireEvent.change(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
      {
        target: { value: "### First\n\nSibling" }
      }
    )
    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const editor = await screen.findByTestId("notes-wysiwyg-editor")
    const originalText = editor.querySelector("h3")!.firstChild!
    const sibling = editor.lastElementChild
    editor.focus()
    const range = document.createRange()
    range.setStart(originalText, 5)
    range.collapse(true)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    expect(window.getSelection()?.anchorOffset).toBe(5)
    const originalCommand = document.execCommand
    document.execCommand = vi.fn(
      (command: string, _showUI?: boolean, value?: string) => {
        let text: Node | null = null
        let offset = 5
        if (command === "formatBlock" && value === "<p>") {
          const heading = editor.querySelector("h3")!
          const paragraph = document.createElement("p")
          paragraph.textContent = heading.textContent
          heading.replaceWith(paragraph)
          text = paragraph.firstChild
        } else if (command === "insertOrderedList") {
          const paragraph = editor.firstElementChild!
          const list = document.createElement("ol")
          const item = document.createElement("li")
          item.textContent = paragraph.textContent
          list.append(item)
          paragraph.replaceWith(list)
          text = item.firstChild
          offset = 0
        }
        if (text) {
          const next = document.createRange()
          next.setStart(text, offset)
          next.collapse(true)
          window.getSelection()?.removeAllRanges()
          window.getSelection()?.addRange(next)
        }
        return true
      }
    )
    try {
      fireEvent.mouseDown(screen.getByTestId("notes-toolbar-numbered"))
      fireEvent.click(screen.getByTestId("notes-toolbar-numbered"))
      const text = editor.querySelector("ol > li")!.firstChild
      expect(text).not.toBe(originalText)
      expect(window.getSelection()?.anchorNode).toBe(text)
      expect(window.getSelection()?.anchorOffset).toBe(5)
      expect(editor.lastElementChild).toBe(sibling)
      fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))
      await waitFor(() =>
        expect(
          screen.getByPlaceholderText(
            "Write your note here... (Markdown supported)"
          )
        ).toHaveValue("1. First\n\nSibling")
      )
    } finally {
      document.execCommand = originalCommand
    }
  })

  it("keeps a native empty-list caret inside its item when its text offset stays zero", async () => {
    renderPage()
    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const editor = await screen.findByTestId("notes-wysiwyg-editor")
    editor.focus()
    const range = document.createRange()
    range.selectNodeContents(editor.firstElementChild!)
    range.collapse(true)
    window.getSelection()?.removeAllRanges()
    window.getSelection()?.addRange(range)
    const originalCommand = document.execCommand
    document.execCommand = vi.fn((command: string) => {
      if (command === "insertOrderedList") {
        const list = document.createElement("ol")
        const item = document.createElement("li")
        item.append(document.createElement("br"))
        list.append(item)
        editor.firstElementChild!.replaceWith(list)
        const next = document.createRange()
        next.setStart(item, 0)
        next.collapse(true)
        window.getSelection()?.removeAllRanges()
        window.getSelection()?.addRange(next)
      }
      return true
    })
    try {
      fireEvent.mouseDown(screen.getByTestId("notes-toolbar-numbered"))
      fireEvent.click(screen.getByTestId("notes-toolbar-numbered"))
      expect(window.getSelection()?.anchorNode).toBe(editor.querySelector("li"))
      expect(window.getSelection()?.anchorOffset).toBe(0)
    } finally {
      document.execCommand = originalCommand
    }
  })

  it("restores the WYSIWYG selection before creating a link after prompt focus steals it", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: "Link target here" } })

    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const richEditor = await screen.findByTestId("notes-wysiwyg-editor")
    ;(richEditor as HTMLDivElement).innerHTML = "<p>Link target here</p>"

    const textNode = richEditor.querySelector("p")?.firstChild
    expect(textNode?.nodeType).toBe(Node.TEXT_NODE)

    const selection = window.getSelection()
    const range = document.createRange()
    range.setStart(textNode as Text, 0)
    range.setEnd(textNode as Text, "Link target".length)
    selection?.removeAllRanges()
    selection?.addRange(range)

    const selectionSnapshots: string[] = []
    const docWithExec = document as Document & {
      execCommand?: ReturnType<typeof vi.fn>
    }
    const originalExecCommand = docWithExec.execCommand
    docWithExec.execCommand = vi.fn((command: string) => {
      if (command === "createLink") {
        selectionSnapshots.push(window.getSelection()?.toString() || "")
      }
      return true
    })

    mockPromptModal.mockImplementation(async () => {
      window.getSelection()?.removeAllRanges()
      return "https://example.com"
    })

    try {
      fireEvent.click(screen.getByTestId("notes-toolbar-link"))

      await waitFor(() => {
        expect(docWithExec.execCommand).toHaveBeenCalledWith(
          "createLink",
          false,
          "https://example.com"
        )
      })
      expect(selectionSnapshots[0]).toBe("Link target")
    } finally {
      docWithExec.execCommand = originalExecCommand
    }
  })
})
