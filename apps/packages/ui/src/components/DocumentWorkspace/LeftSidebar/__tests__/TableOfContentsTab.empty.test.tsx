import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { TocItem } from "../../types"
import { TableOfContentsTab } from "../TableOfContentsTab"

const state = vi.hoisted(() => ({
  activeDocumentId: 1,
  activeDocumentType: "pdf",
  currentPage: 1,
  setCurrentPage: vi.fn()
}))
const outline = vi.hoisted(() => ({
  data: { documentId: 1, source: "embedded", partial: false, items: [] as Partial<TocItem>[] },
  resolvePage: vi.fn(),
  isCurrent: () => true,
  isLoading: false,
  error: null as Error | null
}))
vi.mock("@/store/document-workspace", () => ({
  useDocumentWorkspaceStore: Object.assign((select: (value: typeof state) => unknown) => select(state), { getState: () => state })
}))
vi.mock("@/hooks/document-workspace", () => ({ usePdfOutline: () => outline }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))

describe("empty document Contents", () => {
  beforeEach(() => {
    state.activeDocumentId = 1
    state.currentPage = 1
    outline.data.documentId = 1
    outline.resolvePage.mockReset()
    outline.data.items = []
    outline.data.source = "embedded"
    outline.data.partial = false
    outline.isLoading = false
    outline.error = null
    vi.clearAllMocks()
  })

  it("offers the existing hide action with supported foreground tokens", () => {
    const onHide = vi.fn()
    render(<TableOfContentsTab onHide={onHide} />)
    expect(screen.getByText("No table of contents")).toHaveClass("text-text")
    expect(screen.getByText("This document doesn't have an embedded outline")).toHaveClass("text-text-muted")
    fireEvent.click(screen.getByRole("button", { name: "Hide contents" }))
    expect(onHide).toHaveBeenCalledOnce()
  })

  it.each(["loading", "error", "populated"])("does not mislabel %s Contents as empty", (mode) => {
    if (mode === "loading") outline.isLoading = true
    if (mode === "error") outline.error = new Error("Unavailable")
    if (mode === "populated") outline.data.items = [{ title: "Chapter one", page: 3 }]
    render(<TableOfContentsTab onHide={vi.fn()} />)
    expect(screen.queryByRole("button", { name: "Hide contents" })).not.toBeInTheDocument()
    expect(screen.queryByText("No table of contents")).not.toBeInTheDocument()
    if (mode === "populated") {
      fireEvent.click(screen.getByText("Chapter one"))
      expect(state.setCurrentPage).toHaveBeenCalledWith(3)
    }
  })

  it("resolves a generated entry lazily and never highlights its printed target as a physical page", async () => {
    let finish!: (page: number) => void
    outline.resolvePage.mockReturnValue(new Promise<number>(resolve => { finish = resolve }))
    outline.data.source = "generated"
    outline.data.partial = true
    outline.data.items = [{ title: "Education (printed 319)", page: 0, navigationNodeId: "education" }]
    state.currentPage = 319
    render(<TableOfContentsTab />)
    expect(screen.getByText("Generated navigation")).toBeInTheDocument()
    expect(screen.getByText("Some navigation entries are unavailable.")).toBeInTheDocument()
    expect(screen.getByRole("treeitem")).not.toHaveAttribute("aria-current")
    fireEvent.click(screen.getByText("Education (printed 319)"))
    expect(state.setCurrentPage).not.toHaveBeenCalled()
    expect(screen.getByText("Finding PDF page…")).toBeInTheDocument()
    await act(async () => finish(352))
    expect(state.setCurrentPage).toHaveBeenCalledWith(352)
    expect(screen.queryByText("Finding PDF page…")).not.toBeInTheDocument()
  })

  it("keeps an unresolved generated destination visible and unavailable without jumping", async () => {
    outline.resolvePage.mockRejectedValue(new Error("Unresolved"))
    outline.data.source = "generated"
    outline.data.items = [{ title: "Afterword", page: 0, navigationNodeId: "afterword" }]
    render(<TableOfContentsTab />)
    const entry = screen.getByRole("treeitem")
    fireEvent.keyDown(entry, { key: "Enter" })
    await waitFor(() => expect(screen.getByText("Page unavailable")).toBeInTheDocument())
    expect(screen.getByRole("button", { name: /Afterword/ })).toHaveAttribute("aria-disabled", "true")
    fireEvent.keyDown(entry, { key: "Enter" })
    expect(outline.resolvePage).toHaveBeenCalledOnce()
    expect(state.setCurrentPage).not.toHaveBeenCalled()
  })

  it("aborts a selected section and ignores its late physical page after switching documents", async () => {
    let finish!: (page: number) => void
    outline.resolvePage.mockReturnValue(new Promise<number>(resolve => { finish = resolve }))
    outline.data.source = "generated"
    outline.data.items = [{ title: "Old section", page: 0, navigationNodeId: "old" }]
    const view = render(<TableOfContentsTab />)
    fireEvent.click(screen.getByText("Old section"))
    const signal = outline.resolvePage.mock.calls[0][1]
    state.activeDocumentId = 2
    outline.data = { ...outline.data, documentId: 2, items: [{ title: "New section", page: 0, navigationNodeId: "new" }] }
    view.rerender(<TableOfContentsTab />)
    expect(signal.aborted).toBe(true)
    await act(async () => finish(352))
    expect(state.setCurrentPage).not.toHaveBeenCalled()
    expect(screen.getByText("New section")).toBeInTheDocument()
    expect(screen.queryByText("Page unavailable")).not.toBeInTheDocument()
  })

  it("explains an empty generated map and retains the Hide contents action", () => {
    outline.data.source = "generated"
    render(<TableOfContentsTab onHide={vi.fn()} />)
    expect(screen.getByText("No verified page navigation is available.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Hide contents" })).toBeInTheDocument()
  })

  it("retains tree keyboard navigation and activates only the focused child", () => {
    outline.data.source = "generated"
    outline.data.items = [{ title: "Parent", page: 1, children: [{ title: "Child", page: 4, level: 2 }] }]
    render(<TableOfContentsTab />)
    const [parent, child] = screen.getAllByRole("treeitem")
    parent.focus()
    fireEvent.keyDown(parent, { key: "ArrowRight" })
    expect(child).toHaveFocus()
    expect(child).toHaveAttribute("tabindex", "0")
    fireEvent.keyDown(child, { key: "Enter" })
    expect(state.setCurrentPage.mock.calls).toEqual([[4]])
    fireEvent.keyDown(child, { key: "ArrowLeft" })
    expect(parent).toHaveFocus()
    fireEvent.keyDown(parent, { key: "ArrowLeft" })
    expect(screen.getAllByRole("treeitem")).toHaveLength(1)
  })
})
