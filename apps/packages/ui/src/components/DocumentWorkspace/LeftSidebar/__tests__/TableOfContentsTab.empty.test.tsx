import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { TableOfContentsTab } from "../TableOfContentsTab"

const state = vi.hoisted(() => ({
  activeDocumentId: 1,
  activeDocumentType: "pdf",
  currentPage: 1,
  setCurrentPage: vi.fn()
}))
const outline = vi.hoisted(() => ({
  data: { documentId: 1, items: [] as Array<{ title: string; page: number }> },
  isLoading: false,
  error: null as Error | null
}))
vi.mock("@/store/document-workspace", () => ({
  useDocumentWorkspaceStore: (select: (value: typeof state) => unknown) => select(state)
}))
vi.mock("@/hooks/document-workspace", () => ({ usePdfOutline: () => outline }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))

describe("empty document Contents", () => {
  beforeEach(() => {
    outline.data.items = []
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
})
