import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { EpubViewer } from "../EpubViewer"
import { PdfDocument } from "../PdfViewer/PdfDocument"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@/store/document-workspace", () => {
  const state = {
    currentPage: 1,
    setCurrentPage: vi.fn(),
    setTotalPages: vi.fn(),
    currentCfi: null,
    setCurrentCfi: vi.fn(),
    setCurrentPercentage: vi.fn(),
    setCurrentChapterTitle: vi.fn(),
    annotations: [],
    epubTheme: "light",
    epubScrollMode: "paginated",
    epubSpreadMode: "none",
    epubFontSize: 100,
    epubFontFamily: "serif",
    epubLineHeight: 1.5
  }

  const useDocumentWorkspaceStore = (selector: (store: typeof state) => unknown) =>
    selector(state)
  useDocumentWorkspaceStore.getState = () => state

  return { useDocumentWorkspaceStore }
})

vi.mock("@/hooks/document-workspace/useEpubSettings", () => ({
  EPUB_THEMES: {
    light: { body: {} }
  },
  FONT_FAMILY_CSS: {
    serif: "serif"
  }
}))

vi.mock("../TextSelectionPopover", () => ({
  TextSelectionPopover: () => null
}))

vi.mock("../EpubViewer/EpubSearch", () => ({
  EpubSearch: () => null
}))

vi.mock("epubjs", () => ({
  default: () => ({
    ready: Promise.reject(new Error("EPUB load failed")),
    destroy: vi.fn()
  })
}))

vi.mock("react-pdf", () => ({
  pdfjs: {
    version: "4.10.38",
    GlobalWorkerOptions: {
      workerSrc: ""
    }
  },
  Document: ({ error }: { error?: React.ReactNode }) => (
    <div data-testid="pdf-document">{error}</div>
  )
}))

vi.mock("../PdfViewer/PdfPage", () => ({
  PdfPage: () => null
}))

vi.mock("@/hooks/document-workspace/useTextSelection", () => ({
  useTextSelection: () => ({
    selection: null,
    clearSelection: vi.fn()
  })
}))

vi.mock("@/utils/browser-runtime", () => ({
  getBrowserRuntime: () => null,
  isExtensionRuntime: () => false
}))

vi.mock("antd", () => ({
  Spin: () => <div>Loading</div>
}))

describe("document viewer design-system alerts", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders the EPUB missing-url state through the design-system Alert", () => {
    const { container } = render(<EpubViewer url="" documentId={7} />)

    const alert = screen.getByRole("alert")
    expect(alert).toHaveTextContent("No document URL")
    expect(alert).toHaveTextContent("Please select a document to view")
    expect(alert.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })

  it("renders EPUB load errors through the design-system Alert", async () => {
    const { container } = render(<EpubViewer url="/broken.epub" documentId={7} />)

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Failed to load EPUB")
    expect(alert).toHaveTextContent("EPUB load failed")
    expect(alert.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })

  it("renders the PDF missing-url state through the design-system Alert", () => {
    const { container } = render(
      <PdfDocument
        documentId={11}
        currentPage={1}
        zoomLevel={100}
        viewMode="single"
        onLoadSuccess={vi.fn()}
        onLoadError={vi.fn()}
        onPageChange={vi.fn()}
      />
    )

    const alert = screen.getByRole("alert")
    expect(alert).toHaveTextContent("No document URL")
    expect(alert).toHaveTextContent("Please select a document to view")
    expect(alert.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })

  it("provides PDF load errors through the design-system Alert", async () => {
    const { container } = render(
      <PdfDocument
        url="/broken.pdf"
        documentId={11}
        currentPage={1}
        zoomLevel={100}
        viewMode="single"
        onLoadSuccess={vi.fn()}
        onLoadError={vi.fn()}
        onPageChange={vi.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Failed to load PDF")
    })
    const alert = screen.getByRole("alert")
    expect(alert).toHaveTextContent("An error occurred while loading the document")
    expect(alert.closest('[data-ds-component="Alert"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
  })
})
