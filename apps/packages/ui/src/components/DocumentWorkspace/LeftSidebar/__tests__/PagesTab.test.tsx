import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { PagesTab } from "../PagesTab"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValue?: string,
      values?: Record<string, string | number | undefined>
    ) => {
      const template = defaultValue || _key
      if (!values) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token: string) => {
        const value = values[token]
        return value === undefined ? `{{${token}}}` : String(value)
      })
    }
  })
}))

vi.mock("react-pdf", () => ({
  Document: ({
    children,
    onLoadError
  }: {
    children?: React.ReactNode
    onLoadError?: (error: Error) => void
  }) => {
    React.useEffect(() => {
      const timeoutId = setTimeout(() => {
        onLoadError?.(new Error("Failed to parse PDF"))
      }, 0)

      return () => clearTimeout(timeoutId)
    }, [onLoadError])

    return <div data-testid="mock-document">{children}</div>
  },
  Page: ({
    pageNumber,
    onLoadSuccess,
    onRenderSuccess
  }: {
    pageNumber: number
    onLoadSuccess?: (page: typeof pdfPage) => void
    onRenderSuccess?: (page: typeof pdfPage) => void
  }) => {
    React.useEffect(() => {
      onLoadSuccess?.(pdfPage)
      onRenderSuccess?.(pdfPage)
      // eslint-disable-next-line react-hooks/exhaustive-deps -- react-pdf reads its callbacks at call time
    }, [])
    return <div data-testid={`mock-page-${pageNumber}`} />
  }
}))

const pdfPage = vi.hoisted(() => ({ cleanup: vi.fn(() => true) }))

describe("PagesTab", () => {
  beforeEach(() => {
    useDocumentWorkspaceStore.getState().reset()
    useDocumentWorkspaceStore.setState({
      activeDocumentId: 42,
      activeDocumentType: "pdf",
      openDocuments: [
        {
          id: 42,
          title: "Research paper",
          type: "pdf",
          url: "blob://paper.pdf"
        }
      ],
      totalPages: 1
    })
  })

  afterEach(() => {
    cleanup()
    useDocumentWorkspaceStore.getState().reset()
    vi.clearAllMocks()
  })

  it("renders PDF load errors with the design-system Alert", async () => {
    render(<PagesTab />)

    const message = await screen.findByText("Failed to parse PDF")

    expect(
      message.closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.getByText("Failed to load document")).toBeInTheDocument()
  })

  it("releases pdf.js render data for a thumbnail once painted and again on unmount", async () => {
    const view = render(<PagesTab />)
    await screen.findByTestId("mock-page-1")

    expect(pdfPage.cleanup).toHaveBeenCalledTimes(1)
    view.unmount()
    expect(pdfPage.cleanup).toHaveBeenCalledTimes(2)
  })
})
