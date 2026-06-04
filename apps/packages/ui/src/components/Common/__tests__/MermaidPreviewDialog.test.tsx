import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MermaidPreviewDialog } from "../MermaidPreviewDialog"

vi.mock("antd", () => ({
  Modal: ({ children, open, title }: any) =>
    open ? (
      <div aria-label={title} role="dialog">
        <h2>{title}</h2>
        {children}
      </div>
    ) : null,
  Tooltip: ({ children }: any) => <>{children}</>
}))

describe("MermaidPreviewDialog", () => {
  const source = "sequenceDiagram\n  Alice->>Bob: hello"
  const generatedSvg =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40"><text>Rendered diagram</text></svg>'
  let writeText: ReturnType<typeof vi.fn>
  let createObjectURL: ReturnType<typeof vi.fn>
  let revokeObjectURL: ReturnType<typeof vi.fn>
  let clickSpy: ReturnType<typeof vi.spyOn>
  let blobParts: BlobPart[] | undefined

  beforeEach(() => {
    writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    })

    createObjectURL = vi.fn(() => "blob:mermaid-preview")
    revokeObjectURL = vi.fn()
    vi.stubGlobal("URL", {
      ...URL,
      createObjectURL,
      revokeObjectURL
    })

    blobParts = undefined
    vi.stubGlobal(
      "Blob",
      vi.fn(function BlobMock(this: Blob, parts: BlobPart[]) {
        blobParts = parts
      }) as unknown as typeof Blob
    )
    clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("renders generated SVG content in the preview viewport", () => {
    render(
      <MermaidPreviewDialog
        generatedSvg={generatedSvg}
        onClose={vi.fn()}
        open
        source={source}
      />
    )

    expect(
      screen.getByRole("dialog", { name: "Mermaid diagram preview" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("mermaid-preview-canvas").innerHTML).toContain(
      "Rendered diagram"
    )
  })

  it("updates the zoom label and transform, then resets them", () => {
    render(
      <MermaidPreviewDialog
        generatedSvg={generatedSvg}
        onClose={vi.fn()}
        open
        source={source}
      />
    )

    const canvas = screen.getByTestId("mermaid-preview-canvas")

    expect(screen.getByText("100%")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }))

    expect(screen.getByText("125%")).toBeInTheDocument()
    expect(canvas).toHaveStyle("transform: translate(0px, 0px) scale(1.25)")

    fireEvent.click(screen.getByRole("button", { name: "Reset zoom and pan" }))

    expect(screen.getByText("100%")).toBeInTheDocument()
    expect(canvas).toHaveStyle("transform: translate(0px, 0px) scale(1)")
  })

  it("changes pan translation when the viewport is dragged", () => {
    render(
      <MermaidPreviewDialog
        generatedSvg={generatedSvg}
        onClose={vi.fn()}
        open
        source={source}
      />
    )

    const viewport = screen.getByLabelText("Mermaid diagram viewport")
    const canvas = screen.getByTestId("mermaid-preview-canvas")

    fireEvent.pointerDown(viewport, { clientX: 10, clientY: 15, pointerId: 1 })
    fireEvent.pointerMove(viewport, { clientX: 32, clientY: 44, pointerId: 1 })
    fireEvent.pointerUp(viewport, { pointerId: 1 })

    expect(canvas).toHaveStyle("transform: translate(22px, 29px) scale(1)")
  })

  it("copies Mermaid source instead of generated SVG", async () => {
    render(
      <MermaidPreviewDialog
        generatedSvg={generatedSvg}
        onClose={vi.fn()}
        open
        source={source}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Copy Mermaid source" }))

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith(source)
    })
    expect(writeText).not.toHaveBeenCalledWith(generatedSvg)
  })

  it("downloads the generated SVG", () => {
    render(
      <MermaidPreviewDialog
        generatedSvg={generatedSvg}
        onClose={vi.fn()}
        open
        source={source}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Download Mermaid SVG" }))

    expect(blobParts).toEqual([generatedSvg])
    expect(createObjectURL).toHaveBeenCalled()
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mermaid-preview")
    expect(clickSpy).toHaveBeenCalled()
  })

  it("calls onClose from the close control", () => {
    const onClose = vi.fn()
    render(
      <MermaidPreviewDialog
        generatedSvg={generatedSvg}
        onClose={onClose}
        open
        source={source}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Close Mermaid preview" })
    )

    expect(onClose).toHaveBeenCalled()
  })

  it("shows raw source fallback and omits SVG download without generated SVG", () => {
    render(
      <MermaidPreviewDialog
        generatedSvg={undefined}
        onClose={vi.fn()}
        open
        source={source}
      />
    )

    expect(
      screen.getByText(
        (_, element) =>
          element?.tagName === "PRE" && element.textContent === source
      )
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Download Mermaid SVG" })
    ).not.toBeInTheDocument()
  })
})
