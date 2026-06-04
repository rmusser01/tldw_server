import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MermaidDiagramBlock } from "../MermaidDiagramBlock"
import type { MermaidRenderState } from "../Mermaid"

const mermaidMock = vi.hoisted(() => ({
  renderState: {
    status: "success",
    svg: '<svg xmlns="http://www.w3.org/2000/svg"><text>Generated</text></svg>'
  } as MermaidRenderState
}))

vi.mock("../Mermaid", async () => {
  const ReactModule = await import("react")

  const Mermaid = ({
    code,
    onRenderStateChange
  }: {
    code: string
    onRenderStateChange?: (state: MermaidRenderState) => void
  }) => {
    ReactModule.useEffect(() => {
      onRenderStateChange?.(mermaidMock.renderState)
    }, [code, onRenderStateChange])

    return ReactModule.createElement(
      "div",
      {
        "aria-label": "Mock Mermaid diagram",
        role: "img"
      },
      code
    )
  }

  return {
    Mermaid,
    default: Mermaid
  }
})

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

describe("MermaidDiagramBlock", () => {
  const source = "graph TD\n  A-->B"
  let writeText: ReturnType<typeof vi.fn>
  let createObjectURL: ReturnType<typeof vi.fn>
  let revokeObjectURL: ReturnType<typeof vi.fn>
  let clickSpy: ReturnType<typeof vi.spyOn>
  let blobParts: BlobPart[] | undefined

  beforeEach(() => {
    mermaidMock.renderState = {
      status: "success",
      svg: '<svg xmlns="http://www.w3.org/2000/svg"><text>Generated</text></svg>'
    }
    writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    })

    createObjectURL = vi.fn(() => "blob:mermaid")
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

  it("renders compact Mermaid block chrome with the renderer", async () => {
    render(<MermaidDiagramBlock blockIndex={3} source={source} />)

    expect(screen.getByText("mermaid")).toBeInTheDocument()
    expect(
      screen.getByRole("img", { name: "Mock Mermaid diagram" })
    ).toHaveTextContent("graph TD A-->B")

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Download Mermaid SVG" })
      ).toBeInTheDocument()
    })
  })

  it("copies the Mermaid source instead of generated SVG", async () => {
    render(<MermaidDiagramBlock source={source} />)

    fireEvent.click(screen.getByRole("button", { name: "Copy Mermaid source" }))

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith(source)
    })
    expect(writeText).not.toHaveBeenCalledWith(expect.stringContaining("<svg"))
  })

  it("does not show copied state when clipboard API is unavailable", async () => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: undefined
    })

    render(<MermaidDiagramBlock source={source} />)

    const copyButton = screen.getByRole("button", {
      name: "Copy Mermaid source"
    })
    fireEvent.click(copyButton)

    await act(async () => {
      await Promise.resolve()
    })
    expect(copyButton.querySelector(".text-success")).not.toBeInTheDocument()
  })

  it("does not show copied state when clipboard write fails", async () => {
    writeText.mockRejectedValueOnce(new Error("clipboard denied"))

    render(<MermaidDiagramBlock source={source} />)

    const copyButton = screen.getByRole("button", {
      name: "Copy Mermaid source"
    })
    fireEvent.click(copyButton)

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith(source)
    })
    expect(copyButton.querySelector(".text-success")).not.toBeInTheDocument()
  })

  it("downloads only the generated SVG after render success", async () => {
    render(<MermaidDiagramBlock source={source} />)

    const downloadButton = await screen.findByRole("button", {
      name: "Download Mermaid SVG"
    })
    fireEvent.click(downloadButton)

    expect(blobParts).toEqual([mermaidMock.renderState.svg])
    expect(blobParts?.join("")).not.toContain(source)
    expect(createObjectURL).toHaveBeenCalled()
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mermaid")
    expect(clickSpy).toHaveBeenCalled()
  })

  it("sanitizes unsafe generated SVG before inline download", async () => {
    mermaidMock.renderState = {
      status: "success",
      svg: '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><script>alert(1)</script><text>Safe inline diagram</text></svg>'
    }

    render(<MermaidDiagramBlock source={source} />)

    const downloadButton = await screen.findByRole("button", {
      name: "Download Mermaid SVG"
    })
    fireEvent.click(downloadButton)

    const downloadedSvg = blobParts?.join("")
    expect(downloadedSvg).toContain("Safe inline diagram")
    expect(downloadedSvg).not.toContain("<script")
    expect(downloadedSvg).not.toContain("onload")
  })

  it("opens the Mermaid preview dialog with the generated SVG", async () => {
    render(<MermaidDiagramBlock source={source} />)

    const previewButton = await screen.findByRole("button", {
      name: "Open Mermaid preview"
    })
    fireEvent.click(previewButton)

    expect(
      screen.getByRole("dialog", { name: "Mermaid diagram preview" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("mermaid-preview-canvas").innerHTML).toContain(
      "<svg"
    )
  })

  it("shows a raw source fallback when rendering fails", async () => {
    mermaidMock.renderState = {
      status: "error",
      error: "Parse error"
    }

    render(<MermaidDiagramBlock source={source} />)

    expect(
      await screen.findByText("Unable to render Mermaid diagram.")
    ).toBeInTheDocument()
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
