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
const artifactsStoreMock = vi.hoisted(() => ({
  openArtifact: vi.fn(),
  isPinned: false
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

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: () => artifactsStoreMock
}))

describe("MermaidDiagramBlock", () => {
  const source = "graph TD\n  A-->B"
  let writeText: ReturnType<typeof vi.fn>
  let createObjectURL: ReturnType<typeof vi.fn>
  let revokeObjectURL: ReturnType<typeof vi.fn>
  let clickSpy: ReturnType<typeof vi.spyOn>
  let blobParts: BlobPart[] | undefined

  beforeEach(() => {
    artifactsStoreMock.openArtifact.mockClear()
    artifactsStoreMock.isPinned = false
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

  it("uses the already sanitized generated SVG for inline download", async () => {
    mermaidMock.renderState = {
      status: "success",
      svg: '<svg xmlns="http://www.w3.org/2000/svg"><text>Safe inline diagram</text></svg>'
    }

    render(<MermaidDiagramBlock source={source} />)

    const downloadButton = await screen.findByRole("button", {
      name: "Download Mermaid SVG"
    })
    fireEvent.click(downloadButton)

    const downloadedSvg = blobParts?.join("")
    expect(downloadedSvg).toContain("Safe inline diagram")
    expect(downloadedSvg).toBe(mermaidMock.renderState.svg)
  })

  it("uses unique header ids even when block indexes repeat across messages", () => {
    render(
      <>
        <MermaidDiagramBlock blockIndex={0} source={source} />
        <MermaidDiagramBlock blockIndex={0} source="graph TD\n  C-->D" />
      </>
    )

    const labelledBlocks = Array.from(
      document.querySelectorAll("[aria-labelledby]")
    )
    const headerIds = labelledBlocks.map((block) =>
      block.getAttribute("aria-labelledby")
    )

    expect(headerIds).toHaveLength(2)
    expect(new Set(headerIds).size).toBe(2)
    headerIds.forEach((headerId) => {
      expect(headerId).toBeTruthy()
      expect(document.getElementById(headerId || "")).toBeInTheDocument()
    })
  })

  it("does not show the artifact action by default", () => {
    render(<MermaidDiagramBlock blockIndex={0} source={source} />)

    expect(
      screen.queryByRole("button", { name: "View Mermaid diagram" })
    ).not.toBeInTheDocument()
  })

  it("opens a diagram artifact when the artifact action is enabled", () => {
    render(
      <MermaidDiagramBlock
        artifactContextId="assistant-message-123"
        blockIndex={2}
        enableArtifactAction
        source={source}
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "View Mermaid diagram" })
    )

    expect(artifactsStoreMock.openArtifact).toHaveBeenCalledTimes(1)
    const artifact = artifactsStoreMock.openArtifact.mock.calls[0]?.[0]
    expect(artifact).toEqual(
      expect.objectContaining({
        content: source,
        kind: "diagram",
        language: "mermaid",
        lineCount: 2,
        title: "Mermaid diagram 3"
      })
    )
    expect(artifact.id).toContain("assistant-message-123")
    expect(
      document.getElementById(`artifact-origin-${artifact.id}`)
    ).toBeInTheDocument()
  })

  it("includes artifact context in repeated diagram origin ids", () => {
    render(
      <>
        <MermaidDiagramBlock
          artifactContextId="assistant-a"
          blockIndex={0}
          enableArtifactAction
          source={source}
        />
        <MermaidDiagramBlock
          artifactContextId="assistant-b"
          blockIndex={0}
          enableArtifactAction
          source={source}
        />
      </>
    )

    const origins = Array.from(document.querySelectorAll("[data-artifact-origin]"))
    const originIds = origins.map((origin) =>
      origin.getAttribute("data-artifact-origin")
    )

    expect(originIds).toHaveLength(2)
    expect(new Set(originIds).size).toBe(2)
    expect(originIds[0]).toContain("assistant-a")
    expect(originIds[1]).toContain("assistant-b")
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
