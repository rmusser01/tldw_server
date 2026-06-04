import React from "react"
import { render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const { initializeMock, renderMock } = vi.hoisted(() => ({
  initializeMock: vi.fn(),
  renderMock: vi.fn()
}))

vi.mock("mermaid", () => ({
  default: {
    initialize: initializeMock,
    render: renderMock
  }
}))

import Mermaid from "../Mermaid"

const code = "graph TD; A-->B"
const svg = "<svg><text>Rendered diagram</text></svg>"

describe("Mermaid", () => {
  beforeEach(() => {
    renderMock.mockResolvedValue({ svg })
    document.documentElement.className = ""
    window.matchMedia = vi.fn().mockReturnValue({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn()
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    initializeMock.mockReset()
    renderMock.mockReset()
  })

  it("reports generated SVG and initializes Mermaid with strict security", async () => {
    const onRenderStateChange = vi.fn()

    const { container } = render(
      <Mermaid code={code} onRenderStateChange={onRenderStateChange} />
    )

    await waitFor(() => {
      expect(renderMock).toHaveBeenCalledWith(
        expect.stringMatching(/^mermaid-/),
        code
      )
    })

    expect(initializeMock).toHaveBeenCalledWith({
      startOnLoad: false,
      theme: "default",
      securityLevel: "strict"
    })

    await waitFor(() => {
      expect(onRenderStateChange).toHaveBeenCalledWith({
        status: "success",
        svg,
        theme: "default"
      })
    })

    expect(onRenderStateChange).toHaveBeenCalledWith({
      status: "rendering",
      theme: "default"
    })
    expect(container.querySelector("svg")?.textContent).toBe("Rendered diagram")
  })

  it("sanitizes generated SVG before rendering and reporting it", async () => {
    const unsafeSvg =
      '<svg><script>alert("xss")</script><text onclick="alert(1)">Rendered diagram</text></svg>'
    renderMock.mockResolvedValueOnce({ svg: unsafeSvg })
    const onRenderStateChange = vi.fn()

    const { container } = render(
      <Mermaid code={code} onRenderStateChange={onRenderStateChange} />
    )

    await waitFor(() => {
      expect(container.querySelector("svg")?.textContent).toBe(
        "Rendered diagram"
      )
    })

    expect(container.querySelector("script")).not.toBeInTheDocument()
    expect(container.querySelector("text")?.getAttribute("onclick")).toBeNull()
    expect(onRenderStateChange).toHaveBeenCalledWith({
      status: "success",
      svg: "<svg><text>Rendered diagram</text></svg>",
      theme: "default"
    })
  })

  it("reports render failures without throwing", async () => {
    renderMock.mockRejectedValueOnce(new Error("Invalid Mermaid graph"))
    const onRenderStateChange = vi.fn()

    const { findByText } = render(
      <Mermaid code={code} onRenderStateChange={onRenderStateChange} />
    )

    expect(await findByText("Invalid Mermaid graph")).toBeInTheDocument()
    expect(onRenderStateChange).toHaveBeenCalledWith({
      status: "error",
      error: "Invalid Mermaid graph",
      theme: "default"
    })
  })

  it("ignores stale render completions after code changes", async () => {
    let resolveFirstRender:
      | ((value: { svg: string }) => void)
      | undefined
    renderMock
      .mockImplementationOnce(
        () =>
          new Promise<{ svg: string }>((resolve) => {
            resolveFirstRender = resolve
          })
      )
      .mockResolvedValueOnce({
        svg: "<svg><text>Fresh diagram</text></svg>"
      })
    const onRenderStateChange = vi.fn()

    const { container, rerender } = render(
      <Mermaid
        code="graph TD; Old-->Diagram"
        onRenderStateChange={onRenderStateChange}
      />
    )

    await waitFor(() => {
      expect(renderMock).toHaveBeenCalledTimes(1)
    })

    rerender(
      <Mermaid
        code="graph TD; New-->Diagram"
        onRenderStateChange={onRenderStateChange}
      />
    )

    await waitFor(() => {
      expect(container.querySelector("svg")?.textContent).toBe("Fresh diagram")
    })

    resolveFirstRender?.({
      svg: "<svg><text>Stale diagram</text></svg>"
    })

    await waitFor(() => {
      expect(container.querySelector("svg")?.textContent).toBe("Fresh diagram")
    })
    expect(onRenderStateChange).not.toHaveBeenCalledWith({
      status: "success",
      svg: "<svg><text>Stale diagram</text></svg>",
      theme: "default"
    })
  })
})
